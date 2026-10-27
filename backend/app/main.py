# -*- coding: utf-8 -*-
"""FastAPI 后端入口：把 LangGraph 智能体封装为 REST 接口。

对应开发前文件 §3.6-L2 后端承载（FastAPI）+ §3.7 混合路线。
接口：
- POST /api/evaluate      提交一次试讲（meta + multimodal）→ 返回完整评价报告
- GET  /api/users/{uid}/history  读取用户历史试讲
- GET  /api/traces        列出轨迹文件
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app.graph import graph
from app.memory import memory_store
from app.tools import perceive
from app.trace import trace_logger
import uuid, shutil
from pathlib import Path

UPLOAD_DIR = settings.BASE_DIR / "datasets" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="师范生试讲多模态智能评价 Agent", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class EvaluateRequest(BaseModel):
    user_id: str = "anonymous"
    subject: str
    lesson: str
    lesson_type: str = "新授课"
    grade_level: str = ""
    duration_seconds: int = 600
    recorded_at: str = ""
    asr_text: str = ""
    board_text: str = ""
    timeline: list = Field(default_factory=list)
    # 多模态扩展（阶段3）：posture / expression / audio_feats
    posture: dict | None = None
    expression: dict | None = None
    audio_feats: dict | None = None


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    meta = {
        "user_id": req.user_id,
        "subject": req.subject,
        "lesson": req.lesson,
        "lesson_type": req.lesson_type,
        "grade_level": req.grade_level,
        "duration_seconds": req.duration_seconds,
        "recorded_at": req.recorded_at if hasattr(req, "recorded_at") else None,
    }
    multimodal = {}
    if req.asr_text:
        multimodal["asr_text"] = req.asr_text
    if req.board_text:
        multimodal["board_text"] = req.board_text
    if req.timeline:
        multimodal["timeline"] = req.timeline
    for k in ("posture", "expression", "audio_feats"):
        v = getattr(req, k, None)
        if v:
            multimodal[k] = v
    try:
        result = graph.run_agent(meta, multimodal, user_id=req.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评价失败: {e}")
    return result.get("final_report", {})


@app.post("/api/evaluate_video")
async def evaluate_video(
    video: UploadFile = File(...),
    user_id: str = Form("anonymous"),
    subject: str = Form("数学"),
    lesson: str = Form(""),
    lesson_type: str = Form("新授课"),
    grade_level: str = Form(""),
    duration_seconds: int = Form(600),
) -> dict:
    """上传试讲视频 → 自动跑 ASR/OCR/姿态/时间轴感知 → 送入 Agent 评价。"""
    # 1. 保存视频
    ext = Path(video.filename or "video.mp4").suffix or ".mp4"
    trial_id = f"trial_{uuid.uuid4().hex[:10]}"
    video_path = UPLOAD_DIR / f"{trial_id}{ext}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # 2. 多模态感知（ASR+OCR+姿态+时间轴）
    try:
        sensing = perceive.perceive_video(str(video_path), trial_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"多模态感知失败: {e}")

    # 3. 送入 Agent 评价
    meta = {
        "user_id": user_id, "subject": subject, "lesson": lesson,
        "lesson_type": lesson_type, "grade_level": grade_level,
        "duration_seconds": duration_seconds,
    }
    try:
        result = graph.run_agent(meta, sensing["multimodal"], user_id=user_id, try_id=trial_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评价失败: {e}")

    out = result.get("final_report", {})
    out["sensing_steps"] = sensing["steps"]          # 感知工具执行记录
    out["sensing_elapsed"] = sensing["elapsed_sec"]   # 感知耗时
    out["trial_id"] = trial_id
    return out


@app.get("/api/users/{user_id}/history")
def history(user_id: str, limit: int = 20) -> dict:
    memory_store.init_db()
    trials = memory_store.get_user_trials(user_id, limit=limit)
    return {"user_id": user_id, "trials": trials, "count": len(trials)}


@app.get("/api/users/{user_id}/growth")
def growth(user_id: str, limit: int = 20) -> dict:
    """P0：个人成长时间线（成长曲线 + 短板热力图数据）。"""
    memory_store.init_db()
    return memory_store.get_growth_timeline(user_id, limit=limit)


# ---------------- 知识库管理（P1） ----------------
@app.get("/api/kb/documents")
def kb_documents() -> dict:
    """列出知识库文档（从 Chroma 聚合）。"""
    import chromadb
    client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
    col = client.get_collection(settings.COLLECTION)
    data = col.get(include=["metadatas"])
    files = {}
    for m in data["metadatas"]:
        f = m.get("source_file", "?")
        if f not in files:
            files[f] = {"name": f, "type": m.get("doc_type", "?"),
                        "subject": m.get("subject", "?"), "chunks": 0}
        files[f]["chunks"] += 1
    return {"documents": list(files.values()), "total": len(files)}


@app.post("/api/kb/search")
def kb_search(body: dict) -> dict:
    """检索测试。"""
    from app.rag.retriever import retrieve
    q = body.get("query", "")
    if not q:
        return {"results": []}
    hits = retrieve(q, top_k=body.get("top_k", 5))
    return {"results": hits}


@app.post("/api/kb/upload")
async def kb_upload(file: UploadFile):
    """上传文件入库。"""
    import shutil
    save_dir = Path("datasets/knowledge_base/uploaded")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "filename": file.filename, "msg": "文件已保存，请运行入库脚本更新知识库"}


@app.delete("/api/kb/documents/{source_file:path}")
def kb_delete(source_file: str):
    """删除指定文档（从 Chroma 中删除该 source_file 的所有片段）。"""
    import chromadb
    client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
    col = client.get_collection(settings.COLLECTION)
    col.delete(where={"source_file": source_file})
    return {"ok": True, "deleted": source_file}


# ---------------- 老师看板（P2） ----------------
@app.get("/api/dashboard/class")
def dashboard_class() -> dict:
    """班级聚合数据（从记忆表聚合）。"""
    memory_store.init_db()
    conn = memory_store._connect()
    rows = conn.execute(
        "SELECT user_id, scores_json, overall_percent FROM evaluation ORDER BY created_at"
    ).fetchall()
    conn.close()
    import json
    users = {}
    for r in rows:
        uid = r["user_id"]
        if uid not in users:
            users[uid] = {"scores": [], "dims": {}}
        try:
            s = json.loads(r["scores_json"])
            users[uid]["scores"].append(r["overall_percent"])
            for d in ("D1", "D2", "D3", "D4"):
                users[uid]["dims"].setdefault(d, []).append(s.get(d, {}).get("total", 0))
        except:
            pass
    # 聚合
    all_scores = [u["scores"][-1] for u in users.values() if u["scores"]]
    dim_avg = {}
    for d in ("D1", "D2", "D3", "D4"):
        vals = [v for u in users.values() for v in u["dims"].get(d, [])]
        dim_avg[d] = round(sum(vals)/len(vals), 2) if vals else 0
    students = [{"user_id": uid, "last_score": u["scores"][-1] if u["scores"] else 0,
                 "avg_score": round(sum(u["scores"])/len(u["scores"]), 1) if u["scores"] else 0}
                for uid, u in users.items()]
    return {
        "student_count": len(users),
        "class_avg": round(sum(all_scores)/len(all_scores), 1) if all_scores else 0,
        "dim_avg": dim_avg,
        "students": students,
    }


@app.get("/api/traces")
def traces() -> dict:
    files = trace_logger.list_traces()
    return {"count": len(files), "files": [f.name for f in files[-20:]]}


@app.get("/api/triage/pending")
def triage_pending(limit: int = 50) -> dict:
    """待专家复评的试讲列表。"""
    memory_store.init_db()
    trials = memory_store.get_pending_reviews(limit=limit)
    return {"count": len(trials), "trials": trials}


@app.post("/api/expert-review")
def expert_review(body: dict) -> dict:
    """专家提交复评意见。
    body: {try_id, expert_id, final_scores: {D1:{total}, D2:{total}, ..., overall_percent}, comments}
    """
    memory_store.init_db()
    try_id = body.get("try_id", "")
    expert_id = body.get("expert_id", "expert")
    final_scores = body.get("final_scores", {})
    comments = body.get("comments", "")
    if not try_id:
        return {"error": "try_id 不能为空"}
    result = memory_store.submit_expert_review(try_id, expert_id, final_scores, comments)
    return {"status": "ok", **result}


@app.get("/api/messages")
def get_messages(user_id: str) -> dict:
    """查询用户未读消息。"""
    memory_store.init_db()
    msgs = memory_store.get_unread_messages(user_id)
    return {"count": len(msgs), "messages": msgs}


@app.post("/api/messages/read")
def read_messages(body: dict) -> dict:
    """标记消息已读。body: {user_id, message_ids?: []}"""
    memory_store.init_db()
    user_id = body.get("user_id", "")
    message_ids = body.get("message_ids")
    if not user_id:
        return {"error": "user_id 不能为空"}
    count = memory_store.mark_messages_read(user_id, message_ids)
    return {"status": "ok", "marked": count}


@app.get("/api/chat")
def chat(body: dict) -> dict:
    """试讲陪练聊天助手。
    body: {user_id, message, session_id?}
    """
    from app.chat import service as chat_service
    memory_store.init_db()
    user_id = body.get("user_id", "")
    message = body.get("message", "")
    session_id = body.get("session_id")
    if not user_id or not message:
        return {"error": "user_id 和 message 不能为空"}
    result = chat_service.chat(user_id, message, session_id)
    return {"status": "ok", **result}


@app.get("/api/calibration")
def calibration() -> dict:
    """查询系统校准状态（机器分vs专家分偏差）。"""
    memory_store.init_db()
    return memory_store.compute_calibration_offset()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "trial-eval-agent"}


# 挂载前端静态页面（放在所有 /api 路由之后，避免被静态文件接管）
FRONTEND_DIR = settings.BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
