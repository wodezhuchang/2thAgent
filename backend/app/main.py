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
import sys
import base64
import hashlib
import hmac
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header
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

# 知识库上传/待入库目录（增量入库扫描该目录，撤销上传即删除其中文件）
KB_UPLOAD_DIR = settings.BASE_DIR / "datasets" / "knowledge_base" / "uploaded"
KB_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="师范生试讲多模态智能评价 Agent", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------- 认证（多角色改造）：JWT（HS256，零依赖自实现） ----------------

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _make_token(payload: dict, secret: str, exp_seconds: int = 7 * 24 * 3600) -> str:
    body = dict(payload)
    body["exp"] = int(time.time()) + exp_seconds
    header = {"alg": "HS256", "typ": "JWT"}
    h = _b64url(json.dumps(header, separators=(",", ":")).encode())
    p = _b64url(json.dumps(body, separators=(",", ":")).encode())
    sig = _b64url(hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())
    return f"{h}.{p}.{sig}"


def _verify_token(token: str, secret: str) -> dict | None:
    try:
        h, p, s = token.split(".")
        expected = _b64url(hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(s, expected):
            return None
        pad = "=" * (-len(p) % 4)
        payload = json.loads(base64.urlsafe_b64decode(p + pad))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def _public_user(user: dict) -> dict:
    return {k: user.get(k) for k in ("user_id", "username", "name", "role", "account_type", "org")}


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
def login(req: LoginRequest) -> dict:
    """登录：校验账号密码 → 签发 JWT。"""
    memory_store.init_db()
    user = memory_store.get_user_by_username(req.username)
    if not user or not memory_store._verify_password(req.password, user.get("password_hash") or ""):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = _make_token({"sub": user["user_id"], "role": user["role"]}, settings.JWT_SECRET)
    return {"token": token, "user": _public_user(user)}


@app.get("/api/auth/me")
def auth_me(authorization: str | None = Header(None)) -> dict:
    """按 token 返回当前用户（登录态恢复）。"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    payload = _verify_token(authorization[7:], settings.JWT_SECRET)
    if not payload:
        raise HTTPException(status_code=401, detail="token 无效或过期")
    user = memory_store.get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return {"user": _public_user(user)}


class EvaluateRequest(BaseModel):
    user_id: str = "anonymous"
    expert_id: str = ""
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
    # 发送指定老师：显式指定 expert_id；未指定或未绑定则拒绝
    if not req.expert_id:
        raise HTTPException(status_code=400, detail="请先指定发送给哪位老师")
    if not memory_store.is_bound(req.expert_id, req.user_id):
        raise HTTPException(status_code=400, detail="该老师与你尚未建立绑定关系，无法发送")
    meta = {
        "user_id": req.user_id,
        "expert_id": req.expert_id,
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
    expert_id: str = Form(""),
) -> dict:
    """上传试讲视频 → 自动跑 ASR/OCR/姿态/时间轴感知 → 送入 Agent 评价。

    expert_id：学生显式指定的发送对象（必须已与该老师建立绑定），防止错发/悬空。
    """
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
    # 发送指定老师：显式指定 expert_id；未指定或未绑定则拒绝（避免错发/悬空/无关专家无义务处理）
    if not expert_id:
        raise HTTPException(status_code=400, detail="请先指定发送给哪位老师")
    if not memory_store.is_bound(expert_id, user_id):
        raise HTTPException(status_code=400, detail="该老师与你尚未建立绑定关系，无法发送")
    meta = {
        "user_id": user_id,
        "expert_id": expert_id,
        "subject": subject, "lesson": lesson,
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


@app.get("/api/try/{try_id}")
def get_trial_report(try_id: str) -> dict:
    """按 try_id 返回完整 AgentReport（供学生端报告页真实取数）。"""
    memory_store.init_db()
    report = memory_store.get_full_report(try_id)
    if not report:
        raise HTTPException(status_code=404, detail="该次试讲记录不存在")
    return report


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
    """上传文件到待入库目录（仅保存，不自动入库；入库调 POST /api/kb/index）。"""
    save_dir = KB_UPLOAD_DIR
    save_path = save_dir / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "filename": file.filename, "msg": "文件已保存到待入库目录，可执行增量入库"}


@app.get("/api/kb/uploads")
def kb_uploads() -> dict:
    """列出待入库目录文件（含类型/学科推断、大小、是否已入库）。"""
    import chromadb
    client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
    try:
        col = client.get_collection(settings.COLLECTION)
        metas = col.get(include=["metadatas"])["metadatas"]
        indexed_src = {m.get("source_file") for m in metas if m and m.get("source_file")}
    except Exception:
        indexed_src = set()

    sys.path.insert(0, str(settings.APP_DIR / "scripts"))
    import kb_manager as kb
    items = []
    for p in sorted(KB_UPLOAD_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() not in kb.ALL_EXTS:
            continue
        doc_type, subject = kb.classify_doc(p.name)
        items.append({
            "name": p.name,
            "size": p.stat().st_size,
            "doc_type": doc_type,
            "subject": subject,
            "already_indexed": p.name in indexed_src,
            "modified_at": p.stat().st_mtime,
        })
    return {"uploads": items, "total": len(items)}


@app.post("/api/kb/index")
def kb_index() -> dict:
    """增量入库：把待入库目录中新增/变更的文件写入向量库（按 source_file + 文件哈希去重）。"""
    import chromadb
    from app.config import settings

    sys.path.insert(0, str(settings.APP_DIR / "scripts"))
    import kb_manager as kb

    model = kb.get_embedder(settings.EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
    col = kb.load_collection(client, settings.COLLECTION)

    existing: dict[str, str] = {}
    try:
        for m in col.get(include=["metadatas"])["metadatas"]:
            if m and m.get("source_file"):
                existing[m["source_file"]] = m.get("file_hash", "")
    except Exception:
        existing = {}

    results = {"indexed": [], "skipped": [], "errors": []}
    files = [p for p in sorted(KB_UPLOAD_DIR.iterdir())
             if p.is_file() and p.suffix.lower() in kb.ALL_EXTS]
    for path in files:
        rel = path.name
        fhash = kb.file_hash(path)
        if rel in existing and existing[rel] == fhash:
            results["skipped"].append({"name": rel, "reason": "未变更"})
            continue
        try:
            text = kb.read_doc(path)
            doc_type, subject = kb.classify_doc(path.name)
            chunks = kb.split_doc_into_chunks(path, rel, doc_type, subject, text, fhash)
            if not chunks:
                results["errors"].append({"name": rel, "reason": "未提取到文本"})
                continue
            action = "update" if rel in existing else "add"
            if rel in existing:
                col.delete(where={"source_file": rel})
            col.add(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                embeddings=[model.encode(c["text"], normalize_embeddings=True).tolist() for c in chunks],
                metadatas=[c["meta"] for c in chunks],
            )
            results["indexed"].append({"name": rel, "action": action, "chunks": len(chunks)})
        except Exception as e:  # noqa: BLE001
            results["errors"].append({"name": rel, "reason": str(e)})

    results["collection_count"] = col.count()
    return results


@app.delete("/api/kb/uploads/{filename:path}")
def kb_upload_delete(filename: str):
    """撤销上传：删除待入库目录中尚未入库的文件。"""
    target = (KB_UPLOAD_DIR / filename).resolve()
    if KB_UPLOAD_DIR.resolve() not in target.parents and target != KB_UPLOAD_DIR.resolve():
        raise HTTPException(status_code=400, detail="非法路径")
    if not target.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    target.unlink()
    return {"ok": True, "removed": filename}


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
    """班级聚合数据（从记忆表聚合）：班级整体分析。
    返回班级均分、四维均分、学生列表（含最弱维度）、班级均分随时间趋势。
    """
    memory_store.init_db()
    conn = memory_store._connect()
    rows = conn.execute(
        "SELECT user_id, scores_json, overall_percent, created_at FROM evaluation ORDER BY created_at"
    ).fetchall()
    conn.close()
    import json
    users = {}
    trend = []
    # rows 已按 created_at 升序：逐条维护每学生最近分，实时更新班级整体均分 → 班级趋势
    for r in rows:
        uid = r["user_id"]
        if uid not in users:
            users[uid] = {"scores": [], "dims": {}, "weakest": "", "last": None}
        try:
            s = json.loads(r["scores_json"]) if r["scores_json"] else {}
        except Exception:
            s = {}
        users[uid]["scores"].append(r["overall_percent"])
        users[uid]["last"] = r["overall_percent"]
        for d in ("D1", "D2", "D3", "D4"):
            users[uid]["dims"].setdefault(d, []).append(s.get(d, {}).get("total", 0))
        wd = min(("D1", "D2", "D3", "D4"), key=lambda d: s.get(d, {}).get("total", 0) or 0)
        users[uid]["weakest"] = wd
        vals = [u["last"] for u in users.values() if u["last"] is not None]
        if vals:
            trend.append({"date": (r["created_at"] or "")[:10],
                          "avg": round(sum(vals) / len(vals), 1)})
    # 聚合
    all_scores = [u["scores"][-1] for u in users.values() if u["scores"]]
    dim_avg = {}
    for d in ("D1", "D2", "D3", "D4"):
        vals = [v for u in users.values() for v in u["dims"].get(d, [])]
        dim_avg[d] = round(sum(vals) / len(vals), 2) if vals else 0
    students = [{"user_id": uid, "last_score": u["scores"][-1] if u["scores"] else 0,
                 "avg_score": round(sum(u["scores"]) / len(u["scores"]), 1) if u["scores"] else 0,
                 "weakest_dim": u["weakest"]}
                for uid, u in users.items()]
    return {
        "student_count": len(users),
        "class_avg": round(sum(all_scores) / len(all_scores), 1) if all_scores else 0,
        "dim_avg": dim_avg,
        "class_trend": trend,
        "students": students,
    }


@app.get("/api/teacher/{teacher_id}/students")
def teacher_students(teacher_id: str) -> dict:
    """教师端「学生洞察」：教师名下学生的学情聚合指标（阶段1）。"""
    memory_store.init_db()
    students = memory_store.get_teacher_students_dashboard(teacher_id)
    return {"teacher_id": teacher_id, "count": len(students), "students": students}


@app.get("/api/students/{student_id}/reviews")
def student_reviews(student_id: str) -> dict:
    """某学生的专家复评历史（机器分 vs 专家分、争议维度、意见）（阶段2）。"""
    memory_store.init_db()
    reviews = memory_store.get_student_reviews(student_id)
    return {"student_id": student_id, "count": len(reviews), "reviews": reviews}


@app.get("/api/students/{student_id}/training-plans")
def student_training_plans(student_id: str) -> dict:
    """某学生的历次训练计划（阶段2）。"""
    memory_store.init_db()
    plans = memory_store.get_student_training_plans(student_id)
    return {"student_id": student_id, "count": len(plans), "plans": plans}


@app.get("/api/traces")
def traces() -> dict:
    files = trace_logger.list_traces()
    return {"count": len(files), "files": [f.name for f in files[-20:]]}


@app.get("/api/triage/pending")
def triage_pending(expert_id: str | None = None, limit: int = 50) -> dict:
    """待专家复评的试讲列表。expert_id 为空返回全部（admin 公共池）。"""
    memory_store.init_db()
    trials = memory_store.get_pending_reviews(expert_id=expert_id, limit=limit)
    return {"count": len(trials), "trials": trials}


@app.get("/api/users")
def list_users(role: str | None = None) -> dict:
    """列出账号（可按角色过滤），供 admin 管理 / 建立关系。"""
    memory_store.init_db()
    return {"users": memory_store.list_users(role=role)}


class RelationshipRequest(BaseModel):
    teacher_id: str
    student_id: str


@app.post("/api/relationships")
def create_relationship(req: RelationshipRequest) -> dict:
    """建立师生指导关系（评价分发依据）。"""
    memory_store.init_db()
    memory_store.add_teacher_student(req.teacher_id, req.student_id)
    return {"status": "ok"}


class BindingRequestModel(BaseModel):
    student_id: str
    teacher_id: str


@app.post("/api/bindings/request")
def binding_request(req: BindingRequestModel) -> dict:
    """学生发起绑定申请（线下协商后线上发起，专家确认后生效）。"""
    memory_store.init_db()
    return memory_store.request_binding(req.student_id, req.teacher_id)


@app.get("/api/bindings/requests")
def binding_requests(teacher_id: str | None = None, student_id: str | None = None,
                     status: str | None = None) -> dict:
    """查询绑定申请（专家查看待确认 / 学生查看自己的申请）。"""
    memory_store.init_db()
    items = memory_store.list_binding_requests(
        teacher_id=teacher_id, student_id=student_id, status=status)
    return {"requests": items, "count": len(items)}


@app.post("/api/bindings/confirm")
def binding_confirm(req: BindingRequestModel) -> dict:
    """专家确认绑定 → 写入 teacher_student，学生试讲即可发送到该专家。"""
    memory_store.init_db()
    return memory_store.confirm_binding(req.teacher_id, req.student_id)


@app.post("/api/bindings/reject")
def binding_reject(req: BindingRequestModel) -> dict:
    """专家拒绝绑定申请。"""
    memory_store.init_db()
    return memory_store.reject_binding(req.teacher_id, req.student_id)


class ApplyByCodeRequest(BaseModel):
    student_id: str
    invite_code: str


@app.post("/api/bindings/apply-by-code")
def binding_apply_by_code(req: ApplyByCodeRequest) -> dict:
    """学生凭教师线下提供的邀请口令发起绑定申请。

    学生端不暴露老师池，只有拿到老师口令的学生才能申请到该老师，防恶意申请骚扰。
    """
    memory_store.init_db()
    code = req.invite_code.strip()
    teacher = memory_store.get_teacher_by_invite_code(code)
    if not teacher:
        return {"ok": False, "reason": "邀请口令无效，请向老师确认口令"}
    if not req.student_id:
        return {"ok": False, "reason": "student_id 不能为空"}
    result = memory_store.request_binding(req.student_id, teacher["user_id"])
    result["teacher_id"] = teacher["user_id"]
    result["teacher_name"] = teacher.get("name") or teacher["user_id"]
    return result


@app.get("/api/users/{user_id}/invite-code")
def user_invite_code(user_id: str) -> dict:
    """查询账号邀请口令（教师端展示，便于线下告知指定学生）。"""
    memory_store.init_db()
    user = memory_store.get_user_by_id(user_id)
    if not user:
        return {"error": "用户不存在"}
    return {"invite_code": memory_store.ensure_invite_code(user_id)}


@app.get("/api/users/{user_id}/teachers")
def user_teachers(user_id: str) -> dict:
    """学生已绑定的老师列表（供发送试讲时指定发送对象）。"""
    memory_store.init_db()
    return {"teachers": memory_store.list_student_teachers(user_id)}


class ReturnRequest(BaseModel):
    try_id: str
    teacher_id: str
    reason: str = ""


@app.post("/api/expert-review/return")
def expert_review_return(req: ReturnRequest) -> dict:
    """教师退回试讲：标记 returned，通知学生重新上传给正确老师（不指明具体老师）。"""
    memory_store.init_db()
    return memory_store.return_review(req.try_id, req.teacher_id, req.reason)


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
    # 知识库精选回流（旁路，失败不影响专家复评主流程）
    try:
        from app.rag.recycle import recycle_disputed_case
        result["recycle"] = recycle_disputed_case(try_id)
    except Exception as e:  # noqa: BLE001
        result["recycle"] = {"recycled": False, "reason": f"回流模块异常: {e}", "try_id": try_id}
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


@app.post("/api/chat")
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


@app.get("/api/chat/sessions")
def chat_sessions(user_id: str) -> dict:
    """学生端会话列表（按 session 聚合，供主题分类 + 时间查找）。"""
    memory_store.init_db()
    if not user_id:
        return {"error": "user_id 不能为空"}
    return {"sessions": memory_store.list_chat_sessions(user_id)}


@app.get("/api/chat/sessions/{session_id}")
def chat_session_detail(session_id: str, user_id: str) -> dict:
    """某会话完整对话内容（时间正序，供点开回看）。"""
    memory_store.init_db()
    if not user_id:
        return {"error": "user_id 不能为空"}
    return {"session_id": session_id,
            "messages": memory_store.get_session_messages(user_id, session_id)}


@app.get("/api/calibration")
def calibration() -> dict:
    """查询系统校准状态（机器分vs专家分偏差）。"""
    memory_store.init_db()
    return memory_store.compute_calibration_offset()


@app.post("/api/expert-chat")
def expert_chat(body: dict) -> dict:
    """专家端教研助手聊天。
    body: {expert_id, message, student_id?, session_id?}
    """
    from app.chat import service as chat_service
    memory_store.init_db()
    expert_id = body.get("expert_id", "")
    message = body.get("message", "")
    student_id = body.get("student_id")
    session_id = body.get("session_id")
    if not expert_id or not message:
        return {"error": "expert_id 和 message 不能为空"}
    result = chat_service.expert_chat(expert_id, message, student_id, session_id)
    return {"status": "ok", **result}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "trial-eval-agent"}


# 挂载前端静态页面（放在所有 /api 路由之后，避免被静态文件接管）
FRONTEND_DIR = settings.BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
