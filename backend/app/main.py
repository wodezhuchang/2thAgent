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


@app.get("/api/traces")
def traces() -> dict:
    files = trace_logger.list_traces()
    return {"count": len(files), "files": [f.name for f in files[-20:]]}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "trial-eval-agent"}


# 挂载前端静态页面（放在所有 /api 路由之后，避免被静态文件接管）
FRONTEND_DIR = settings.BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
