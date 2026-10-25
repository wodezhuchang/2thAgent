# -*- coding: utf-8 -*-
"""轨迹日志（L6 可观测性与治理层）：append-only 运行日志。

每次 Agent 运行（try_id）记录一条带时间戳的 JSON 行，含各阶段步骤与关键中间结果，
支持回放与归因（对应开发前文件 §5 归因分析 / §3.6.4"每次运行可追溯"）。
"""
from __future__ import annotations
import json
import time
from pathlib import Path

from app.config import settings


class TraceLogger:
    """append-only 轨迹日志。每个 try_id 一个 .jsonl 文件。"""

    def __init__(self, try_id: str, trace_dir: Path | None = None):
        self.try_id = try_id
        self.trace_dir = trace_dir or settings.TRACE_DIR
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.trace_dir / f"{try_id}.jsonl"
        self._meta = {"try_id": try_id, "started_at": time.time(), "steps": []}

    def _append(self, event: dict) -> None:
        event = {"ts": round(time.time(), 3), **event}
        self._meta["steps"].append(event)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log(self, step: str, **data) -> None:
        self._append({"step": step, **data})

    def log_llm(self, step: str, model: str, prompt_preview: str, status: str) -> None:
        self._append({"step": step, "event": "llm_call", "model": model,
                      "prompt_preview": prompt_preview[:500], "status": status})

    def finalize(self, status: str, summary: dict | None = None) -> None:
        self._append({"step": "finalize", "status": status, "summary": summary or {},
                      "elapsed_s": round(time.time() - self._meta["started_at"], 2)})

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]


def list_traces(trace_dir: Path | None = None) -> list[Path]:
    d = trace_dir or settings.TRACE_DIR
    if not d.exists():
        return []
    return sorted(d.glob("*.jsonl"))
