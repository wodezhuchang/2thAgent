# -*- coding: utf-8 -*-
"""感知档案存储器（senses store）：把多模态感知的原始结果持久化落盘。

背景：阶段3工具验证时只返回结果 dict，未落盘，导致感知数据（OCR 板书、姿态关键点、
时间轴）无历史可查。本模块统一解决：每次试讲的感知结果写入
    datasets/<trial_id>/senses/{asr,ocr,timeline,pose}.json
作为可追溯、可复现、可对比的感知档案，支撑记忆闭环与答辩展示。

设计：
- 工具保持纯函数（返回结构化结果，无副作用），落盘由本模块统一完成，职责清晰。
- 每个档案文件含：来源视频、工具版本、生成时间、原始感知结果 + 聚合指标。
- 提供 list_senses / load_sense 便于复现与追溯。
"""
from __future__ import annotations
import json
import time
import os
from pathlib import Path
from typing import Any

# 感知档案根目录（相对项目根）
SENSES_ROOT = Path(r"D:\2thAgent\datasets")


def _trial_dir(trial_id: str) -> Path:
    return SENSES_ROOT / trial_id / "senses"


def save_sense(trial_id: str, tool_name: str, result: dict,
               video_source: str = "", tool_version: str = "") -> str:
    """把某工具的感知结果写入档案，返回写入的文件路径。

    tool_name ∈ {"asr", "ocr", "timeline", "pose"}
    """
    _trial_dir(trial_id).mkdir(parents=True, exist_ok=True)
    record = {
        "tool": tool_name,
        "video_source": video_source,
        "tool_version": tool_version,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "result": result,
    }
    path = _trial_dir(trial_id) / f"{tool_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return str(path)


def list_senses(trial_id: str) -> list[dict]:
    """列出某试讲的感知档案（文件元信息）。"""
    d = _trial_dir(trial_id)
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        out.append({"tool": p.stem, "path": str(p), "size": p.stat().st_size})
    return out


def load_sense(trial_id: str, tool_name: str) -> dict | None:
    """读取某工具的感知档案。"""
    p = _trial_dir(trial_id) / f"{tool_name}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_all_senses(trial_id: str, senses: dict, video_source: str = "") -> dict:
    """批量落盘：senses = {tool_name: result_dict}，返回 {tool: filepath}。"""
    saved = {}
    for tool_name, result in senses.items():
        if result is not None:
            saved[tool_name] = save_sense(trial_id, tool_name, result,
                                          video_source=video_source)
    return saved
