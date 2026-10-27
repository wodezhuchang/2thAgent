# -*- coding: utf-8 -*-
"""多模态感知调度器（阶段3 · 图改造核心）。

把"一条真实试讲视频"统一调度四类感知工具，得到可送入评价的多模态信号，并落盘为感知档案。

流程（对应开发前文件 §2.1 多模态感知模块 + §3.6.2-L3 工具层）：
    视频 → registry.call(ASR) ──┬── asr_text/feats
         → registry.call(OCR) ──┼── board_text
         → registry.call(timeline) ─── timeline（用 ASR 分段）
         → registry.call(pose) ──┴── posture 指标 + 关键点
    结果统一经 senses_store 落盘 → 返回 {multimodal, senses, traces}

设计：工具保持纯函数（经注册表调度，可插拔/可观测），落盘由 senses_store 统一完成，
保证"感知结果有历史可查"（可追溯/可复现/可对比）。
"""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from app.tools.registry import registry
from app.tools import perception_tools, timeline_tool, pose_tool  # 导入即注册
from app.tools import senses_store


def perceive_video(video_path: str, trial_id: str,
                   asr_segments: list | None = None) -> dict:
    """统一调度四类感知工具，返回可送入评价的多模态信号 + 落盘档案。

    优化：ASR / OCR / pose 三个独立任务并行跑，timeline 依赖 ASR 结果。
    """
    steps: list[dict] = []
    t0 = time.time()

    # 并行跑三个独立任务：ASR、OCR、pose
    results = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(registry.call, "asr_transcribe", video_or_audio=video_path): "asr",
            ex.submit(registry.call, "ocr_board", video_or_image=video_path, max_frames=30): "ocr",
            ex.submit(registry.call, "pose_analyze", video_path=video_path, n_frames=20): "pose",
        }
        for f in as_completed(futures):
            name = futures[f]
            try:
                results[name] = f.result()
            except Exception as e:
                results[name] = None
                if name == "asr":
                    steps.append({"tool": "asr_transcribe", "ok": False, "error": str(e)})
                elif name == "ocr":
                    steps.append({"tool": "ocr_board", "ok": False, "error": str(e)})
                elif name == "pose":
                    steps.append({"tool": "pose_analyze", "ok": False, "error": str(e)})

    asr = results.get("asr")
    ocr = results.get("ocr")
    pose = results.get("pose")

    if asr:
        asr_text = asr.get("text", "")
        steps.append({"tool": "asr_transcribe", "ok": True,
                      "words": len(asr.get("words", [])), "segments": len(asr.get("segments", []))})
    else:
        asr_text = ""

    if ocr:
        board_text = ocr.get("text", "")
        steps.append({"tool": "ocr_board", "ok": True, "lines": len(ocr.get("lines", []))})
    else:
        board_text = ""

    if pose:
        posture = pose.get("metrics", {})
        steps.append({"tool": "pose_analyze", "ok": True, "detected": pose.get("detected", 0)})
    else:
        posture = {}

    # 时间轴（依赖 ASR 分段，ASR 完成后跑）
    timeline = []
    tl = None
    if asr:
        try:
            tl = registry.call("timeline_slice", segments=asr.get("segments", []))
            timeline = tl.get("timeline", [])
            steps.append({"tool": "timeline_slice", "ok": True, "phases": len(timeline)})
        except Exception as e:
            steps.append({"tool": "timeline_slice", "ok": False, "error": str(e)})

    # 组装多模态信号（评价器消费的字段，对齐 llm_client.build_user_prompt）
    multimodal = {}
    if asr_text:
        multimodal["asr_text"] = asr_text
    if board_text:
        multimodal["board_text"] = board_text
    if timeline:
        multimodal["timeline"] = timeline
    if posture:
        multimodal["posture"] = posture
    if asr and asr.get("feats"):
        multimodal["audio_feats"] = asr["feats"]

    # 落盘为感知档案（可追溯/可复现）
    senses = senses_store.save_all_senses(
        trial_id, {"asr": asr, "ocr": ocr, "timeline": tl if asr else None,
                   "pose": pose}, video_source=video_path)

    return {
        "multimodal": multimodal,
        "senses": senses,
        "steps": steps,
        "elapsed_sec": round(time.time() - t0, 1),
    }
