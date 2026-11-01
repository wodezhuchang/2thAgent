# -*- coding: utf-8 -*-
"""评分稳定性回归脚本。

用途：固定用 demo_preteaching 的感知档案（不重跑 ASR/OCR/pose），
只重跑 LLM 打分，跑 N 次，记录总分与各维度分的极差/标准差，
作为后续调 prompt/权重/RAG 的标尺。

用法：
    cd D:\\2thAgent\\backend
    python -m app.scripts.regression_score [--runs 5]
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path

# 确保 backend 目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import settings
from app.llm import llm_client
from app.evaluator import evaluator
from app.rag import retriever


SENSES_DIR = settings.BASE_DIR / "datasets" / "demo_preteaching" / "senses"

META = {
    "user_id": "demo_user",
    "subject": "数学",
    "lesson": "一次函数",
    "lesson_type": "新授课",
    "grade_level": "初中",
    "duration_seconds": 180,
}


def load_senses() -> dict:
    """从 demo 感知档案加载多模态信号（跳过 ASR/OCR/pose 重跑）。

    senses 文件结构：{tool, video_source, ..., result: {...}}
    """
    def _load(name: str):
        return json.loads((SENSES_DIR / name).read_text(encoding="utf-8")).get("result", {})

    asr = _load("asr.json")
    ocr = _load("ocr.json")
    timeline = _load("timeline.json")
    pose = _load("pose.json")

    mm = {}
    if asr.get("text"):
        mm["asr_text"] = asr["text"]
    if asr.get("feats"):
        mm["audio_feats"] = asr["feats"]
    if ocr.get("text"):
        mm["board_text"] = ocr["text"]
    if timeline.get("timeline"):
        mm["timeline"] = timeline["timeline"]
    if pose.get("metrics"):
        mm["posture"] = pose["metrics"]
    return mm


def run_once(client, meta, context, memory_summary, multimodal) -> dict:
    scores, _ = evaluator.eval_all_dimensions(client, meta, context, memory_summary, multimodal)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5, help="跑几次")
    args = parser.parse_args()

    multimodal = load_senses()
    print(f"[load] multimodal keys: {list(multimodal.keys())}")
    print(f"[load] asr_text len={len(multimodal.get('asr_text',''))}, "
          f"board_text len={len(multimodal.get('board_text',''))}, "
          f"posture keys={list(multimodal.get('posture',{}).keys())}")

    # RAG 检索（确定性，只跑一次；此处跳过依赖，用通用上下文占位——测的是 LLM 方差不是 RAG）
    context = ("[课标依据] 初中数学新课标要求：数学教学应面向全体学生，注重启发式、探究式教学，"
               "突出学生主体地位。评课量表：教学目标明确、重难点突出、环节完整、语言准确清晰、"
               "板书规范、教姿教态自然、有师生互动。")
    print(f"[rag] skipped (offline), context len={len(context)}")

    memory_summary = "（回归测试，无历史对比）"
    client = llm_client.get_llm_client()

    # 跑 N 次
    all_overall = []
    all_dims = {"D1": [], "D2": [], "D3": [], "D4": []}
    dim_key_map = {"D1": "D1_teaching_design", "D2": "D2_language",
                   "D3": "D3_blackboard", "D4": "D4_presence"}

    for i in range(args.runs):
        print(f"\n--- Run {i+1}/{args.runs} ---")
        scores = run_once(client, META, context, memory_summary, multimodal)
        overall = scores.get("overall_percent", 0)
        all_overall.append(overall)
        dims_str = []
        for d, key in dim_key_map.items():
            v = scores.get(key, {}).get("total", 0)
            all_dims[d].append(v)
            dims_str.append(f"{d}={v}")
        print(f"  overall={overall}  {'  '.join(dims_str)}")

    # 统计
    print("\n" + "=" * 60)
    print(f"总分（百分制）: {all_overall}")
    print(f"  mean={statistics.mean(all_overall):.1f}  "
          f"range={max(all_overall)-min(all_overall):.1f}  "
          f"std={statistics.stdev(all_overall):.1f}")
    for d, vals in all_dims.items():
        print(f"{d}（0-5制）: mean={statistics.mean(vals):.2f}  "
              f"range={max(vals)-min(vals):.2f}  std={statistics.stdev(vals):.2f}  "
              f"vals={vals}")
    print("=" * 60)
    print(f"判定：总分极差 {'✅ ≤3' if max(all_overall)-min(all_overall) <= 3 else '❌ >3，需继续压随机性'}")


if __name__ == "__main__":
    main()
