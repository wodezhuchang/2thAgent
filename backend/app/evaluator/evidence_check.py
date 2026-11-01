# -*- coding: utf-8 -*-
"""证据链检查与智能分流（第一阶段新增）。

在 synthesize 之后、reflect 之前执行：
1. compute_confidence：根据多模态信号成功率 + RAG支持度，计算每个维度的客观置信度
2. detect_hard_risk：检测高风险硬规则（互动不足/缺环节/板书空）
3. triage：根据置信度和硬规则结果，分 A/B/C 三路

置信度公式：confidence = 证据成功率 × RAG支持度
- 证据成功率：该维度对应的感知工具是否成功产出数据
- RAG支持度：默认0.8（有召回就当有支持，后续可细化）
"""
from __future__ import annotations
import re
from typing import Any


# 提问词表（用于互动次数统计）
_QUESTION_WORDS = [
    "大家看", "大家想", "谁来说", "谁能回答", "请同学", "请大家",
    "想一想", "思考一下", "我们一起来", "有没有同学", "谁来分享",
]


def compute_confidence(multimodal: dict[str, Any]) -> dict[str, float]:
    """计算 D1-D4 每个维度的置信度（0-1）。

    返回 {D1: float, D2: float, D3: float, D4: float}
    """
    # 证据成功率：该维度有没有对应的感知数据
    # D1 教学设计：依赖时间轴
    has_timeline = bool(multimodal.get("timeline"))
    # D2 讲授语言：依赖 ASR 文本
    has_asr = bool(multimodal.get("asr_text"))
    # D3 板书：依赖 OCR
    has_board = bool(multimodal.get("board_text")) and len(multimodal.get("board_text", "")) >= 5
    # D4 教态：依赖姿态数据
    has_pose = bool(multimodal.get("posture"))

    # RAG支持度：默认0.8（有检索就当有支持，后续可按维度细化）
    rag_support = 0.8

    confidence = {
        "D1": round((1.0 if has_timeline else 0.3) * rag_support, 2),
        "D2": round((1.0 if has_asr else 0.3) * rag_support, 2),
        "D3": round((1.0 if has_board else 0.3) * rag_support, 2),
        "D4": round((1.0 if has_pose else 0.3) * rag_support, 2),
    }
    return confidence


def detect_hard_risk(multimodal: dict[str, Any], duration_sec: int = 0) -> list[str]:
    """检测高风险硬规则，返回触发的风险列表（空列表=无风险）。

    根据视频时长分档控制阀值：
    - 短视频（<5分钟）：跳过「缺小结/练习」规则（片段式试讲本就不要求完整环节）
    - 中长视频（≥5分钟）：全部硬规则生效
    """
    risks = []

    # 规则1：互动次数过少（提问词出现 < 2次）—— 所有时长都生效
    asr_text = multimodal.get("asr_text", "")
    if asr_text:
        question_count = sum(asr_text.count(w) for w in _QUESTION_WORDS)
        if question_count < 2:
            risks.append(f"互动次数过少（提问词仅{question_count}次，要求≥2次）")

    # 规则2：时间轴缺环节 —— 仅对 ≥5 分钟视频生效
    if duration_sec >= 300:
        timeline = multimodal.get("timeline", [])
        if timeline:
            segments = [t.get("segment", "") for t in timeline]
            if "小结" not in segments:
                risks.append("时间轴缺少「小结」环节")
            if "练习" not in segments:
                risks.append("时间轴缺少「练习」环节")

    # 规则3：板书为空（OCR字数 < 10）—— 所有时长都生效
    board_text = multimodal.get("board_text", "")
    if not board_text or len(board_text.strip()) < 10:
        risks.append("板书内容过少（OCR识别字数<10）")

    return risks


def video_duration_label(duration_sec: int) -> str:
    """返回视频时长分档标签，用于报告展示。"""
    if duration_sec <= 0:
        return "未知时长"
    if duration_sec < 300:
        return f"短视频片段（{duration_sec//60}分{duration_sec%60}秒）"
    if duration_sec < 900:
        return f"标准课时（{duration_sec//60}分钟）"
    return f"长课时（{duration_sec//60}分钟）"


def triage(confidence: dict[str, float], multimodal: dict[str, Any],
           duration_sec: int = 0) -> dict[str, Any]:
    """智能分流：根据置信度和硬规则结果，返回分流决定。

    返回：
    {
        "route": "A" | "B" | "C",
        "reason": str,
        "weak_dims": list[str],
        "hard_risks": list[str],
        "duration_tier": str,   # 时长分档标签
    }
    """
    # 先查硬规则（传入时长做分档）
    hard_risks = detect_hard_risk(multimodal, duration_sec)
    tier = video_duration_label(duration_sec)
    if hard_risks:
        return {
            "route": "C",
            "reason": "触发高风险硬规则",
            "weak_dims": [],
            "hard_risks": hard_risks,
            "duration_tier": tier,
        }

    # 再查低置信度维度
    weak_dims = [d for d, c in confidence.items() if c < 0.5]
    if weak_dims:
        return {
            "route": "B",
            "reason": f"存在低置信度维度：{', '.join(weak_dims)}",
            "weak_dims": weak_dims,
            "hard_risks": [],
            "duration_tier": tier,
        }

    # 否则自动通过
    return {
        "route": "A",
        "reason": "证据齐全，置信度达标",
        "weak_dims": [],
        "hard_risks": [],
        "duration_tier": tier,
    }
