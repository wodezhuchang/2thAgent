# -*- coding: utf-8 -*-
"""时间轴切片工具（阶段3 · 支撑 D1 教学设计评价）。

输入：ASR 分段（start/end/text）。
输出：教学环节时间轴（导入/新授/练习/小结）+ 各环节时长占比。

实现：关键词规则 + 时序连续性。
- 每段根据内容关键词打分归属环节；相邻同环节合并。
- 首段默认"导入"，末段倾向"小结"，避免空档。
"""
from __future__ import annotations
from typing import Any

from app.tools.registry import ToolSpec, ToolRegistry, registry


# 各环节关键词（针对试讲场景）。值=权重，强信号给高分。
_SEG_RULES = {
    "导入": {"大家好": 2, "同学们": 1, "今天": 1, "我们来讲": 2, "之前学过": 2, "还记得": 2,
             "复习": 2, "上节课": 2, "回顾": 2, "开始": 1},
    "新授": {"定义": 2, "形式是": 1, "就是": 1, "表示": 1, "代表": 1, "叫做": 2, "等于": 1,
             "称为": 2, "概念": 1, "结论": 1, "规律": 1, "证明": 1, "推导": 1, "性质": 1,
             "图像": 1, "我们看": 1, "注意": 1},
    "练习": {"练习": 3, "例题": 3, "做几": 3, "做题": 3, "请做": 3,
             "大家算": 2, "算一算": 2, "尝试": 2, "满足定义": 2},
    "小结": {"总结": 2, "小结": 2, "结束": 3, "到此": 3, "回家": 2, "回去": 2, "作业": 3,
             "巩固": 2, "谢谢": 3, "下课": 3, "归纳": 2, "回顾一下": 2, "做两个练习": 3},
}


def _score_segment(text: str) -> dict[str, int]:
    scores = {"导入": 0, "新授": 0, "练习": 0, "小结": 0}
    for seg, kws in _SEG_RULES.items():
        for kw, w in kws.items():
            if kw in text:
                scores[seg] += w
    return scores


def _pick_segment(text: str, idx: int, n: int, prev: str | None,
                  next_text: str | None = None) -> str:
    scores = _score_segment(text)
    # 首段（前1段）强归导入
    if idx == 0:
        scores["导入"] += 5
    # 末段（后1段）强归小结
    if idx >= n - 1:
        scores["小结"] += 5
    total = sum(scores.values())
    if total == 0:
        if idx == 0:
            return "导入"
        if idx >= n - 1:
            return "小结"
        return prev or "新授"
    best = max(scores, key=scores.get)
    return best


_ORDER = {"导入": 0, "新授": 1, "练习": 2, "小结": 3}


def _monotonic(labels: list[dict], n: int) -> list[dict]:
    """教学环节单向推进约束：导入→新授→练习→小结，环节不可回退。

    若判定结果回退到更早环节（如中段又标成'导入'），则提升到当前进度环节。
    """
    out = list(labels)
    cur = _ORDER["导入"]
    for i in range(n):
        seg = out[i]["segment"]
        rank = _ORDER.get(seg, 1)
        if rank < cur:
            # 回退 → 提升到当前环节
            seg = [k for k, v in _ORDER.items() if v == cur][0]
            out[i]["segment"] = seg
        else:
            cur = rank
    return out


def _smooth(labels: list[dict], n: int) -> list[dict]:
    """窗口平滑：去除孤立跳变（单段短暂切换的环节并入邻居）。"""
    out = list(labels)
    for i in range(1, n - 1):
        if out[i]["segment"] != out[i - 1]["segment"] and out[i]["segment"] != out[i + 1]["segment"]:
            # 孤立段并入前一段
            out[i]["segment"] = out[i - 1]["segment"]
    return out


def _timeline_slice(segments: list[dict]) -> dict:
    """把 ASR 分段映射为教学环节时间轴。"""
    if not segments:
        return {"timeline": [], "segments": segments, "phases": {}}
    # 先按时间排序
    segs = sorted(segments, key=lambda s: s["start"])
    n = len(segs)
    # 逐段判定环节
    labeled = []
    prev = None
    for i, s in enumerate(segs):
        seg_name = _pick_segment(s.get("text", ""), i, n, prev)
        labeled.append({"start": s["start"], "end": s["end"],
                        "segment": seg_name, "text": s.get("text", "")})
        prev = seg_name
    # 窗口平滑：去除孤立跳变
    labeled = _smooth(labeled, n)
    # 教学环节单向推进：导入→新授→练习→小结，不可回退
    labeled = _monotonic(labeled, n)
    # 相邻同环节合并
    phases = []
    for item in labeled:
        if phases and phases[-1]["segment"] == item["segment"]:
            phases[-1]["end"] = item["end"]
            phases[-1]["texts"].append(item["text"])
        else:
            phases.append({"segment": item["segment"], "start": item["start"],
                           "end": item["end"], "texts": [item["text"]]})
    # 合并后的起止时间 + 各环节时长
    total = max((p["end"] for p in phases), default=0) - min((p["start"] for p in phases), default=0)
    for p in phases:
        p["duration"] = round(p["end"] - p["start"], 2)
        p["text"] = "".join(p["texts"])[:120]
        p.pop("texts", None)
    # 各环节总时长占比
    phase_dur = {}
    for p in phases:
        phase_dur[p["segment"]] = phase_dur.get(p["segment"], 0) + p["duration"]
    ratios = {k: round(v / total, 3) if total else 0 for k, v in phase_dur.items()}
    return {"timeline": phases, "segments": labeled,
            "phase_durations": {k: round(v, 1) for k, v in phase_dur.items()},
            "phase_ratios": ratios, "total_sec": round(total, 1)}


TIMELINE_SPEC = ToolSpec(
    name="timeline_slice",
    description="把 ASR 语音分段切片为教学环节时间轴（导入/新授/练习/小结），输出各环节起止时间、时长与占比。"
                "供 D1 教学设计（环节结构/时间分配）评价使用。",
    parameters={"type": "object", "properties": {
        "segments": {"type": "array", "description": "ASR 分段列表，每项含 start/end/text"},
        },
        "required": ["segments"]},
    category="time",
)


def register_timeline_tool(reg: ToolRegistry = registry) -> None:
    reg.register(TIMELINE_SPEC, _timeline_slice)


register_timeline_tool()
