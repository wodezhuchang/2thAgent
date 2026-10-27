# -*- coding: utf-8 -*-
"""LangGraph 图状态（State）定义。

对应 §3.7 边界：编排状态交给 LangGraph，领域逻辑（评价/记忆/反思）为自研节点函数。
"""
from __future__ import annotations
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # ---- 输入（meta 为试讲元信息，multimodal 为多模态信号，阶段3 丰富）----
    user_id: str
    try_id: str
    meta: dict[str, Any]
    multimodal: dict[str, Any]

    # ---- 运行中产生的中间状态 ----
    retrieved_context: str          # RAG 检索片段拼装的评课依据
    memory_summary: str             # 历史记忆摘要
    scores: dict[str, Any]          # 四维评分（对齐 schema）
    dim_results: dict[str, Any]     # 各维度原始结果（含 comment）
    confidence: dict[str, float]    # 各维度置信度（D1-D4，0-1，客观计算）
    triage_result: dict[str, Any]   # 智能分流结果 {route: A/B/C, reason, weak_dims}
    report: dict[str, Any]          # 完整评价报告（合成后）
    reflection: dict[str, Any]      # Reflexion 自查结果
    progress: dict[str, Any]        # 记忆模块返回的进步对比

    # ---- 输出 ----
    final_report: dict[str, Any]
    errors: list[str]
    trace: list[dict]
