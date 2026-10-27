# -*- coding: utf-8 -*-
"""LangGraph 图组装（阶段2 主干）。

范式（对应 §3.6.1）：Plan-and-Execute 主链 + 维度内 ReAct + 收尾 Reflexion。
主链 6 步固定：解析 → RAG 检索 → 记忆读取 → 四维评价 → 合成报告 → Reflexion →
记忆写入 → 训练规划。轨迹日志 append-only 贯穿全程（L6 可观测性）。

边界（§3.7.4）：LangGraph 只负责编排/状态流转；评价、记忆、反思、检索为自研节点。
"""
from __future__ import annotations
import json
import os
import uuid

from langgraph.graph import StateGraph, START, END

from app.config import settings
from app.graph.state import AgentState
from app.rag import retriever
from app.memory import memory_store
from app.llm import llm_client
from app.evaluator import evaluator
from app.trace import trace_logger
from app.tools.registry import registry, add_on_call_hook
from app.tools import agent_tools  # 导入即注册内置工具
from app.tools import perceive


def _gen_try_id() -> str:
    return f"trial_{uuid.uuid4().hex[:10]}"


def node_perceive(state: AgentState) -> dict:
    """② 多模态感知：若有试讲视频，经工具注册表统一调度 ASR/OCR/时间轴/姿态，
    提取多模态信号并落盘感知档案；否则沿用前端传入的多模态信号。"""
    meta = state["meta"] or {}
    multimodal = dict(state.get("multimodal") or {})
    trace = state.get("trace") or []
    video_path = meta.get("video_path") or multimodal.get("video_path")
    if video_path:
        try:
            pv = perceive.perceive_video(video_path, state["try_id"])
            multimodal = {**multimodal, **pv["multimodal"]}
            multimodal["senses"] = pv["senses"]  # 感知档案路径，供追溯
            trace.extend(pv["steps"])
            trace.append({"step": "perceive", "elapsed_sec": pv["elapsed_sec"],
                          "tools_ok": sum(1 for s in pv["steps"] if s.get("ok"))})
        except Exception as e:
            trace.append({"step": "perceive_error", "error": str(e)})
    else:
        trace.append({"step": "perceive_skip", "reason": "no video_path in meta/multimodal"})
    return {"multimodal": multimodal, "trace": trace}


# ---------------- 节点 ----------------

def node_parse(state: AgentState) -> dict:
    """① 解析：补齐 try_id / 校验 meta。"""
    meta = dict(state.get("meta") or {})
    if not state.get("try_id"):
        state["try_id"] = _gen_try_id()
        meta["try_id"] = state["try_id"]
    user_id = state.get("user_id") or meta.get("user_id") or "anonymous"
    meta["user_id"] = user_id
    # 记录到日志
    state["trace"] = state.get("trace") or []
    return {"meta": meta, "user_id": user_id, "try_id": state["try_id"]}


def node_retrieve(state: AgentState) -> dict:
    """③ RAG 检索：经工具注册表调度 rag_retrieve 工具，召回课标/量表/课例。"""
    meta = state["meta"]
    subject = meta.get("subject")
    lesson = meta.get("lesson", "")
    lesson_type = meta.get("lesson_type", "")
    query = f"{subject} {lesson} {lesson_type} 教学设计 评价 标准 优秀课例" if subject else \
            f"{lesson} {lesson_type} 教学设计 评价 标准"
    state["trace"] = state.get("trace") or []
    try:
        result = registry.call("rag_retrieve", query=query, subject=subject,
                               top_k=settings.RAG_TOP_K)
        context = result["context"]
        n_hits = len(result["hits"])
        state["trace"].append({"step": "tool_call", "tool": "rag_retrieve",
                               "category": "rag", "query": query})
    except Exception as e:
        context = "(RAG 检索失败: %s)" % e
        n_hits = 0
    state["retrieved_context"] = context
    state["trace"].append({"step": "rag_retrieve", "query": query, "hits": n_hits,
                           "via": "tool:rag_retrieve"})
    return {"retrieved_context": context, "trace": state["trace"]}


def node_memory_read(state: AgentState) -> dict:
    """记忆读取：取用户最近一次评价，生成摘要供对比。"""
    memory_store.init_db()
    prev = memory_store.get_prev_evaluation(state["user_id"])
    if prev:
        summary = (f"上次试讲（{prev.get('try_id')}）：总评 {prev.get('overall_percent')} 分，"
                   f"短板：{json.dumps(prev.get('weaknesses', []), ensure_ascii=False)}")
    else:
        summary = "（无历史记录，首次评价）"
    state["memory_summary"] = summary
    state["trace"].append({"step": "memory_read", "has_prev": bool(prev)})
    return {"memory_summary": summary, "trace": state["trace"]}


def node_evaluate(state: AgentState) -> dict:
    """④ 四维评价：调用 LLM 逐维打分。"""
    client = llm_client.get_llm_client()
    context = state.get("retrieved_context", "")
    mem = state.get("memory_summary", "")
    meta = state["meta"]
    multimodal = state.get("multimodal") or {}
    scores, dim_results = evaluator.eval_all_dimensions(
        client, meta, context, mem, multimodal)
    state["scores"] = scores
    state["dim_results"] = dim_results
    state["trace"].append({"step": "evaluate", "overall": scores.get("overall_percent")})
    return {"scores": scores, "dim_results": dim_results, "trace": state["trace"]}


def node_synthesize(state: AgentState) -> dict:
    """⑤ 合成报告：时间轴评语 + 短板 + 进步对比 + 训练规划。"""
    client = llm_client.get_llm_client()
    context = state.get("retrieved_context", "")
    mem = state.get("memory_summary", "")
    meta = state["meta"]
    syn = evaluator.synthesize(client, meta, state["scores"], state["dim_results"],
                               context, mem)
    # 组装完整报告（对齐 JSON Schema）
    report = {
        "meta": {**meta, "recorded_at": meta.get("recorded_at", "")},
        "disclaimer": settings.DISCLAIMER,
        "scores": state["scores"],
        "timeline_feedback": syn.get("timeline_feedback", []),
        "weaknesses": syn.get("weaknesses", []),
        "progress_compare": syn.get("progress_compare", {}),
        "training_plan": syn.get("training_plan", {}),
    }
    state["report"] = report
    state["trace"].append({"step": "synthesize"})
    return {"report": report, "trace": state["trace"]}


def node_evidence_check(state: AgentState) -> dict:
    """⑥ 证据链检查：计算每个维度的客观置信度。"""
    from app.evaluator import evidence_check
    multimodal = state.get("multimodal") or {}
    confidence = evidence_check.compute_confidence(multimodal)
    state["confidence"] = confidence
    state["trace"].append({"step": "evidence_check", "confidence": confidence})
    return {"confidence": confidence, "trace": state["trace"]}


def node_triage(state: AgentState) -> dict:
    """⑦ 智能分流：根据置信度和硬规则分 A/B/C 三路。"""
    from app.evaluator import evidence_check
    confidence = state.get("confidence") or {}
    multimodal = state.get("multimodal") or {}
    result = evidence_check.triage(confidence, multimodal)
    state["triage_result"] = result
    state["trace"].append({"step": "triage", "route": result["route"],
                           "reason": result["reason"]})
    return {"triage_result": result, "trace": state["trace"]}


def node_reflect(state: AgentState) -> dict:
    """⑥ Reflexion 自查：检验幻觉/违背课标/遗漏，必要时修正。"""
    client = llm_client.get_llm_client()
    context = state.get("retrieved_context", "")
    refl = evaluator.reflect(client, state["meta"], state["report"], context)
    if refl.get("needs_fix"):
        state["report"] = refl.get("fixed_report") or state["report"]
        state["trace"].append({"step": "reflect_fix", "issues": refl.get("issues", [])})
    else:
        state["trace"].append({"step": "reflect_ok"})
    state["reflection"] = refl
    return {"report": state["report"], "reflection": refl, "trace": state["trace"]}


def node_memory_write(state: AgentState) -> dict:
    """记忆写入：落 evaluation + progress + training_plan。"""
    report = state.get("report") or {}
    scores = report.get("scores") or {}
    # scores 键为 D1_teaching_design 等，需转成 D1 供记忆模块计算
    scores_dim = {}
    for skey, sval in scores.items():
        if isinstance(sval, dict) and skey.startswith("D"):
            dim = skey.split("_")[0]
            scores_dim[dim] = sval
    meta = state["meta"]
    try:
        prog = memory_store.save_evaluation(
            user_id=state["user_id"], try_id=state["try_id"], scores=scores_dim,
            timeline_feedback=report.get("timeline_feedback", []),
            weaknesses=report.get("weaknesses", []),
            reflection_log=state.get("reflection"),
            raw_multimodal=state.get("multimodal"),
            prompt_version=settings.PROMPT_VERSION)
        state["progress"] = prog
        # 训练规划
        tp = report.get("training_plan", {})
        if tp:
            memory_store.save_training_plan(
                eval_id=prog["eval_id"],
                next_focus=tp.get("next_focus", ""),
                specific_tasks=tp.get("specific_tasks", []),
                reference_case=tp.get("reference_case", ""),
                estimated_repeat_times=tp.get("estimated_repeat_times", 1))
        state["trace"].append({"step": "memory_write", "eval_id": prog["eval_id"],
                               "trend": prog["trend"]})
        # 写入分流结果到 evaluation 表
        triage = state.get("triage_result") or {}
        route = triage.get("route", "A")
        review_status = "auto_pass" if route == "A" else "pending_expert"
        conn = memory_store._connect()
        conn.execute(
            "UPDATE evaluation SET triage_route = ?, review_status = ? WHERE eval_id = ?",
            (route, review_status, prog["eval_id"]))
        conn.commit()
        conn.close()
    except Exception as e:
        import traceback as _tb
        err_detail = f"memory_write: {e}\n{_tb.format_exc()}"
        state["errors"] = state.get("errors", []) + [err_detail]
        state["trace"].append({"step": "memory_write_error", "error": str(e)})
        # 同时写到 stderr 方便排查
        print(f"[memory_write ERROR] {e}", flush=True)
        prog = None
    return {"progress": prog, "trace": state["trace"]}


def node_finalize(state: AgentState) -> dict:
    """输出最终报告。"""
    report = state["report"]
    report["progress_compare"] = {
        **(report.get("progress_compare") or {}),
        "vs_previous_try_id": (state.get("progress") or {}).get("prev_try_id"),
        "trend": (state.get("progress") or {}).get("trend", report.get("progress_compare", {}).get("trend", "first_try")),
    }
    state["final_report"] = report
    state["trace"].append({"step": "finalize"})
    return {"final_report": report, "report": report, "trace": state["trace"]}


# ---------------- 图构建 ----------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("parse", node_parse)
    g.add_node("perceive", node_perceive)
    g.add_node("retrieve", node_retrieve)
    g.add_node("memory_read", node_memory_read)
    g.add_node("evaluate", node_evaluate)
    g.add_node("synthesize", node_synthesize)
    g.add_node("evidence_check", node_evidence_check)
    g.add_node("triage", node_triage)
    g.add_node("reflect", node_reflect)
    g.add_node("memory_write", node_memory_write)
    g.add_node("finalize", node_finalize)

    g.add_edge(START, "parse")
    g.add_edge("parse", "perceive")
    g.add_edge("perceive", "retrieve")
    g.add_edge("retrieve", "memory_read")
    g.add_edge("memory_read", "evaluate")
    g.add_edge("evaluate", "synthesize")
    g.add_edge("synthesize", "evidence_check")
    g.add_edge("evidence_check", "triage")
    g.add_edge("triage", "reflect")
    g.add_edge("reflect", "memory_write")
    g.add_edge("memory_write", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def run_agent(meta: dict, multimodal: dict | None = None,
              user_id: str | None = None, try_id: str | None = None) -> dict:
    """端到端运行一次试讲评价。返回 {final_report, trace, try_id}。"""
    state: AgentState = {
        "meta": meta,
        "multimodal": multimodal or {},
        "user_id": user_id or meta.get("user_id", "anonymous"),
        "try_id": try_id,
        "trace": [],
        "errors": [],
    }
    # 用轨迹日志器包裹
    tracer = trace_logger.TraceLogger(state["try_id"] or _gen_try_id())
    # 工具调用轨迹钩子：把工具调度写入 trace_logger（append-only，保证可追溯）
    def _tool_hook(name: str, category: str, kwargs: dict) -> None:
        tracer.log("tool_call", tool=name, category=category,
                   args={k: str(v)[:80] for k, v in kwargs.items()})
    add_on_call_hook(_tool_hook)
    graph = build_graph()
    # 附加 tracing：在每个节点调用后写日志（简化：结果返回后写一条）
    try:
        result = graph.invoke(dict(state))
        tracer.log("run_complete", status="ok", try_id=result.get("try_id"),
                   overall=(result.get("final_report") or {}).get("scores", {}).get("overall_percent"))
        tracer.finalize("ok")
    except Exception as e:
        tracer.log("run_error", error=str(e))
        tracer.finalize("error")
        raise
    result["trace"] = result.get("trace", [])
    return result
