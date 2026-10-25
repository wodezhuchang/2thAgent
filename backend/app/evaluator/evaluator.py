# -*- coding: utf-8 -*-
"""评价核心（领域逻辑，全自研——对应 §3.7 自研边界）。

实现：
- eval_all_dimensions：四维（D1-D4）并行评价，每维调用 LLM 输出该维二级指标分与评语。
- reflect：Reflexion 自查环，检验是否违背课标/量表、有无幻觉、是否遗漏短板，必要时重跑。
- 输出严格对齐《Agent_SystemPrompt与输出模板_v1.0.md》JSON Schema。
"""
from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.config import settings
from app.llm import llm_client


def _dim_prompt(dim: str, meta: dict, context: str, memory_summary: str,
                multimodal: dict) -> tuple[str, str]:
    """构建单个维度的 System+User Prompt。"""
    inds = settings.DIM_INDICATORS[dim]
    names = {k: settings.INDICATOR_NAMES[k] for k in inds}
    dim_names = {"D1": "教学设计合理性", "D2": "讲授语言表达", "D3": "课堂板书规范", "D4": "教姿教态"}
    sys_extra = (
        f"本次仅评价【{dim_names[dim]}】维度。\n"
        f"该维度二级指标及含义：{json.dumps(names, ensure_ascii=False)}\n"
        f"请对每个二级指标按 0-5 打分（5=优秀，3=基本达标，1=差），并给出该维度1-2句总体评语。\n"
        f"依据系统注入的课标/量表/课例，不要凭空打分。\n"
        f"严格输出 JSON：{{\"{dim}\": {{\"items\": {{各指标:分数}}, \"comment\": \"总体评语\"}}}}"
    )
    system = llm_client.build_system_prompt(context, memory_summary) + "\n" + sys_extra
    user = llm_client.build_user_prompt(meta, multimodal)
    return system, user


def eval_single_dimension(client, dim: str, meta: dict, context: str,
                          memory_summary: str, multimodal: dict) -> dict:
    """评价单个维度，返回 {dim, items:{指标:分}, comment}。"""
    system, user = _dim_prompt(dim, meta, context, memory_summary, multimodal)
    out = llm_client.chat_json(client, system, user)
    # 容错解析：可能返回 {D1:{...}} 或直接 {items:...}
    if dim in out:
        out = out[dim]
    items = out.get("items") or {}
    # 只保留合法指标键，缺失补0
    valid = {k: float(items.get(k, 0)) for k in settings.DIM_INDICATORS[dim]}
    return {"dim": dim, "items": valid, "comment": out.get("comment", "")}


def eval_all_dimensions(client, meta: dict, context: str, memory_summary: str,
                        multimodal: dict) -> dict:
    """四维并行评价，聚合为 scores 结构（对齐 JSON Schema）。

    阶段2 串行遍历；阶段3 改为 ThreadPoolExecutor 并行（LLM 调用为 IO 密集，
    线程池合适；chat_json 每次新建请求，无共享可变状态，线程安全）。
    """
    dims = list(settings.DIM_WEIGHTS.keys())
    with ThreadPoolExecutor(max_workers=len(dims)) as ex:
        futures = {ex.submit(eval_single_dimension, client, dim, meta, context,
                             memory_summary, multimodal): dim for dim in dims}
        results = {}
        for f in futures:
            dim = futures[f]
            try:
                results[dim] = f.result()
            except Exception as e:
                # 单维失败兜底：全 0 + 说明
                results[dim] = {"dim": dim,
                                "items": {k: 0 for k in settings.DIM_INDICATORS[dim]},
                                "comment": f"该维度评价失败: {e}"}

    # 组装 scores（对齐 schema：total 为该维平均，weight 取配置）
    scores = {}
    for dim, w in settings.DIM_WEIGHTS.items():
        items = results[dim]["items"]
        total = round(sum(items.values()) / len(items), 2)
        key = {"D1": "D1_teaching_design", "D2": "D2_language",
               "D3": "D3_blackboard", "D4": "D4_presence"}[dim]
        scores[key] = {"total": total, "items": items, "weight": w}
    # 百分制
    overall = 0.0
    for dim, w in settings.DIM_WEIGHTS.items():
        items = results[dim]["items"]
        dim_avg = sum(items.values()) / len(items)
        overall += w * dim_avg
    scores["overall_percent"] = round(overall * 20, 1)
    return scores, results


def synthesize(client, meta: dict, scores: dict, dim_results: dict,
               context: str, memory_summary: str) -> dict:
    """合成：时间轴评语 + 短板清单 + 进步对比 + 训练规划（一次调用）。"""
    system = llm_client.build_system_prompt(context, memory_summary) + (
        "\n基于已给出的四维评分，完成最后三项：\n"
        "1. timeline_feedback：分段式细节评语，每条约含 timestamp_start/end、segment、issue、suggestion。\n"
        "2. weaknesses：短板清单（2-5条，可落地）。\n"
        "3. progress_compare：对比用户历史（trend 取 improved/declined/flat/first_try）。\n"
        "4. training_plan：next_focus、specific_tasks（2-4条）、reference_case、estimated_repeat_times。\n"
        "严格输出 JSON：{timeline_feedback, weaknesses, progress_compare, training_plan}"
    )
    user = llm_client.build_user_prompt(meta, {"__scores__": scores}) + \
        f"\n\n当前四维评分：{json.dumps(scores, ensure_ascii=False)}"
    return llm_client.chat_json(client, system, user)


def reflect(client, meta: dict, report: dict, context: str) -> dict:
    """Reflexion 自查环：检验是否有幻觉/违背课标/遗漏短板，返回是否修正。"""
    system = (
        "你是评课质检员。检查以下评价报告是否符合课标/量表、有无明显幻觉或违背师范评课规范、"
        "有无遗漏明显短板。只输出 JSON：{needs_fix:bool, issues:[...], fixed_report:当前报告或修正后的报告}"
    )
    user = f"评课依据：{context[:2000]}\n\n待检报告：{json.dumps(report, ensure_ascii=False)}"
    try:
        out = llm_client.chat_json(client, system, user)
    except Exception:
        return {"needs_fix": False, "issues": [], "fixed_report": report}
    if out.get("needs_fix"):
        return {"needs_fix": True, "issues": out.get("issues", []),
                "fixed_report": out.get("fixed_report") or report}
    return {"needs_fix": False, "issues": out.get("issues", []), "fixed_report": report}
