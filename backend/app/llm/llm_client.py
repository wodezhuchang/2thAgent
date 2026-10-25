# -*- coding: utf-8 -*-
"""LLM 接入与 Prompt 构建（L1 模型层 + L5 上下文层）。

- LLM 客户端：OpenAI 兼容 SDK 接 DeepSeek（api_key 从环境变量读，不硬编码）。
- build_eval_prompt：把"系统提示 + 检索到的评课依据 + 历史记忆 + 试讲多模态信号"
  拼装为上下文稳定的 Prompt（防上下文漂移，对应 L5）。
- 四维并行评价与 Reflexion 的调用在 evaluator.py 中实现。
"""
from __future__ import annotations
import json
import os
from typing import Any

from openai import OpenAI

from app.config import settings


def get_llm_client() -> OpenAI:
    api_key = os.environ.get(settings.LLM_API_KEY_ENV, "")
    return OpenAI(base_url=settings.LLM_BASE_URL, api_key=api_key or "sk-dummy")


SYSTEM_PROMPT = """【角色】
你是一名资深师范教育学科教学专家与微格教学评课员，擅长从多维度对师范生试讲进行
专业、结构化、可落地的评价。你的服务对象是在校师范生，你的目标不是打分本身，而是
帮助他们在自主练习中持续改进教学技能。

【任务边界（必须遵守）】
1. 你的评价是"AI 辅助练习建议"，不作为课程最终考核结果；专业评判以任课教师为准。
2. 你不生成完整试讲教案，不替代教师的专业判断，不做无边界闲聊。
3. 你只依据系统注入的课标、评课量表、优秀课例（RAG 检索结果）与试讲多模态信号进行
   评价，不凭主观自由发挥。

【评价依据】
- 评价指标体系：《{indicator_version}》
- 课标/量表/课例（本次检索到的权威依据，如无则说明"未检索到强依据"）：
{retrieved_context}
- 用户历史（供对比）：{memory_summary}

【评价流程】（Plan-and-Execute）
1. 解析用户信息：学科、学段、课题、课型、试讲时长。
2. 对 D1 教学设计、D2 讲授语言、D3 板书、D4 教姿教态四个维度逐项评价，依据检索到的
   量表与锚点打分（0–5）。
3. 若发现某维度证据不足或与检索到的课例冲突，可先检索再校准（ReAct）。
4. 汇总为结构化评分表 + 分段式细节评语（定位时间轴）+ 短板清单。
5. 读取该用户上次试讲记录，对比本次，判断进步/退步/持平。
6. 生成下一轮针对性训练规划（明确练什么、参考哪个示范案例）。
7. 输出前做一次自查（Reflexion）：检查是否违背课标/量表、有无幻觉、是否遗漏短板，
   如发现问题则修正后输出。

【输出要求】
严格按 JSON Schema 输出，不要输出模板之外的自由文本。
"""


def build_system_prompt(retrieved_context: str, memory_summary: str) -> str:
    return SYSTEM_PROMPT.format(
        indicator_version=settings.INDICATOR_VERSION,
        retrieved_context=retrieved_context,
        memory_summary=memory_summary,
    )


def _json_schema() -> str:
    return json.dumps({
        "meta": {"user_id": "string", "try_id": "string", "subject": "string",
                 "lesson": "string", "lesson_type": "string", "grade_level": "string",
                 "duration_seconds": "number", "recorded_at": "ISO8601"},
        "disclaimer": "本评价为 AI 辅助练习建议，不作为课程最终考核结果，专业评判以任课教师为准。",
        "scores": {
            "D1_teaching_design": {"total": 0, "items": {k: 0 for k in settings.DIM_INDICATORS["D1"]},
                                    "weight": settings.DIM_WEIGHTS["D1"]},
            "D2_language": {"total": 0, "items": {k: 0 for k in settings.DIM_INDICATORS["D2"]},
                            "weight": settings.DIM_WEIGHTS["D2"]},
            "D3_blackboard": {"total": 0, "items": {k: 0 for k in settings.DIM_INDICATORS["D3"]},
                              "weight": settings.DIM_WEIGHTS["D3"]},
            "D4_presence": {"total": 0, "items": {k: 0 for k in settings.DIM_INDICATORS["D4"]},
                            "weight": settings.DIM_WEIGHTS["D4"]},
            "overall_percent": 0,
        },
        "timeline_feedback": [
            {"timestamp_start": "mm:ss", "timestamp_end": "mm:ss", "segment": "导入/新授/...",
             "issue": "string", "suggestion": "string"}
        ],
        "weaknesses": ["string"],
        "progress_compare": {
            "vs_previous_try_id": "string", "trend": "improved|declined|flat|first_try",
            "dimension_deltas": {k: 0 for k in settings.DIM_WEIGHTS},
            "still_weak": ["string"],
        },
        "training_plan": {
            "next_focus": "string", "specific_tasks": ["string"],
            "reference_case": "string", "estimated_repeat_times": "number",
        },
    }, ensure_ascii=False)


def build_user_prompt(meta: dict, multimodal: dict) -> str:
    """构建用户侧 Prompt：试讲元信息 + 多模态信号（转录/板书/姿态等）。"""
    lines = []
    lines.append(f"试讲信息：学科={meta.get('subject')}，课题={meta.get('lesson')}，"
                 f"课型={meta.get('lesson_type')}，学段={meta.get('grade_level')}，"
                 f"时长={meta.get('duration_seconds')}秒")
    # 多模态信号（阶段3 逐步丰富；当前用可用的）
    if multimodal.get("asr_text"):
        lines.append(f"\n【讲授语音转录】\n{multimodal['asr_text']}")
    if multimodal.get("board_text"):
        lines.append(f"\n【板书 OCR 文字】\n{multimodal['board_text']}")
    if multimodal.get("timeline"):
        lines.append(f"\n【环节时间轴】\n{json.dumps(multimodal['timeline'], ensure_ascii=False)}")
    for k in ("posture", "expression", "audio_feats"):
        if multimodal.get(k):
            lines.append(f"\n【{k}特征】\n{json.dumps(multimodal[k], ensure_ascii=False)}")
    if not any(k in multimodal for k in ("asr_text", "board_text", "timeline", "posture", "expression", "audio_feats")):
        lines.append("\n（本次未提供多模态信号，仅依据教案元信息与通用师范规范评价）")
    return "\n".join(lines)


def chat_json(client: OpenAI, system: str, user: str, temperature: float | None = None) -> Any:
    """调用 LLM 并解析 JSON 输出（容错：剥离代码块围栏）。"""
    resp = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    content = content.strip()
    if content.startswith("```"):
        # 剥离 ```json ... ```
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip("` \n")
    return json.loads(content)
