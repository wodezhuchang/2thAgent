# -*- coding: utf-8 -*-
"""Agent 可插拔工具集合（L3 工具层）。

把领域能力封装为"插件式工具"注册进 ToolRegistry，Agent 通过工具名调度。
当前已封装：
- rag_retrieve：RAG 知识库检索（课标/量表/课例）——对应阶段3 第①步。
后续（同文件追加）：asr_transcribe / ocr_board / timeline_slice / posture_analyze 等。

每个工具 = 一个普通函数 + 一份 ToolSpec（元数据，对齐 MCP tool 结构）。
"""
from __future__ import annotations
from typing import Any

from app.tools.registry import ToolSpec, ToolRegistry, registry
from app.rag import retriever
from app.config import settings


# ============================================================
# ① RAG 检索工具
# ============================================================
def _rag_retrieve(query: str, subject: str | None = None,
                  doc_type: str | None = None, top_k: int | None = None) -> dict:
    """调用 RAG 检索，返回召回片段列表 + 拼装好的评课依据上下文。"""
    hits = retriever.retrieve(query, subject=subject, doc_type=doc_type,
                              top_k=top_k or settings.RAG_TOP_K)
    context = retriever.build_context(hits)
    return {"hits": hits, "context": context}


RAG_SPEC = ToolSpec(
    name="rag_retrieve",
    description="从评课知识库检索课标、评课量表、优秀试讲案例，返回召回片段与拼装好的评课依据上下文。"
                "用于评价前锚定专业规范。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索查询，如'初中数学一次函数教学设计评价标准'"},
            "subject": {"type": "string", "description": "学科过滤（可选）"},
            "doc_type": {"type": "string", "description": "类型过滤：curriculum/rubric/case/rule（可选）"},
            "top_k": {"type": "integer", "description": "召回条数（可选，默认配置）"},
        },
        "required": ["query"],
    },
    output_schema={"type": "object", "properties": {"hits": "array", "context": "string"}},
    category="rag",
)


# ============================================================
# 工具注册：这里集中注册所有内置工具
# ============================================================
def register_builtin_tools(reg: ToolRegistry = registry) -> None:
    reg.register(RAG_SPEC, _rag_retrieve)


# 模块导入即注册（保证任何入口拿到注册表即有工具）
register_builtin_tools()
