# -*- coding: utf-8 -*-
"""RAG 检索封装（L4 记忆层 / RAG 检索链路）。

阶段1 已建成向量库（Chroma + BGE）。此处封装为 LangGraph 节点可调用的
retrieve() 函数：按"学科+课题+课型+维度"构造查询，召回课标/量表/课例片段，
并组装为可注入 Prompt 的上下文文本。

对应开发前文件 §3.6.3 工具调用（RAG 检索工具）+ §3.5 检索时机。
"""
from __future__ import annotations
import sys
from pathlib import Path

# 复用阶段1 的 kb_manager（含加载集合/embedding 逻辑），避免重复实现
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kb_manager as kb  # noqa: E402

from app.config import settings  # noqa: E402


def retrieve(query: str, subject: str | None = None,
             doc_type: str | None = None, top_k: int | None = None,
             collection: str | None = None) -> list[dict]:
    """RAG 检索：返回召回片段列表（含文本、来源、类型、距离）。

    返回每个片段 dict: {text, source, doc_type, subject, distance, indicator}
    """
    model = kb.get_embedder(settings.EMBED_MODEL)
    import chromadb
    client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
    col = kb.load_collection(client, collection or settings.COLLECTION, create=False)
    where = {}
    if doc_type:
        where["doc_type"] = doc_type
    if subject:
        where["subject"] = subject
    emb = model.encode(query, normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=[emb],
                    n_results=top_k or settings.RAG_TOP_K,
                    where=where or None,
                    include=["documents", "metadatas", "distances"])
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({
            "text": doc,
            "source": meta.get("source_file", "?"),
            "doc_type": meta.get("doc_type", "?"),
            "subject": meta.get("subject", "?"),
            "distance": float(dist),
            "indicator": meta.get("indicator"),
        })
    return hits


def build_context(hits: list[dict], max_chars: int = 4000) -> str:
    """把召回片段组装为可注入 Prompt 的评课依据文本。"""
    if not hits:
        return "(本次未检索到强评课依据，请基于通用师范教学规范评价)"
    parts = []
    used = 0
    for i, h in enumerate(hits, 1):
        src = f"{h['indicator']}" if h.get("indicator") else h["source"]
        seg = f"[{i}] {h['text'].strip()}\n     (来源: {src})"
        if used + len(seg) > max_chars:
            break
        parts.append(seg)
        used += len(seg)
    return "\n".join(parts)
