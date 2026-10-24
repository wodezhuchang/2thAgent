# -*- coding: utf-8 -*-
"""
知识库向量化入库脚本（阶段 1 / RAG 链路骨架）
================================================
功能：
  1. 读取评课细则知识库语料（knowledge_base/*.md）
  2. 按标题块解析 + 文本 chunk 切块
  3. 用 BGE embedding 向量化
  4. 写入 Chroma 向量库（持久化）
  5. 提供检索 demo（供 E1 评审与阶段 2 使用）

用法：
  python scripts/build_kb.py            # 全量构建
  python scripts/build_kb.py --search "导入环节时间过长怎么评价"
  python scripts/build_kb.py --clean    # 清空重建

依赖：chromadb, sentence-transformers, numpy
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

# 项目根目录（脚本位于 backend/app/scripts/，向上三级到项目根）
ROOT = Path(__file__).resolve().parents[3]
KB_DIR = ROOT / "knowledge_base"
DB_DIR = ROOT / "backend" / "data" / "chroma"

DEFAULT_EMBED = "BAAI/bge-small-zh-v1.5"
DEFAULT_COLLECTION = "trial_evaluation_rules"

# 标题块解析：识别 "### 细则 D1.1 目标定位" / "## D1 教学设计合理性" 等
BLOCK_RE = re.compile(r"^(#{2,3})\s+(.*)$")


def parse_rule_docs(md_text: str) -> list[dict]:
    """把 md 语料解析为若干带元数据的块。

    返回 [{doc_id, dimension, indicator, indicator_name, title, content, tags}]
    以 "### 细则 Dxx.x" 为切分点；无细则标题的行并入前一块。
    """
    lines = md_text.splitlines()
    blocks: list[dict] = []
    cur: dict | None = None
    for ln in lines:
        m = BLOCK_RE.match(ln.strip())
        if m and "细则" in m.group(2):
            if cur is not None:
                blocks.append(cur)
            title = m.group(2).strip()          # 如 "细则 D1.1 目标定位"
            ind = re.search(r"(D\d\.\d)", title)
            indicator = ind.group(1) if ind else "unknown"
            dim = indicator.split(".")[0] if indicator != "unknown" else "unknown"
            name = title.split(" ", 1)[1] if " " in title else ""
            cur = {
                "doc_id": f"rule_{indicator}",
                "dimension": dim,
                "indicator": indicator,
                "indicator_name": name,
                "title": title,
                "content": [],
                "tags": [dim, name],
            }
        else:
            if cur is not None:
                cur["content"].append(ln)
    if cur is not None:
        blocks.append(cur)
    for b in blocks:
        b["content"] = "\n".join(b["content"]).strip()
    return blocks


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 30) -> list[str]:
    """简单滑动窗口切块，按段落优先，保证块间有重叠以利检索。"""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 > chunk_size and buf:
            chunks.append(buf)
            buf = buf[-overlap:] if overlap else ""
        buf = (buf + "\n" + p).strip()
    if buf:
        chunks.append(buf)
    # 仍可能超过 chunk_size 的超长段，硬切
    final: list[str] = []
    for c in chunks:
        while len(c) > chunk_size:
            final.append(c[:chunk_size])
            c = c[chunk_size - overlap :]
        if c.strip():
            final.append(c)
    return final


def build_kb(model_name: str, collection_name: str) -> None:
    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"[1/5] 加载 embedding 模型: {model_name}")
    model = SentenceTransformer(model_name)

    print("[2/5] 解析评课细则语料")
    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        print("!! knowledge_base 下没有 .md 语料", file=sys.stderr)
        sys.exit(1)

    docs: list[dict] = []
    for f in md_files:
        print(f"    - {f.name}")
        blocks = parse_rule_docs(f.read_text(encoding="utf-8"))
        for b in blocks:
            for ch in chunk_text(b["content"]):
                docs.append({
                    "id": hashlib.md5(f"{b['doc_id']}:{ch}".encode()).hexdigest()[:16],
                    "text": ch,
                    "meta": {
                        "doc_id": b["doc_id"],
                        "dimension": b["dimension"],
                        "indicator": b["indicator"],
                        "indicator_name": b["indicator_name"],
                        "title": b["title"],
                        "tags": ",".join(b["tags"]),
                        "source": f.name,
                    },
                })
    print(f"    共 {len(docs)} 个检索片段")

    print("[3/5] 生成 embedding")
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    print(f"[4/5] 写入 Chroma（持久化: {DB_DIR}）")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    col.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        embeddings=[emb.tolist() for emb in embeddings],
        metadatas=[d["meta"] for d in docs],
    )
    print(f"    集合 {collection_name} 写入 {col.count()} 条")

    print("[5/5] 完成")
    print(f"    知识库路径: {DB_DIR}")


def search(query: str, model_name: str, collection_name: str, top_k: int = 5) -> None:
    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"检索: {query}\n")
    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_collection(collection_name)
    emb = model.encode(query, normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=[emb], n_results=top_k, include=["documents", "metadatas", "distances"])
    for i, (doc, meta, dist) in enumerate(zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    )):
        print(f"--- top{i+1} (dist={dist:.4f}) [{meta.get('indicator')} {meta.get('indicator_name')}] ---")
        print(doc[:200])
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="评课细则知识库构建/检索")
    ap.add_argument("--model", default=DEFAULT_EMBED)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--search", help="检索模式，给出查询词")
    ap.add_argument("--clean", action="store_true", help="清空重建")
    args = ap.parse_args()

    if args.search:
        search(args.search, args.model, args.collection)
    else:
        build_kb(args.model, args.collection)


if __name__ == "__main__":
    main()
