# -*- coding: utf-8 -*-
"""独立验证向量库真实内容（不依赖 build_kb.py 的输出）"""
import chromadb
from sentence_transformers import SentenceTransformer

DB = r"D:\2thAgent\backend\data\chroma"
client = chromadb.PersistentClient(path=DB)
col = client.get_collection("trial_evaluation_rules")

print("=== 集合真实条数 ===")
print("count =", col.count())

print("=== 抽样文档(前3条) ===")
data = col.get(limit=3, include=["documents", "metadatas"])
for d, m in zip(data["documents"], data["metadatas"]):
    print(f"[{m.get('indicator')} {m.get('indicator_name')}] {d[:60]}...")

print("=== 实际检索: 板书错别字 ===")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
q = model.encode("师范生试讲时板书出现错别字如何评价", normalize_embeddings=True).tolist()
res = col.query(query_embeddings=[q], n_results=3)
for doc, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"dist={dist:.3f} [{m.get('indicator')} {m.get('indicator_name')}] {doc[:50]}...")
