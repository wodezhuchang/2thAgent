# -*- coding: utf-8 -*-
"""检查：哪些源文件未入库（数据缺口）"""
import sys
sys.path.insert(0, r"D:\2thAgent\backend\app\scripts")
import kb_manager as kb
import chromadb

client = chromadb.PersistentClient(path=str(kb.DB_DIR))
col = client.get_collection(kb.DEFAULT_COLLECTION)
metas = col.get(include=["metadatas"])["metadatas"]
in_db = {m["source_file"] for m in metas if m and m.get("source_file")}

all_files = kb.scan_files()
not_in = [f.relative_to(kb.KB_DIR).as_posix() for f in all_files if f.relative_to(kb.KB_DIR).as_posix() not in in_db]
print(f"扫描文件总数: {len(all_files)}")
print(f"已入库: {len(in_db)}")
print(f"未入库: {len(not_in)}")
for f in not_in:
    print("  -", f)
