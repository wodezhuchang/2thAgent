# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"D:\2thAgent\backend\app\scripts")
import kb_manager as kb

files = kb.scan_files()
print("扫描到文档数:", len(files))
for f in files[:8]:
    rel = f.relative_to(kb.KB_DIR).as_posix()
    dt, sub = kb.classify_doc(f.name)
    print(f"  {rel} -> type={dt}, subject={sub}")

# 统计类型分布
from collections import Counter
types = Counter(kb.classify_doc(f.name)[0] for f in files)
subs = Counter(kb.classify_doc(f.name)[1] for f in files)
print("类型分布:", dict(types))
print("学科分布(前10):", dict(subs.most_common(10)))
