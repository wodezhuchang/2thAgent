# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"D:\2thAgent\backend\app\scripts")
import kb_manager as kb
from pathlib import Path

# 测试一个之前失败的义务教育课标
p = Path(r"D:\2thAgent\knowledge_base\外部权威资料\义务教育_数学_义务教育数学课程标准2022年版.pdf")
text = kb.parse_pdf(p)
print("提取字符数:", len(text))
print("前200字:", text[:200].replace("\n", " "))
