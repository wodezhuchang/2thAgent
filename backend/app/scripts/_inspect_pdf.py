# -*- coding: utf-8 -*-
from pypdf import PdfReader
from pathlib import Path

p = Path(r"D:\2thAgent\knowledge_base\外部权威资料\义务教育_数学_义务教育数学课程标准2022年版.pdf")
r = PdfReader(str(p))
print("页数:", len(r.pages))
pg = r.pages[0]
print("页面对象属性:", [k for k in pg.keys()])
# 检查是否有 XObject 图片
xobj = pg.get("/Resources", {}).get("/XObject", {}) if pg.get("/Resources") else {}
print("XObject 条目:", {k: str(v) for k, v in list(xobj.items())[:5]})
# 检查是否有字体
fonts = pg.get("/Resources", {}).get("/Font", {}) if pg.get("/Resources") else {}
print("Font 条目:", {k: str(v) for k, v in list(fonts.items())[:5]})
