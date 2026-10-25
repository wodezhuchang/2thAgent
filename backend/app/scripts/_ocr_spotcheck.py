# -*- coding: utf-8 -*-
"""OCR 质量抽检：对 4 个代表课标抽样页重新 OCR，检查识别质量。"""
import pymupdf
from rapidocr_onnxruntime import RapidOCR
from pathlib import Path

# (文件名, 抽检页码1, 页码2)
SAMPLES = [
    (r"外部权威资料\义务教育_数学_义务教育数学课程标准2022年版.pdf", 4, 60),
    (r"外部权威资料\义务教育_英语_义务教育英语课程标准2022年版.pdf", 5, 90),
    (r"外部权威资料\义务教育_科学_义务教育科学课程标准2022年版.pdf", 4, 100),
    (r"外部权威资料\义务教育_历史_义务教育历史课程标准2022年版.pdf", 4, 50),
]
BASE = Path(r"D:\2thAgent\knowledge_base")
ocr = RapidOCR()

for rel, p1, p2 in SAMPLES:
    path = BASE / rel
    doc = pymupdf.open(str(path))
    print("=" * 60)
    print(f"文件: {Path(rel).name} (共{len(doc)}页)  抽检页: {p1}, {p2}")
    for pno in (p1, p2):
        page = doc[pno - 1]
        pix = page.get_pixmap(dpi=200)
        result, _ = ocr(pix.tobytes("png"))
        text = "\n".join(line[1] for line in result) if result else "(无识别)"
        print(f"\n--- 第{pno}页 (识别{len(result) if result else 0}行) ---")
        print(text[:220].replace("\n", " "))
    doc.close()
