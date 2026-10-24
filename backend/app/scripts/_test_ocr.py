# -*- coding: utf-8 -*-
"""测试: PyMuPDF渲染扫描PDF页 + RapidOCR识别中文"""
import pymupdf
from rapidocr_onnxruntime import RapidOCR

pdf = r"D:\2thAgent\knowledge_base\外部权威资料\义务教育_数学_义务教育数学课程标准2022年版.pdf"
doc = pymupdf.open(pdf)
page = doc[2]  # 取第3页
pix = page.get_pixmap(dpi=200)
img_path = r"D:\2thAgent\backend\app\scripts\_tmp_page.png"
pix.save(img_path)
print("渲染页尺寸:", pix.width, "x", pix.height)

ocr = RapidOCR()
result, elapse = ocr(img_path)
if result:
    texts = [line[1] for line in result]
    joined = "\n".join(texts)
    print("识别行数:", len(texts))
    print("前300字:", joined[:300])
else:
    print("OCR 未识别到内容")
