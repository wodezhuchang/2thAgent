# -*- coding: utf-8 -*-
"""统计哪些PDF是扫描型（无文本层）"""
from pypdf import PdfReader
from pathlib import Path

base = Path(r"D:\2thAgent\knowledge_base")
pdfs = [p for p in base.rglob("*.pdf")]
scanned = []
textual = []
for p in sorted(pdfs):
    try:
        r = PdfReader(str(p))
        pg = r.pages[0]
        fonts = pg.get("/Resources", {}).get("/Font", {}) if pg.get("/Resources") else {}
        has_font = bool(fonts)
        (textual if has_font else scanned).append((p.name, len(r.pages)))
    except Exception as e:
        scanned.append((p.name, f"ERR:{e}"))

print(f"文本型 PDF（可pypdf提取）: {len(textual)}")
for n, pp in textual:
    print("   ", n, pp, "页")
print(f"\n扫描型 PDF（需OCR）: {len(scanned)}")
for n, pp in scanned:
    print("   ", n, pp, "页")
