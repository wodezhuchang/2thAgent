# -*- coding: utf-8 -*-
"""
知识库管理系统（阶段 1 / RAG 链路）
====================================
统一管理 knowledge_base 下所有权威资料（课标/微格评课量表/优秀试讲案例/评课细则），
支持 PDF / MD / TXT 多源解析、自动分类、增量追加（按文件哈希去重）、文件更新检测、
状态查看、检索。

命令用法（在 backend 目录用 .venv 的 python 运行）：
  python app/scripts/kb_manager.py --update            # 增量更新：扫描并追加/更新已变更文件
  python app/scripts/kb_manager.py --rebuild           # 全量重建（清空后重建）
  python app/scripts/kb_manager.py --status            # 查看已入库文件清单与统计
  python app/scripts/kb_manager.py --search "查询词"    # 检索 Top-K
  python app/scripts/kb_manager.py --search "查询词" --top 8 --doc_type curriculum
  python app/scripts/kb_manager.py --clean             # 清空向量库

设计要点：
  - 去重：以 (相对路径, 文件内容 sha256) 为指纹，未变文件跳过，已变文件删除旧块后重入库。
  - 分类：按文件名关键词推断 doc_type（curriculum/rubric/case/rule），不依赖目录结构。
  - 可追溯：每个块带 source_file / doc_type / subject / file_hash / chunk_index 元数据。
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

# 项目根（脚本位于 backend/app/scripts/，向上三级）
ROOT = Path(__file__).resolve().parents[3]
KB_DIR = ROOT / "knowledge_base"
DB_DIR = ROOT / "backend" / "data" / "chroma"
DEFAULT_EMBED = "BAAI/bge-small-zh-v1.5"
DEFAULT_COLLECTION = "trial_evaluation_rules"
CHUNK_SIZE = 220
CHUNK_OVERLAP = 30

# 支持的扩展名
TEXT_EXTS = {".md", ".txt"}
PDF_EXTS = {".pdf"}
ALL_EXTS = TEXT_EXTS | PDF_EXTS

# 文件名关键词 -> doc_type
TYPE_KEYWORDS = {
    "curriculum": ["课程标准", "课程方案"],
    "rubric": ["评价", "量表", "评分", "考核", "评课"],
    "case": ["教学设计", "试讲案例", "教学案例", "说课"],
}
# 文件所属学科（从文件名中常见学科关键词识别）
SUBJECTS = ["语文", "数学", "英语", "物理", "化学", "生物", "生物学", "历史", "地理",
            "政治", "思想政治", "道德与法治", "科学", "艺术", "音乐", "美术", "体育",
            "体育与健康", "信息科技", "信息技术", "通用技术", "劳动", "课程方案", "学前"]


def file_hash(path: Path) -> str:
    """计算文件内容 sha256（用于变更检测）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(8192), b""):
            h.update(blk)
    return h.hexdigest()


def classify_doc(filename: str) -> tuple[str, str]:
    """根据文件名返回 (doc_type, subject)。"""
    doc_type = "general"
    for typ, kws in TYPE_KEYWORDS.items():
        if any(k in filename for k in kws):
            doc_type = typ
            break
    subject = "general"
    for s in SUBJECTS:
        if s in filename:
            subject = s
            break
    return doc_type, subject


# 扫描型PDF文本过少阈值（低于则触发OCR）
OCR_FALLBACK_THRESHOLD = 200
# OCR 渲染 DPI（越高越清晰，越慢）
OCR_DPI = 200


def _extract_pdf_pypdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts)


def _extract_pdf_ocr(path: Path) -> str:
    """扫描型 PDF：PyMuPDF 渲染每页为图片，RapidOCR 识别。"""
    import pymupdf
    from rapidocr_onnxruntime import RapidOCR

    doc = pymupdf.open(str(path))
    ocr = RapidOCR()
    page_texts = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=OCR_DPI)
        png_bytes = pix.tobytes("png")
        # RapidOCR 支持传入 bytes
        result, _ = ocr(png_bytes)
        if result:
            lines = [line[1] for line in result]
            page_texts.append("\n".join(lines))
        if (i + 1) % 10 == 0:
            print(f"      OCR 进度 {path.stem}: {i+1}/{len(doc)} 页", flush=True)
    return "\n\n".join(page_texts)


def parse_pdf(path: Path) -> str:
    """提取 PDF 文本；若 pypdf 提取过少（扫描件）则自动转 OCR。"""
    text = _extract_pdf_pypdf(path)
    if len(text.strip()) >= OCR_FALLBACK_THRESHOLD:
        return text
    print(f"      {path.name} 为扫描型，走 OCR ...", flush=True)
    try:
        return _extract_pdf_ocr(path)
    except Exception as e:
        print(f"      !! OCR 失败 {path.name}: {e}", file=sys.stderr)
        return text


def read_doc(path: Path) -> str:
    """按扩展名读取文档文本。"""
    if path.suffix.lower() in PDF_EXTS:
        return parse_pdf(path)
    # md/txt 用 utf-8 读，失败则尝试 gbk
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_rule_blocks(text: str) -> list[dict]:
    """把评课细则 md 解析为带 indicator 的块（仅对 rule 类文档生效）。"""
    lines = text.splitlines()
    blocks: list[dict] = []
    cur: dict | None = None
    for ln in lines:
        m = re.match(r"^(#{2,3})\s+(.*)$", ln.strip())
        if m and "细则" in m.group(2):
            if cur is not None:
                blocks.append(cur)
            title = m.group(2).strip()
            ind = re.search(r"(D\d\.\d)", title)
            indicator = ind.group(1) if ind else "unknown"
            name = title.split(" ", 1)[1] if " " in title else ""
            cur = {"doc_id": f"rule_{indicator}", "indicator": indicator,
                   "indicator_name": name, "title": title, "content": []}
        else:
            if cur is not None:
                cur["content"].append(ln)
    if cur is not None:
        blocks.append(cur)
    for b in blocks:
        b["content"] = "\n".join(b["content"]).strip()
    return blocks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """按段落优先的滑动窗口切块。"""
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text.strip()]
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
    final: list[str] = []
    for c in chunks:
        while len(c) > chunk_size:
            final.append(c[:chunk_size])
            c = c[chunk_size - overlap:]
        if c.strip():
            final.append(c)
    return final


def split_doc_into_chunks(path: Path, rel: str, doc_type: str, subject: str,
                          text: str, fhash: str) -> list[dict]:
    """把单文档切成带元数据的 chunk 列表。"""
    chunks: list[dict] = []
    # 评课细则走标题块，其他文档直接切
    if doc_type == "rule" or ("细则" in path.name and path.suffix.lower() in TEXT_EXTS):
        blocks = parse_rule_blocks(text)
        for b in blocks:
            for i, ch in enumerate(chunk_text(b["content"])):
                chunks.append({
                    "id": hashlib.md5(f"{rel}:{b['doc_id']}:{i}".encode()).hexdigest()[:16],
                    "text": ch,
                    "meta": {
                        "doc_type": "rule",
                        "subject": subject,
                        "source_file": rel,
                        "file_hash": fhash,
                        "indicator": b["indicator"],
                        "indicator_name": b["indicator_name"],
                        "title": b["title"],
                        "chunk_index": i,
                    },
                })
    else:
        for i, ch in enumerate(chunk_text(text)):
            chunks.append({
                "id": hashlib.md5(f"{rel}:{i}".encode()).hexdigest()[:16],
                "text": ch,
                "meta": {
                    "doc_type": doc_type,
                    "subject": subject,
                    "source_file": rel,
                    "file_hash": fhash,
                    "title": path.stem,
                    "chunk_index": i,
                },
            })
    return chunks


# 排除的文件名（纯链接清单等，无实质正文，不入库）
EXCLUDE_NAMES = ["可信链接清单", "链接清单", "readme", "说明"]


def scan_files() -> list[Path]:
    """递归收集 knowledge_base 下所有支持的文档文件，排除链接清单类。"""
    files = []
    for p in KB_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALL_EXTS:
            if any(k in p.stem for k in EXCLUDE_NAMES):
                continue
            files.append(p)
    return sorted(files)


def get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def load_collection(client, name, create: bool = True):
    try:
        return client.get_collection(name)
    except Exception:
        if create:
            return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
        raise


def cmd_update(model_name: str, collection_name: str) -> None:
    import chromadb

    model = get_embedder(model_name)
    files = scan_files()
    if not files:
        print("!! knowledge_base 下没有可入库文档")
        return

    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = load_collection(client, collection_name)

    # 读取现有索引：source_file -> set(file_hash)
    existing: dict[str, str] = {}
    try:
        all_meta = col.get(include=["metadatas"])["metadatas"]
        for m in all_meta:
            if m and m.get("source_file"):
                existing[m["source_file"]] = m.get("file_hash", "")
    except Exception:
        existing = {}

    added = updated = skipped = 0
    for path in files:
        rel = path.relative_to(KB_DIR).as_posix()
        fhash = file_hash(path)
        if rel in existing and existing[rel] == fhash:
            skipped += 1
            print(f"  跳过(未变) {rel}")
            continue
        # 变化或新增：删除旧块
        if rel in existing:
            col.delete(where={"source_file": rel})
            updated += 1
            print(f"  更新 {rel}")
        else:
            added += 1
            print(f"  新增 {rel}")
        text = read_doc(path)
        doc_type, subject = classify_doc(path.name)
        chunks = split_doc_into_chunks(path, rel, doc_type, subject, text, fhash)
        if not chunks:
            print(f"    !! {rel} 未提取到文本，跳过")
            continue
        col.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=[model.encode(c["text"], normalize_embeddings=True).tolist() for c in chunks],
            metadatas=[c["meta"] for c in chunks],
        )

    print(f"\n完成: 新增 {added} / 更新 {updated} / 跳过 {skipped}，集合共 {col.count()} 条")


def cmd_rebuild(model_name: str, collection_name: str) -> None:
    import chromadb
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    cmd_update(model_name, collection_name)


def cmd_status(collection_name: str) -> None:
    import chromadb
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        col = client.get_collection(collection_name)
    except Exception:
        print("!! 向量库不存在，请先运行 --update 或 --rebuild")
        return
    metas = col.get(include=["metadatas"])["metadatas"]
    from collections import Counter, defaultdict
    by_type = Counter(m.get("doc_type", "?") for m in metas if m)
    by_subject = Counter(m.get("subject", "?") for m in metas if m)
    files = defaultdict(int)
    for m in metas:
        if m:
            files[m.get("source_file", "?")] += 1
    print(f"集合: {collection_name} | 总片段: {col.count()}")
    print("\n按类型:", dict(by_type))
    print("按学科:", dict(by_subject))
    print(f"\n已入库文件 ({len(files)}):")
    for f in sorted(files):
        print(f"  {f}  ({files[f]} 片段)")


def cmd_search(query: str, model_name: str, collection_name: str, top_k: int,
               doc_type: str | None = None, subject: str | None = None) -> None:
    import chromadb
    model = get_embedder(model_name)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = load_collection(client, collection_name, create=False)
    where = {}
    if doc_type:
        where["doc_type"] = doc_type
    if subject:
        where["subject"] = subject
    emb = model.encode(query, normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=[emb], n_results=top_k,
                    where=where or None, include=["documents", "metadatas", "distances"])
    print(f"检索: {query} (doc_type={doc_type}, subject={subject})\n")
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        ind = f"[{meta.get('indicator')} {meta.get('indicator_name')}]" if meta.get("indicator") else ""
        print(f"--- top {dist:.4f} {ind} ({meta.get('doc_type')}/{meta.get('subject')}) ---")
        print(f"来源: {meta.get('source_file')}")
        print(doc[:180].replace("\n", " "))
        print()


def cmd_clean(collection_name: str) -> None:
    import chromadb
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(collection_name)
        print(f"已清空集合 {collection_name}")
    except Exception:
        print("集合不存在，无需清理")


def main() -> None:
    ap = argparse.ArgumentParser(description="知识库管理系统")
    ap.add_argument("--update", action="store_true", help="增量更新")
    ap.add_argument("--rebuild", action="store_true", help="全量重建")
    ap.add_argument("--status", action="store_true", help="查看状态")
    ap.add_argument("--clean", action="store_true", help="清空")
    ap.add_argument("--search", help="检索查询词")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--doc_type", help="按类型过滤检索")
    ap.add_argument("--subject", help="按学科过滤检索")
    ap.add_argument("--model", default=DEFAULT_EMBED)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = ap.parse_args()

    if args.clean:
        cmd_clean(args.collection)
    elif args.status:
        cmd_status(args.collection)
    elif args.search:
        cmd_search(args.search, args.model, args.collection, args.top, args.doc_type, args.subject)
    elif args.rebuild:
        cmd_rebuild(args.model, args.collection)
    else:
        cmd_update(args.model, args.collection)


if __name__ == "__main__":
    main()
