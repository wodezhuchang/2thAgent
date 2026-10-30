# -*- coding: utf-8 -*-
"""知识库精选回流：专家复评 vs 机器初评的争议课例，脱敏后回流入向量库。

设计约束（延续既有分层方案，不重构稳定架构）：
- 校准数据仍走现有 evaluation + expert_review 表（compute_calibration_offset 已统计人机差异），本模块不重复造表。
- 向量库仍是单一 Chroma collection（settings.COLLECTION），这里只是新增 doc_type=case 的"争议课例"片段。
- 不改动 memory_store / graph / retriever / kb_index，只在 expert-review 提交后旁路触发。
- 精选回流门槛：总分差或某维度分差超过阈值，且专家填写了校正意见，避免知识库无限膨胀/噪声。
- 幂等：同一 try_id 只回流一次。
- 脱敏：case 文本不包含学生姓名 / user_id / 学号等个人信息，仅含学科·课题·课型与评分、专家意见。
- 任何异常都不上抛，不影响专家复评主流程。
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from app.config import settings

# 复用 kb_manager 的 embedder / collection 加载，与既有 RAG 同一套
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kb_manager as kb  # noqa: E402

# 回流门槛（可按需调整）
RECYCLE_OVERALL_THRESHOLD = 8.0   # |机器分 - 专家分| 总分差 >= 8 触发
RECYCLE_DIM_THRESHOLD = 2.0       # 任一维度分差 >= 2 触发
DIM_NAMES = {"D1": "教学设计", "D2": "讲授语言", "D3": "课堂板书", "D4": "教姿教态"}


def _connect_db(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else settings.MEMORY_DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _load_case_data(try_id: str) -> dict | None:
    """聚合试讲元数据、机器评价、专家复评三段数据。"""
    conn = _connect_db()
    try:
        trial = conn.execute("SELECT * FROM trial WHERE try_id = ?", (try_id,)).fetchone()
        ev = conn.execute(
            "SELECT overall_percent, scores_json, timeline_feedback, weaknesses "
            "FROM evaluation WHERE try_id = ?", (try_id,)).fetchone()
        er = conn.execute(
            "SELECT expert_id, final_scores, comments, disagree_dims, score_deltas "
            "FROM expert_review WHERE try_id = ?", (try_id,)).fetchone()
    finally:
        conn.close()
    if not ev or not er:
        return None
    meta = json.loads((trial["meta"] or "{}")) if trial else {}
    return {
        "try_id": try_id,
        "subject": (trial["subject"] if trial else "") or "general",
        "lesson": meta.get("lesson", ""),
        "lesson_type": (trial["lesson_type"] if trial else "") or "",
        "grade_level": (trial["grade_level"] if trial else "") or "",
        "expert_id": er["expert_id"] or "",
        "machine_overall": float(ev["overall_percent"] or 0),
        "machine_scores": json.loads(ev["scores_json"] or "{}"),
        "timeline_feedback": json.loads(ev["timeline_feedback"] or "[]") or [],
        "weaknesses": json.loads(ev["weaknesses"] or "[]") or [],
        "expert_scores": json.loads(er["final_scores"] or "{}"),
        "comments": er["comments"] or "",
        "disagree_dims": json.loads(er["disagree_dims"] or "[]") or [],
        "score_deltas": json.loads(er["score_deltas"] or "{}") or {},
    }


def _expert_overall(data: dict) -> float:
    return float(data["expert_scores"].get("overall_percent") or data["machine_overall"])


def _should_recycle(data: dict) -> tuple[bool, str]:
    """判定是否达到精选回流门槛。"""
    if not data["comments"].strip():
        return False, "专家未填写校正意见，不回流"
    delta = abs(data["machine_overall"] - _expert_overall(data))
    dim_max = max((abs(v) for v in data["score_deltas"].values()), default=0.0)
    if delta >= RECYCLE_OVERALL_THRESHOLD or dim_max >= RECYCLE_DIM_THRESHOLD:
        return True, f"总分差{delta:.0f}或维度差{dim_max:.1f}超阈值"
    return False, f"差异未达阈值（总分差{delta:.0f}，维度差{dim_max:.1f}）"


def _build_case_text(data: dict) -> str:
    """构造脱敏的争议课例文本（不含学生姓名/user_id/学号等个人信息）。"""
    head = data["subject"]
    if data["lesson"]:
        head += f"·{data['lesson']}"
    if data["lesson_type"]:
        head += f"·{data['lesson_type']}"
    lines = [f"争议课例（{head}）：机器初评与专家复评存在显著分歧，供评课参照边界。"]
    e_overall = _expert_overall(data)
    # 口径与系统校准一致：正 = 专家比机器高 / 机器偏低
    lines.append(
        f"机器评分 {data['machine_overall']} 分，专家评分 {e_overall} 分，"
        f"差异 {e_overall - data['machine_overall']:+.0f}。"
    )
    if data["disagree_dims"]:
        parts = [f"{DIM_NAMES.get(d, d)}({data['score_deltas'].get(d, '')})"
                 for d in data["disagree_dims"]]
        lines.append("分歧维度：" + "、".join(parts) + "。")
    if data["weaknesses"]:
        lines.append("机器识别短板：" + "；".join(str(w) for w in data["weaknesses"][:3]))
    if data["timeline_feedback"]:
        segs = [str(t.get("suggestion", "")) for t in data["timeline_feedback"][:2] if t.get("suggestion")]
        if segs:
            lines.append("机器细节评语摘录：" + "；".join(segs))
    if data["comments"]:
        lines.append("专家校正意见：" + str(data["comments"]).strip())
    return "\n".join(lines)


def recycle_disputed_case(try_id: str, db_path: Path | None = None) -> dict:
    """专家复评提交后旁路触发：争议案例脱敏回流入向量库（幂等）。

    返回 {"recycled": bool, "reason": str, "try_id": str, ...}。
    任何异常都会被捕获并返回 recycled=False，绝不阻断专家复评主流程。
    """
    try:
        data = _load_case_data(try_id)
        if not data:
            return {"recycled": False, "reason": "未找到试讲/复评记录", "try_id": try_id}

        ok, reason = _should_recycle(data)
        if not ok:
            return {"recycled": False, "reason": reason, "try_id": try_id}

        import chromadb
        client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
        col = kb.load_collection(client, settings.COLLECTION)
        # 幂等：同 try_id 已回流则跳过
        already = False
        try:
            for m in col.get(include=["metadatas"])["metadatas"]:
                if m and m.get("try_id") == try_id:
                    already = True
                    break
        except Exception:  # noqa: BLE001
            already = False
        if already:
            return {"recycled": False, "reason": "该 try_id 已回流，跳过", "try_id": try_id}

        text = _build_case_text(data)
        emb = kb.get_embedder(settings.EMBED_MODEL).encode(text, normalize_embeddings=True).tolist()
        doc_id = hashlib.md5(("recycled:" + try_id).encode()).hexdigest()[:16]
        meta = {
            "doc_type": "case",
            "subject": data["subject"],
            "source_file": f"recycled/{try_id}",
            "file_hash": try_id,
            "title": f"争议课例·{data['subject'] or '通用'}·{data['lesson_type'] or '试讲'}",
            "chunk_index": 0,
            "try_id": try_id,
            "expert_id": data["expert_id"],
            "machine_overall": data["machine_overall"],
            "expert_overall": _expert_overall(data),
        }
        col.add(ids=[doc_id], documents=[text], embeddings=[emb], metadatas=[meta])
        return {"recycled": True, "reason": "已回流入库为 case", "try_id": try_id,
                "doc_id": doc_id, "subject": data["subject"]}
    except Exception as e:  # noqa: BLE001
        return {"recycled": False, "reason": f"回流失败（不影响主流程）: {e}", "try_id": try_id}
