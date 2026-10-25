# -*- coding: utf-8 -*-
"""长期记忆模块（L4 记忆层）：SQLite 结构化试讲档案。

对应《记忆存储结构设计_v1.0.md》：user / trial / evaluation / progress / training_plan 五表。
- 写入：评价完成后落库（evaluation + progress + training_plan）。
- 读取：新一次评价开始时，按 user_id 读取最近一次 evaluation + progress，供进步对比。
短期会话上下文由 LangGraph checkpointer 承担（见 graph 模块），本模块只管长期跨会话档案。

设计要点：
- 所有写操作幂等可重放；try_id 由外部生成，重复写覆盖。
- 进度对比（progress）由 save_evaluation 内部根据前一条自动派生。
- 时间戳统一 ISO8601 字符串。
"""
from __future__ import annotations
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    p = db_path or settings.MEMORY_DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: Path | None = None) -> None:
    """建表（幂等）。"""
    conn = _connect(db_path)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS user (
        user_id      TEXT PRIMARY KEY,
        name         TEXT,
        subject_major TEXT,
        grade_level  TEXT,
        created_at   TEXT
    );
    CREATE TABLE IF NOT EXISTS trial (
        try_id          TEXT PRIMARY KEY,
        user_id         TEXT,
        subject         TEXT,
        lesson_type     TEXT,
        grade_level     TEXT,
        duration_seconds INTEGER,
        video_path      TEXT,
        recorded_at     TEXT,
        meta            TEXT
    );
    CREATE TABLE IF NOT EXISTS evaluation (
        eval_id         TEXT PRIMARY KEY,
        try_id          TEXT,
        user_id         TEXT,
        scores_json     TEXT,
        overall_percent REAL,
        timeline_feedback TEXT,
        weaknesses      TEXT,
        reflection_log  TEXT,
        raw_multimodal  TEXT,
        prompt_version  TEXT,
        created_at      TEXT
    );
    CREATE TABLE IF NOT EXISTS progress (
        user_id       TEXT,
        try_id        TEXT,
        prev_try_id   TEXT,
        trend         TEXT,
        dimension_deltas TEXT,
        still_weak    TEXT,
        resolved      TEXT,
        PRIMARY KEY (user_id, try_id)
    );
    CREATE TABLE IF NOT EXISTS training_plan (
        plan_id     TEXT PRIMARY KEY,
        eval_id     TEXT,
        next_focus  TEXT,
        specific_tasks TEXT,
        reference_case TEXT,
        estimated_repeat_times INTEGER,
        status      TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_eval_user ON evaluation(user_id);
    CREATE INDEX IF NOT EXISTS idx_trial_user ON trial(user_id);
    """)
    conn.commit()
    conn.close()


# ---------------- 写入 ----------------

def upsert_user(user_id: str, name: str | None = None, subject_major: str | None = None,
                grade_level: str | None = None, db_path: Path | None = None) -> None:
    conn = _connect(db_path)
    conn.execute("""
        INSERT INTO user (user_id, name, subject_major, grade_level, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            name=excluded.name, subject_major=excluded.subject_major,
            grade_level=excluded.grade_level
    """, (user_id, name, subject_major, grade_level, _now()))
    conn.commit(); conn.close()


def save_trial(try_id: str, user_id: str, subject: str, lesson_type: str,
               grade_level: str, duration_seconds: int, video_path: str = "",
               meta: dict | None = None, recorded_at: str | None = None,
               db_path: Path | None = None) -> None:
    conn = _connect(db_path)
    conn.execute("""
        INSERT INTO trial (try_id, user_id, subject, lesson_type, grade_level,
                           duration_seconds, video_path, recorded_at, meta)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(try_id) DO UPDATE SET
            user_id=excluded.user_id, subject=excluded.subject,
            lesson_type=excluded.lesson_type, grade_level=excluded.grade_level,
            duration_seconds=excluded.duration_seconds, video_path=excluded.video_path,
            recorded_at=excluded.recorded_at, meta=excluded.meta
    """, (try_id, user_id, subject, lesson_type, grade_level, duration_seconds,
          video_path, recorded_at or _now(), json.dumps(meta, ensure_ascii=False)))
    conn.commit(); conn.close()


def _compute_dim_score(scores: dict, dim: str) -> float:
    """由二级指标分合成维度分（0-5）。对齐指标体系 §6：Dn = Σ(二级权重×得分)。"""
    items = scores.get(dim, {}).get("items", {})
    if not items:
        return 0.0
    # 二级指标默认等权
    return round(sum(items.values()) / len(items), 2)


def _compute_overall(scores: dict) -> float:
    """百分制总评：Σ 维度权重 × 维度分 × 20。对齐指标体系 §6。"""
    total = 0.0
    for dim, w in settings.DIM_WEIGHTS.items():
        dim_score = _compute_dim_score(scores, dim)
        total += w * dim_score
    return round(total * 20, 1)


def get_prev_evaluation(user_id: str, db_path: Path | None = None) -> dict | None:
    """读取该用户最近一次评价（不含当前），用于进步对比。"""
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT * FROM evaluation WHERE user_id=? ORDER BY created_at DESC, eval_id DESC LIMIT 1",
        (user_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return _row_to_eval_dict(row)


def _row_to_eval_dict(row) -> dict:
    d = dict(row)
    for k in ("scores_json", "timeline_feedback", "weaknesses", "reflection_log", "raw_multimodal"):
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except Exception:
                pass
    return d


def save_evaluation(user_id: str, try_id: str, scores: dict,
                    timeline_feedback: list, weaknesses: list,
                    reflection_log: dict | None = None,
                    raw_multimodal: dict | None = None,
                    prompt_version: str | None = None,
                    eval_id: str | None = None,
                    db_path: Path | None = None) -> dict:
    """落 evaluation + 派生 progress + 返回 progress 对比信息。"""
    eval_id = eval_id or f"eval_{uuid.uuid4().hex[:10]}"
    prev = get_prev_evaluation(user_id, db_path)

    overall = _compute_overall(scores)
    # 进步对比
    dim_deltas = {}
    still_weak = []
    resolved = []
    trend = "first_try"
    if prev:
        prev_scores = prev.get("scores_json") or {}
        prev_weak = prev.get("weaknesses") or []
        for dim in settings.DIM_WEIGHTS:
            cur = _compute_dim_score(scores, dim)
            old = _compute_dim_score(prev_scores, dim)
            dim_deltas[dim] = round(cur - old, 2)
        deltas_sum = sum(dim_deltas.values())
        trend = "improved" if deltas_sum > 0.15 else ("declined" if deltas_sum < -0.15 else "flat")
        # 持续短板：上次短板本次仍在 weaknesses；已改善：上次短板本次不在
        cur_weak = set(weaknesses)
        for w in prev_weak:
            # 简单按关键词命中判断（短板通常为短语）
            if any(w in cw or cw in w for cw in cur_weak):
                still_weak.append(w)
            else:
                resolved.append(w)

    conn = _connect(db_path)
    conn.execute("""
        INSERT INTO evaluation (eval_id, try_id, user_id, scores_json, overall_percent,
                                timeline_feedback, weaknesses, reflection_log, raw_multimodal,
                                prompt_version, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (eval_id, try_id, user_id,
          json.dumps(scores, ensure_ascii=False), overall,
          json.dumps(timeline_feedback, ensure_ascii=False),
          json.dumps(weaknesses, ensure_ascii=False),
          json.dumps(reflection_log or {}, ensure_ascii=False),
          json.dumps(raw_multimodal or {}, ensure_ascii=False),
          prompt_version or settings.PROMPT_VERSION, _now()))
    conn.execute("""
        INSERT INTO progress (user_id, try_id, prev_try_id, trend, dimension_deltas, still_weak, resolved)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(user_id, try_id) DO UPDATE SET
            prev_try_id=excluded.prev_try_id, trend=excluded.trend,
            dimension_deltas=excluded.dimension_deltas,
            still_weak=excluded.still_weak, resolved=excluded.resolved
    """, (user_id, try_id, (prev or {}).get("try_id"), trend,
          json.dumps(dim_deltas, ensure_ascii=False),
          json.dumps(still_weak, ensure_ascii=False),
          json.dumps(resolved, ensure_ascii=False)))
    conn.commit(); conn.close()

    return {
        "eval_id": eval_id,
        "overall_percent": overall,
        "prev_try_id": (prev or {}).get("try_id"),
        "trend": trend,
        "dimension_deltas": dim_deltas,
        "still_weak": still_weak,
        "resolved": resolved,
    }


def save_training_plan(eval_id: str, next_focus: str, specific_tasks: list,
                       reference_case: str, estimated_repeat_times: int,
                       plan_id: str | None = None, status: str = "pending",
                       db_path: Path | None = None) -> None:
    conn = _connect(db_path)
    conn.execute("""
        INSERT INTO training_plan (plan_id, eval_id, next_focus, specific_tasks,
                                   reference_case, estimated_repeat_times, status)
        VALUES (?,?,?,?,?,?,?)
    """, (plan_id or f"plan_{uuid.uuid4().hex[:10]}", eval_id, next_focus,
          json.dumps(specific_tasks, ensure_ascii=False), reference_case,
          estimated_repeat_times, status))
    conn.commit(); conn.close()


# ---------------- 读取 ----------------

def get_user_trials(user_id: str, limit: int = 20, db_path: Path | None = None) -> list[dict]:
    """读取用户历史试讲（倒序）。"""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT t.try_id, t.subject, t.lesson_type, t.recorded_at, "
        "       e.overall_percent, e.created_at "
        "FROM trial t LEFT JOIN evaluation e ON e.try_id=t.try_id "
        "WHERE t.user_id=? ORDER BY t.recorded_at DESC LIMIT ?",
        (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_evaluation(eval_id: str, db_path: Path | None = None) -> dict | None:
    conn = _connect(db_path)
    row = conn.execute("SELECT * FROM evaluation WHERE eval_id=?", (eval_id,)).fetchone()
    conn.close()
    return _row_to_eval_dict(row) if row else None


def get_training_plan(eval_id: str, db_path: Path | None = None) -> dict | None:
    conn = _connect(db_path)
    row = conn.execute("SELECT * FROM training_plan WHERE eval_id=?", (eval_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ---------------- 成长可视化（P0：个人成长曲线 + 短板热力图） ----------------

def get_growth_timeline(user_id: str, limit: int = 20, db_path: Path | None = None) -> dict:
    """返回用户历次试讲的成长时间线（按时间升序），供成长曲线/短板热力图可视化。

    返回：{
      "points": [ {index, try_id, date, overall, dims:{D1,D2,D3,D4}, items:{D1.1:.., ..}}, ... ],
      "dim_names": {...}, "item_names": {...}
    }
    """
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT eval_id, try_id, scores_json, overall_percent, created_at "
        "FROM evaluation WHERE user_id=? ORDER BY created_at ASC, eval_id ASC LIMIT ?",
        (user_id, limit)).fetchall()
    conn.close()

    dim_keys = {"D1_teaching_design": "D1", "D2_language": "D2",
                "D3_blackboard": "D3", "D4_presence": "D4"}
    points = []
    for i, row in enumerate(rows):
        scores = {}
        try:
            scores = json.loads(row["scores_json"]) if row["scores_json"] else {}
        except Exception:
            scores = {}
        dims, items = {}, {}
        for d in ("D1", "D2", "D3", "D4"):
            node = scores.get(d, {})
            dims[d] = round(float(node.get("total", 0) or 0), 2)
            for ind, val in (node.get("items") or {}).items():
                items[ind] = round(float(val or 0), 2)
        points.append({
            "index": i + 1,
            "try_id": row["try_id"],
            "date": (row["created_at"] or "")[:10],
            "overall": round(float(row["overall_percent"] or 0), 1),
            "dims": dims,
            "items": items,
        })
    return {"points": points, "count": len(points),
            "dim_names": {"D1": "教学设计", "D2": "讲授语言", "D3": "课堂板书", "D4": "教姿教态"},
            "item_names": settings.INDICATOR_NAMES}
