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
import hashlib
import hmac
import secrets
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
    CREATE TABLE IF NOT EXISTS expert_review (
        review_id      TEXT PRIMARY KEY,
        try_id         TEXT,
        expert_id      TEXT,
        reviewed_at    TEXT,
        final_scores   TEXT,
        comments       TEXT,
        disagree_dims  TEXT,
        score_deltas   TEXT
    );
    CREATE TABLE IF NOT EXISTS message (
        message_id     TEXT PRIMARY KEY,
        user_id        TEXT,
        type           TEXT,
        related_try_id TEXT,
        title          TEXT,
        content        TEXT,
        is_read        INTEGER DEFAULT 0,
        created_at     TEXT
    );
    CREATE TABLE IF NOT EXISTS chat_message (
        message_id   TEXT PRIMARY KEY,
        user_id      TEXT,
        role         TEXT,
        content      TEXT,
        intent       TEXT,
        session_id   TEXT,
        created_at   TEXT
    );
    CREATE TABLE IF NOT EXISTS teacher_student (
        teacher_id TEXT NOT NULL,
        student_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (teacher_id, student_id)
    );
    -- 绑定申请（线下协商 → 学生发起 → 专家确认后写入 teacher_student）
    CREATE TABLE IF NOT EXISTS binding_request (
        student_id TEXT NOT NULL,
        teacher_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',  -- pending / accepted / rejected
        created_at TEXT NOT NULL,
        updated_at TEXT,
        PRIMARY KEY (student_id, teacher_id)
    );
    CREATE INDEX IF NOT EXISTS idx_binding_req_teacher ON binding_request(teacher_id, status);
    CREATE INDEX IF NOT EXISTS idx_eval_user ON evaluation(user_id);
    CREATE INDEX IF NOT EXISTS idx_trial_user ON trial(user_id);
    CREATE INDEX IF NOT EXISTS idx_message_user ON message(user_id, is_read);
    CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_message(user_id, session_id);
    """)
    # 迁移：检查 evaluation 表是否已有新字段
    cols = [row[1] for row in conn.execute("PRAGMA table_info(evaluation)").fetchall()]
    if "review_status" not in cols:
        conn.execute("ALTER TABLE evaluation ADD COLUMN review_status TEXT DEFAULT 'pending'")
    if "triage_route" not in cols:
        conn.execute("ALTER TABLE evaluation ADD COLUMN triage_route TEXT")
    if "expert_score" not in cols:
        conn.execute("ALTER TABLE evaluation ADD COLUMN expert_score REAL")
    if "calibration_note" not in cols:
        conn.execute("ALTER TABLE evaluation ADD COLUMN calibration_note TEXT")
    # 多角色改造：evaluation 归属（谁试讲 / 推给哪位专家）
    if "student_id" not in cols:
        conn.execute("ALTER TABLE evaluation ADD COLUMN student_id TEXT")
    if "expert_id" not in cols:
        conn.execute("ALTER TABLE evaluation ADD COLUMN expert_id TEXT")
    # user 表补登录/角色字段（多角色账号体系）
    ucols = [row[1] for row in conn.execute("PRAGMA table_info(user)").fetchall()]
    if "username" not in ucols:
        conn.execute("ALTER TABLE user ADD COLUMN username TEXT")
    if "password_hash" not in ucols:
        conn.execute("ALTER TABLE user ADD COLUMN password_hash TEXT")
    if "role" not in ucols:
        conn.execute("ALTER TABLE user ADD COLUMN role TEXT DEFAULT 'student'")
    if "account_type" not in ucols:
        conn.execute("ALTER TABLE user ADD COLUMN account_type TEXT DEFAULT 'real'")
    if "org" not in ucols:
        conn.execute("ALTER TABLE user ADD COLUMN org TEXT DEFAULT ''")
    if "invite_code" not in ucols:
        conn.execute("ALTER TABLE user ADD COLUMN invite_code TEXT")
    # message 定向推送
    mcols = [row[1] for row in conn.execute("PRAGMA table_info(message)").fetchall()]
    if "recipient_id" not in mcols:
        conn.execute("ALTER TABLE message ADD COLUMN recipient_id TEXT")
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


# ---------------- 账号与师生关系（多角色改造） ----------------

def _hash_password(password: str) -> str:
    """pbkdf2_hmac 哈希（零依赖），返回 salt$dk。"""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100_000)
    return f"{salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split('$', 1)
        dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'),
                                 bytes.fromhex(salt_hex), 100_000)
        return hmac.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


def create_user(user_id: str, username: str, password: str,
                name: str = '', role: str = 'student',
                account_type: str = 'real', org: str = '',
                db_path: Path | None = None) -> None:
    """创建账号（幂等）。password 明文入库前哈希。teacher 自动生成邀请口令。"""
    invite_code = _gen_invite_code() if role == 'teacher' else None
    conn = _connect(db_path)
    conn.execute("""
        INSERT INTO user (user_id, name, subject_major, grade_level, created_at,
                          username, password_hash, role, account_type, org, invite_code)
        VALUES (?, ?, '', '', ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            username=excluded.username, password_hash=excluded.password_hash,
            role=excluded.role, account_type=excluded.account_type, org=excluded.org
    """, (user_id, name, _now(), username, _hash_password(password),
          role, account_type, org, invite_code))
    conn.commit(); conn.close()


def get_user_by_username(username: str, db_path: Path | None = None) -> dict | None:
    conn = _connect(db_path)
    row = conn.execute("SELECT * FROM user WHERE username=?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: str, db_path: Path | None = None) -> dict | None:
    conn = _connect(db_path)
    row = conn.execute("SELECT * FROM user WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def _gen_invite_code() -> str:
    """生成 8 位邀请口令（去易混淆字符 O/0/I/1）。"""
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    return ''.join(secrets.choice(alphabet) for _ in range(8))


def ensure_invite_code(user_id: str, db_path: Path | None = None) -> str:
    """确保账号有邀请口令（教师），无则生成回写。返回口令。"""
    conn = _connect(db_path)
    row = conn.execute("SELECT invite_code FROM user WHERE user_id=?",
                       (user_id,)).fetchone()
    if row and row["invite_code"]:
        conn.close()
        return row["invite_code"]
    code = _gen_invite_code()
    conn.execute("UPDATE user SET invite_code=? WHERE user_id=?", (code, user_id))
    conn.commit(); conn.close()
    return code


def get_teacher_by_invite_code(code: str, db_path: Path | None = None) -> dict | None:
    """按邀请口令解析教师账号（用于学生凭口令申请绑定）。"""
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT * FROM user WHERE invite_code=? AND role='teacher'", (code,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_users(role: str | None = None, db_path: Path | None = None) -> list[dict]:
    conn = _connect(db_path)
    if role:
        rows = conn.execute(
            "SELECT user_id, username, name, role, account_type, org FROM user WHERE role=?",
            (role,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT user_id, username, name, role, account_type, org FROM user").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_teacher_student(teacher_id: str, student_id: str,
                        db_path: Path | None = None) -> None:
    """建立指导关系（分发依据）。"""
    conn = _connect(db_path)
    conn.execute(
        "INSERT INTO teacher_student (teacher_id, student_id, created_at) "
        "VALUES (?,?,?) ON CONFLICT(teacher_id, student_id) DO NOTHING",
        (teacher_id, student_id, _now()))
    conn.commit(); conn.close()


def resolve_expert_for_student(student_id: str,
                               db_path: Path | None = None) -> str | None:
    """由学生解析其指导专家（teacher_id）。用于评价分发。"""
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT teacher_id FROM teacher_student WHERE student_id=? LIMIT 1",
        (student_id,)).fetchone()
    conn.close()
    return row["teacher_id"] if row else None


# ---------------- 师生绑定（线下协商 → 线上确认） ----------------

def request_binding(student_id: str, teacher_id: str,
                    db_path: Path | None = None) -> dict:
    """学生发起绑定申请（幂等）。已有生效绑定则返回 already_bound。"""
    conn = _connect(db_path)
    bound = conn.execute(
        "SELECT 1 FROM teacher_student WHERE teacher_id=? AND student_id=?",
        (teacher_id, student_id)).fetchone()
    if bound:
        conn.close()
        return {"ok": True, "status": "already_bound"}
    conn.execute("""
        INSERT INTO binding_request (student_id, teacher_id, status, created_at, updated_at)
        VALUES (?,?, 'pending', ?, ?)
        ON CONFLICT(student_id, teacher_id) DO UPDATE SET
            status='pending', updated_at=excluded.updated_at
    """, (student_id, teacher_id, _now(), _now()))
    conn.commit(); conn.close()
    return {"ok": True, "status": "pending"}


def list_binding_requests(teacher_id: str | None = None,
                          student_id: str | None = None,
                          status: str | None = None,
                          db_path: Path | None = None) -> list[dict]:
    """查询绑定申请（专家待确认列表 / 学生查看自己的申请）。"""
    conn = _connect(db_path)
    sql = ("SELECT br.student_id, br.teacher_id, br.status, br.created_at, "
           "u.name AS teacher_name, u2.name AS student_name "
           "FROM binding_request br "
           "LEFT JOIN user u ON u.user_id = br.teacher_id "
           "LEFT JOIN user u2 ON u2.user_id = br.student_id WHERE 1=1")
    args = []
    if teacher_id:
        sql += " AND br.teacher_id = ?"; args.append(teacher_id)
    if student_id:
        sql += " AND br.student_id = ?"; args.append(student_id)
    if status:
        sql += " AND br.status = ?"; args.append(status)
    sql += " ORDER BY br.created_at DESC"
    rows = conn.execute(sql, args).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def confirm_binding(teacher_id: str, student_id: str,
                    db_path: Path | None = None) -> dict:
    """专家确认绑定：写入 teacher_student，更新申请为 accepted。"""
    conn = _connect(db_path)
    req = conn.execute(
        "SELECT 1 FROM binding_request WHERE student_id=? AND teacher_id=? AND status='pending'",
        (student_id, teacher_id)).fetchone()
    if not req:
        conn.close()
        return {"ok": False, "reason": "不存在待确认的绑定申请"}
    conn.execute("""
        UPDATE binding_request SET status='accepted', updated_at=?
        WHERE student_id=? AND teacher_id=?
    """, (_now(), student_id, teacher_id))
    conn.execute(
        "INSERT INTO teacher_student (teacher_id, student_id, created_at) "
        "VALUES (?,?,?) ON CONFLICT(teacher_id, student_id) DO NOTHING",
        (teacher_id, student_id, _now()))
    conn.commit(); conn.close()
    return {"ok": True, "status": "accepted"}


def reject_binding(teacher_id: str, student_id: str,
                   db_path: Path | None = None) -> dict:
    """专家拒绝绑定申请。"""
    conn = _connect(db_path)
    conn.execute("""
        UPDATE binding_request SET status='rejected', updated_at=?
        WHERE student_id=? AND teacher_id=?
    """, (_now(), student_id, teacher_id))
    conn.commit(); conn.close()
    return {"ok": True, "status": "rejected"}


def is_bound(teacher_id: str, student_id: str,
             db_path: Path | None = None) -> bool:
    """是否已建立师生绑定。"""
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT 1 FROM teacher_student WHERE teacher_id=? AND student_id=?",
        (teacher_id, student_id)).fetchone()
    conn.close()
    return bool(row)


def list_student_teachers(student_id: str, db_path: Path | None = None) -> list[dict]:
    """学生已绑定的老师列表（供发送试讲时指定发送对象）。"""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT ts.teacher_id, u.name AS teacher_name, u.subject_major, u.org "
        "FROM teacher_student ts LEFT JOIN user u ON u.user_id = ts.teacher_id "
        "WHERE ts.student_id = ? ORDER BY ts.created_at", (student_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_teacher_students(teacher_id: str, db_path: Path | None = None) -> list[dict]:
    """专家已绑定的学生列表。"""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT ts.student_id, u.name AS student_name "
        "FROM teacher_student ts LEFT JOIN user u ON u.user_id = ts.student_id "
        "WHERE ts.teacher_id = ? ORDER BY ts.created_at", (teacher_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_teacher_students_dashboard(teacher_id: str, db_path: Path | None = None) -> list[dict]:
    """教师端「学生洞察」聚合：教师名下每个学生的关键学情指标。

    返回每项：{user_id, name, last_score, avg_score, delta, trend,
               weakest_dim, weakest_value, pending_count, review_count,
               trial_count, last_at}
    """
    import json
    conn = _connect(db_path)
    sts = conn.execute(
        "SELECT ts.student_id, u.name AS student_name "
        "FROM teacher_student ts LEFT JOIN user u ON u.user_id = ts.student_id "
        "WHERE ts.teacher_id = ? ORDER BY ts.created_at", (teacher_id,)).fetchall()

    result = []
    for st in sts:
        uid = st["student_id"]
        rows = conn.execute(
            "SELECT e.eval_id, e.try_id, e.overall_percent, e.scores_json, e.review_status, e.created_at "
            "FROM evaluation e WHERE e.user_id=? ORDER BY e.created_at ASC, e.eval_id ASC",
            (uid,)).fetchall()

        rc = conn.execute(
            "SELECT COUNT(*) AS c FROM expert_review er JOIN evaluation e ON e.try_id=er.try_id "
            "WHERE e.user_id=?", (uid,)).fetchone()["c"]

        if not rows:
            result.append({
                "user_id": uid, "name": st["student_name"] or uid,
                "last_score": 0, "avg_score": 0, "delta": 0, "trend": "flat",
                "weakest_dim": "", "weakest_value": 0,
                "pending_count": 0, "review_count": int(rc or 0),
                "trial_count": 0, "last_at": "",
            })
            continue

        last = rows[-1]
        prev = rows[-2] if len(rows) >= 2 else None
        last_score = round(float(last["overall_percent"] or 0), 1)
        avg_score = round(sum(float(r["overall_percent"] or 0) for r in rows) / len(rows), 1)
        delta = round(last_score - round(float(prev["overall_percent"] or 0), 1), 1) if prev else 0.0
        trend = "up" if delta > 0 else ("down" if delta < 0 else "flat")

        try:
            scores = json.loads(last["scores_json"] or "{}")
        except Exception:
            scores = {}
        dims = {d: round(float(scores.get(d, {}).get("total", 0) or 0), 2)
                for d in ("D1", "D2", "D3", "D4")}
        wd = min(dims, key=lambda d: dims[d])
        pending = sum(1 for r in rows
                      if (r["review_status"] is None or r["review_status"] not in
                          ("auto_pass", "expert_reviewed", "returned")))

        result.append({
            "user_id": uid, "name": st["student_name"] or uid,
            "last_score": last_score, "avg_score": avg_score,
            "delta": delta, "trend": trend,
            "weakest_dim": wd, "weakest_value": dims[wd],
            "pending_count": pending, "review_count": int(rc or 0),
            "trial_count": len(rows), "last_at": (last["created_at"] or "")[:10],
        })
    conn.close()
    return result


def get_student_reviews(user_id: str, db_path: Path | None = None) -> list[dict]:
    """某学生的全部专家复评历史（机器分 vs 专家分、争议维度、意见）。"""
    import json
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT e.try_id, e.overall_percent AS machine_overall, e.scores_json, "
        "       er.final_scores, er.comments, er.disagree_dims, er.reviewed_at, "
        "       t.subject, t.lesson_type "
        "FROM evaluation e JOIN expert_review er ON e.try_id = er.try_id "
        "LEFT JOIN trial t ON t.try_id = e.try_id "
        "WHERE e.user_id = ? ORDER BY er.reviewed_at ASC", (user_id,)).fetchall()
    conn.close()

    result = []
    for r in rows:
        machine_overall = round(float(r["machine_overall"] or 0), 1)
        expert = {}
        try:
            expert = json.loads(r["final_scores"] or "{}")
        except Exception:
            expert = {}
        expert_overall = round(float(expert.get("overall_percent", 0)), 1)
        try:
            disagree = json.loads(r["disagree_dims"] or "[]")
        except Exception:
            disagree = []
        result.append({
            "try_id": r["try_id"],
            "subject": r["subject"] or r["lesson_type"] or "试讲",
            "machine_overall": machine_overall,
            "expert_overall": expert_overall,
            "delta": round(expert_overall - machine_overall, 1),
            "disagree_dims": disagree,
            "comments": r["comments"] or "",
            "reviewed_at": (r["reviewed_at"] or ""),
        })
    return result


def get_student_training_plans(user_id: str, db_path: Path | None = None) -> list[dict]:
    """某学生的历次训练计划（按时间升序）。"""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT tp.next_focus, tp.specific_tasks, tp.reference_case, tp.status, "
        "       e.try_id, e.overall_percent, e.created_at "
        "FROM training_plan tp JOIN evaluation e ON e.eval_id = tp.eval_id "
        "WHERE e.user_id = ? ORDER BY e.created_at ASC", (user_id,)).fetchall()
    conn.close()
    return [{
        "try_id": r["try_id"],
        "next_focus": r["next_focus"] or "",
        "specific_tasks": r["specific_tasks"] or "",
        "reference_case": r["reference_case"] or "",
        "status": r["status"] or "pending",
        "overall_percent": round(float(r["overall_percent"] or 0), 1),
        "date": (r["created_at"] or "")[:10],
    } for r in rows]


def send_message(user_id: str, title: str, content: str,
                 msg_type: str = "system", related_try_id: str = "",
                 recipient_id: str | None = None,
                 db_path: Path | None = None) -> str:
    """发送一条消息到 message 表（学生端消息铃铛可见）。返回 message_id。"""
    import uuid
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    conn = _connect(db_path)
    conn.execute("""
        INSERT INTO message (message_id, user_id, type, related_try_id, title, content,
                             is_read, created_at, recipient_id)
        VALUES (?,?,?,?,?,?,0,?,?)
    """, (msg_id, user_id, msg_type, related_try_id, title, content, _now(), recipient_id))
    conn.commit(); conn.close()
    return msg_id


def return_review(try_id: str, teacher_id: str, reason: str = "",
                  db_path: Path | None = None) -> dict:
    """教师退回试讲：标记 evaluation 为 returned，并通知学生重新上传给正确老师。
    按需求不指明具体老师，仅提示重新上传。"""
    conn = _connect(db_path)
    ev = conn.execute(
        "SELECT user_id, expert_id FROM evaluation WHERE try_id=?",
        (try_id,)).fetchone()
    if not ev:
        conn.close()
        return {"ok": False, "reason": "该试讲记录不存在"}
    if ev["expert_id"] and ev["expert_id"] != teacher_id:
        conn.close()
        return {"ok": False, "reason": "该试讲不属于你，无法退回"}
    conn.execute(
        "UPDATE evaluation SET review_status='returned' WHERE try_id=?", (try_id,))
    conn.commit(); conn.close()
    note = "该试讲未被采用，请核对科目/标题/发送对象后重新上传给正确的老师。"
    if reason:
        note += f"（{reason}）"
    send_message(ev["user_id"], "试讲被退回，请重新上传",
                 note, msg_type="review_returned", related_try_id=try_id,
                 db_path=db_path)
    return {"ok": True, "try_id": try_id}


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
                    student_id: str | None = None,
                    expert_id: str | None = None,
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
                                prompt_version, created_at, student_id, expert_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (eval_id, try_id, user_id,
          json.dumps(scores, ensure_ascii=False), overall,
          json.dumps(timeline_feedback, ensure_ascii=False),
          json.dumps(weaknesses, ensure_ascii=False),
          json.dumps(reflection_log or {}, ensure_ascii=False),
          json.dumps(raw_multimodal or {}, ensure_ascii=False),
          prompt_version or settings.PROMPT_VERSION, _now(),
          student_id or user_id, expert_id))
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
    """读取用户历史试讲（倒序）。以 evaluation 为主表聚合，保证无论 trial 表是否完整都能返回真实历史。"""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT e.try_id, t.subject, t.lesson_type, e.overall_percent, e.created_at "
        "FROM evaluation e LEFT JOIN trial t ON t.try_id=e.try_id "
        "WHERE e.user_id=? ORDER BY e.created_at DESC LIMIT ?",
        (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# 维度短键 -> 前端报告长键
_DIM_LONG_KEYS = {
    "D1": "D1_teaching_design", "D2": "D2_language",
    "D3": "D3_blackboard", "D4": "D4_presence",
}


def get_full_report(try_id: str, db_path: Path | None = None) -> dict | None:
    """按 try_id 聚合完整 AgentReport（evaluation + trial + training_plan + progress）。

    供前端报告页 `/api/try/{try_id}` 使用；无该次记录时返回 None。
    """
    conn = _connect(db_path)
    ev = conn.execute("SELECT * FROM evaluation WHERE try_id=?", (try_id,)).fetchone()
    if not ev:
        conn.close()
        return None
    tr = conn.execute("SELECT * FROM trial WHERE try_id=?", (try_id,)).fetchone()
    plan = conn.execute(
        "SELECT * FROM training_plan WHERE eval_id=?", (ev["eval_id"],)).fetchone()
    prog = conn.execute(
        "SELECT * FROM progress WHERE user_id=? AND try_id=?",
        (ev["user_id"], try_id)).fetchone()
    conn.close()

    ev_d = _row_to_eval_dict(ev)
    scores = {}
    for dim, long in _DIM_LONG_KEYS.items():
        node = (ev_d.get("scores_json") or {}).get(dim, {})
        scores[long] = {
            "total": node.get("total", 0),
            "weight": settings.DIM_WEIGHTS.get(dim, 0),
            "items": node.get("items", {}),
        }
    scores["overall_percent"] = ev["overall_percent"]

    tr_meta = {}
    if tr and tr.get("meta"):
        try:
            tr_meta = json.loads(tr["meta"])
        except Exception:
            tr_meta = {}

    report: dict = {
        "meta": {
            "user_id": ev["user_id"], "try_id": try_id,
            "subject": (tr and tr["subject"]) or tr_meta.get("subject") or "",
            "lesson": tr_meta.get("lesson") or "",
            "lesson_type": (tr and tr["lesson_type"]) or tr_meta.get("lesson_type") or "",
            "grade_level": (tr and tr["grade_level"]) or tr_meta.get("grade_level") or "",
            "duration_seconds": (tr and tr["duration_seconds"]) or tr_meta.get("duration_seconds") or 600,
            "recorded_at": ev["created_at"],
        },
        "scores": scores,
        "timeline_feedback": ev_d.get("timeline_feedback") or [],
        "weaknesses": ev_d.get("weaknesses") or [],
        "review_status": (ev["review_status"] if "review_status" in ev.keys() else None) or "pending_expert",
    }
    if plan:
        report["training_plan"] = {
            "next_focus": plan["next_focus"],
            "specific_tasks": json.loads(plan["specific_tasks"] or "[]"),
            "reference_case": plan["reference_case"],
            "estimated_repeat_times": plan["estimated_repeat_times"],
        }
    if prog:
        report["progress_compare"] = {
            "vs_previous_try_id": prog["prev_try_id"],
            "trend": prog["trend"],
            "dimension_deltas": json.loads(prog["dimension_deltas"] or "{}"),
            "still_weak": json.loads(prog["still_weak"] or "[]"),
        }
    return report


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
        "SELECT e.eval_id, e.try_id, e.scores_json, e.overall_percent, e.created_at, "
        "       (er.review_id IS NOT NULL) AS expert_reviewed "
        "FROM evaluation e LEFT JOIN expert_review er ON er.try_id=e.try_id "
        "WHERE e.user_id=? ORDER BY e.created_at ASC, e.eval_id ASC LIMIT ?",
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
            "expert_reviewed": bool(row["expert_reviewed"]),
        })
    return {"points": points, "count": len(points),
            "dim_names": {"D1": "教学设计", "D2": "讲授语言", "D3": "课堂板书", "D4": "教姿教态"},
            "item_names": settings.INDICATOR_NAMES}


def get_pending_reviews(expert_id: str | None = None, limit: int = 50,
                        db_path: Path | None = None) -> list[dict]:
    """查询待专家复评的试讲列表（非 auto_pass 的都需要复评）。

    expert_id 为空返回全部（admin 公共池）；非空则按专家归属过滤。
    """
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT e.try_id, e.user_id, e.overall_percent, e.triage_route, e.expert_id, "
        "e.created_at, t.subject, t.lesson_type "
        "FROM evaluation e LEFT JOIN trial t ON e.try_id = t.try_id "
        "WHERE (e.review_status IS NULL OR e.review_status != 'auto_pass') "
        "AND (? IS NULL OR e.expert_id = ?) "
        "ORDER BY e.created_at DESC LIMIT ?",
        (expert_id, expert_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------- 专家复评 ----------------

def submit_expert_review(try_id: str, expert_id: str, final_scores: dict,
                          comments: str, db_path: Path | None = None) -> dict:
    """专家提交复评意见，写入 expert_review 表，更新 evaluation 状态，发消息给学生。"""
    import uuid
    conn = _connect(db_path)
    # 先查机器分，计算差异
    eval_row = conn.execute(
        "SELECT eval_id, user_id, overall_percent, scores_json FROM evaluation WHERE try_id = ?",
        (try_id,)).fetchone()
    if not eval_row:
        conn.close()
        raise ValueError(f"try_id {try_id} 不存在")

    machine_scores = json.loads(eval_row["scores_json"] or "{}")
    machine_overall = float(eval_row["overall_percent"] or 0)
    expert_overall = round(float(final_scores.get("overall_percent", 0)), 1)
    delta = round(expert_overall - machine_overall, 1)

    # 分歧维度
    disagree_dims = []
    score_deltas = {}
    for d in ("D1", "D2", "D3", "D4"):
        m_dim = machine_scores.get(d, {}).get("total", 0)
        e_dim = final_scores.get(d, {}).get("total", 0)
        d_diff = round(e_dim - m_dim, 1)
        score_deltas[d] = d_diff
        if abs(d_diff) >= 5:
            disagree_dims.append(d)

    review_id = f"rev_{uuid.uuid4().hex[:12]}"
    conn.execute("""
        INSERT INTO expert_review (review_id, try_id, expert_id, reviewed_at,
                                   final_scores, comments, disagree_dims, score_deltas)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (review_id, try_id, expert_id, _now(),
          json.dumps(final_scores, ensure_ascii=False),
          comments,
          json.dumps(disagree_dims, ensure_ascii=False),
          json.dumps(score_deltas, ensure_ascii=False)))

    # 更新 evaluation 状态
    conn.execute("""
        UPDATE evaluation SET review_status = 'expert_reviewed',
                              expert_score = ?,
                              calibration_note = ?
        WHERE try_id = ?
    """, (expert_overall, json.dumps(score_deltas, ensure_ascii=False), try_id))

    # 给学生发消息
    user_id = eval_row["user_id"]
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    diff_text = f"（机器分 {machine_overall} → 专家分 {expert_overall}，差 {delta:+}）"
    conn.execute("""
        INSERT INTO message (message_id, user_id, type, related_try_id, title, content, is_read, created_at)
        VALUES (?, ?, 'expert_review_ready', ?, ?, ?, 0, ?)
    """, (msg_id, user_id, try_id,
          "你的试讲专家复核已完成",
          f"试讲 try_id={try_id} 的专家复核意见已出。{diff_text} 点击查看详情。",
          _now()))
    conn.commit()
    conn.close()
    return {"review_id": review_id, "machine_overall": machine_overall,
            "expert_overall": expert_overall, "delta": delta,
            "disagree_dims": disagree_dims}


def get_unread_messages(user_id: str, db_path: Path | None = None) -> list[dict]:
    """查询用户未读消息。"""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM message WHERE (user_id = ? OR recipient_id = ?) AND is_read = 0 "
        "ORDER BY created_at DESC",
        (user_id, user_id)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_messages_read(user_id: str, message_ids: list[str] | None = None,
                       db_path: Path | None = None) -> int:
    """标记消息为已读。不传 message_ids 则全部标记已读。"""
    conn = _connect(db_path)
    if message_ids:
        placeholders = ",".join("?" * len(message_ids))
        conn.execute(
            f"UPDATE message SET is_read = 1 WHERE (user_id = ? OR recipient_id = ?) "
            f"AND message_id IN ({placeholders})",
            (user_id, user_id, *message_ids))
    else:
        conn.execute("UPDATE message SET is_read = 1 WHERE user_id = ? OR recipient_id = ?",
                     (user_id, user_id))
    conn.commit()
    count = conn.total_changes
    conn.close()
    return count


# ---------------- chat 对话 ----------------

def save_chat_message(user_id: str, role: str, content: str, intent: str,
                      session_id: str, db_path: Path | None = None) -> str:
    """存一条 chat 消息。"""
    import uuid
    msg_id = f"chat_{uuid.uuid4().hex[:12]}"
    conn = _connect(db_path)
    conn.execute("""
        INSERT INTO chat_message (message_id, user_id, role, content, intent, session_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (msg_id, user_id, role, content, intent, session_id, _now()))
    conn.commit()
    conn.close()
    return msg_id


def get_chat_history(user_id: str, session_id: str, limit: int = 20,
                    db_path: Path | None = None) -> list[dict]:
    """查某 session 的对话历史。"""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT role, content, intent, created_at FROM chat_message "
        "WHERE user_id = ? AND session_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, session_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


def list_chat_sessions(user_id: str, db_path: Path | None = None) -> list[dict]:
    """按 session 聚合该用户的对话会话列表。

    每条含：session_id、title（首条 user 消息截断）、intent（首条 user 消息意图，用于主题分类）、
    first_at / last_at、msg_count。
    """
    conn = _connect(db_path)
    rows = conn.execute("""
        SELECT cm.session_id,
               (SELECT c2.content FROM chat_message c2
                 WHERE c2.user_id=cm.user_id AND c2.session_id=cm.session_id AND c2.role='user'
                 ORDER BY c2.created_at ASC LIMIT 1) AS title,
               (SELECT c3.intent FROM chat_message c3
                 WHERE c3.user_id=cm.user_id AND c3.session_id=cm.session_id AND c3.role='user'
                 ORDER BY c3.created_at ASC LIMIT 1) AS intent,
               MIN(cm.created_at) AS first_at,
               MAX(cm.created_at) AS last_at,
               COUNT(*) AS msg_count
        FROM chat_message cm
        WHERE cm.user_id = ?
        GROUP BY cm.session_id
        ORDER BY last_at DESC
    """, (user_id,)).fetchall()
    conn.close()
    items = []
    for r in rows:
        d = dict(r)
        d["title"] = (d.get("title") or "").strip()[:40]
        d["intent"] = d.get("intent") or "casual"
        items.append(d)
    return items


def get_session_messages(user_id: str, session_id: str,
                         db_path: Path | None = None) -> list[dict]:
    """查某会话全部消息（时间正序，用于回看完整对话）。"""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT role, content, intent, created_at FROM chat_message "
        "WHERE user_id = ? AND session_id = ? ORDER BY created_at ASC",
        (user_id, session_id)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------- 系统校准 ----------------

def compute_calibration_offset(db_path: Path | None = None) -> dict:
    """计算系统偏差：机器分均值 - 专家分均值。
    攒够足够样本后返回各维度偏差，样本不足返回空。
    """
    conn = _connect(db_path)
    rows = conn.execute("""
        SELECT e.scores_json, e.overall_percent, er.final_scores
        FROM evaluation e
        JOIN expert_review er ON e.try_id = er.try_id
        WHERE e.review_status = 'expert_reviewed'
    """).fetchall()
    conn.close()

    if len(rows) < 5:
        return {"sample_size": len(rows), "offsets": {}, "note": "样本不足，至少需要5个专家复评"}

    offsets = {"overall": [], "D1": [], "D2": [], "D3": [], "D4": []}
    for row in rows:
        machine = json.loads(row["scores_json"] or "{}")
        expert = json.loads(row["final_scores"] or "{}")
        # 总分偏差
        m_overall = float(row["overall_percent"] or 0)
        e_overall = float(expert.get("overall_percent", 0))
        offsets["overall"].append(e_overall - m_overall)
        # 各维度偏差
        for d in ("D1", "D2", "D3", "D4"):
            m_dim = machine.get(d, {}).get("total", 0)
            e_dim = expert.get(d, {}).get("total", 0)
            offsets[d].append(e_dim - m_dim)

    # 算平均值
    result = {"sample_size": len(rows), "offsets": {}}
    for k, vals in offsets.items():
        if vals:
            result["offsets"][k] = round(sum(vals) / len(vals), 2)
    result["note"] = "正值=机器分偏低需上调，负值=机器分偏高需下调"
    return result


def get_calibrated_score(overall_percent: float, dimension_scores: dict,
                         db_path: Path | None = None) -> dict:
    """根据系统偏差校准分数。样本不足则原样返回。"""
    cal = compute_calibration_offset(db_path)
    offsets = cal.get("offsets", {})
    if not offsets:
        return {"raw": overall_percent, "calibrated": overall_percent,
                "offset": 0, "calibrated": False, "note": cal.get("note", "")}

    offset = offsets.get("overall", 0)
    calibrated = round(overall_percent + offset, 1)
    return {
        "raw": overall_percent,
        "calibrated": calibrated,
        "offset": offset,
        "calibrated": True,
        "sample_size": cal.get("sample_size", 0),
        "note": f"基于{cal.get('sample_size')}个专家复评校准"
    }
