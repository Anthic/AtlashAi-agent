"""
memory/history.py
────────────────────────────────────────────────────────────
Production history store — Supabase PostgreSQL.

Schema (auto-created on first run):
    research_history (
        id          SERIAL PRIMARY KEY,
        job_id      TEXT,
        topic       TEXT NOT NULL,
        report      TEXT,
        critique    TEXT,
        score       FLOAT DEFAULT 0,
        fact_score  FLOAT DEFAULT 0,
        urls        JSONB,
        time_sec    FLOAT DEFAULT 0,
        created_at  BIGINT NOT NULL
    )
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

DATABASE_URL = (
    os.getenv("DATABASE_URL_POOLER")   # Session Pooler (preferred) — port 5432 or 6543
    or os.getenv("DATABASE_URL")       # Direct connection fallback
    or ""
)

# ── DDL ────────────────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS research_history (
    id         SERIAL PRIMARY KEY,
    job_id     TEXT,
    topic      TEXT    NOT NULL,
    report     TEXT,
    critique   TEXT,
    score      FLOAT   DEFAULT 0,
    fact_score FLOAT   DEFAULT 0,
    urls       JSONB   DEFAULT '[]',
    time_sec   FLOAT   DEFAULT 0,
    created_at BIGINT  NOT NULL
);
"""

_MIGRATIONS = [
    "ALTER TABLE research_history ADD COLUMN IF NOT EXISTS job_id TEXT;",
    "CREATE INDEX IF NOT EXISTS research_history_job_id_idx ON research_history (job_id);",
    "ALTER TABLE research_history ADD COLUMN IF NOT EXISTS user_id TEXT;",
    "CREATE INDEX IF NOT EXISTS research_history_user_id_idx ON research_history (user_id);",
]

# ── Connection pool ────────────────────────────────────────────────────────────

import threading
from contextlib import contextmanager

_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    """Lazily create a ThreadedConnectionPool (None if DB not configured/reachable)."""
    global _pool
    if _pool is not None:
        return _pool
    if not DATABASE_URL:
        log.warning("History: DATABASE_URL not set — history disabled")
        return None
    with _pool_lock:
        if _pool is not None:
            return _pool
        try:
            from psycopg2.pool import ThreadedConnectionPool
            _pool = ThreadedConnectionPool(
                1, int(os.getenv("DB_POOL_MAX", "8")), dsn=DATABASE_URL, connect_timeout=10
            )
        except ImportError:
            log.error("History: psycopg2 not installed — run: pip install psycopg2-binary")
        except Exception as exc:
            log.warning("History: DB pool creation failed (%s) — history will be skipped", exc)
        return _pool


@contextmanager
def _conn():
    """Yield a pooled connection (or None). Commits on success, rolls back on error."""
    pool = _get_pool()
    if pool is None:
        yield None
        return
    c = pool.getconn()
    broken = False
    try:
        yield c
        c.commit()
    except Exception:
        try:
            c.rollback()
        except Exception:
            broken = True
        raise
    finally:
        try:
            if getattr(c, "closed", 0):
                broken = True
            pool.putconn(c, close=broken)
        except Exception:
            pass


def _init_db() -> None:
    """Ensure the table exists (idempotent — safe to call every time)."""
    try:
        with _conn() as c:
            if c is None:
                return
            with c.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
                for stmt in _MIGRATIONS:
                    cur.execute(stmt)
        log.info("History: Supabase table ready")
    except Exception as exc:
        log.warning("History: could not init DB — %s", exc)


# Run table creation once at import time (fast — CREATE TABLE IF NOT EXISTS)
try:
    _init_db()
except Exception:
    pass


# ── Public API ─────────────────────────────────────────────────────────────────

def save_research(result: Dict[str, Any]) -> int:
    """Persist a completed research result to Supabase. Returns row id or -1."""
    try:
        with _conn() as c:
            if c is None:
                return -1
            with c.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO research_history
                        (job_id, user_id, topic, report, critique, score, fact_score, urls, time_sec, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        result.get("job_id"),
                        result.get("user_id"),
                        result.get("topic", ""),
                        result.get("report", ""),
                        result.get("critique", ""),
                        float(result.get("critique_score", 0) or 0),
                        float(result.get("fact_check_score") or 0.0),
                        json.dumps(result.get("verified_urls", [])),
                        float(result.get("time_sec", 0) or 0),
                        int(time.time()),
                    ),
                )
                row_id: int = cur.fetchone()[0]
        log.info("History: saved research id=%d topic=%r user_id=%r", row_id, result.get("topic"), result.get("user_id"))
        return row_id
    except Exception as exc:
        log.exception("History: save_research failed — %s", exc)
        return -1


def get_recent(limit: int = 10, user_id: Optional[str] = None) -> List[Dict]:
    """Return the `limit` most recent research entries (newest first) for a specific user_id."""
    if not user_id:
        # Prevent any multi-tenant data leakage by returning empty if user_id is not specified
        return []
    limit = max(1, min(int(limit), 100))
    try:
        with _conn() as c:
            if c is None:
                return []
            with c.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, job_id, user_id, topic, score, fact_score, time_sec, created_at
                    FROM research_history
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        log.exception("History: get_recent failed — %s", exc)
        return []


def _fetch_one_by_clause(where_clause: str, params: tuple) -> Optional[Dict]:
    try:
        with _conn() as c:
            if c is None:
                return None
            with c.cursor() as cur:
                cur.execute(f"SELECT * FROM research_history WHERE {where_clause}", params)
                row = cur.fetchone()
                if not row:
                    return None
                cols = [d[0] for d in cur.description]
                result = dict(zip(cols, row))
        if isinstance(result.get("urls"), str):
            result["urls"] = json.loads(result["urls"])
        return result
    except Exception as exc:
        log.exception("History: get_by_id failed — %s", exc)
        return None


def get_by_id(record_id: Any, user_id: Optional[str] = None) -> Optional[Dict]:
    """
    Retrieve a full research record by numeric id or job_id.

    When `user_id` is given the record is returned only if it belongs to that
    user (ownership check; ids are sequential so unscoped lookups are guessable).
    Callers must pass the user_id verified by the Node backend.
    """
    if record_id is None:
        return None
    record_str = str(record_id).strip()
    clause, params = (("id = %s", (int(record_str),)) if record_str.isdigit()
                      else ("job_id = %s", (record_str,)))
    if user_id:
        clause += " AND user_id = %s"
        params = params + (user_id,)
    return _fetch_one_by_clause(clause, params)


def find_similar(topic: str, limit: int = 3, user_id: Optional[str] = None) -> List[Dict]:
    """Find past research with topics similar to the given one (scoped to user_id if given)."""
    words = topic.strip().split()
    if not words:
        return []
    try:
        with _conn() as c:
            if c is None:
                return []
            where = "(" + " OR ".join("topic ILIKE %s" for _ in words) + ")"
            params = [f"%{w}%" for w in words]
            if user_id:
                where += " AND user_id = %s"
                params.append(user_id)
            params.append(limit)
            with c.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, topic, score, fact_score, time_sec, created_at
                    FROM research_history
                    WHERE {where}
                    ORDER BY created_at DESC LIMIT %s
                    """,
                    params,
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        log.exception("History: find_similar failed — %s", exc)
        return []


def history_stats() -> Dict[str, Any]:
    """Aggregate statistics about stored research sessions."""
    try:
        with _conn() as c:
            if c is None:
                return {"status": "db_unavailable"}
            with c.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) AS total, AVG(score) AS avg_score, AVG(time_sec) AS avg_time "
                    "FROM research_history"
                )
                cols = [d[0] for d in cur.description]
                row = cur.fetchone()
        return dict(zip(cols, row)) if row else {}
    except Exception as exc:
        log.exception("History: stats failed — %s", exc)
        return {}
