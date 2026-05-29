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

# ── Connection ─────────────────────────────────────────────────────────────────

def _get_conn():
    """
    Open a psycopg2 connection to Supabase PostgreSQL.
    Returns None (gracefully) if DATABASE_URL is not configured or server is unreachable.
    """
    if not DATABASE_URL:
        log.warning("History: DATABASE_URL not set — history disabled")
        return None
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
        return conn
    except ImportError:
        log.error("History: psycopg2 not installed — run: pip install psycopg2-binary")
        return None
    except Exception as exc:
        log.warning("History: DB connection failed (%s) — history will be skipped", exc)
        return None


def _init_db() -> None:
    """Ensure the table exists (idempotent — safe to call every time)."""
    conn = None
    try:
        conn = _get_conn()
        if conn is None:
            return
        with conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
                for stmt in _MIGRATIONS:
                    cur.execute(stmt)
        log.info("History: Supabase table ready")
    except Exception as exc:
        log.warning("History: could not init DB — %s", exc)
    finally:
        if conn is not None:
            conn.close()


# Run table creation once at import time (fast — CREATE TABLE IF NOT EXISTS)
try:
    _init_db()
except Exception:
    pass


# ── Public API ─────────────────────────────────────────────────────────────────

def save_research(result: Dict[str, Any]) -> int:
    """
    Persist a completed research result to Supabase.
    """
    conn = None
    try:
        conn = _get_conn()
        if conn is None:
            return -1
        with conn:
            with conn.cursor() as cur:
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
                        float(result.get("critique_score", 0)),
                        float(result.get("fact_check_score") or 0.0),
                        json.dumps(result.get("verified_urls", [])),
                        float(result.get("time_sec", 0)),
                        int(time.time()),
                    ),
                )
                row_id: int = cur.fetchone()[0]
        log.info("History: saved research id=%d topic=%r user_id=%r", row_id, result.get("topic"), result.get("user_id"))
        return row_id
    except Exception as exc:
        log.exception("History: save_research failed — %s", exc)
        return -1
    finally:
        if conn is not None:
            conn.close()


def get_recent(limit: int = 10, user_id: Optional[str] = None) -> List[Dict]:
    """Return the `limit` most recent research entries (newest first) for a specific user_id."""
    if not user_id:
        # Prevent any multi-tenant data leakage by returning empty if user_id is not specified
        return []
    conn = None
    try:
        conn = _get_conn()
        if conn is None:
            return []
        with conn.cursor() as cur:
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
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        return rows
    except Exception as exc:
        log.exception("History: get_recent failed — %s", exc)
        return []
    finally:
        if conn is not None:
            conn.close()


def _fetch_one_by_clause(where_clause: str, params: tuple) -> Optional[Dict]:
    conn = None
    try:
        conn = _get_conn()
        if conn is None:
            return None
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM research_history WHERE {where_clause}",
                params,
            )
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
    finally:
        if conn is not None:
            conn.close()


def get_by_id(record_id: Any) -> Optional[Dict]:
    """Retrieve a full research record by numeric id or job_id."""
    if record_id is None:
        return None
    record_str = str(record_id).strip()
    if record_str.isdigit():
        return _fetch_one_by_clause("id = %s", (int(record_str),))
    return _fetch_one_by_clause("job_id = %s", (record_str,))
    


def find_similar(topic: str, limit: int = 3) -> List[Dict]:
    """Find past research with topics similar to the given one (full-text LIKE)."""
    conn = None
    try:
        conn = _get_conn()
        if conn is None:
            return []
        words = topic.strip().split()
        if not words:
            return []
        # Build parameterised OR clause
        where = " OR ".join("topic ILIKE %s" for _ in words)
        params = [f"%{w}%" for w in words] + [limit]
        with conn.cursor() as cur:
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
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        return rows
    except Exception as exc:
        log.exception("History: find_similar failed — %s", exc)
        return []
    finally:
        if conn is not None:
            conn.close()


def history_stats() -> Dict[str, Any]:
    """Aggregate statistics about stored research sessions."""
    conn = None
    try:
        conn = _get_conn()
        if conn is None:
            return {"status": "db_unavailable"}
        with conn.cursor() as cur:
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
    finally:
        if conn is not None:
            conn.close()
