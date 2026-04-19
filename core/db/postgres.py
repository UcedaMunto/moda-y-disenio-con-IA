import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor


DB_NAME = os.getenv("POSTGRES_DB", "fabricdb")
DB_USER = os.getenv("POSTGRES_USER", "almapa")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "asdfghjkl")
DB_HOST = os.getenv("POSTGRES_HOST", "db")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))


def build_dsn() -> str:
    return (
        f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} "
        f"host={DB_HOST} port={DB_PORT}"
    )


@contextmanager
def get_connection():
    conn = psycopg2.connect(build_dsn())
    try:
        yield conn
    finally:
        conn.close()


def init_database() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id BIGSERIAL PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    candidate_path TEXT NOT NULL,
                    label TEXT NOT NULL CHECK (label IN ('approve', 'reject')),
                    score INTEGER NULL CHECK (score BETWEEN 1 AND 5),
                    comment TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_feedback_project_id
                ON feedback_events(project_id);
                """
            )
        conn.commit()


def insert_feedback(
    project_id: str,
    candidate_path: str,
    label: str,
    score: int | None,
    comment: str | None,
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback_events(project_id, candidate_path, label, score, comment)
                VALUES (%s, %s, %s, %s, %s);
                """,
                (project_id, candidate_path, label, score, comment),
            )
        conn.commit()


def feedback_summary(project_id: str) -> dict:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    AVG(score)::float AS avg_score,
                    COALESCE(SUM(CASE WHEN label = 'approve' THEN 1 ELSE 0 END), 0) AS approve_count,
                    COALESCE(SUM(CASE WHEN label = 'reject' THEN 1 ELSE 0 END), 0) AS reject_count
                FROM feedback_events
                WHERE project_id = %s;
                """,
                (project_id,),
            )
            row = cur.fetchone() or {
                "total": 0,
                "avg_score": None,
                "approve_count": 0,
                "reject_count": 0,
            }

    return {
        "project_id": project_id,
        "total": int(row["total"] or 0),
        "labels": {
            "approve": int(row["approve_count"] or 0),
            "reject": int(row["reject_count"] or 0),
        },
        "avg_score": row["avg_score"],
        "storage": "postgres",
    }


def candidate_ranking(project_id: str, limit: int = 20) -> dict:
    safe_limit = max(1, min(limit, 200))
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    candidate_path,
                    COUNT(*) AS total_feedback,
                    COALESCE(SUM(CASE WHEN label = 'approve' THEN 1 ELSE 0 END), 0) AS approve_count,
                    COALESCE(SUM(CASE WHEN label = 'reject' THEN 1 ELSE 0 END), 0) AS reject_count,
                    AVG(score)::float AS avg_score,
                    (
                        COALESCE(SUM(CASE WHEN label = 'approve' THEN 1 ELSE 0 END), 0)::float
                        / NULLIF(COUNT(*), 0)
                    ) AS approval_ratio
                FROM feedback_events
                WHERE project_id = %s
                GROUP BY candidate_path
                ORDER BY approval_ratio DESC NULLS LAST, avg_score DESC NULLS LAST, total_feedback DESC
                LIMIT %s;
                """,
                (project_id, safe_limit),
            )
            rows = cur.fetchall() or []

    items: list[dict] = []
    for row in rows:
        items.append(
            {
                "candidate_path": row["candidate_path"],
                "total_feedback": int(row["total_feedback"] or 0),
                "approve_count": int(row["approve_count"] or 0),
                "reject_count": int(row["reject_count"] or 0),
                "avg_score": row["avg_score"],
                "approval_ratio": row["approval_ratio"],
            }
        )

    return {
        "project_id": project_id,
        "limit": safe_limit,
        "items": items,
        "storage": "postgres",
    }
