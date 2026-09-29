"""Audit log: every task run, every step, every safety decision.

Two backends behind the same :class:`AuditLog` protocol:

* :class:`SQLiteAuditLog` — zero-dependency default (also ``:memory:`` for tests).
* :class:`PostgresAuditLog` — production, using a ``psycopg_pool`` connection
  pool. The ``psycopg`` import is lazy so the API boots without a database.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from src.api.schemas import TaskResponse
from src.config import Settings

DDL = [
    """CREATE TABLE IF NOT EXISTS task_runs (
        trace_id TEXT PRIMARY KEY,
        task_id TEXT,
        task TEXT NOT NULL,
        success INTEGER NOT NULL,
        total_steps INTEGER NOT NULL,
        blocked_actions INTEGER NOT NULL,
        verifier_failures INTEGER NOT NULL,
        latency_ms INTEGER NOT NULL,
        tokens_in INTEGER NOT NULL,
        tokens_out INTEGER NOT NULL,
        cost_usd REAL NOT NULL,
        reasoner TEXT,
        vm TEXT,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS task_steps (
        trace_id TEXT NOT NULL,
        step INTEGER NOT NULL,
        action TEXT NOT NULL,
        screenshot_path TEXT,
        reasoning TEXT,
        execution_result TEXT,
        safety_blocked INTEGER NOT NULL,
        safety_reason TEXT,
        verifier_ok INTEGER NOT NULL,
        verifier_note TEXT,
        latency_ms INTEGER NOT NULL,
        cost_usd REAL NOT NULL,
        PRIMARY KEY (trace_id, step)
    )""",
]

RUN_COLUMNS = (
    "trace_id, task_id, task, success, total_steps, blocked_actions, verifier_failures, "
    "latency_ms, tokens_in, tokens_out, cost_usd, reasoner, vm, created_at"
)
STEP_COLUMNS = (
    "trace_id, step, action, screenshot_path, reasoning, execution_result, safety_blocked, "
    "safety_reason, verifier_ok, verifier_note, latency_ms, cost_usd"
)


def _run_row(response: TaskResponse) -> tuple[Any, ...]:
    return (
        response.trace_id,
        response.task_id,
        response.task,
        int(response.success),
        response.total_steps,
        response.blocked_actions,
        response.verifier_failures,
        response.latency_ms,
        response.tokens_in,
        response.tokens_out,
        response.cost_usd,
        response.reasoner,
        response.vm,
        datetime.now(UTC).isoformat(),
    )


def _step_rows(response: TaskResponse) -> list[tuple[Any, ...]]:
    return [
        (
            response.trace_id,
            step.step,
            json.dumps(step.action.model_dump(exclude_none=True)),
            step.screenshot_path,
            step.reasoning,
            step.execution_result,
            int(step.safety_blocked),
            step.safety_reason,
            int(step.verifier_ok),
            step.verifier_note,
            step.latency_ms,
            step.cost_usd,
        )
        for step in response.steps
    ]


@runtime_checkable
class AuditLog(Protocol):
    """Sink for task results."""

    def record(self, response: TaskResponse) -> None:
        """Persist a task run and its steps."""

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent task runs (newest first)."""

    def close(self) -> None:
        """Release connections."""


class SQLiteAuditLog:
    """SQLite-backed audit log (default; ``:memory:`` for tests)."""

    name = "sqlite"

    def __init__(self, path: str = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        for ddl in DDL:
            self._conn.execute(ddl)
        self._conn.commit()

    def record(self, response: TaskResponse) -> None:
        """Insert the run and its steps in one transaction."""
        placeholders = ", ".join("?" for _ in RUN_COLUMNS.split(","))
        with self._conn:
            self._conn.execute(
                f"INSERT OR REPLACE INTO task_runs ({RUN_COLUMNS}) VALUES ({placeholders})",
                _run_row(response),
            )
            step_placeholders = ", ".join("?" for _ in STEP_COLUMNS.split(","))
            self._conn.executemany(
                f"INSERT OR REPLACE INTO task_steps ({STEP_COLUMNS}) VALUES ({step_placeholders})",
                _step_rows(response),
            )

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the latest runs."""
        rows = self._conn.execute(
            "SELECT * FROM task_runs ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 1000)),)
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()


class PostgresAuditLog:
    """PostgreSQL audit log with a ``psycopg_pool`` connection pool.

    Args:
        dsn: ``postgresql://user:pass@host:5432/db``.
        pool: Optional pre-built pool (tests inject a fake).
        min_size / max_size: Pool bounds.
    """

    name = "postgres"

    def __init__(self, dsn: str, *, pool: Any = None, min_size: int = 1, max_size: int = 4) -> None:
        if pool is None:
            from psycopg_pool import ConnectionPool

            pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size, open=True, timeout=10)
        self._pool = pool
        with self._pool.connection() as conn:
            for ddl in DDL:
                conn.execute(
                    ddl.replace("INTEGER NOT NULL", "INTEGER NOT NULL").replace("REAL", "DOUBLE PRECISION")
                )
            conn.commit()

    def record(self, response: TaskResponse) -> None:
        """Insert the run and its steps."""
        run_placeholders = ", ".join("%s" for _ in RUN_COLUMNS.split(","))
        step_placeholders = ", ".join("%s" for _ in STEP_COLUMNS.split(","))
        with self._pool.connection() as conn:
            conn.execute(
                f"INSERT INTO task_runs ({RUN_COLUMNS}) VALUES ({run_placeholders}) "
                "ON CONFLICT (trace_id) DO NOTHING",
                _run_row(response),
            )
            for row in _step_rows(response):
                conn.execute(
                    f"INSERT INTO task_steps ({STEP_COLUMNS}) VALUES ({step_placeholders}) "
                    "ON CONFLICT (trace_id, step) DO NOTHING",
                    row,
                )
            conn.commit()

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the latest runs."""
        columns = [c.strip() for c in RUN_COLUMNS.split(",")]
        with self._pool.connection() as conn:
            cur = conn.execute(
                f"SELECT {RUN_COLUMNS} FROM task_runs ORDER BY created_at DESC LIMIT %s",
                (max(1, min(limit, 1000)),),
            )
            return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]

    def close(self) -> None:
        """Close the pool."""
        self._pool.close()


def build_audit_log(settings: Settings) -> AuditLog:
    """Select PostgreSQL when ``DATABASE_URL`` is set, else SQLite.

    A PostgreSQL connection failure is logged and falls back to SQLite so the
    API stays usable in development; the fallback is explicit in the logs.
    """
    if settings.database_url:
        try:
            return PostgresAuditLog(settings.database_url)
        except (ImportError, OSError, RuntimeError) as exc:
            logger.error(f"PostgreSQL audit log unavailable ({exc!r}); using SQLite fallback")
    return SQLiteAuditLog(settings.audit_sqlite_path)
