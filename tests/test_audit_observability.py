from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

from src.api.schemas import Action, Step, TaskResponse
from src.observability import (
    configure_langsmith,
    configure_logging,
    estimate_cost_usd,
    log_node,
    new_trace_id,
)
from src.storage.audit import PostgresAuditLog, SQLiteAuditLog, build_audit_log


def _response(trace: str = "abc") -> TaskResponse:
    return TaskResponse(
        trace_id=trace,
        task="t",
        task_id="ff-01",
        success=True,
        steps=[
            Step(step=1, action=Action(type="click", coords=(1, 2)), reasoning="r", latency_ms=3),
            Step(step=2, action=Action(type="type", text="rm -rf /"), safety_blocked=True, safety_reason="x"),
        ],
        total_steps=2,
        latency_ms=10,
        cost_usd=0.01,
    )


def test_sqlite_audit_roundtrip(tmp_path):
    log = SQLiteAuditLog(str(tmp_path / "a" / "audit.sqlite3"))
    log.record(_response("t1"))
    log.record(_response("t2"))
    rows = log.recent(1)
    assert len(rows) == 1 and rows[0]["task_id"] == "ff-01"
    assert len(log.recent(10)) == 2
    steps = log._conn.execute("SELECT count(*) FROM task_steps").fetchone()[0]
    assert steps == 4
    log.close()


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, store):
        self.store = store

    def execute(self, sql, params=None):
        self.store.append((sql, params))
        if sql.startswith("SELECT"):
            return FakeCursor(
                [("t", "ff-01", "task", 1, 2, 0, 0, 10, 0, 0, 0.0, "stub", "fake", "2026-01-01")]
            )
        return FakeCursor([])

    def commit(self):
        self.store.append(("COMMIT", None))


class FakePool:
    def __init__(self):
        self.store: list = []
        self.closed = False

    @contextmanager
    def connection(self):
        yield FakeConn(self.store)

    def close(self):
        self.closed = True


def test_postgres_audit_uses_pool():
    pool = FakePool()
    log = PostgresAuditLog("postgresql://x", pool=pool)
    log.record(_response())
    inserts = [s for s, _ in pool.store if s.startswith("INSERT")]
    assert len(inserts) == 3  # 1 run + 2 steps
    rows = log.recent(5)
    assert rows[0]["task_id"] == "ff-01"
    log.close()
    assert pool.closed


def test_build_audit_log_falls_back_to_sqlite(settings, mocker):
    assert isinstance(build_audit_log(settings), SQLiteAuditLog)
    pg = settings.model_copy(
        update={"database_url": "postgresql://u:p@nohost:1/db", "audit_sqlite_path": ":memory:"}
    )
    mocker.patch("src.storage.audit.PostgresAuditLog.__init__", side_effect=OSError("no db"))
    assert isinstance(build_audit_log(pg), SQLiteAuditLog)


def test_cost_estimate_and_unknown_model():
    assert estimate_cost_usd("claude-sonnet-4-5-20250929", 1_000_000, 0) == pytest.approx(3.0)
    assert estimate_cost_usd("claude-sonnet-4-5-20250929", 0, 1_000_000) == pytest.approx(15.0)
    assert estimate_cost_usd("unknown-model", 100, 100) == 0.0


def test_trace_ids_are_unique_and_logging_configures():
    assert new_trace_id() != new_trace_id()
    configure_logging("DEBUG")
    log_node(
        "t",
        "observe",
        "in",
        {"step": 1, "task": "x" * 100, "steps_taken": [1, 2], "action": None, "observation": None},
    )


def test_configure_langsmith(settings):
    assert configure_langsmith(settings) is False
    assert "LANGCHAIN_TRACING_V2" not in os.environ
    on = settings.model_copy(update={"langchain_tracing_v2": True, "langchain_api_key": "ls-test"})
    assert configure_langsmith(on) is True
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_PROJECT"] == "computer-use-agent"
    os.environ.pop("LANGCHAIN_TRACING_V2", None)
    os.environ.pop("LANGCHAIN_API_KEY", None)
