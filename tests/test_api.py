from __future__ import annotations

import json

from src.api.schemas import MAX_TASK_LENGTH


def test_health_returns_ok(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert data["vm_mode"] == "fake"
    assert data["llm_enabled"] is False
    assert data["model"] == "claude-sonnet-4-5-20250929"
    assert data["extra"]["tasks"] == 20


def test_tasks_endpoint_lists_catalogue(api_client):
    r = api_client.get("/api/tasks")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 20
    assert {"id", "category", "task", "app", "max_steps"} <= set(body[0])
    assert "plan" not in body[0] and "checker" not in body[0]


def test_safety_rules_endpoint(api_client):
    r = api_client.get("/api/safety/rules")
    assert r.status_code == 200
    assert "sudo " in r.json()["typed_blocklist"]


def test_run_task_empty_is_422(api_client):
    assert api_client.post("/api/run-task", json={"task": ""}).status_code == 422
    assert api_client.post("/api/run-task", json={"task": "   "}).status_code == 422


def test_run_task_oversized_is_422(api_client):
    r = api_client.post("/api/run-task", json={"task": "x" * (MAX_TASK_LENGTH + 1)})
    assert r.status_code == 422


def test_run_task_malformed_json_is_422(api_client):
    r = api_client.post("/api/run-task", content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    r = api_client.post("/api/run-task", json={"task": "ok task", "max_steps": "many"})
    assert r.status_code == 422
    r = api_client.post("/api/run-task", json={"task": "ok task", "unexpected": 1})
    assert r.status_code == 422


def test_run_task_unknown_task_id_is_404(api_client):
    r = api_client.post("/api/run-task", json={"task": "x y", "task_id": "zz-99"})
    assert r.status_code == 404


def test_run_task_on_fake_vm_returns_trace(api_client):
    r = api_client.post("/api/run-task", json={"task": "Open the My Documents folder", "max_steps": 6})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["trace_id"]
    assert body["steps"][-1]["action"]["type"] == "task_complete"
    recent = api_client.get("/api/audit/recent?limit=5").json()
    assert recent[0]["trace_id"] == body["trace_id"]


def test_run_task_with_catalogue_id_uses_checker(api_client):
    r = api_client.post("/api/run-task", json={"task": "run de-05", "task_id": "de-05", "max_steps": 6})
    assert r.status_code == 200
    body = r.json()
    assert body["task_id"] == "de-05"
    assert body["ground_truth_checked"] is True
    # The fake VM has no DOM: the plan's target never appears, so the checker fails honestly.
    assert body["success"] is False


def test_run_task_runtime_error_is_500(api_client, mocker):
    mocker.patch("src.api.main.run_task", side_effect=RuntimeError("chromium exploded"))
    r = api_client.post("/api/run-task", json={"task": "anything here"})
    assert r.status_code == 500
    assert "chromium exploded" in r.json()["detail"]


def test_audit_recent_validates_limit(api_client):
    assert api_client.get("/api/audit/recent?limit=0").status_code == 422
    assert api_client.get("/api/audit/recent?limit=5000").status_code == 422


def test_eval_results_endpoint(api_client, tmp_path, mocker):
    runs = tmp_path / "eval" / "runs"
    runs.mkdir(parents=True)
    (runs / "2026-01-01-x.json").write_text(
        json.dumps({"name": "x", "llm_used": False, "aggregate": {"overall": {}}})
    )
    (runs / "2026-01-01-safety.json").write_text(
        json.dumps({"name": "safety", "llm_used": False, "with_safety": {}})
    )
    mocker.patch("src.api.main.REPO_ROOT", tmp_path)
    body = api_client.get("/api/eval/results").json()
    assert [r["name"] for r in body["runs"]] == ["safety", "x"]


def test_cors_is_restricted(api_client):
    r = api_client.options(
        "/api/tasks",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert r.headers.get("access-control-allow-origin") != "https://evil.example"
    r = api_client.get("/api/tasks", headers={"Origin": "http://testserver"})
    assert r.headers.get("access-control-allow-origin") == "http://testserver"
