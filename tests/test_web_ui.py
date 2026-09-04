"""Cheap tests for the env split. No live docker/ollama/lmstudio."""

import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import self_improving_agent as agent
import web_ui


def _boom(name):
    def inner(*_a, **_k):
        raise AssertionError(f"probe_env must not call {name}")
    return inner


def test_probe_env_stays_fast_without_docker_or_models(monkeypatch):
    monkeypatch.setattr(agent, "docker_available", _boom("docker_available"))
    monkeypatch.setattr(agent, "docker_image_present", _boom("docker_image_present"))
    monkeypatch.setattr(web_ui, "ollama_models", _boom("ollama_models"))
    monkeypatch.setattr(web_ui, "lmstudio_models", _boom("lmstudio_models"))

    env = web_ui.probe_env()
    assert env["ollama"] == {"up": False, "models": []}
    assert env["lmstudio"] == {"up": False, "models": []}
    assert env["docker"] == {"up": False, "image": False}
    assert set(env["tasks"]) == set(agent.BUILTIN_TASKS)
    for name, meta in env["tasks"].items():
        task = agent.BUILTIN_TASKS[name]
        assert meta["func"] == task["func_name"]
        assert meta["cases"] == len(task["cases"])
        assert meta["spec"] == task["spec"]


def test_page_has_static_backend_and_task_options():
    # HTML options render before /api/env; JS BACKENDS is the client fallback.
    assert "const BACKENDS" in web_ui.JS
    for value in ("mock", "mock-stuck", "ollama", "lmstudio", "anthropic", "groq"):
        assert f'value="{value}"' in web_ui.PAGE
        assert f'"{value}"' in web_ui.JS
    for name, task in agent.BUILTIN_TASKS.items():
        assert f'value="{name}"' in web_ui.PAGE
        assert task["func_name"] in web_ui.PAGE


def test_task_options_html_lists_builtins():
    html = web_ui._task_options_html()
    for name in agent.BUILTIN_TASKS:
        assert f'value="{name}"' in html
    assert 'value="__custom"' in html


def test_api_env_http_does_not_probe_docker(monkeypatch):
    monkeypatch.setattr(agent, "docker_available", _boom("docker_available"))
    monkeypatch.setattr(agent, "docker_image_present", _boom("docker_image_present"))
    monkeypatch.setattr(web_ui, "ollama_models", _boom("ollama_models"))
    monkeypatch.setattr(web_ui, "lmstudio_models", _boom("lmstudio_models"))

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_ui.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/api/env"
        with urllib.request.urlopen(url, timeout=2) as r:
            data = json.loads(r.read())
        assert set(data["tasks"]) == set(agent.BUILTIN_TASKS)
        assert data["docker"] == {"up": False, "image": False}
        assert data["lmstudio"] == {"up": False, "models": []}
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_api_history_legacy_lines_and_cases(tmp_path, monkeypatch):
    log = tmp_path / "attempts.jsonl"
    monkeypatch.setattr(agent, "LOG_PATH", log)
    old = {
        "run_id": "old1", "ts": "2020-01-01T00:00:00+00:00",
        "backend": "mock", "model": "mock", "task": "roman",
        "attempt": 1, "passed": 3, "total": 11, "success": False,
        "seconds": 0.1, "error": "nope", "code": "x",
    }
    new = {
        **old,
        "run_id": "new1", "ts": "2020-01-02T00:00:00+00:00",
        "cases": [
            {"id": 0, "input": "1", "ok": True},
            {"id": 1, "input": "4", "ok": False, "error": "got 'IIII'"},
        ],
    }
    log.write_text(
        json.dumps(old) + "\n" + json.dumps(new) + "\n", encoding="utf-8")

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_ui.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/api/history"
        with urllib.request.urlopen(url, timeout=2) as r:
            data = json.loads(r.read())
        by_id = {row["run_id"]: row for row in data}
        assert "cases" not in by_id["old1"]["points"][0]
        assert by_id["old1"]["points"][0]["passed"] == 3
        assert by_id["new1"]["points"][0]["cases"][1]["ok"] is False
        assert by_id["new1"]["points"][0]["cases"][0]["input"] == "1"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_attempt_bar_uses_case_outcomes():
    assert "function bar(passed, total, cases)" in web_ui.JS
    assert "bar(ev.passed, ev.total, ev.cases)" in web_ui.JS


def test_curve_heatmap_uses_cases_and_skips_legacy():
    assert "function drawHeatmap(runs)" in web_ui.JS
    assert 'id="heatmap"' in web_ui.HTML
    assert "no case log" in web_ui.JS
    assert "function grayBar(" in web_ui.JS
    assert "drawHeatmap(slice)" in web_ui.JS
    assert "polyline" in web_ui.JS


def test_attempt_card_shows_telemetry():
    assert "function telemetryLine(ev)" in web_ui.JS
    assert "telemetryLine(ev)" in web_ui.JS
    assert "elapsed_ms" in web_ui.JS
    assert "tok" in web_ui.JS


def test_api_history_forwards_telemetry_when_present(tmp_path, monkeypatch):
    log = tmp_path / "attempts.jsonl"
    monkeypatch.setattr(agent, "LOG_PATH", log)
    rec = {
        "run_id": "tel1", "ts": "2020-01-03T00:00:00+00:00",
        "backend": "anthropic", "model": "claude", "task": "roman",
        "attempt": 1, "passed": 3, "total": 11, "success": False,
        "seconds": 1.2, "elapsed_ms": 1200, "tokens": 800,
        "error": "nope", "code": "x",
        "cases": [{"id": 0, "input": "1", "ok": True}],
    }
    log.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web_ui.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/api/history"
        with urllib.request.urlopen(url, timeout=2) as r:
            data = json.loads(r.read())
        pt = data[0]["points"][0]
        assert pt["elapsed_ms"] == 1200
        assert pt["tokens"] == 800
        assert pt["cases"][0]["ok"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
