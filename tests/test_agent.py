"""Cheap, deterministic tests. No paid APIs."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import self_improving_agent as agent


def test_builtin_task_names():
    assert set(agent.BUILTIN_TASKS) == {
        "roman", "parens", "csv", "expr", "roman_parse",
    }


def test_validate_task_accepts_well_formed_task():
    task = {
        "name": "fizz",
        "func_name": "fizzbuzz",
        "spec": "Write fizzbuzz(n).",
        "cases": [[[3], "Fizz"], [[5], "Buzz"]],
    }
    assert agent.validate_task(task) is task


def test_validate_task_rejects_missing_key():
    with pytest.raises(ValueError, match="missing required key"):
        agent.validate_task({"name": "x", "func_name": "f", "spec": "s"})


def test_validate_task_rejects_bad_case_shape():
    with pytest.raises(ValueError, match="each case must be"):
        agent.validate_task({
            "name": "x", "func_name": "f", "spec": "s",
            "cases": [{"args": [1], "expected": 1}],
        })


def test_load_task_from_file(tmp_path):
    payload = {
        "name": "dbl",
        "func_name": "double",
        "spec": "Write double(n) returning 2*n.",
        "cases": [[[2], 4]],
    }
    path = tmp_path / "task.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    task = agent.load_task(SimpleNamespace(task_file=str(path), task="roman"))
    assert task["func_name"] == "double"


def test_extract_code_prefers_longest_fenced_block():
    text = (
        "Sure.\n"
        "```python\ndef tiny():\n    return 1\n```\n"
        "```python\ndef bigger():\n    x = 1\n    return x + 1\n```\n"
    )
    code = agent.extract_code(text)
    assert "def bigger()" in code
    assert "def tiny()" not in code


def test_extract_code_falls_back_to_def_line():
    text = "Here you go:\ndef int_to_roman(num):\n    return 'I'\n"
    assert agent.extract_code(text).startswith("def int_to_roman")


def test_mock_roman_loop_solves(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "HERE", tmp_path)
    monkeypatch.setattr(agent, "LOG_PATH", tmp_path / "attempts.jsonl")
    monkeypatch.setattr(agent.time, "sleep", lambda *_a, **_k: None)

    events = list(agent.iter_loop(
        agent.MockBackend(), agent.BUILTIN_TASKS["roman"],
        attempts=5, backend_name="mock",
    ))
    done = [e for e in events if e["type"] == "done"]
    assert done and done[0]["success"] is True
    assert (tmp_path / done[0]["solution"]).is_file()
    attempts = [e for e in events if e["type"] == "attempt"]
    assert attempts[-1]["passed"] == attempts[-1]["total"]
    last_cases = attempts[-1]["cases"]
    assert len(last_cases) == attempts[-1]["total"]
    assert all(c["ok"] for c in last_cases)
    first_cases = attempts[0]["cases"]
    assert any(not c["ok"] for c in first_cases)
    assert attempts[0]["passed"] == sum(1 for c in first_cases if c["ok"])
    logged = [
        json.loads(line)
        for line in (tmp_path / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert logged[-1]["cases"] == last_cases


def _tiny_task():
    return {
        "name": "dbl",
        "func_name": "double",
        "spec": "Write double(n) returning 2*n.",
        "cases": [[[1], 2], [[2], 4], [[3], 6]],
    }


def test_runner_writes_per_case_outcomes():
    passed, total, error, cases = agent.run_tests_subprocess(
        "def double(n):\n    return n * 2 if n != 3 else 0\n", _tiny_task())
    assert (passed, total) == (2, 3)
    assert [c["ok"] for c in cases] == [True, True, False]
    assert cases[0] == {"id": 0, "input": "1", "ok": True}
    assert cases[2]["id"] == 2
    assert "3" in cases[2]["input"]
    assert cases[2].get("error")
    assert "error" not in cases[0]
    assert "expected" in error or "0" in error


def test_runner_load_failure_marks_all_cases():
    passed, total, error, cases = agent.run_tests_subprocess(
        "def other():\n    return 1\n", _tiny_task())
    assert passed == 0
    assert total == 3
    assert "double" in error
    assert len(cases) == 3
    assert all(c["ok"] is False for c in cases)
    assert [c["id"] for c in cases] == [0, 1, 2]
    assert all(c.get("error") for c in cases)


def test_grouped_runs_accepts_legacy_lines(tmp_path, monkeypatch):
    log = tmp_path / "attempts.jsonl"
    monkeypatch.setattr(agent, "LOG_PATH", log)
    old = {
        "run_id": "abc123", "ts": "2020-01-01T00:00:00+00:00",
        "backend": "mock", "model": "mock", "task": "roman",
        "attempt": 1, "passed": 3, "total": 11, "success": False,
        "seconds": 0.1, "error": "nope", "code": "def int_to_roman(num): pass",
    }
    log.write_text(json.dumps(old) + "\n", encoding="utf-8")
    runs = agent.grouped_runs()
    assert list(runs) == ["abc123"]
    rec = runs["abc123"][0]
    assert rec["passed"] == 3
    assert "cases" not in rec


@pytest.mark.parametrize("task_name", list(agent.BUILTIN_TASKS))
def test_mock_improves_or_solves_each_builtin(tmp_path, monkeypatch, task_name):
    """Run mock is an honest demo on every built-in, not only roman."""
    _silence_loop(tmp_path, monkeypatch)
    task = agent.BUILTIN_TASKS[task_name]
    events = list(agent.iter_loop(
        agent.MockBackend(), task, attempts=5, backend_name="mock",
    ))
    attempts = [e for e in events if e["type"] == "attempt"]
    assert attempts
    done = [e for e in events if e["type"] == "done"]
    assert done and done[0].get("success")
    assert attempts[-1]["passed"] == attempts[-1]["total"]
    assert f"def {task['func_name']}(" in attempts[0]["code"]
    assert len(attempts[-1]["cases"]) == attempts[-1]["total"]
    assert all(c["ok"] for c in attempts[-1]["cases"])


def test_mock_custom_task_is_dummy(tmp_path, monkeypatch):
    """Custom JSON is not a built-in: mock must not pretend to solve it."""
    _silence_loop(tmp_path, monkeypatch)
    task = {
        "name": "custom",
        "func_name": "double",
        "spec": "Write double(n) returning 2*n.",
        "cases": [[[2], 4], [[0], 0]],
    }
    events = list(agent.iter_loop(
        agent.MockBackend(), task, attempts=2, backend_name="mock",
    ))
    attempts = [e for e in events if e["type"] == "attempt"]
    assert attempts
    assert all(a["passed"] < a["total"] for a in attempts)
    assert all(len(a["cases"]) == a["total"] for a in attempts)
    assert all(not c["ok"] for a in attempts for c in a["cases"])
    done = [e for e in events if e["type"] == "done"]
    assert done and done[0]["success"] is False



# ----------------------------------------------------------------------------
# Prompt capture: zero-escape (no anchor) vs ratchet (has_anchor)
# ----------------------------------------------------------------------------

ZERO_ROMAN = (
    "```python\n"
    "def int_to_roman(num):\n"
    "    return 'NOPE'\n"
    "```"
)


class RecordingBackend:
    """Canned reply plus a log of every complete() prompt. No LLM."""

    model = "recording"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, system, messages, temperature=0.2, max_tokens=2000):
        self.calls.append([dict(m) for m in messages])
        return self.reply


def _silence_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "HERE", tmp_path)
    monkeypatch.setattr(agent, "LOG_PATH", tmp_path / "attempts.jsonl")
    monkeypatch.setattr(agent.time, "sleep", lambda *_a, **_k: None)


def test_zero_score_ordinary_retry_uses_fresh(tmp_path, monkeypatch):
    """best_passed == 0: next attempt is FRESH, not RETRY anchored on broken code."""
    _silence_loop(tmp_path, monkeypatch)
    backend = RecordingBackend(ZERO_ROMAN)
    task = agent.BUILTIN_TASKS["roman"]

    events = list(agent.iter_loop(
        backend, task, attempts=2, backend_name="recording", stall_limit=10,
    ))
    attempts = [e for e in events if e["type"] == "attempt"]
    assert attempts[0]["passed"] == 0
    assert attempts[0]["best"] == 0
    assert len(backend.calls) == 2

    retry = backend.calls[1]
    assert len(retry) == 1
    assert retry[0]["role"] == "user"
    assert retry[0]["content"] == agent.FRESH.format(
        spec=task["spec"], error=attempts[0]["error"])
    assert "Keep everything that already works" not in retry[0]["content"]
    assert all(m["role"] != "assistant" for m in retry)


def test_zero_score_plateau_escalates_fresh_reseed(tmp_path, monkeypatch):
    """No passing tests ever: plateau reseeds with FRESH_RESEED, has_anchor false."""
    _silence_loop(tmp_path, monkeypatch)
    backend = RecordingBackend(ZERO_ROMAN)
    task = agent.BUILTIN_TASKS["roman"]

    events = list(agent.iter_loop(
        backend, task, attempts=5, backend_name="recording", stall_limit=3,
    ))
    escalates = [e for e in events if e["type"] == "escalate"]
    assert len(escalates) == 1
    esc = escalates[0]
    assert esc["has_anchor"] is False
    assert esc["fresh_context"] is True
    assert esc["anchor_passed"] == 0

    attempts = [e for e in events if e["type"] == "attempt"]
    plateau = next(a for a in attempts if a["attempt"] == esc["attempt"])
    reseed = backend.calls[esc["attempt"]]  # next complete() after escalate
    assert len(reseed) == 1
    assert reseed[0]["content"] == agent.FRESH_RESEED.format(
        spec=task["spec"], error=plateau["error"])
    assert "A near-complete solution already exists" not in reseed[0]["content"]


def test_mock_stuck_plateau_escalates_reseed_with_anchor(tmp_path, monkeypatch):
    """mock-stuck on csv: best_passed > 0, so escalate uses RESEED and has_anchor."""
    _silence_loop(tmp_path, monkeypatch)
    backend = agent.StuckMockBackend()
    calls = []
    orig = backend.complete

    def capture(system, messages, temperature=0.2, max_tokens=2000):
        calls.append([dict(m) for m in messages])
        return orig(system, messages, temperature=temperature, max_tokens=max_tokens)

    backend.complete = capture
    task = agent.BUILTIN_TASKS["csv"]

    events = list(agent.iter_loop(
        backend, task, attempts=5, backend_name="mock-stuck", stall_limit=3,
    ))
    attempts = [e for e in events if e["type"] == "attempt"]
    assert attempts[0]["passed"] > 0
    assert attempts[0]["passed"] < attempts[0]["total"]

    escalates = [e for e in events if e["type"] == "escalate"]
    assert len(escalates) == 1
    esc = escalates[0]
    assert esc["has_anchor"] is True
    assert esc["fresh_context"] is True
    assert esc["anchor_passed"] == attempts[0]["passed"]

    reseed = calls[esc["attempt"]]
    assert len(reseed) == 1
    assert reseed[0]["content"] == agent.RESEED.format(
        spec=task["spec"],
        passed=attempts[0]["passed"],
        total=attempts[0]["total"],
        error=attempts[0]["error"],
        code=attempts[0]["code"],
    )


def test_build_backend_lmstudio_default_url(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.delenv("LMSTUDIO_HOST_URL", raising=False)

    be = agent.build_backend("lmstudio")
    assert isinstance(be, agent.OpenAICompatBackend)
    assert be.model == "local-model"
    assert captured["base_url"] == "http://localhost:1234/v1"
    assert captured["api_key"] == "lmstudio"


def test_build_backend_lmstudio_respects_host_url_env(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("LMSTUDIO_HOST_URL", "http://example.invalid:9999/v1")

    be = agent.build_backend("lmstudio", model="my-local")
    assert isinstance(be, agent.OpenAICompatBackend)
    assert be.model == "my-local"
    assert captured["base_url"] == "http://example.invalid:9999/v1"
    assert captured["api_key"] == "lmstudio"


def test_tokens_from_usage_openai_and_anthropic():
    oa = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    assert agent.tokens_from_usage(oa) == {"prompt": 10, "completion": 20, "total": 30}
    ant = SimpleNamespace(input_tokens=4, output_tokens=6)
    assert agent.tokens_from_usage(ant)["total"] == 10
    assert agent.tokens_from_usage(None) is None
    assert agent.tokens_from_usage({"total_tokens": 0}) is None
    assert agent.tokens_from_usage({"prompt": 1, "completion": 2, "total": 3})["total"] == 3


def test_take_last_tokens_clears_and_skips_missing():
    be = SimpleNamespace()
    assert agent.take_last_tokens(be) is None
    be.last_usage = {"prompt_tokens": 100, "completion_tokens": 700, "total_tokens": 800}
    assert agent.take_last_tokens(be) == 800
    assert be.last_usage is None


def test_loop_records_elapsed_ms_without_tokens(tmp_path, monkeypatch):
    _silence_loop(tmp_path, monkeypatch)
    events = list(agent.iter_loop(
        agent.MockBackend(), agent.BUILTIN_TASKS["roman"],
        attempts=1, backend_name="mock",
    ))
    att = next(e for e in events if e["type"] == "attempt")
    assert isinstance(att["elapsed_ms"], int)
    assert att["elapsed_ms"] >= 0
    assert "tokens" not in att
    logged = json.loads((tmp_path / "attempts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert logged["elapsed_ms"] == att["elapsed_ms"]
    assert "tokens" not in logged


class _UsageBackend:
    model = "usage-fake"

    def __init__(self, reply):
        self.reply = reply
        self.last_usage = None

    def complete(self, system, messages, temperature=0.2, max_tokens=2000):
        self.last_usage = {
            "prompt_tokens": 100, "completion_tokens": 700, "total_tokens": 800,
        }
        return self.reply


def test_loop_records_tokens_when_backend_reports(tmp_path, monkeypatch):
    _silence_loop(tmp_path, monkeypatch)
    events = list(agent.iter_loop(
        _UsageBackend(ZERO_ROMAN), agent.BUILTIN_TASKS["roman"],
        attempts=1, backend_name="usage",
    ))
    att = next(e for e in events if e["type"] == "attempt")
    assert att["tokens"] == 800
    logged = json.loads((tmp_path / "attempts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert logged["tokens"] == 800
    assert logged["elapsed_ms"] == att["elapsed_ms"]
