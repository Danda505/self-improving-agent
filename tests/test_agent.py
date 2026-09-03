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
