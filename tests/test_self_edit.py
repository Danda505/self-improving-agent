"""Self-edit propose / verify-or-revert. No live LLM."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import self_edit

OLD = '''"""web_ui.py -- tiny fixture for self-edit tests"""

x = 1
y = 2
'''

NEW = '''"""web_ui.py -- tiny fixture for self-edit tests"""

x = 1
y = 3
'''

GATE_LABELS = ["compiles", "imports", "loop still works", "server boots"]


class FakeBackend:
    model = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, system, messages, temperature=0.2, max_tokens=32000):
        self.calls.append({"system": system, "messages": messages,
                           "temperature": temperature, "max_tokens": max_tokens})
        return self.reply


def _sandbox(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    (root / "web_ui.py").write_text(OLD, encoding="utf-8")
    monkeypatch.setattr(self_edit, "HERE", root)
    monkeypatch.setattr(self_edit, "BACKUP_DIR", root / "backups")
    return root


def test_gate_still_has_four_checks():
    assert [label for label, _ in self_edit.GATE] == GATE_LABELS
    assert len(self_edit.GATE) == 4


def test_parse_reply_extracts_file_and_code(tmp_path, monkeypatch):
    _sandbox(tmp_path, monkeypatch)
    name, content = self_edit.parse_reply(
        "FILE: web_ui.py\n```python\n" + NEW + "```\n")
    assert name == "web_ui.py"
    assert content.startswith('"""web_ui.py')
    assert "y = 3" in content


def test_parse_reply_rejects_protected(tmp_path, monkeypatch):
    _sandbox(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="protected"):
        self_edit.parse_reply(
            "FILE: self_edit.py\n```python\nx = 1\n```\n")


def test_propose_returns_diff_without_live_llm(tmp_path, monkeypatch):
    _sandbox(tmp_path, monkeypatch)
    backend = FakeBackend("FILE: web_ui.py\n```python\n" + NEW + "```\n")
    name, content, diff = self_edit.propose(backend, "change y to 3", "web_ui.py")
    assert name == "web_ui.py"
    assert "y = 3" in content
    assert "y = 2" in diff and "y = 3" in diff
    assert backend.calls
    assert backend.calls[0]["max_tokens"] == self_edit.MAX_EDIT_TOKENS


def _pass_gate():
    return [(label, lambda: (True, "ok")) for label in GATE_LABELS]


def test_apply_edit_keeps_file_when_gate_passes(tmp_path, monkeypatch):
    root = _sandbox(tmp_path, monkeypatch)
    monkeypatch.setattr(self_edit, "GATE", _pass_gate())
    steps = list(self_edit.apply_edit("web_ui.py", NEW))
    stages = [s["stage"] for s in steps]
    assert stages[0] == "backup" and steps[0]["ok"]
    assert "write" in stages
    assert stages[-1] == "done" and steps[-1]["ok"]
    assert "reverted" not in stages
    assert (root / "web_ui.py").read_text(encoding="utf-8") == NEW
    assert all(s["ok"] for s in steps)


def test_apply_edit_reverts_when_gate_fails(tmp_path, monkeypatch):
    root = _sandbox(tmp_path, monkeypatch)
    failing = [
        ("compiles", lambda: (True, "ok")),
        ("imports", lambda: (False, "import exploded")),
        ("loop still works", lambda: (True, "should not run")),
        ("server boots", lambda: (True, "should not run")),
    ]
    monkeypatch.setattr(self_edit, "GATE", failing)
    steps = list(self_edit.apply_edit("web_ui.py", NEW))
    by_stage = {s["stage"]: s for s in steps}
    assert by_stage["write"]["ok"] is True
    assert by_stage["imports"]["ok"] is False
    assert by_stage["reverted"]["ok"] is False
    assert "done" not in by_stage
    assert "loop still works" not in by_stage
    assert "server boots" not in by_stage
    assert (root / "web_ui.py").read_text(encoding="utf-8") == OLD
