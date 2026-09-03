#!/usr/bin/env python3
"""
self_edit.py -- let the agent rewrite its own source, with a safety net.

You type a request ("add a dark/light toggle", "show tokens per second").
A strong model rewrites one file. You see a diff. Nothing is written until
you approve it, and nothing survives that breaks the app.

The honest framing: this is a self-EDITING agent, not a self-improving one.
There is no scoreboard for "did it add live weather" the way there is for
"did 15 tests pass", so the loop cannot tell whether an edit was an
improvement -- only whether the program still works afterwards. You are the
scoreboard here. That is a real difference and it is why this lives in a
separate file from the part that actually measures something.

Safety, in order:
  1. self_edit.py is not editable by the agent -- the referee does not play.
  2. Every write is preceded by a timestamped backup of the whole folder.
  3. Nothing is written until you approve the diff.
  4. After writing, a verification gate runs: compile, import, a real mock
     agent run, and (for web_ui.py) an actual server boot on a spare port.
  5. Any failure reverts the file automatically.
"""

import ast
import difflib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKUP_DIR = HERE / "backups"
VERIFY_PORT = 8765
BOOT_TIMEOUT = 60
# Enough to emit a ~900-line file whole. Rewrites are not cheap; this is the
# main reason a self-edit costs more than a task run.
MAX_EDIT_TOKENS = 32000

# The agent may not edit these. self_edit.py runs the verification gate; if
# the agent could rewrite it, it could rewrite the thing that catches its
# mistakes. attempts.jsonl is data, not code.
PROTECTED = {"self_edit.py", "attempts.jsonl"}

EDIT_SYSTEM = """You are editing the source of a small Python application \
that runs a self-improving-agent loop with a web UI.

Rules:
- Reply with the COMPLETE new content of exactly ONE file.
- Put it in a single ```python code block. Nothing before or after it except \
one short line naming the file you chose, in the form: FILE: web_ui.py
- Preserve everything that already works. You are making a targeted change, \
not a rewrite. Keep existing function names, routes and behaviour intact \
unless the request requires changing them.
- The app must keep running. It uses only the Python standard library plus \
optional `openai`, `anthropic` and `matplotlib`. Do not add dependencies \
that require installation unless the user explicitly asked for one.
- No placeholders, no "... rest of file unchanged". Output the whole file."""


# ----------------------------------------------------------------------------
# What the agent is allowed to see and touch
# ----------------------------------------------------------------------------

def editable_files():
    out = []
    for p in sorted(HERE.glob("*.py")):
        if p.name in PROTECTED or p.name.startswith("solution_"):
            continue
        out.append({"name": p.name, "lines": len(p.read_text(
            encoding="utf-8", errors="replace").splitlines()),
            "bytes": p.stat().st_size})
    return out


def read_file(name):
    p = (HERE / name).resolve()
    if p.parent != HERE or p.name in PROTECTED or not p.exists():
        raise ValueError(f"not an editable file: {name}")
    return p.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------
# Backups
# ----------------------------------------------------------------------------

def make_backup(label="edit"):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = BACKUP_DIR / f"{stamp}-{label}"
    dest.mkdir(parents=True, exist_ok=True)
    for p in HERE.glob("*.py"):
        shutil.copy2(p, dest / p.name)
    return dest


def list_backups():
    if not BACKUP_DIR.exists():
        return []
    return sorted((d.name for d in BACKUP_DIR.iterdir() if d.is_dir()),
                  reverse=True)


def restore_backup(name):
    src = BACKUP_DIR / name
    if not src.is_dir():
        raise ValueError(f"no such backup: {name}")
    restored = []
    for p in src.glob("*.py"):
        if p.name in PROTECTED:
            continue
        shutil.copy2(p, HERE / p.name)
        restored.append(p.name)
    return restored


# ----------------------------------------------------------------------------
# Proposing an edit
# ----------------------------------------------------------------------------

def build_context(target=None):
    """The files the model gets to read. One target keeps the reply small."""
    names = [f["name"] for f in editable_files()]
    if target:
        names = [target] + [n for n in names if n != target]
    parts = []
    for n in names:
        parts.append(f"=== {n} ===\n{read_file(n)}")
    return "\n\n".join(parts)


def parse_reply(reply):
    """Pull FILE: name and the code block out of the model's answer."""
    name = None
    for line in reply.splitlines():
        s = line.strip()
        if s.upper().startswith("FILE:"):
            name = s.split(":", 1)[1].strip().strip("`*")
            break
    import re
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply,
                        re.DOTALL | re.IGNORECASE)
    if not blocks:
        raise ValueError("the model did not return a code block")
    content = max(blocks, key=len).strip() + "\n"
    if not name:
        # fall back: guess from a module docstring line like "web_ui.py -- ..."
        first = content.lstrip('"\'\n ').split("\n", 1)[0]
        for f in editable_files():
            if f["name"] in first:
                name = f["name"]
                break
    if not name:
        raise ValueError("could not tell which file the model meant to edit; "
                         "it must start its reply with 'FILE: <name>'")
    if name in PROTECTED:
        raise ValueError(f"{name} is protected and cannot be edited")
    if name not in {f["name"] for f in editable_files()}:
        raise ValueError(f"unknown file: {name}")
    return name, content


def propose(backend, request, target=None):
    """Ask the model for a rewrite. Returns (file, new_content, diff)."""
    user = (f"{build_context(target)}\n\n"
            f"=== REQUEST ===\n{request}\n\n"
            "Rewrite exactly one file to satisfy this. Start your reply with "
            "'FILE: <filename>' then the complete file in one code block.")
    # A whole-file rewrite is thousands of tokens. The default budget for a
    # single function would truncate it silently, halfway through a line.
    reply = backend.complete(EDIT_SYSTEM, [{"role": "user", "content": user}],
                             temperature=0.2, max_tokens=MAX_EDIT_TOKENS)
    name, content = parse_reply(reply)
    old = read_file(name)

    # Truncation looks like a syntax error but has a completely different
    # cause and fix, so name it specifically.
    if len(content) < len(old) * 0.6:
        raise ValueError(
            f"the reply looks truncated -- {len(content)} characters back for "
            f"a {len(old)}-character file. The model ran out of output budget. "
            f"Try a smaller, more specific request.")

    try:
        ast.parse(content)
    except SyntaxError as e:
        hint = ("  This often means the reply was cut off mid-file."
                if e.lineno and e.lineno > content.count("\n") - 5 else "")
        raise ValueError(f"the model produced invalid Python "
                         f"(line {e.lineno}): {e.msg}.{hint}")

    diff = "".join(difflib.unified_diff(
        old.splitlines(keepends=True), content.splitlines(keepends=True),
        fromfile=f"{name} (current)", tofile=f"{name} (proposed)", n=3))
    if not diff.strip():
        raise ValueError("the model returned the file unchanged")
    return name, content, diff


# ----------------------------------------------------------------------------
# The verification gate. Every check runs in a subprocess so a broken edit
# cannot take down the server that is running this.
# ----------------------------------------------------------------------------

def _run(cmd, timeout, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    try:
        p = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True,
                           timeout=timeout, env=e)
        return p.returncode == 0, (p.stdout + p.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except Exception as ex:
        return False, f"{type(ex).__name__}: {ex}"


def check_compiles():
    return _run([sys.executable, "-m", "py_compile",
                 "self_improving_agent.py", "web_ui.py"], 60)


def check_imports():
    return _run([sys.executable, "-c",
                 "import self_improving_agent, web_ui; print('imports ok')"], 60)


def check_loop_still_works():
    """A real end-to-end run on the mock backend. Catches edits that leave the
    file importable but break the actual loop."""
    ok, out = _run([sys.executable, "self_improving_agent.py",
                    "--backend", "mock", "--task", "roman", "--attempts", "5"], 90)
    if ok and "ALL TESTS PASSED" not in out:
        return False, "mock run finished but did not solve the task:\n" + out
    return ok, out


def check_server_boots():
    """Start the UI on a spare port and make a real request to it."""
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "web_ui.py"], cwd=str(HERE),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, "SIA_PORT": str(VERIFY_PORT),
                 "SIA_NO_BROWSER": "1"})
        deadline = time.time() + BOOT_TIMEOUT
        last = "no response"
        while time.time() < deadline:
            if proc.poll() is not None:
                return False, "server exited immediately:\n" + (
                    proc.stdout.read()[-1200:] if proc.stdout else "")
            # /api/ping does no work. /api/env is the fallback, and it can be
            # genuinely slow (it probes Docker), so give it real time rather
            # than timing out and retrying forever.
            for path, per_try in (("/api/ping", 5), ("/api/env", 20)):
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{VERIFY_PORT}{path}",
                            timeout=per_try) as r:
                        json.loads(r.read())
                    return True, f"server answered on {path}"
                except Exception as e:
                    last = f"{path}: {type(e).__name__}"
            time.sleep(0.5)
        return False, (f"server did not answer within {BOOT_TIMEOUT}s "
                       f"(last: {last})")
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


GATE = [
    ("compiles", check_compiles),
    ("imports", check_imports),
    ("loop still works", check_loop_still_works),
    ("server boots", check_server_boots),
]


def apply_edit(name, content):
    """Backup, write, verify, revert on any failure. Yields progress dicts."""
    old = read_file(name)
    backup = make_backup("pre-" + name.replace(".py", ""))
    yield {"stage": "backup", "ok": True, "detail": backup.name}

    (HERE / name).write_text(content, encoding="utf-8")
    yield {"stage": "write", "ok": True, "detail": f"{name} written"}

    for label, fn in GATE:
        ok, detail = fn()
        yield {"stage": label, "ok": ok, "detail": detail.strip()[-800:]}
        if not ok:
            (HERE / name).write_text(old, encoding="utf-8")
            yield {"stage": "reverted", "ok": False,
                   "detail": f"{name} restored; nothing was kept. "
                             f"Backup also at backups/{backup.name}"}
            return

    yield {"stage": "done", "ok": True,
           "detail": f"{name} passed every check and was kept. "
                     "Restart the server to load it."}


# ----------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="self-edit from the command line")
    p.add_argument("request", nargs="?", help="what to change")
    p.add_argument("--backend", default="anthropic")
    p.add_argument("--model", default=None)
    p.add_argument("--file", default=None, help="which file to target")
    p.add_argument("--list-backups", action="store_true")
    p.add_argument("--restore", metavar="NAME")
    p.add_argument("--verify-only", action="store_true",
                   help="run the gate against the current files")
    a = p.parse_args()

    if a.list_backups:
        for b in list_backups():
            print(" ", b)
        return
    if a.restore:
        print("restored:", ", ".join(restore_backup(a.restore)))
        return
    if a.verify_only:
        for label, fn in GATE:
            ok, detail = fn()
            print(f"  {'PASS' if ok else 'FAIL'}  {label}")
            if not ok:
                print("       " + detail.replace("\n", "\n       ")[:600])
                sys.exit(1)
        print("\nall checks passed")
        return
    if not a.request:
        p.error("give a request, or use --verify-only / --list-backups")

    import self_improving_agent as agent
    backend = agent.build_backend(a.backend, a.model)
    print(f"asking {getattr(backend, 'model', a.backend)} to edit...\n")
    name, content, diff = propose(backend, a.request, a.file)
    print(diff)
    if input(f"\napply this to {name}? [y/N] ").strip().lower() != "y":
        print("nothing written")
        return
    for step in apply_edit(name, content):
        print(f"  {'ok  ' if step['ok'] else 'FAIL'} {step['stage']}")
        if not step["ok"]:
            print("       " + step["detail"].replace("\n", "\n       ")[:800])


if __name__ == "__main__":
    main()
