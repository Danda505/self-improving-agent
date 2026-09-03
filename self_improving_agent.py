#!/usr/bin/env python3
"""
self_improving_agent.py -- a feedback loop with a scoreboard.

The model writes a function, the function is run against a fixed test suite in a
sandbox, and failures are fed back verbatim so the model can rewrite. That's the
whole trick. The test suite is what makes "improvement" mean anything.

Backends:
    anthropic   paid API, claude-sonnet-5
    ollama      free, local, OpenAI-compatible endpoint on localhost:11434
    groq        free tier, hosted, OpenAI-compatible
    mock        no LLM at all -- canned buggy-then-correct answers, for testing

Runners (where candidate code executes):
    subprocess  default. Isolation with a timeout, NOT a sandbox.
    docker      python:3.11-slim, --network=none, memory and pid capped.

Usage:
    python self_improving_agent.py --backend ollama --model qwen2.5-coder:7b
    python self_improving_agent.py --backend ollama --runner docker
    python self_improving_agent.py --plot
    python web_ui.py                     # browser UI for all of the above
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / "attempts.jsonl"
TIMEOUT_SECONDS = 10
DOCKER_IMAGE = "python:3.11-slim"


# ----------------------------------------------------------------------------
# Tasks: a spec for the model + cases the machine can check without a human
# ----------------------------------------------------------------------------

BUILTIN_TASKS = {
    "roman": {
        "name": "roman",
        "func_name": "int_to_roman",
        "spec": (
            "Write a Python function `int_to_roman(num)` that converts an "
            "integer in the range 1..3999 into its Roman numeral string, using "
            "standard subtractive notation (4 is 'IV', 900 is 'CM')."
        ),
        "cases": [
            [[1], "I"], [[3], "III"], [[4], "IV"], [[9], "IX"], [[14], "XIV"],
            [[40], "XL"], [[90], "XC"], [[400], "CD"], [[1994], "MCMXCIV"],
            [[2024], "MMXXIV"], [[3999], "MMMCMXCIX"],
        ],
    },
    "parens": {
        "name": "parens",
        "func_name": "is_balanced",
        "spec": (
            "Write a Python function `is_balanced(s)` that returns True if all "
            "brackets in the string s -- (), [] and {} -- are correctly matched "
            "and nested, and False otherwise. Characters that are not brackets "
            "are ignored."
        ),
        "cases": [
            [[""], True], [["()"], True], [["([{}])"], True], [["(]"], False],
            [["([)]"], False], [["((("], False], [["a(b)c[d]"], True],
            [["}{"], False],
        ],
    },
    "csv": {
        "name": "csv",
        "func_name": "parse_csv_line",
        "spec": (
            "Write a Python function `parse_csv_line(line)` that splits a "
            "single line of CSV into a list of field strings.\n\n"
            "Rules:\n"
            "- Fields are separated by commas.\n"
            "- A field may be wrapped in double quotes. Commas inside quotes "
            "are literal text, not separators.\n"
            "- Inside a quoted field, two double quotes ("
            "\"\") mean one literal double quote character.\n"
            "- Quotes are removed from the output; the field's contents are "
            "returned, not the quoting.\n"
            "- Whitespace is never stripped. Empty fields become empty "
            "strings. An empty line returns a list containing one empty "
            "string.\n\n"
            "Do not use the csv module or any import. Parse the string "
            "yourself."
        ),
        "cases": [
            [["a,b,c"], ["a", "b", "c"]],
            [[""], [""]],
            [["a,,c"], ["a", "", "c"]],
            [['"a,b",c'], ["a,b", "c"]],
            [['"a""b",c'], ['a"b', "c"]],
            [['"",x'], ["", "x"]],
            [['a,"b",c'], ["a", "b", "c"]],
            [['"a,b","c,d"'], ["a,b", "c,d"]],
            [["a,b,"], ["a", "b", ""]],
            [[",a"], ["", "a"]],
            [['"he said ""hi"""'], ['he said "hi"']],
            [['x," y ",z'], ["x", " y ", "z"]],
            [['"a""""b"'], ['a""b']],
            [["a b,c"], ["a b", "c"]],
            [['"multi,word,field"'], ["multi,word,field"]],
        ],
    },
    "expr": {
        "name": "expr",
        "func_name": "evaluate",
        "spec": (
            "Write a Python function `evaluate(expr)` that evaluates an "
            "arithmetic expression string and returns the integer result.\n\n"
            "Rules:\n"
            "- The expression contains non-negative integers, the operators "
            "+ - and *, parentheses, and spaces.\n"
            "- Standard precedence: * binds tighter than + and -.\n"
            "- + and - are left-associative, so 10-2-3 is 5, not 11.\n"
            "- Spaces may appear anywhere and are ignored.\n"
            "- Results may be negative.\n\n"
            "Do not use eval(), exec(), ast, or any import. Parse the string "
            "yourself."
        ),
        "cases": [
            [["2+3*4"], 14],
            [["(2+3)*4"], 20],
            [["10-2-3"], 5],
            [["2*3+4*5"], 26],
            [["((1+2)*(3+4))"], 21],
            [["7"], 7],
            [["1+2*3-4"], 3],
            [["10-(2-3)"], 11],
            [["2*(3+(4*5))"], 46],
            [[" 2 + 3 "], 5],
            [["100-10*5"], 50],
            [["(1+2)*(3-4)"], -3],
            [["2*3*4-5"], 19],
            [["1+2+3+4"], 10],
        ],
    },
    "roman_parse": {
        "name": "roman_parse",
        "func_name": "roman_to_int",
        "spec": (
            "Write a Python function `roman_to_int(s)` that converts a Roman "
            "numeral string (valid, uppercase, 1..3999, standard subtractive "
            "notation) into its integer value."
        ),
        "cases": [
            [["I"], 1], [["IV"], 4], [["IX"], 9], [["XIV"], 14], [["XL"], 40],
            [["XC"], 90], [["CD"], 400], [["MCMXCIV"], 1994],
            [["MMXXIV"], 2024], [["MMMCMXCIX"], 3999],
        ],
    },
}


def load_task(args):
    if args.task_file:
        task = json.loads(Path(args.task_file).read_text(encoding="utf-8"))
        validate_task(task)
        return task
    return BUILTIN_TASKS[args.task]


def validate_task(task):
    for key in ("name", "func_name", "spec", "cases"):
        if key not in task:
            raise ValueError(f"task is missing required key: {key}")
    if not isinstance(task["cases"], list) or not task["cases"]:
        raise ValueError("task needs a non-empty 'cases' list")
    for c in task["cases"]:
        if not (isinstance(c, list) and len(c) == 2 and isinstance(c[0], list)):
            raise ValueError(f"each case must be [[args...], expected]; got {c!r}")
    return task


# ----------------------------------------------------------------------------
# Backends: every one of these is complete(system, messages) -> str
# ----------------------------------------------------------------------------

class AnthropicBackend:
    """Anthropic takes `system` as its own parameter, not a message."""

    def __init__(self, model=None):
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "the anthropic SDK is not installed. Run:\n"
                "    python -m pip install anthropic")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. In PowerShell:\n"
                '    $env:ANTHROPIC_API_KEY = "sk-ant-..."\n'
                "then start the server again from that same window.")
        self.model = model or "claude-sonnet-5"
        self.client = anthropic.Anthropic()
        # Newer models reject `temperature` outright. We only find out by
        # asking, so try once and remember the answer.
        self.supports_temperature = True

    # Above this the SDK refuses a plain request -- it could exceed the
    # 10-minute non-streaming limit, so it insists on streaming.
    STREAM_ABOVE = 8000

    def _call(self, kwargs):
        if kwargs["max_tokens"] > self.STREAM_ABOVE:
            parts = []
            with self.client.messages.stream(**kwargs) as s:
                for chunk in s.text_stream:
                    parts.append(chunk)
            return "".join(parts)
        resp = self.client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if b.type == "text")

    def complete(self, system, messages, temperature=0.2, max_tokens=2000):
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=system,
                      messages=messages)
        if self.supports_temperature:
            kwargs["temperature"] = temperature
        try:
            return self._call(kwargs)
        except Exception as e:
            if self.supports_temperature and "temperature" in str(e).lower():
                self.supports_temperature = False
                kwargs.pop("temperature", None)
                return self._call(kwargs)
            raise


class OpenAICompatBackend:
    """Ollama and Groq both speak the OpenAI chat format.
    Here `system` is just the first message in the list."""

    def __init__(self, model, base_url, api_key):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "the openai SDK is not installed (it is the client for Ollama "
                "and Groq too). Run:\n    python -m pip install openai")
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=300.0)

    def complete(self, system, messages, temperature=0.2, max_tokens=2000):
        full = [{"role": "system", "content": system}] + messages
        resp = self.client.chat.completions.create(
            model=self.model, messages=full, temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


class MockBackend:
    """No LLM. A fixed buggy-then-correct sequence so you can exercise the loop,
    the logging and the chart without an API key or a GPU."""

    RESPONSES = [
        """Here you go:
````````python
def int_to_roman(num):
    vals = [(1000,'M'),(500,'D'),(100,'C'),(50,'L'),(10,'X'),(5,'V'),(1,'I')]
    out = ''
    for v, s in vals:
        while num >= v:
            out += s
            num -= v
    return out
```````""",
        """Fixed:
``````python
def int_to_roman(num):
    vals = [(1000,'M'),(900,'CM'),(500,'D'),(100,'C'),(90,'XC'),
            (50,'L'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    out = ''
    for v, s in vals:
        while num >= v:
            out += s
            num -= v
    return out
`````""",
        """```python
def int_to_roman(num):
    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
            (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    out = ''
    for v, s in vals:
        while num >= v:
            out += s
            num -= v
    return out
````""",
    ]

    model = "mock"

    def __init__(self, model=None):
        self.i = 0

    def complete(self, system, messages, temperature=0.2, max_tokens=2000):
        time.sleep(0.4)  # so the UI stream is visible
        r = self.RESPONSES[min(self.i, len(self.RESPONSES) - 1)]
        self.i += 1
        return r


class StuckMockBackend(MockBackend):
    """Always returns the same 12/15 answer, so you can watch stagnation
    detection fire without burning a real model on it."""

    model = "mock-stuck"
    STUCK = '''```python
def parse_csv_line(line):
    result, field, in_quotes = [], "", False
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            result.append(field); field = ""
        else:
            field += char
    result.append(field)
    return result
```'''

    def complete(self, system, messages, temperature=0.2, max_tokens=2000):
        time.sleep(0.2)
        return self.STUCK


def build_backend(backend, model=None):
    if backend == "anthropic":
        return AnthropicBackend(model)
    if backend == "ollama":
        return OpenAICompatBackend(
            model=model or "qwen2.5-coder:7b",
            base_url=os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434/v1"),
            api_key="ollama",  # required by the client, ignored by the server
        )
    if backend == "groq":
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("set GROQ_API_KEY first (free key from console.groq.com)")
        return OpenAICompatBackend(
            model=model or "llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1", api_key=key,
        )
    if backend == "mock":
        return MockBackend()
    if backend == "mock-stuck":
        return StuckMockBackend()
    raise ValueError(f"unknown backend: {backend}")


# ----------------------------------------------------------------------------
# Pulling code out of a chatty reply
# ----------------------------------------------------------------------------

FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text):
    """Small models wrap code in prose, stray fences and 'Sure!' preambles.
    Prefer the longest fenced block; fall back to the raw text."""
    blocks = FENCE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("def ", "import ", "from ")):
            return "\n".join(lines[i:]).strip()
    return text.strip()


# ----------------------------------------------------------------------------
# The scoreboard. Candidate code and cases go in over stdin, a JSON verdict
# comes back on stdout -- same protocol whether we run locally or in Docker.
# ----------------------------------------------------------------------------

RUNNER_SRC = r'''
import json, sys, traceback

payload = json.loads(sys.stdin.read())
code, cases, func_name = payload["code"], payload["cases"], payload["func_name"]

def emit(passed, error):
    print("===RESULT===" + json.dumps(
        {"passed": passed, "total": len(cases), "error": error}))
    sys.exit(0)

ns = {}
try:
    exec(compile(code, "candidate.py", "exec"), ns)
except Exception:
    emit(0, "code failed to load:\n" + traceback.format_exc(limit=3))

fn = ns.get(func_name)
if not callable(fn):
    emit(0, "no function named %r was defined" % func_name)

passed, failures = 0, []
for args, expected in cases:
    shown = ", ".join(repr(a) for a in args)
    try:
        got = fn(*args)
    except Exception:
        failures.append("%s(%s) raised:\n%s"
                        % (func_name, shown, traceback.format_exc(limit=2)))
        continue
    if got == expected:
        passed += 1
    else:
        failures.append("%s(%s) -> %r, expected %r"
                        % (func_name, shown, got, expected))

emit(passed, "\n".join(failures[:5]) if failures else "")
'''


def _parse_verdict(stdout, stderr, total):
    marker = "===RESULT==="
    if marker not in stdout:
        return 0, total, (f"test runner produced no result.\n"
                          f"stdout: {stdout[-400:]}\nstderr: {stderr[-400:]}")
    r = json.loads(stdout.split(marker, 1)[1].strip())
    return r["passed"], r["total"], r["error"]


def run_tests_subprocess(code, task):
    """Isolation with a timeout. NOT a sandbox -- this code can still reach
    your filesystem and network."""
    total = len(task["cases"])
    payload = json.dumps({"code": code, "cases": task["cases"],
                          "func_name": task["func_name"]})
    try:
        proc = subprocess.run(
            [sys.executable, "-c", RUNNER_SRC], input=payload,
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return 0, total, (f"the code did not finish within {TIMEOUT_SECONDS}s "
                          "-- probably an infinite loop")
    return _parse_verdict(proc.stdout, proc.stderr, total)


# `docker info` hangs for a long time when Docker Desktop is not running, and
# the UI probes it on every page load. Short timeout, and cache the answer so
# one slow probe cannot stall the interface repeatedly.
_docker_cache = {"ok": None, "image": None, "at": 0.0}
_DOCKER_CACHE_SECONDS = 20


def docker_available(force=False):
    now = time.time()
    if (not force and _docker_cache["ok"] is not None
            and now - _docker_cache["at"] < _DOCKER_CACHE_SECONDS):
        return _docker_cache["ok"]
    try:
        p = subprocess.run(["docker", "info"], capture_output=True,
                           text=True, timeout=4)
        ok = p.returncode == 0
    except Exception:
        ok = False
    _docker_cache.update(ok=ok, at=now)
    return ok


def docker_image_present(force=False):
    now = time.time()
    if (not force and _docker_cache["image"] is not None
            and now - _docker_cache["at"] < _DOCKER_CACHE_SECONDS):
        return _docker_cache["image"]
    if not docker_available():
        _docker_cache.update(image=False)
        return False
    try:
        p = subprocess.run(["docker", "image", "inspect", DOCKER_IMAGE],
                           capture_output=True, text=True, timeout=10)
        present = p.returncode == 0
    except Exception:
        present = False
    _docker_cache.update(image=present)
    return present


def docker_pull():
    subprocess.run(["docker", "pull", DOCKER_IMAGE], timeout=600)


def run_tests_docker(code, task):
    """A real sandbox: no network, capped memory and processes, read-only
    filesystem, non-root, killed on timeout."""
    total = len(task["cases"])
    payload = json.dumps({"code": code, "cases": task["cases"],
                          "func_name": task["func_name"]})
    name = "sia-" + uuid.uuid4().hex[:10]
    cmd = [
        "docker", "run", "--rm", "-i", "--name", name,
        "--network=none",           # no exfiltration, no downloads
        "--memory=256m", "--memory-swap=256m",
        "--cpus=1", "--pids-limit=64",
        "--read-only",              # filesystem is immutable
        "--tmpfs", "/tmp:rw,size=16m",
        "--user", "65534:65534",    # nobody
        "--security-opt", "no-new-privileges",
        DOCKER_IMAGE, "python", "-c", RUNNER_SRC,
    ]
    try:
        proc = subprocess.run(cmd, input=payload, capture_output=True,
                              text=True, timeout=TIMEOUT_SECONDS + 15)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return 0, total, (f"the code did not finish within {TIMEOUT_SECONDS}s "
                          "-- probably an infinite loop")
    if proc.returncode == 137:
        return 0, total, "the container was killed -- ran out of memory"
    return _parse_verdict(proc.stdout, proc.stderr, total)


def get_runner(name):
    if name == "docker":
        if not docker_available(force=True):
            raise RuntimeError(
                "Docker is not responding. Start Docker Desktop and wait for "
                "the whale icon to stop animating, then try again.")
        if not docker_image_present():
            docker_pull()
        return run_tests_docker
    return run_tests_subprocess


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def log_attempt(record):
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_log(task_filter=None):
    if not LOG_PATH.exists():
        return []
    rows = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if task_filter:
        rows = [r for r in rows if r.get("task") == task_filter]
    return rows


def grouped_runs(task_filter=None):
    runs = {}
    for r in read_log(task_filter):
        runs.setdefault(r["run_id"], []).append(r)
    for attempts in runs.values():
        attempts.sort(key=lambda a: a["attempt"])
    return runs


# ----------------------------------------------------------------------------
# The loop, as a stream of events. The CLI prints them; the web UI streams
# them to the browser. Same loop either way.
# ----------------------------------------------------------------------------

SYSTEM = (
    "You are a Python programmer. Reply with ONE Python function inside a "
    "single ```python code block. No explanation, no example usage, no tests, "
    "no print statements. Only the function definition."
)

# The ratchet: every retry is anchored on the BEST attempt so far, never on a
# regression. A model that just made things worse is shown its own best version
# and asked to fix only what still fails -- so it cannot wander off from a worse
# starting point, and it cannot lose ground it already earned.
RETRY = (
    "That version passes {passed} of {total} tests. It still fails these:\n\n"
    "{error}\n\n"
    "Keep everything that already works. Change only what is needed to fix the "
    "failing cases above. Reply with only the complete function in one code "
    "block."
)

# On a plateau we reseed from a clean context -- same anchor (the best code so
# far), but freshly stated, because repeating an identical prompt reproduces an
# identical answer. Crucially we do NOT tell the model to abandon the approach:
# an earlier version did, threw away a nearly-correct solution, and the score
# collapsed from 12/15 to 2/15. Keep the good work; fix only the gaps.
RESEED = (
    "{spec}\n\n"
    "A near-complete solution already exists. It passes {passed} of {total} "
    "tests and fails only these:\n\n{error}\n\n"
    "Here it is:\n\n```python\n{code}\n```\n\n"
    "Study the failing cases and fix them, keeping everything that already "
    "works. Reply with only the complete function in one code block."
)


def iter_loop(backend, task, attempts=10, runner=None, should_stop=None,
              backend_name="?", stall_limit=3, escalate=True, temp_step=0.0):
    """Yields dicts: start, attempt, escalate, done, error, stopped.

    Ratchet: the best-scoring attempt is always kept, saved to disk, and used as
    the seed for the next prompt -- so a regression can never cost ground. On a
    plateau (no improvement for `stall_limit` attempts) the loop reseeds from a
    clean context still anchored on that best attempt.

    temp_step: temperature added per escalation level. Default 0.0 leaves
    temperature fixed, so the effect of reseeding can be measured without also
    changing sampling. (An earlier version changed both at once and could not
    tell which mattered; on a 7B the temperature rise made things worse.)
    """
    runner = runner or run_tests_subprocess
    run_id = uuid.uuid4().hex[:8]
    model_name = getattr(backend, "model", backend_name)
    total = len(task["cases"])
    base_temp = 0.2

    yield {"type": "start", "run_id": run_id, "backend": backend_name,
           "model": model_name, "task": task["name"],
           "func_name": task["func_name"], "total": total,
           "max_attempts": attempts,
           "stall_limit": stall_limit if escalate else 0}

    messages = [{"role": "user", "content": task["spec"]}]
    best_passed, best_code, best_error, best_attempt = -1, None, "", 0
    stall = 0
    level = 0
    temperature = base_temp

    def save_best():
        if best_code is None:
            return None
        p = HERE / f"best_{task['name']}_{run_id}.py"
        p.write_text(best_code + "\n", encoding="utf-8")
        return p.name

    for attempt in range(1, attempts + 1):
        if should_stop and should_stop():
            yield {"type": "stopped", "run_id": run_id, "attempt": attempt}
            return

        t0 = time.time()
        try:
            reply = backend.complete(SYSTEM, messages, temperature=temperature)
        except Exception as e:
            yield {"type": "error", "run_id": run_id,
                   "message": f"{type(e).__name__}: {e}"}
            return
        elapsed = time.time() - t0

        code = extract_code(reply)
        try:
            passed, total, error = runner(code, task)
        except Exception as e:
            yield {"type": "error", "run_id": run_id,
                   "message": f"test runner failed: {type(e).__name__}: {e}"}
            return

        improved = passed > best_passed
        if improved:
            best_passed, best_code, best_error, best_attempt = \
                passed, code, error, attempt
            stall = 0
            save_best()
        else:
            stall += 1

        record = {
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "backend": backend_name, "model": model_name,
            "task": task["name"], "attempt": attempt,
            "passed": passed, "total": total, "success": passed == total,
            "seconds": round(elapsed, 2), "error": error, "code": code,
            "temperature": round(temperature, 2), "level": level,
            "best": best_passed, "improved": improved,
        }
        log_attempt(record)
        yield {"type": "attempt", **record}

        if passed == total:
            path = HERE / f"solution_{task['name']}_{run_id}.py"
            path.write_text(code + "\n", encoding="utf-8")
            yield {"type": "done", "run_id": run_id, "success": True,
                   "attempts_used": attempt, "solution": path.name,
                   "code": code}
            return

        # Plateau -> reseed from a clean context, still anchored on the best.
        if escalate and stall >= stall_limit:
            level += 1
            stall = 0
            if temp_step:
                temperature = round(min(base_temp + temp_step * level, 1.0), 2)
            yield {"type": "escalate", "run_id": run_id, "attempt": attempt,
                   "level": level, "temperature": round(temperature, 2),
                   "reason": f"no improvement for {stall_limit} attempts",
                   "fresh_context": True, "anchor_passed": best_passed}
            messages = [{"role": "user", "content": RESEED.format(
                spec=task["spec"], passed=best_passed, total=total,
                error=best_error, code=best_code)}]
            continue

        # Ordinary retry -- ALWAYS anchored on the best attempt, not the latest.
        # This is the ratchet: a worse attempt is discarded, the best is what
        # the model sees and refines.
        messages = [
            {"role": "user", "content": task["spec"]},
            {"role": "assistant", "content": f"```python\n{best_code}\n```"},
            {"role": "user", "content": RETRY.format(
                passed=best_passed, total=total, error=best_error)},
        ]

    # Gave up -- but never lose the best. It is already saved; report where.
    yield {"type": "done", "run_id": run_id, "success": False,
           "attempts_used": attempts, "best": best_passed, "total": total,
           "best_attempt": best_attempt,
           "best_solution": f"best_{task['name']}_{run_id}.py"}


def run_loop_cli(backend, task, args, runner):
    ok = False
    for ev in iter_loop(backend, task, args.attempts, runner,
                        backend_name=args.backend,
                        stall_limit=args.stall_limit,
                        escalate=not args.no_escalate,
                        temp_step=args.temp_step):
        if ev["type"] == "start":
            print(f"task     : {ev['task']}  ({ev['func_name']})")
            print(f"backend  : {ev['backend']}  model: {ev['model']}")
            print(f"runner   : {args.runner}")
            print(f"run id   : {ev['run_id']}")
            print(f"attempts : up to {ev['max_attempts']}\n")
        elif ev["type"] == "attempt":
            bar = "#" * ev["passed"] + "." * (ev["total"] - ev["passed"])
            print(f"attempt {ev['attempt']:>2}: [{bar}] "
                  f"{ev['passed']}/{ev['total']}  ({ev['seconds']}s)")
            if ev["error"] and ev["passed"] != ev["total"]:
                print(f"           {ev['error'].splitlines()[0][:100]}")
        elif ev["type"] == "escalate":
            print(f"           -- stuck: {ev['reason']}. reseeding from best "
                  f"({ev['anchor_passed']}/{ev.get('total', '?')}) at level "
                  f"{ev['level']}, temperature {ev['temperature']}")
        elif ev["type"] == "done":
            if ev["success"]:
                ok = True
                print("\nALL TESTS PASSED\n")
                print(ev["code"])
                print(f"\nsaved -> {ev['solution']}")
            else:
                print(f"\ngave up after {ev['attempts_used']} attempts "
                      f"(best: {ev['best']}/{ev['total']} on attempt "
                      f"{ev['best_attempt']})")
                print(f"best kept -> {ev['best_solution']}")
        elif ev["type"] == "error":
            print(f"\nerror: {ev['message']}")
    return ok


# ----------------------------------------------------------------------------
# The improvement curve (CLI)
# ----------------------------------------------------------------------------

def plot(args):
    runs = grouped_runs(args.task_filter)
    if not runs:
        sys.exit("no runs in the log yet -- run the agent first")

    print(f"\n{len(runs)} run(s) in {LOG_PATH.name}\n")
    for run_id, attempts in runs.items():
        head = attempts[0]
        total = head["total"]
        flags = [a["success"] for a in attempts]
        status = (f"solved on attempt {flags.index(True) + 1}"
                  if any(flags) else "unsolved")
        print(f"  {run_id}  {head['model']:<28} {head['task']:<12} {status}")
        for a in attempts:
            bar = "#" * a["passed"] + "." * (total - a["passed"])
            print(f"      {a['attempt']:>2} [{bar}] {a['passed']}/{total}")
        print()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(pip install matplotlib for a PNG chart)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for run_id, attempts in runs.items():
        xs = [a["attempt"] for a in attempts]
        ys = [100.0 * a["passed"] / a["total"] for a in attempts]
        ax.plot(xs, ys, marker="o",
                label=f"{attempts[0]['model']} / {attempts[0]['task']} ({run_id})")
    ax.set_ylim(-5, 105)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("attempt")
    ax.set_ylabel("tests passed (%)")
    ax.set_title("Improvement curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out = HERE / "improvement_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"chart -> {out.name}")


# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="mock",
                   choices=["anthropic", "ollama", "groq", "mock", "mock-stuck"])
    p.add_argument("--stall-limit", type=int, default=3,
                   help="attempts without improvement before reseeding")
    p.add_argument("--no-escalate", action="store_true",
                   help="disable reseeding on plateau (for A/B runs)")
    p.add_argument("--temp-step", type=float, default=0.0,
                   help="temperature added per reseed level (default 0: fixed)")
    p.add_argument("--model", default=None, help="override the default model")
    p.add_argument("--runner", default="subprocess",
                   choices=["subprocess", "docker"])
    p.add_argument("--task", default="roman", choices=list(BUILTIN_TASKS))
    p.add_argument("--task-file", default=None, help="path to a custom task JSON")
    p.add_argument("--attempts", type=int, default=10)
    p.add_argument("--plot", action="store_true", help="show the improvement curve")
    p.add_argument("--task-filter", default=None, help="with --plot, one task only")
    args = p.parse_args()

    if args.plot:
        plot(args)
        return

    try:
        task = load_task(args)
        runner = get_runner(args.runner)
        backend = build_backend(args.backend, args.model)
    except Exception as e:
        sys.exit(f"error: {e}")

    sys.exit(0 if run_loop_cli(backend, task, args, runner) else 1)


if __name__ == "__main__":
    main()
