#!/usr/bin/env python3
"""
self_improving_agent.py -- a feedback loop with a scoreboard.

The model writes a function, the function is run against a fixed test suite in a
sandbox, and failures are fed back verbatim so the model can rewrite. That's the
whole trick. The test suite is what makes "improvement" mean anything.

Backends:
    anthropic   paid API, claude-sonnet-5
    ollama      free, local, OpenAI-compatible endpoint on localhost:11434
    lmstudio    local, OpenAI-compatible endpoint on localhost:1234
    groq        free tier, hosted, OpenAI-compatible
    mock        no LLM at all -- canned buggy-then-correct answers per built-in task
    mock-stuck  always the same partial answer, to exercise plateau reseeding

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
import difflib
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

def tokens_from_usage(usage):
    """Normalize OpenAI-compat / Anthropic usage to prompt/completion/total.

    Local servers often omit usage or send zeros; return None in those cases
    so the UI can stay quiet instead of showing '0 tok'.
    """
    if usage is None:
        return None
    if isinstance(usage, int):
        return {"prompt": 0, "completion": 0, "total": usage} if usage else None
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens", usage.get("input_tokens",
                           usage.get("prompt")))
        completion = usage.get("completion_tokens", usage.get("output_tokens",
                                usage.get("completion")))
        total = usage.get("total_tokens", usage.get("total"))
    else:
        prompt = getattr(usage, "prompt_tokens", None)
        if prompt is None:
            prompt = getattr(usage, "input_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if completion is None:
            completion = getattr(usage, "output_tokens", None)
        total = getattr(usage, "total_tokens", None)
    if prompt is None and completion is None and total is None:
        return None
    prompt = int(prompt or 0)
    completion = int(completion or 0)
    total = int(total) if total is not None else prompt + completion
    if total <= 0:
        return None
    return {"prompt": prompt, "completion": completion, "total": total}


def take_last_tokens(backend):
    """Total tokens from the last complete(), or None if nothing was reported."""
    raw = getattr(backend, "last_usage", None)
    if hasattr(backend, "last_usage"):
        backend.last_usage = None
    info = tokens_from_usage(raw)
    if not info:
        return None
    return int(info["total"])


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

    def _remember_usage(self, resp):
        try:
            self.last_usage = tokens_from_usage(getattr(resp, "usage", None))
        except Exception:
            self.last_usage = None

    def _call(self, kwargs):
        self.last_usage = None
        if kwargs["max_tokens"] > self.STREAM_ABOVE:
            parts = []
            with self.client.messages.stream(**kwargs) as s:
                for chunk in s.text_stream:
                    parts.append(chunk)
                try:
                    self._remember_usage(s.get_final_message())
                except Exception:
                    pass
            return "".join(parts)
        resp = self.client.messages.create(**kwargs)
        self._remember_usage(resp)
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
        self.last_usage = None
        full = [{"role": "system", "content": system}] + messages
        resp = self.client.chat.completions.create(
            model=self.model, messages=full, temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            self.last_usage = tokens_from_usage(getattr(resp, "usage", None))
        except Exception:
            self.last_usage = None
        return resp.choices[0].message.content or ""


class MockBackend:
    """No LLM. A canned buggy-then-correct sequence per built-in task, keyed
    off the spec in `messages` (that is how the loop passes the task). Custom
    JSON gets a dummy stub, not a pretended solution."""

    RESPONSES_FOR = {
        "roman": [
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
        ],
        "parens": [
            """```python
def is_balanced(s):
    depth = 0
    for c in s:
        if c == '(':
            depth += 1
        elif c == ')':
            if depth == 0:
                return False
            depth -= 1
    return depth == 0
```""",
            """```python
def is_balanced(s):
    pairs = {')': '(', ']': '['}
    stack = []
    for c in s:
        if c in '([':
            stack.append(c)
        elif c in pairs:
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
    return not stack
```""",
            """```python
def is_balanced(s):
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for c in s:
        if c in '([{':
            stack.append(c)
        elif c in pairs:
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
    return not stack
```""",
        ],
        "csv": [
            """```python
def parse_csv_line(line):
    return line.split(',')
```""",
            """```python
def parse_csv_line(line):
    result, field, in_quotes = [], "", False
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            result.append(field)
            field = ""
        else:
            field += char
    result.append(field)
    return result
```""",
            """```python
def parse_csv_line(line):
    result, field, in_quotes = [], "", False
    i = 0
    while i < len(line):
        char = line[i]
        if in_quotes:
            if char == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    field += '"'
                    i += 1
                else:
                    in_quotes = False
            else:
                field += char
        else:
            if char == '"':
                in_quotes = True
            elif char == ',':
                result.append(field)
                field = ""
            else:
                field += char
        i += 1
    result.append(field)
    return result
```""",
        ],
        "expr": [
            """```python
def evaluate(expr):
    expr = expr.replace(' ', '')
    total = 0
    for part in expr.split('+'):
        total += int(part)
    return total
```""",
            """```python
def evaluate(expr):
    s = expr.replace(' ', '')
    i = 0
    def peek():
        return s[i] if i < len(s) else ''
    def num():
        nonlocal i
        n = 0
        while peek().isdigit():
            n = n * 10 + int(s[i])
            i += 1
        return n
    def factor():
        nonlocal i
        if peek() == '(':
            i += 1
            v = expr_ltr()
            i += 1
            return v
        return num()
    def expr_ltr():
        nonlocal i
        v = factor()
        while peek() in ('+', '-', '*'):
            op = peek()
            i += 1
            r = factor()
            if op == '+':
                v += r
            elif op == '-':
                v -= r
            else:
                v *= r
        return v
    return expr_ltr()
```""",
            """```python
def evaluate(expr):
    s = expr.replace(' ', '')
    i = 0
    def peek():
        return s[i] if i < len(s) else ''
    def num():
        nonlocal i
        n = 0
        while peek().isdigit():
            n = n * 10 + int(s[i])
            i += 1
        return n
    def factor():
        nonlocal i
        if peek() == '(':
            i += 1
            v = parse_expr()
            i += 1
            return v
        return num()
    def term():
        nonlocal i
        v = factor()
        while peek() == '*':
            i += 1
            v *= factor()
        return v
    def parse_expr():
        nonlocal i
        v = term()
        while peek() in ('+', '-'):
            op = peek()
            i += 1
            r = term()
            v = v + r if op == '+' else v - r
        return v
    return parse_expr()
```""",
        ],
        "roman_parse": [
            """```python
def roman_to_int(s):
    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    return sum(vals[c] for c in s)
```""",
            """```python
def roman_to_int(s):
    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    n = 0
    for i, c in enumerate(s):
        v = vals[c]
        if i + 1 < len(s) and vals[s[i + 1]] > v and c == 'I':
            n -= v
        else:
            n += v
    return n
```""",
            """```python
def roman_to_int(s):
    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    n = 0
    for i, c in enumerate(s):
        v = vals[c]
        if i + 1 < len(s) and vals[s[i + 1]] > v:
            n -= v
        else:
            n += v
    return n
```""",
        ],
    }
    RESPONSES = RESPONSES_FOR["roman"]

    model = "mock"

    def __init__(self, model=None):
        self.i = 0

    def _builtin_from_messages(self, messages):
        blob = "\n".join(
            (m.get("content") or "") if isinstance(m, dict) else str(m)
            for m in messages
        )
        for name, task in BUILTIN_TASKS.items():
            if task["spec"] in blob:
                return name
        for name, task in BUILTIN_TASKS.items():
            if f"`{task['func_name']}(" in blob:
                return name
        return None

    def _custom_dummy(self, messages):
        blob = "\n".join(
            (m.get("content") or "") if isinstance(m, dict) else str(m)
            for m in messages
        )
        m = re.search(r"function `(\w+)\(", blob)
        if not m:
            m = re.search(r"Write (?:a Python function )?`(\w+)\(", blob)
        if not m:
            m = re.search(r"Write (\w+)\(", blob)
        name = m.group(1) if m else "solution"
        return (
            "```python\n"
            f"def {name}(*args, **kwargs):\n"
            "    return None  # mock does not solve custom tasks\n"
            "```"
        )

    def complete(self, system, messages, temperature=0.2, max_tokens=2000):
        time.sleep(0.4)  # so the UI stream is visible
        key = self._builtin_from_messages(messages)
        seq = self.RESPONSES_FOR.get(key) if key else None
        if not seq:
            return self._custom_dummy(messages)
        r = seq[min(self.i, len(seq) - 1)]
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
    if backend == "lmstudio":
        return OpenAICompatBackend(
            model=model or "local-model",
            base_url=os.environ.get("LMSTUDIO_HOST_URL",
                                    "http://localhost:1234/v1"),
            api_key="lmstudio",  # ignored by LM Studio's local server
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


def compact_code_diff(prev, curr, context=2):
    """Unified diff of two attempt bodies.

    None when there is no previous attempt. Empty string when the code did
    not change. File headers are omitted so the UI can show a short hunk.
    """
    if prev is None:
        return None
    if prev == curr:
        return ""
    a = prev.splitlines(keepends=True)
    b = curr.splitlines(keepends=True)
    if a and not a[-1].endswith("\n"):
        a[-1] += "\n"
    if b and not b[-1].endswith("\n"):
        b[-1] += "\n"
    lines = []
    for line in difflib.unified_diff(a, b, fromfile="prev", tofile="this",
                                     n=context):
        if line.startswith("---") or line.startswith("+++"):
            continue
        lines.append(line if line.endswith("\n") else line + "\n")
    return "".join(lines)


# ----------------------------------------------------------------------------
# The scoreboard. Candidate code and cases go in over stdin, a JSON verdict
# comes back on stdout -- same protocol whether we run locally or in Docker.
# ----------------------------------------------------------------------------

RUNNER_SRC = r'''
import json, sys, traceback

payload = json.loads(sys.stdin.read())
code, cases, func_name = payload["code"], payload["cases"], payload["func_name"]

def case_input(args):
    shown = ", ".join(repr(a) for a in args)
    return shown if len(shown) <= 80 else shown[:77] + "..."

def short_err(text, n=160):
    line = (text or "").strip().splitlines()[-1] if text else ""
    return line[:n]

def blank_cases(error):
    msg = short_err(error)
    out = []
    for i, (args, _) in enumerate(cases):
        entry = {"id": i, "input": case_input(args), "ok": False}
        if msg:
            entry["error"] = msg
        out.append(entry)
    return out

def emit(passed, error, case_results=None):
    print("===RESULT===" + json.dumps(
        {"passed": passed, "total": len(cases), "error": error,
         "cases": case_results if case_results is not None else blank_cases(error)}))
    sys.exit(0)

ns = {}
try:
    exec(compile(code, "candidate.py", "exec"), ns)
except Exception:
    emit(0, "code failed to load:\n" + traceback.format_exc(limit=3))

fn = ns.get(func_name)
if not callable(fn):
    emit(0, "no function named %r was defined" % func_name)

passed, failures, results = 0, [], []
for i, (args, expected) in enumerate(cases):
    shown = case_input(args)
    try:
        got = fn(*args)
    except Exception:
        msg = "%s(%s) raised:\n%s" % (
            func_name, shown, traceback.format_exc(limit=2))
        failures.append(msg)
        results.append({"id": i, "input": shown, "ok": False,
                        "error": short_err(msg)})
        continue
    if got == expected:
        passed += 1
        results.append({"id": i, "input": shown, "ok": True})
    else:
        msg = "%s(%s) -> %r, expected %r" % (func_name, shown, got, expected)
        failures.append(msg)
        results.append({"id": i, "input": shown, "ok": False,
                        "error": short_err(msg)})

emit(passed, "\n".join(failures[:5]) if failures else "", results)
'''


def _case_input(args, limit=80):
    shown = ", ".join(repr(a) for a in args)
    return shown if len(shown) <= limit else shown[:limit - 3] + "..."


def _short_err(text, n=160):
    if not text:
        return ""
    return text.strip().splitlines()[-1][:n]


def _failed_cases(task, error):
    """Scoreboard rows when the runner never got to individual cases."""
    msg = _short_err(error)
    out = []
    for i, pair in enumerate(task["cases"]):
        entry = {"id": i, "input": _case_input(pair[0]), "ok": False}
        if msg:
            entry["error"] = msg
        out.append(entry)
    return out


def _parse_verdict(stdout, stderr, task):
    total = len(task["cases"])
    marker = "===RESULT==="
    if marker not in stdout:
        err = (f"test runner produced no result.\n"
               f"stdout: {stdout[-400:]}\nstderr: {stderr[-400:]}")
        return 0, total, err, _failed_cases(task, err)
    r = json.loads(stdout.split(marker, 1)[1].strip())
    cases = r.get("cases")
    if not isinstance(cases, list):
        cases = _failed_cases(task, r.get("error") or "")
    return r["passed"], r["total"], r["error"], cases


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
        err = (f"the code did not finish within {TIMEOUT_SECONDS}s "
               "-- probably an infinite loop")
        return 0, total, err, _failed_cases(task, err)
    return _parse_verdict(proc.stdout, proc.stderr, task)


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
        err = (f"the code did not finish within {TIMEOUT_SECONDS}s "
               "-- probably an infinite loop")
        return 0, total, err, _failed_cases(task, err)
    if proc.returncode == 137:
        err = "the container was killed -- ran out of memory"
        return 0, total, err, _failed_cases(task, err)
    return _parse_verdict(proc.stdout, proc.stderr, task)


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

# Zero-escape: when NOTHING has ever worked (best score is 0), there is nothing
# worth keeping. Anchoring on broken code just makes the model regenerate the
# same broken code -- so instead we start fresh, telling it only what went wrong.
FRESH = (
    "{spec}\n\n"
    "A previous attempt failed with this error:\n\n{error}\n\n"
    "Write a fresh implementation. Reply with only the complete function in one "
    "code block."
)

FRESH_RESEED = (
    "{spec}\n\n"
    "Several attempts have all failed the same way:\n\n{error}\n\n"
    "The overall structure is the problem, not a detail. Use a completely "
    "different approach. Reply with only the complete function in one code block."
)


DEFAULT_MAX_TOKENS = 100_000  # 0 = no token cap (backends that never report)


def iter_loop(backend, task, attempts=10, runner=None, should_stop=None,
              backend_name="?", stall_limit=3, escalate=True, temp_step=0.0,
              max_tokens=DEFAULT_MAX_TOKENS, max_seconds=0):
    """Yields dicts: start, attempt, escalate, done, error, stopped, capped.

    Ratchet: the best-scoring attempt is always kept, saved to disk, and used as
    the seed for the next prompt -- so a regression can never cost ground. On a
    plateau (no improvement for `stall_limit` attempts) the loop reseeds from a
    clean context still anchored on that best attempt.

    temp_step: temperature added per escalation level. Default 0.0 leaves
    temperature fixed, so the effect of reseeding can be measured without also
    changing sampling. (An earlier version changed both at once and could not
    tell which mattered; on a 7B the temperature rise made things worse.)

    max_tokens: stop before the next API call once reported usage reaches this
    total. 0 disables it. Backends that never report usage are not estimated —
    the attempt cap is then the only budget.
    max_seconds: wall-clock cap for the whole run. 0 disables it.
    """
    runner = runner or run_tests_subprocess
    run_id = uuid.uuid4().hex[:8]
    model_name = getattr(backend, "model", backend_name)
    total = len(task["cases"])
    base_temp = 0.2
    try:
        max_tokens = int(max_tokens or 0)
    except (TypeError, ValueError):
        max_tokens = 0
    try:
        max_seconds = float(max_seconds or 0)
    except (TypeError, ValueError):
        max_seconds = 0
    if max_tokens < 0:
        max_tokens = 0
    if max_seconds < 0:
        max_seconds = 0

    yield {"type": "start", "run_id": run_id, "backend": backend_name,
           "model": model_name, "task": task["name"],
           "func_name": task["func_name"], "total": total,
           "max_attempts": attempts,
           "stall_limit": stall_limit if escalate else 0,
           "max_tokens": max_tokens, "max_seconds": max_seconds}

    messages = [{"role": "user", "content": task["spec"]}]
    best_passed, best_code, best_error, best_attempt = -1, None, "", 0
    stall = 0
    level = 0
    temperature = base_temp
    prev_code = None
    tokens_used = 0
    t_start = time.time()

    def save_best():
        if best_code is None:
            return None
        p = HERE / f"best_{task['name']}_{run_id}.py"
        p.write_text(best_code + "\n", encoding="utf-8")
        return p.name

    def cap_event(reason, attempt):
        ev = {
            "type": "capped", "run_id": run_id, "reason": reason,
            "attempts_used": attempt,
            "best": max(best_passed, 0), "total": total,
            "best_attempt": best_attempt,
            "tokens_used": tokens_used,
            "max_tokens": max_tokens,
            "elapsed_s": round(time.time() - t_start, 1),
            "max_seconds": max_seconds,
        }
        if best_code is not None:
            ev["best_solution"] = f"best_{task['name']}_{run_id}.py"
        return ev

    for attempt in range(1, attempts + 1):
        if should_stop and should_stop():
            yield {"type": "stopped", "run_id": run_id, "attempt": attempt}
            return
        if max_seconds and (time.time() - t_start) >= max_seconds:
            yield cap_event("time", attempt - 1 if attempt > 1 else 0)
            return

        t0 = time.time()
        try:
            reply = backend.complete(SYSTEM, messages, temperature=temperature)
        except Exception as e:
            yield {"type": "error", "run_id": run_id,
                   "message": f"{type(e).__name__}: {e}"}
            return
        elapsed = time.time() - t0
        elapsed_ms = max(0, int(round(elapsed * 1000)))
        tokens = take_last_tokens(backend)
        if tokens is not None:
            tokens_used += tokens

        code = extract_code(reply)
        try:
            passed, total, error, cases = runner(code, task)
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
            "seconds": round(elapsed, 2), "elapsed_ms": elapsed_ms,
            "error": error, "code": code,
            "cases": cases,
            "temperature": round(temperature, 2), "level": level,
            "best": best_passed, "improved": improved,
        }
        if tokens is not None:
            record["tokens"] = tokens
        log_attempt(record)
        ev = {"type": "attempt", **record}
        diff = compact_code_diff(prev_code, code)
        if diff is not None:
            ev["code_diff"] = diff
        prev_code = code
        yield ev

        if passed == total:
            path = HERE / f"solution_{task['name']}_{run_id}.py"
            path.write_text(code + "\n", encoding="utf-8")
            yield {"type": "done", "run_id": run_id, "success": True,
                   "attempts_used": attempt, "solution": path.name,
                   "code": code}
            return

        if max_tokens and tokens_used >= max_tokens:
            yield cap_event("tokens", attempt)
            return
        if max_seconds and (time.time() - t_start) >= max_seconds:
            yield cap_event("time", attempt)
            return

        # Plateau -> reseed from a clean context, still anchored on the best.
        if escalate and stall >= stall_limit:
            level += 1
            stall = 0
            if temp_step:
                temperature = round(min(base_temp + temp_step * level, 1.0), 2)
            has_anchor = best_passed > 0
            yield {"type": "escalate", "run_id": run_id, "attempt": attempt,
                   "level": level, "temperature": round(temperature, 2),
                   "reason": f"no improvement for {stall_limit} attempts",
                   "fresh_context": True, "anchor_passed": best_passed,
                   "has_anchor": has_anchor}
            if has_anchor:
                messages = [{"role": "user", "content": RESEED.format(
                    spec=task["spec"], passed=best_passed, total=total,
                    error=best_error, code=best_code)}]
            else:
                messages = [{"role": "user", "content": FRESH_RESEED.format(
                    spec=task["spec"], error=error)}]
            continue

        # Ordinary retry. With a non-zero best, anchor on it and refine -- the
        # ratchet. With nothing working yet, do NOT refine broken code: start
        # fresh so the model can try a different structure.
        if best_passed > 0:
            messages = [
                {"role": "user", "content": task["spec"]},
                {"role": "assistant", "content": f"```python\n{best_code}\n```"},
                {"role": "user", "content": RETRY.format(
                    passed=best_passed, total=total, error=best_error)},
            ]
        else:
            messages = [{"role": "user", "content": FRESH.format(
                spec=task["spec"], error=error)}]

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
                        temp_step=args.temp_step,
                        max_tokens=args.max_tokens,
                        max_seconds=args.max_seconds):
        if ev["type"] == "start":
            print(f"task     : {ev['task']}  ({ev['func_name']})")
            print(f"backend  : {ev['backend']}  model: {ev['model']}")
            print(f"runner   : {args.runner}")
            print(f"run id   : {ev['run_id']}")
            print(f"attempts : up to {ev['max_attempts']}")
            if ev.get("max_tokens"):
                print(f"token cap: {ev['max_tokens']}")
            if ev.get("max_seconds"):
                print(f"time cap : {ev['max_seconds']}s")
            print()
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
        elif ev["type"] == "capped":
            why = ("token budget"
                   if ev["reason"] == "tokens" else "time limit")
            detail = (f"{ev['tokens_used']}/{ev['max_tokens']} tokens"
                      if ev["reason"] == "tokens"
                      else f"{ev['elapsed_s']}s")
            print(f"\nstopped: {why} reached ({detail}) after "
                  f"{ev['attempts_used']} attempts "
                  f"(best: {ev['best']}/{ev['total']}"
                  + (f" on attempt {ev['best_attempt']}" if ev.get("best_attempt")
                     else "") + ")")
            if ev.get("best_solution"):
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
                   choices=["anthropic", "ollama", "lmstudio", "groq",
                            "mock", "mock-stuck"])
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
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="stop the run after this many reported tokens "
                        f"(default {DEFAULT_MAX_TOKENS}; 0 = no token cap). "
                        "Backends that do not report usage ignore this.")
    p.add_argument("--max-seconds", type=float, default=0,
                   help="stop the run after this many wall-clock seconds "
                        "(0 = no time cap)")
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
