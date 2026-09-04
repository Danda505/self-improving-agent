#!/usr/bin/env python3
"""
web_ui.py -- browser front end for self_improving_agent.py

    python web_ui.py

Then open http://localhost:8000. Attempts stream into the page live as they
happen, so you watch the loop climb rather than reading it afterwards.

Standard library only -- no Flask, no FastAPI, nothing to install. Binds to
127.0.0.1, so it is not reachable from your network.
"""

import json
import os
import threading
import traceback
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import self_improving_agent as agent

try:
    import self_edit
except Exception:          # the self-edit tab is optional
    self_edit = None

# SIA_PORT lets the verification gate boot a second copy on a spare port
HOST, PORT = "127.0.0.1", int(os.environ.get("SIA_PORT", "8000"))
_cancelled = set()
_cancel_lock = threading.Lock()


# ----------------------------------------------------------------------------
# Environment probing, so the UI can grey out what isn't ready
# ----------------------------------------------------------------------------

def ollama_models():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            data = json.loads(r.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def lmstudio_models():
    # LM Studio serves an OpenAI-compatible endpoint on :1234
    try:
        with urllib.request.urlopen("http://localhost:1234/v1/models", timeout=3) as r:
            data = json.loads(r.read())
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []


def probe_env():
    models = ollama_models()
    lm = lmstudio_models()
    return {
        "ollama": {"up": bool(models) or _ollama_reachable(), "models": models},
        "lmstudio": {"up": bool(lm), "models": lm},
        "docker": {"up": agent.docker_available(),
                   "image": agent.docker_image_present()},
        "anthropic": {"key": bool(os.environ.get("ANTHROPIC_API_KEY"))},
        "groq": {"key": bool(os.environ.get("GROQ_API_KEY"))},
        "selfedit": {"available": self_edit is not None,
                     "files": self_edit.editable_files() if self_edit else [],
                     "backups": self_edit.list_backups()[:10] if self_edit else []},
        "tasks": {k: {"func": v["func_name"], "cases": len(v["cases"]),
                      "spec": v["spec"]}
                  for k, v in agent.BUILTIN_TASKS.items()},
    }


def friendly_error(e):
    """Turn SDK exceptions into something worth reading. A wall of traceback
    in the browser tells the user nothing they can act on."""
    name = type(e).__name__
    text = str(e)
    low = (name + " " + text).lower()

    if "authentication" in low or "401" in text:
        return ("Your API key was rejected (401). Check it at "
                "console.anthropic.com, then set it again and restart the "
                "server from the same PowerShell window.")
    if "rate_limit" in low or "429" in text:
        return "Rate limited by the API (429). Wait a moment and try again."
    if "credit" in low or "billing" in low or "quota" in low:
        return ("The API rejected the request for billing reasons. Check "
                "your credit balance at console.anthropic.com.")
    if "not_found" in low or "404" in text:
        return (f"The API did not recognise that model name. {text[:200]}")
    if "overloaded" in low or "529" in text:
        return "The API is overloaded (529). Try again shortly."
    if "connection" in low or "timeout" in low:
        return (f"Could not reach the model. If you selected ollama, make "
                f"sure it is running. ({name})")
    return f"{name}: {text[:400]}"


def _ollama_reachable():
    try:
        urllib.request.urlopen("http://localhost:11434/", timeout=2).read()
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass  # keep the console clean for the agent's own output

    # -- helpers ------------------------------------------------------------
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        raw = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    # -- routes -------------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self._send(200, PAGE, "text/html")
            elif u.path == "/api/ping":
                # Deliberately does no work. The self-edit gate uses this to
                # tell "the server is alive" from "a probe inside /api/env is
                # slow", which are very different things.
                self._send(200, {"ok": True})
            elif u.path == "/api/env":
                self._send(200, probe_env())
            elif u.path == "/api/history":
                self._send(200, self._history())
            elif u.path == "/api/run":
                self._stream_run(q)
            elif u.path == "/api/cancel":
                rid = q.get("run_id", [""])[0]
                with _cancel_lock:
                    _cancelled.add(rid)
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})
        except BrokenPipeError:
            pass
        except Exception:
            try:
                self._send(500, {"error": traceback.format_exc(limit=3)})
            except Exception:
                pass

    # -- self-edit (POST, because bodies are large) --------------------------
    def do_POST(self):
        u = urlparse(self.path)
        try:
            if self_edit is None:
                return self._send(503, {"error": "self_edit.py not available"})
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")

            if u.path == "/api/propose":
                backend = agent.build_backend(body.get("backend", "anthropic"),
                                              body.get("model") or None)
                name, content, diff = self_edit.propose(
                    backend, body["request"], body.get("file") or None)
                self._send(200, {"file": name, "content": content, "diff": diff})

            elif u.path == "/api/apply":
                self._stream_apply(body["file"], body["content"])

            elif u.path == "/api/restore":
                names = self_edit.restore_backup(body["backup"])
                self._send(200, {"ok": True, "restored": names})

            else:
                self._send(404, {"error": "not found"})
        except BrokenPipeError:
            pass
        except (ValueError, RuntimeError, KeyError) as e:
            # expected, actionable failures -- show the message, not a traceback
            msg = f"missing field: {e}" if isinstance(e, KeyError) else str(e)
            try:
                self._send(400, {"error": msg})
            except Exception:
                pass
        except Exception as e:
            # Anything else: a one-line summary for the browser, the full
            # traceback in the console where it belongs.
            traceback.print_exc()
            try:
                self._send(500, {"error": friendly_error(e)})
            except Exception:
                pass

    def _history(self):
        runs = agent.grouped_runs()
        out = []
        for run_id, attempts in runs.items():
            head = attempts[0]
            flags = [a["success"] for a in attempts]
            out.append({
                "run_id": run_id, "model": head["model"],
                "backend": head["backend"], "task": head["task"],
                "total": head["total"], "ts": head["ts"],
                "solved_at": flags.index(True) + 1 if any(flags) else None,
                "points": [{"attempt": a["attempt"], "passed": a["passed"],
                            "total": a["total"], "seconds": a.get("seconds", 0)}
                           for a in attempts],
            })
        out.sort(key=lambda r: r["ts"], reverse=True)
        return out

    def _stream_apply(self, name, content):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for step in self_edit.apply_edit(name, content):
                self.wfile.write(f"data: {json.dumps(step)}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                self.wfile.write(f"data: {json.dumps({'stage':'error','ok':False,'detail':str(e)})}\n\n".encode())
                self.wfile.flush()
            except Exception:
                pass

    # -- the live stream ----------------------------------------------------
    def _stream_run(self, q):
        one = lambda k, d=None: q.get(k, [d])[0]
        backend_name = one("backend", "mock")
        model = one("model") or None
        runner_name = one("runner", "subprocess")
        attempts = int(one("attempts", "10"))
        escalate = one("escalate", "1") == "1"
        stall_limit = int(one("stall_limit", "3"))
        temp_step = float(one("temp_step", "0") or "0")
        task_name = one("task", "roman")
        task_json = one("task_json")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            if task_json:
                task = agent.validate_task(json.loads(task_json))
            else:
                task = agent.BUILTIN_TASKS[task_name]

            emit({"type": "status", "message": "checking runner..."})
            if (runner_name == "docker" and agent.docker_available()
                    and not agent.docker_image_present()):
                emit({"type": "status",
                      "message": f"pulling {agent.DOCKER_IMAGE} (first run only, ~45MB)..."})
            runner = agent.get_runner(runner_name)

            emit({"type": "status", "message": "connecting to model..."})
            backend = agent.build_backend(backend_name, model)
        except Exception as e:
            emit({"type": "error", "message": friendly_error(e)})
            return

        state = {"run_id": None}

        def should_stop():
            with _cancel_lock:
                return state["run_id"] in _cancelled

        try:
            for ev in agent.iter_loop(backend, task, attempts, runner,
                                      should_stop, backend_name,
                                      stall_limit, escalate, temp_step):
                if ev["type"] == "start":
                    state["run_id"] = ev["run_id"]
                ev["runner"] = runner_name
                emit(ev)
        except (BrokenPipeError, ConnectionResetError):
            return  # browser closed the tab; nothing to clean up
        except Exception as e:
            try:
                emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
            except Exception:
                pass
        finally:
            with _cancel_lock:
                _cancelled.discard(state["run_id"])


# ----------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Self-improving agent</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --panel2:#1d212a; --line:#2a2f3a;
    --fg:#e6e8ee; --dim:#8b93a5; --accent:#5eb0ff; --ok:#3fca7c;
    --bad:#ef6461; --warn:#e5b95c;
    --mono: ui-monospace,"Cascadia Code",Consolas,monospace;
  }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 system-ui,"Segoe UI",sans-serif }
  header { padding:16px 22px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:14px }
  h1 { font-size:17px; margin:0; font-weight:600 }
  header .sub { color:var(--dim); font-size:13px }
  header .clock { margin-left:auto; color:var(--dim); font-size:13px;
                  font-family:var(--mono); letter-spacing:.5px;
                  white-space:nowrap }
  .wrap { display:grid; grid-template-columns:310px 1fr; gap:0;
          height:calc(100vh - 57px) }
  aside { border-right:1px solid var(--line); padding:18px; overflow-y:auto;
          background:var(--panel) }
  main { padding:18px 22px; overflow-y:auto }
  label { display:block; font-size:12px; color:var(--dim);
          margin:14px 0 5px; text-transform:uppercase; letter-spacing:.4px }
  select, input, textarea, button {
    width:100%; background:var(--panel2); color:var(--fg);
    border:1px solid var(--line); border-radius:7px; padding:8px 10px;
    font-size:13px; font-family:inherit }
  textarea { font-family:var(--mono); font-size:12px; min-height:120px;
             resize:vertical }
  button { cursor:pointer; font-weight:600; margin-top:16px }
  #run { background:var(--accent); color:#06121f; border-color:transparent }
  #run:hover:not(:disabled) { filter:brightness(1.1) }
  #run:disabled { opacity:.5; cursor:default }
  #stop { background:transparent; border-color:var(--bad); color:var(--bad);
          margin-top:8px }
  .pill { display:inline-block; padding:2px 8px; border-radius:99px;
          font-size:11px; border:1px solid var(--line); color:var(--dim) }
  .pill.on { color:var(--ok); border-color:#245c3d }
  .pill.off { color:var(--dim) }
  .status-row { display:flex; gap:6px; flex-wrap:wrap; margin-top:6px }
  .card { background:var(--panel); border:1px solid var(--line);
          border-radius:10px; padding:14px 16px; margin-bottom:12px }
  .card.pass { border-color:#245c3d }
  .card.fail { border-color:#3d2b2b }
  .card h3 { margin:0 0 10px; font-size:13px; font-weight:600;
             display:flex; justify-content:space-between; align-items:center }
  .muted { color:var(--dim); font-weight:400; font-size:12px }
  .bar { display:flex; gap:3px; margin:8px 0 10px }
  .seg { height:8px; flex:1; border-radius:2px; background:#343a47 }
  .seg.p { background:var(--ok) }
  pre { background:#0b0d11; border:1px solid var(--line); border-radius:7px;
        padding:11px 13px; overflow-x:auto; font-family:var(--mono);
        font-size:12px; margin:0; white-space:pre-wrap; word-break:break-word }
  pre.err { color:#ffb3b1; background:#160f10; border-color:#3d2b2b }
  details summary { cursor:pointer; color:var(--dim); font-size:12px;
                    margin-bottom:8px; user-select:none }
  .banner { padding:11px 14px; border-radius:8px; margin-bottom:12px;
            font-weight:600 }
  .banner.ok { background:#0f2a1c; color:var(--ok); border:1px solid #245c3d }
  .banner.bad { background:#251617; color:var(--bad); border:1px solid #3d2b2b }
  .banner.info { background:#141b26; color:var(--accent);
                 border:1px solid #23374d; font-weight:400 }
  .empty { color:var(--dim); text-align:center; padding:60px 20px }
  svg { width:100%; height:230px }
  .hist { font-size:12px; color:var(--dim); display:flex;
          justify-content:space-between; padding:5px 0;
          border-bottom:1px solid var(--line) }
  .tabs { display:flex; gap:4px; margin-bottom:14px }
  .tab { padding:6px 13px; border-radius:7px; cursor:pointer;
         font-size:13px; color:var(--dim) }
  .tab.active { background:var(--panel); color:var(--fg) }
  .hide { display:none }
  .warn-box { background:#231d10; border:1px solid #4a3c1c; color:var(--warn);
              border-radius:8px; padding:9px 12px; font-size:12px;
              margin-top:14px }
</style>
</head>
<body>
<header>
  <h1>Self-improving agent</h1>
  <span class="sub">write &rarr; test &rarr; feed the error back &rarr; rewrite</span>
  <span class="clock" id="clock"></span>
</header>

<div class="wrap">
<aside>
  <label>Backend</label>
  <select id="backend"></select>
  <div class="status-row" id="envpills"></div>

  <label>Model</label>
  <select id="modelsel" class="hide"></select>
  <input id="model" placeholder="default">

  <label>Task</label>
  <select id="task"></select>

  <div id="customwrap" class="hide">
    <label>Task JSON</label>
    <textarea id="taskjson"></textarea>
  </div>

  <label>Where code runs</label>
  <select id="runner">
    <option value="subprocess">subprocess (fast, not a sandbox)</option>
    <option value="docker">docker (sandboxed, no network)</option>
  </select>

  <label>Max attempts <span id="attlbl" class="muted">10</span></label>
  <input type="range" id="attempts" min="1" max="25" value="10">

  <label>If it plateaus</label>
  <select id="escalate">
    <option value="1">reseed from best — keep the high-water mark</option>
    <option value="0">nothing — keep feeding the same error</option>
  </select>

  <label>Stall limit <span id="stalllbl" class="muted">3</span></label>
  <input type="range" id="stall_limit" min="1" max="10" value="3">

  <label>Temp step <span class="muted">0 = no heat-up</span></label>
  <input id="temp_step" type="number" min="0" max="1" step="0.05" value="0">

  <button id="run">Run</button>
  <button id="stop" class="hide">Stop</button>

  <div class="warn-box" id="sandboxwarn">
    subprocess isolates crashes and infinite loops, but model-written code can
    still touch your files. Switch to docker for anything open-ended.
  </div>
</aside>

<main>
  <div class="tabs">
    <div class="tab active" data-tab="live">Live run</div>
    <div class="tab" data-tab="curve">Improvement curve</div>
    <div class="tab" data-tab="edit">Self-edit</div>
  </div>

  <div id="live">
    <div id="banner"></div>
    <div id="attempts-out"></div>
    <div class="empty" id="idle">Pick a backend and press Run.<br>
      <span style="font-size:12px">Start with <b>mock</b> — it needs no model
      and proves the loop works.</span></div>
  </div>

  <div id="curve" class="hide">
    <div class="card"><svg id="chart" viewBox="0 0 800 230"
         preserveAspectRatio="none"></svg></div>
    <div class="card"><h3>Runs</h3><div id="histlist"></div></div>
  </div>

  <div id="edit" class="hide">
    <div class="warn-box" style="margin:0 0 14px">
      This rewrites the app's own source on your disk. Nothing is written until
      you approve the diff, every write is backed up first, and any edit that
      fails the checks is reverted automatically. Use a strong model — a 7B
      will mangle a 700-line file.
    </div>

    <div class="card">
      <div style="display:flex; gap:10px; align-items:flex-end">
        <div style="flex:1">
          <label style="margin-top:0">Ask for a change</label>
          <textarea id="editreq" style="min-height:64px"
            placeholder="add a button that clears the run history"></textarea>
        </div>
        <div style="width:170px">
          <label style="margin-top:0">Target file</label>
          <select id="editfile"></select>
        </div>
      </div>
      <div style="display:flex; gap:10px">
        <button id="propose" style="flex:1">Propose a change</button>
        <button id="restorebtn" style="width:170px; background:transparent;
                border-color:var(--line)">Restore backup…</button>
      </div>
      <select id="backupsel" class="hide" style="margin-top:10px"></select>
    </div>

    <div id="editout"></div>
  </div>
</main>
</div>

<script>
const $ = id => document.getElementById(id);
let es = null, runId = null, env = null;

const BACKENDS = [
  ["mock",       "mock — no model, canned answers"],
  ["ollama",     "ollama — local, free"],
  ["lmstudio",   "lmstudio — local, LM Studio"],
  ["anthropic",  "anthropic — paid API"],
  ["groq",       "groq — free tier, hosted"],
  ["mock-stuck", "mock-stuck — always 12/15, to test plateau reseeding"],
];

function tickClock() {
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  $("clock").textContent = `${hh}:${mm}:${ss}`;
}
tickClock();
setInterval(tickClock, 1000);

async function loadEnv() {
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 15000);
    const r = await fetch("/api/env", { signal: ctl.signal });
    clearTimeout(t);
    if (!r.ok) throw new Error("server returned " + r.status);
    env = await r.json();
  } catch (e) {
    // Blank dropdowns with no explanation are the worst possible failure.
    $("banner").innerHTML =
      `<div class="banner bad">Could not load settings from the server:
       ${esc(e.message)}. Is web_ui.py still running in PowerShell?</div>`;
    $("idle").classList.add("hide");
    throw e;
  }
  $("backend").innerHTML = BACKENDS
    .map(([v, l]) => `<option value="${v}">${l}</option>`).join("");

  $("task").innerHTML =
    Object.entries(env.tasks).map(([k, t]) =>
      `<option value="${k}">${k} — ${t.func}() · ${t.cases} tests</option>`
    ).join("") + `<option value="__custom">custom…</option>`;

  const pill = (on, txt) =>
    `<span class="pill ${on ? "on" : "off"}">${on ? "●" : "○"} ${txt}</span>`;
  $("envpills").innerHTML =
    pill(env.ollama.up, "ollama") +
    pill(env.lmstudio.up, "lmstudio") +
    pill(env.docker.up, "docker") +
    pill(env.anthropic.key, "anthropic key") +
    pill(env.groq.key, "groq key");

  syncBackend();
  return env;
}

function modelsFor(b) {
  if (b === "ollama") return env.ollama.models;
  if (b === "lmstudio") return env.lmstudio.models;
  return [];
}

function syncBackend() {
  const b = $("backend").value;
  const list = modelsFor(b);
  const useList = list.length > 0;
  if (useList) {
    $("modelsel").innerHTML = list
      .map(m => `<option value="${m}">${m}</option>`).join("");
  }
  $("modelsel").classList.toggle("hide", !useList);
  $("model").classList.toggle("hide", useList);
  $("model").placeholder =
    b === "anthropic" ? "claude-sonnet-5"
    : b === "groq" ? "llama-3.3-70b-versatile"
    : b === "ollama" ? "qwen2.5-coder:7b"
    : b === "lmstudio" ? "(load a model in LM Studio)" : "n/a";
}

function currentModel() {
  const b = $("backend").value;
  return modelsFor(b).length ? $("modelsel").value : $("model").value.trim();
}

$("backend").onchange = syncBackend;
$("attempts").oninput = e => $("attlbl").textContent = e.target.value;
$("stall_limit").oninput = e => $("stalllbl").textContent = e.target.value;
$("task").onchange = e => {
  const custom = e.target.value === "__custom";
  $("customwrap").classList.toggle("hide", !custom);
  if (custom && !$("taskjson").value.trim()) {
    $("taskjson").value = JSON.stringify({
      name: "fizz", func_name: "fizzbuzz",
      spec: "Write fizzbuzz(n) returning 'Fizz', 'Buzz', 'FizzBuzz' or str(n).",
      cases: [[[3], "Fizz"], [[5], "Buzz"], [[15], "FizzBuzz"], [[7], "7"]]
    }, null, 2);
  }
};
$("runner").onchange = e => {
  const d = e.target.value === "docker";
  $("sandboxwarn").textContent = d
    ? "Code runs in python:3.11-slim with no network, 256MB, read-only disk, as nobody. First run pulls the image."
    : "subprocess isolates crashes and infinite loops, but model-written code can still touch your files. Switch to docker for anything open-ended.";
};

document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
  document.querySelectorAll(".tab").forEach(x =>
    x.classList.toggle("active", x === t));
  ["live", "curve", "edit"].forEach(id =>
    $(id).classList.toggle("hide", t.dataset.tab !== id));
  if (t.dataset.tab === "curve") loadHistory();
});

function esc(s) {
  return (s ?? "").toString()
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function bar(passed, total) {
  let s = '<div class="bar">';
  for (let i = 0; i < total; i++)
    s += `<div class="seg ${i < passed ? "p" : ""}"></div>`;
  return s + "</div>";
}

$("run").onclick = () => {
  $("idle").classList.add("hide");
  $("attempts-out").innerHTML = "";
  $("banner").innerHTML = "";
  $("run").disabled = true;
  $("stop").classList.remove("hide");

  const p = new URLSearchParams({
    backend: $("backend").value,
    model: currentModel(),
    runner: $("runner").value,
    attempts: $("attempts").value,
    task: $("task").value,
    escalate: $("escalate").value,
    stall_limit: $("stall_limit").value,
    temp_step: $("temp_step").value,
  });
  if ($("task").value === "__custom") p.set("task_json", $("taskjson").value);

  es = new EventSource("/api/run?" + p);
  es.onmessage = e => handle(JSON.parse(e.data));
  es.onerror = () => { finish(); };
};

$("stop").onclick = () => {
  if (runId) fetch("/api/cancel?run_id=" + runId);
  $("stop").textContent = "stopping…";
};

function finish() {
  if (es) { es.close(); es = null; }
  $("run").disabled = false;
  $("stop").classList.add("hide");
  $("stop").textContent = "Stop";
  runId = null;
}

function handle(ev) {
  const out = $("attempts-out");

  if (ev.type === "status") {
    $("banner").innerHTML = `<div class="banner info">${esc(ev.message)}</div>`;
  }
  else if (ev.type === "start") {
    runId = ev.run_id;
    $("banner").innerHTML =
      `<div class="banner info">${esc(ev.model)} · ${esc(ev.task)} ·
       ${esc(ev.func_name)}() · ${ev.total} tests · ${esc(ev.runner)} ·
       run ${esc(ev.run_id)}</div>`;
  }
  else if (ev.type === "attempt") {
    const ok = ev.passed === ev.total;
    const card = document.createElement("div");
    card.className = "card " + (ok ? "pass" : "fail");
    card.innerHTML =
      `<h3>Attempt ${ev.attempt}
         <span class="muted">${ev.passed}/${ev.total} passed ·
         ${ev.seconds}s</span></h3>` +
      bar(ev.passed, ev.total) +
      (ev.error && !ok
        ? `<pre class="err">${esc(ev.error)}</pre>
           <details style="margin-top:9px"><summary>code</summary>
           <pre>${esc(ev.code)}</pre></details>`
        : `<pre>${esc(ev.code)}</pre>`);
    out.appendChild(card);
    card.scrollIntoView({ behavior: "smooth", block: "end" });
  }
  else if (ev.type === "escalate") {
    const d = document.createElement("div");
    d.className = "card";
    d.style.borderColor = "#4a3c1c";
    d.style.background = "#1c1810";
    d.innerHTML = ev.has_anchor
      ? `<h3 style="color:var(--warn)">Stuck — reseeding from best
         <span class="muted">level ${ev.level} · temperature ${ev.temperature}</span></h3>
       <div class="muted">${esc(ev.reason)}. Restarting from a clean context,
         still using the best code so far` +
      (ev.anchor_passed != null ? ` (${ev.anchor_passed} passed)` : "") +
      `.</div>`
      : `<h3 style="color:var(--warn)">Stuck at 0 — discarding the broken code and starting fresh.
         <span class="muted">level ${ev.level} · temperature ${ev.temperature}</span></h3>
       <div class="muted">${esc(ev.reason)}.</div>`;
    out.appendChild(d);
    d.scrollIntoView({ behavior: "smooth", block: "end" });
  }
  else if (ev.type === "done") {
    $("banner").innerHTML = ev.success
      ? `<div class="banner ok">Solved on attempt ${ev.attempts_used} —
         saved as ${esc(ev.solution)}</div>`
      : `<div class="banner bad">Gave up after ${ev.attempts_used} attempts
         (best ${ev.best}/${ev.total})</div>`;
    finish();
  }
  else if (ev.type === "stopped") {
    $("banner").innerHTML = `<div class="banner bad">Stopped.</div>`;
    finish();
  }
  else if (ev.type === "error") {
    $("banner").innerHTML =
      `<div class="banner bad">${esc(ev.message)}</div>`;
    finish();
  }
}

function shortModel(m) {
  // Trim long fine-tune names to family+size:
  // "qwen3.5-9b-uncensored-hauhaucs-aggressive" -> "qwen3.5-9b"
  const parts = String(m).split("-");
  return parts.length > 2 ? parts.slice(0, 2).join("-") : m;
}

async function loadHistory() {
  const runs = await (await fetch("/api/history")).json();
  $("histlist").innerHTML = runs.length
    ? runs.map(r => {
        const total = r.points[0] ? r.points[0].total : 0;
        const best = Math.max(...r.points.map(p => p.passed));
        return `<div class="hist">
          <span>${esc(shortModel(r.model))} · ${esc(r.task)}</span>
          <span>${r.solved_at
            ? "solved on attempt " + r.solved_at
            : "best " + best + "/" + total + " · " + r.points.length + " tries"}</span>
        </div>`;
      }).join("")
    : `<div class="muted">No runs yet.</div>`;
  drawChart(runs.slice(0, 8));
}

function drawChart(runs) {
  const el = $("chart");
  const W = el.clientWidth || 800, H = 260;
  const PL = 40, PB = 26, PT = 12, PR = 168;
  el.setAttribute("viewBox", `0 0 ${W} ${H}`);
  el.removeAttribute("preserveAspectRatio");

  const maxX = Math.max(2, ...runs.flatMap(r => r.points.map(p => p.attempt)));
  const x = a => PL + (a - 1) / (maxX - 1) * (W - PL - PR);
  const y = pct => PT + (1 - pct / 100) * (H - PT - PB);

  const MODEL_COLORS = ["#5eb0ff", "#3fca7c", "#e5b95c", "#b58cff", "#ef6461", "#4fd0d0"];
  const DASHES = ["", "7 5", "2 4", "10 4 2 4"];
  const models = [...new Set(runs.map(r => shortModel(r.model)))];
  const tasks = [...new Set(runs.map(r => r.task))];
  const colorOf = m => MODEL_COLORS[models.indexOf(shortModel(m)) % MODEL_COLORS.length];
  const dashOf = t => DASHES[tasks.indexOf(t) % DASHES.length];

  let s = "";
  for (const pct of [0, 25, 50, 75, 100]) {
    s += `<line x1="${PL}" y1="${y(pct)}" x2="${W - PR}" y2="${y(pct)}" stroke="#2a2f3a" stroke-width="1"/>`
       + `<text x="${PL - 7}" y="${y(pct) + 3.5}" fill="#8b93a5" font-size="10" text-anchor="end">${pct}%</text>`;
  }
  for (let a = 1; a <= maxX; a++) {
    s += `<text x="${x(a)}" y="${H - 8}" fill="#8b93a5" font-size="10" text-anchor="middle">${a}</text>`;
  }

  runs.forEach(r => {
    const c = colorOf(r.model), dash = dashOf(r.task);
    const pts = r.points.map(p => `${x(p.attempt)},${y(100 * p.passed / p.total)}`);
    s += `<polyline points="${pts.join(" ")}" fill="none" stroke="${c}" stroke-width="2" stroke-dasharray="${dash}" stroke-linejoin="round"/>`;
    r.points.forEach(p => {
      const solved = p.passed === p.total;
      s += `<circle cx="${x(p.attempt)}" cy="${y(100 * p.passed / p.total)}" r="${solved ? 5 : 3}" fill="${c}"${solved ? ' stroke="#fff" stroke-width="1.5"' : ''}><title>${esc(shortModel(r.model))} · ${esc(r.task)} · attempt ${p.attempt} · ${p.passed}/${p.total}</title></circle>`;
    });
    const last = r.points[r.points.length - 1];
    if (last) {
      s += `<text x="${x(last.attempt) + 8}" y="${y(100 * last.passed / last.total) + 3}" fill="${c}" font-size="10">${last.passed}/${last.total}</text>`;
    }
  });

  const lx = W - PR + 12;
  let ly = PT + 6;
  runs.forEach(r => {
    const c = colorOf(r.model), dash = dashOf(r.task);
    s += `<line x1="${lx}" y1="${ly}" x2="${lx + 20}" y2="${ly}" stroke="${c}" stroke-width="2" stroke-dasharray="${dash}"/>`
       + `<text x="${lx + 26}" y="${ly + 3.5}" fill="#e6e8ee" font-size="10">${esc(shortModel(r.model))} · ${esc(r.task)}${r.solved_at ? " ✓" : ""}</text>`;
    ly += 18;
  });

  if (!runs.length) {
    s += `<text x="${(W - PR) / 2}" y="${H / 2}" fill="#8b93a5" font-size="13" text-anchor="middle">No runs logged yet</text>`;
  }
  el.innerHTML = s;
}

// ---- self-edit ------------------------------------------------------------

let proposal = null;

function initSelfEdit() {
  const se = env.selfedit;
  if (!se || !se.available) {
    $("edit").innerHTML =
      `<div class="banner bad">self_edit.py is missing from the folder.</div>`;
    return;
  }
  $("editfile").innerHTML = `<option value="">let it choose</option>` +
    se.files.map(f =>
      `<option value="${f.name}">${f.name} · ${f.lines} lines</option>`).join("");
  $("backupsel").innerHTML = se.backups.length
    ? se.backups.map(b => `<option value="${b}">${b}</option>`).join("")
    : `<option value="">no backups yet</option>`;
}

$("propose").onclick = async () => {
  const req = $("editreq").value.trim();
  if (!req) return;
  const b = $("backend").value;
  if (b === "mock" || b === "mock-stuck") {
    $("editout").innerHTML =
      `<div class="banner bad">Pick a real backend — mock can't write code.</div>`;
    return;
  }
  $("propose").disabled = true;
  $("editout").innerHTML = `<div class="banner info">Thinking… a full file
    rewrite takes a while.</div>`;
  try {
    const r = await fetch("/api/propose", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request: req, backend: b, model: currentModel(),
                             file: $("editfile").value })
    });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    proposal = d;
    $("editout").innerHTML =
      `<div class="banner info">Proposed change to <b>${esc(d.file)}</b> —
         read it before approving.</div>
       <div class="card"><pre id="diffbox">${colorDiff(d.diff)}</pre></div>
       <div style="display:flex; gap:10px">
         <button id="approve" style="flex:1; background:var(--ok);
           color:#06210f; border-color:transparent">Approve and verify</button>
         <button id="reject" style="flex:1; background:transparent;
           border-color:var(--bad); color:var(--bad)">Discard</button>
       </div>`;
    $("approve").onclick = applyProposal;
    $("reject").onclick = () => { proposal = null; $("editout").innerHTML = ""; };
  } catch (e) {
    $("editout").innerHTML = `<div class="banner bad">${esc(e.message)}</div>`;
  } finally {
    $("propose").disabled = false;
  }
};

function colorDiff(diff) {
  return diff.split("\n").map(l => {
    const t = esc(l);
    if (l.startsWith("+++") || l.startsWith("---"))
      return `<span style="color:var(--dim)">${t}</span>`;
    if (l.startsWith("@@"))
      return `<span style="color:var(--accent)">${t}</span>`;
    if (l.startsWith("+"))
      return `<span style="color:var(--ok)">${t}</span>`;
    if (l.startsWith("-"))
      return `<span style="color:var(--bad)">${t}</span>`;
    return t;
  }).join("\n");
}

function applyProposal() {
  if (!proposal) return;
  $("approve").disabled = true;
  $("reject").disabled = true;
  const box = document.createElement("div");
  box.className = "card";
  box.innerHTML = `<h3>Verifying</h3><div id="steps"></div>`;
  $("editout").appendChild(box);

  fetch("/api/apply", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file: proposal.file, content: proposal.content })
  }).then(async r => {
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const p of parts) {
        if (!p.startsWith("data: ")) continue;
        const s = JSON.parse(p.slice(6));
        const row = document.createElement("div");
        row.style.cssText = "padding:5px 0; font-size:13px; border-bottom:1px solid var(--line)";
        row.innerHTML =
          `<span style="color:${s.ok ? "var(--ok)" : "var(--bad)"}">
             ${s.ok ? "✓" : "✗"}</span> ${esc(s.stage)}` +
          (s.detail && !s.ok
            ? `<pre class="err" style="margin-top:6px">${esc(s.detail)}</pre>`
            : s.detail ? ` <span class="muted">${esc(s.detail)}</span>` : "");
        $("steps").appendChild(row);
        if (s.stage === "done")
          box.innerHTML += `<div class="banner ok" style="margin-top:12px">
            Kept. Stop the server (Ctrl+C) and start it again to load the
            new code.</div>`;
        if (s.stage === "reverted")
          box.innerHTML += `<div class="banner bad" style="margin-top:12px">
            Reverted — your files are exactly as they were.</div>`;
      }
    }
    proposal = null;
  });
}

$("restorebtn").onclick = async () => {
  const sel = $("backupsel");
  if (sel.classList.contains("hide")) {
    sel.classList.remove("hide");
    $("restorebtn").textContent = "Restore this one";
    return;
  }
  if (!sel.value) return;
  const r = await fetch("/api/restore", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ backup: sel.value })
  });
  const d = await r.json();
  $("editout").innerHTML = d.error
    ? `<div class="banner bad">${esc(d.error)}</div>`
    : `<div class="banner ok">Restored ${d.restored.join(", ")}. Restart the
       server to load them.</div>`;
};

loadEnv().then(initSelfEdit).catch(() => {});
</script>
</body>
</html>
"""


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"\n  Self-improving agent UI  ->  {url}")
    print("  Ctrl-C to stop\n")
    if not os.environ.get("SIA_NO_BROWSER"):
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
