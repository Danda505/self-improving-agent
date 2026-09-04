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
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.5) as r:
            data = json.loads(r.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def lmstudio_models():
    # LM Studio serves an OpenAI-compatible endpoint on :1234
    try:
        with urllib.request.urlopen("http://localhost:1234/v1/models", timeout=1.5) as r:
            data = json.loads(r.read())
        return [m["id"] for m in data.get("data", [])]
    except Exception:
        return []


def probe_env():
    """Fast: keys, tasks, self-edit. Never waits on docker/ollama/lmstudio.

    Slow probes live on /api/env/status so a hung Docker Desktop cannot
    blank the backend and task dropdowns. Field names stay the same;
    ollama/lmstudio/docker start empty and the client fills them later.
    """
    return {
        "ollama": {"up": False, "models": []},
        "lmstudio": {"up": False, "models": []},
        "docker": {"up": False, "image": False},
        "anthropic": {"key": bool(os.environ.get("ANTHROPIC_API_KEY"))},
        "groq": {"key": bool(os.environ.get("GROQ_API_KEY"))},
        "selfedit": {"available": self_edit is not None,
                     "files": self_edit.editable_files() if self_edit else [],
                     "backups": self_edit.list_backups()[:10] if self_edit else []},
        "tasks": {k: {"func": v["func_name"], "cases": len(v["cases"]),
                      "spec": v["spec"]}
                  for k, v in agent.BUILTIN_TASKS.items()},
    }


def probe_status():
    """Docker, ollama, lmstudio — run in parallel so one hang is the ceiling."""
    out = {
        "ollama": {"up": False, "models": []},
        "lmstudio": {"up": False, "models": []},
        "docker": {"up": False, "image": False},
    }

    def ollama():
        models = ollama_models()
        out["ollama"] = {"up": bool(models) or _ollama_reachable(),
                         "models": models}

    def lmstudio():
        models = lmstudio_models()
        out["lmstudio"] = {"up": bool(models), "models": models}

    def docker():
        out["docker"] = {"up": agent.docker_available(),
                         "image": agent.docker_image_present()}

    threads = [threading.Thread(target=fn) for fn in (ollama, lmstudio, docker)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


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
        urllib.request.urlopen("http://localhost:11434/", timeout=1.5).read()
        return True
    except Exception:
        return False


def clear_history():
    if agent.LOG_PATH.exists():
        agent.LOG_PATH.write_text("", encoding="utf-8")


def _task_options_html():
    lines = []
    for k, v in agent.BUILTIN_TASKS.items():
        n = len(v["cases"])
        lines.append(
            f'          <option value="{k}">{k} — {v["func_name"]}() · '
            f'{n} tests</option>')
    lines.append('          <option value="__custom">custom…</option>')
    return "\n".join(lines)


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

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    # -- routes -------------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self._send(200, PAGE, "text/html")
            elif u.path == "/api/ping":
                # Deliberately does no work. The self-edit gate uses this to
                # tell "the server is alive" from "a slow status probe",
                # which are very different things.
                self._send(200, {"ok": True})
            elif u.path == "/api/env":
                self._send(200, probe_env())
            elif u.path == "/api/env/status":
                self._send(200, probe_status())
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

    # -- POST: history clear + self-edit (bodies are large) -----------------
    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._read_json_body()

            if u.path == "/api/history/clear":
                clear_history()
                return self._send(200, {"ok": True})

            if u.path not in ("/api/propose", "/api/apply", "/api/restore"):
                return self._send(404, {"error": "not found"})

            if self_edit is None:
                return self._send(503, {"error": "self_edit.py not available"})

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

        except BrokenPipeError:
            pass
        except json.JSONDecodeError:
            try:
                self._send(400, {"error": "invalid json"})
            except Exception:
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
                            "total": a["total"], "seconds": a.get("seconds", 0),
                            **({"elapsed_ms": a["elapsed_ms"]}
                               if "elapsed_ms" in a else {}),
                            **({"tokens": a["tokens"]} if "tokens" in a else {}),
                            **({"cases": a["cases"]} if "cases" in a else {})}
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
# Page: CSS / HTML / JS stay inside this file so self-edit can still rewrite it.
# ----------------------------------------------------------------------------

CSS = r"""
  :root {
    --bg:#0c0c0b; --panel:#141412; --panel2:#1c1c18; --hair:#2c2b26;
    --fg:#f3efe4; --muted:#8d897c; --ink:#0c0c0b;
    --brass:#c9a05a; --action:#f3efe4; --action-hover:#fff;
    --ok:#7dba8a; --bad:#e06b66; --warn:#c9a05a;
    --hair-hover:#3f3e37; --escalate-bg:#16140f;
    --code-fg:#d8d4c8; --err-fg:#ffb8b5; --err-bg:#161010;
    --ok-wash:#102016; --bad-wash:#1a1111; --on-ok:#06210f;
    --grain-opacity:.055; --toggle-track:#000; --toggle-track-fg:#fff; --toggle-ring:#5c594e;
    --serif:"Instrument Serif", "Times New Roman", serif;
    --sans:"Instrument Sans", ui-sans-serif, system-ui, sans-serif;
    --mono:"IBM Plex Mono", ui-monospace, "Cascadia Code", Consolas, monospace;
    --ease:cubic-bezier(.22, 1, .36, 1);
  }
  html[data-theme="light"] {
    --bg:#f3efe4; --panel:#ebe6d8; --panel2:#e4dece; --hair:#cfc8b6;
    --fg:#1a1916; --muted:#5e5a50; --ink:#f3efe4;
    --brass:#a67c2e; --action:#1a1916; --action-hover:#2c2a26;
    --ok:#3f8f55; --bad:#c94a45; --warn:#a67c2e;
    --hair-hover:#b8b09a; --escalate-bg:#f0e6d0;
    --code-fg:#2a2824; --err-fg:#9a3530; --err-bg:#f6e8e6;
    --ok-wash:#e3efe6; --bad-wash:#f6e6e4; --on-ok:#06210f;
    --grain-opacity:.028; --toggle-track:#e0ddd4; --toggle-track-fg:#111; --toggle-ring:transparent;
  }
  * { box-sizing:border-box }
  html, body { height:100% }
  html { background:var(--bg); color-scheme:dark }
  html[data-theme="light"] { color-scheme:light }
  body {
    margin:0; background:var(--bg); color:var(--fg);
    font:14px/1.55 var(--sans); letter-spacing:-.005em;
  }
  .grain {
    position:fixed; inset:0; pointer-events:none; z-index:80;
    opacity:var(--grain-opacity); mix-blend-mode:overlay;
    background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='4' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 .7 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
  }
  html[data-theme="light"] .grain { mix-blend-mode:multiply }
  button, select, input, textarea {
    font:inherit; color:inherit; appearance:none; -webkit-appearance:none; border-radius:2px;
  }
  input[type=number]::-webkit-inner-spin-button,
  input[type=number]::-webkit-outer-spin-button { appearance:none; margin:0 }
  header {
    height:64px; padding:0 36px; flex-shrink:0;
    display:flex; align-items:center; gap:18px;
    border-bottom:1px solid transparent;
    background:linear-gradient(to bottom, var(--bg) 70%, transparent);
  }
  .brand { display:flex; align-items:baseline; gap:16px; min-width:0 }
  h1 {
    font-family:var(--serif); font-size:26px; font-weight:400;
    margin:0; letter-spacing:-.03em; line-height:1;
  }
  header .sub {
    color:var(--muted); font-size:10px; letter-spacing:.16em;
    text-transform:uppercase; font-weight:500; white-space:nowrap;
  }
  header .clock {
    color:var(--muted); font-size:12px;
    font-family:var(--mono); letter-spacing:.08em; white-space:nowrap;
  }
  .header-end {
    margin-left:auto; display:flex; align-items:center; gap:14px; flex-shrink:0;
  }
  .theme-toggle {
    display:inline-flex; align-items:center; width:auto; height:32px;
    padding:3px 16px 3px 3px; margin:0; border:none; border-radius:99px;
    background:var(--toggle-track); color:var(--toggle-track-fg);
    box-shadow:inset 0 0 0 1px var(--toggle-ring);
    cursor:pointer; flex-shrink:0; overflow:hidden; font-weight:600;
    letter-spacing:0; line-height:1;
  }
  .theme-thumb {
    width:26px; height:26px; border-radius:50%; background:#fff; color:#111;
    display:grid; place-items:center; flex-shrink:0;
  }
  .theme-thumb svg { display:block; width:14px; height:14px }
  .theme-toggle .icon-sun { display:none }
  .theme-toggle .icon-moon { display:block }
  html[data-theme="light"] .theme-toggle .icon-sun { display:block }
  html[data-theme="light"] .theme-toggle .icon-moon { display:none }
  .theme-caption {
    max-width:0; opacity:0; padding:0; overflow:hidden;
    font-size:8px; font-weight:650; letter-spacing:.12em;
    text-transform:uppercase; line-height:1.15; text-align:left;
    transition:max-width .32s var(--ease), opacity .22s ease, padding .32s var(--ease);
  }
  .theme-caption .cap-light { display:none }
  html[data-theme="light"] .theme-caption .cap-dark { display:none }
  html[data-theme="light"] .theme-caption .cap-light { display:block }
  .theme-toggle:hover,
  .theme-toggle:focus-visible { padding-right:3px }
  .theme-toggle:hover .theme-caption,
  .theme-toggle:focus-visible .theme-caption {
    max-width:4.6em; opacity:1; padding:0 10px 0 8px;
  }
  .wrap {
    display:grid; grid-template-columns:340px 1fr; gap:0;
    height:calc(100vh - 64px);
    transition:grid-template-columns .34s var(--ease);
  }
  .wrap.collapsed { grid-template-columns:56px 1fr }
  aside {
    border-right:1px solid var(--hair); background:var(--panel);
    overflow:hidden; min-width:0;
  }
  .inspector-full {
    height:100%; overflow-y:auto; padding:22px 24px 18px;
    display:flex; flex-direction:column;
  }
  .wrap.collapsed .inspector-full { display:none }
  .inspector-mini {
    display:none; height:100%; flex-direction:column;
    align-items:center; padding:16px 8px; gap:22px;
  }
  .wrap.collapsed .inspector-mini { display:flex }
  .rail-tools { display:flex; justify-content:flex-end; gap:8px; margin:0 0 28px }
  .iconbtn {
    width:32px; height:32px; padding:0; margin:0; cursor:pointer;
    background:transparent; color:var(--muted);
    border:1px solid var(--hair); border-radius:99px;
    font-size:16px; line-height:1;
    transition:color .28s ease, border-color .28s ease;
  }
  .iconbtn:hover { color:var(--fg); border-color:var(--fg) }
  #inspector-pin {
    width:auto; padding:0 12px; font-size:10px; letter-spacing:.14em;
    text-transform:uppercase; font-weight:500; border-radius:99px;
    background:transparent; color:var(--muted); border:1px solid var(--hair);
    transition:color .28s ease, border-color .28s ease;
  }
  #inspector-pin:hover { color:var(--fg) }
  #inspector-pin.on { color:var(--brass); border-color:var(--brass) }
  .mini-item {
    writing-mode:vertical-rl; transform:rotate(180deg);
    font-size:10px; letter-spacing:.16em; text-transform:uppercase;
    color:var(--muted); max-height:28%; overflow:hidden; text-overflow:ellipsis;
    white-space:nowrap; font-weight:500;
  }
  .mini-item.accent { color:var(--brass) }
  .isec { margin-bottom:22px; padding-bottom:0; border-bottom:none }
  .isec h2 {
    font-family:var(--serif); font-size:20px; font-weight:400; font-style:italic;
    letter-spacing:-.02em; text-transform:none; color:var(--fg);
    margin:0 0 12px; line-height:1;
  }
  .isec h2::after { content:" /"; font-style:normal; color:var(--muted) }
  label {
    display:block; font-size:10px; color:var(--muted); margin:14px 0 7px;
    text-transform:uppercase; letter-spacing:.14em; font-weight:500;
  }
  .isec > label:first-of-type { margin-top:0 }
  select, input[type=text], input[type=number], textarea, button {
    width:100%; background:var(--panel2); color:var(--fg);
    border:1px solid var(--hair); padding:10px 12px; font-size:13px;
    transition:border-color .28s ease, background .28s ease, color .28s ease;
  }
  select {
    background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'><path d='M1.2 1.2L6 6l4.8-4.8' stroke='%238d897c' stroke-width='1.2' fill='none'/></svg>");
    background-repeat:no-repeat; background-position:right 12px center;
    background-color:var(--panel2); padding-right:32px; cursor:pointer;
  }
  html[data-theme="light"] select {
    background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'><path d='M1.2 1.2L6 6l4.8-4.8' stroke='%235e5a50' stroke-width='1.2' fill='none'/></svg>");
  }
  select:hover, input[type=text]:hover, input[type=number]:hover, textarea:hover {
    border-color:var(--hair-hover);
  }
  select:focus, input:focus, textarea:focus, button:focus-visible {
    outline:1px solid var(--brass); outline-offset:2px; border-color:var(--brass);
  }
  textarea { font-family:var(--mono); font-size:12px; min-height:120px; resize:vertical; line-height:1.5 }
  input[type=range] {
    -webkit-appearance:none; appearance:none; width:100%; height:22px;
    background:transparent; border:none; padding:8px 0; margin:0; outline:none;
  }
  input[type=range]:focus { outline:none; border-color:transparent }
  input[type=range]:focus-visible { outline:1px solid var(--brass); outline-offset:4px }
  input[type=range]::-webkit-slider-runnable-track {
    height:2px; background:var(--hair); border-radius:99px }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance:none; width:12px; height:12px; margin-top:-5px;
    border-radius:50%; background:var(--brass); border:none; cursor:pointer }
  input[type=range]::-moz-range-track {
    height:2px; background:var(--hair); border-radius:99px; border:none }
  input[type=range]::-moz-range-thumb {
    width:12px; height:12px; border-radius:50%; background:var(--brass);
    border:none; cursor:pointer }
  .inspector-actions {
    margin-top:auto; padding-top:18px; padding-bottom:2px;
    position:sticky; bottom:0; z-index:2;
    background:linear-gradient(to bottom, transparent, var(--panel) 14px);
  }
  button { cursor:pointer; font-weight:500; letter-spacing:.02em }
  #run, #run-mock, #propose {
    background:var(--action); color:var(--ink); border-color:transparent;
    border-radius:99px; font-weight:500;
  }
  #run:hover:not(:disabled), #run-mock:hover, #propose:hover:not(:disabled) {
    background:var(--action-hover); }
  #run:disabled { opacity:.4; cursor:default }
  #stop, #stop-head {
    background:transparent; border-color:var(--bad); color:var(--bad);
    margin-top:10px; border-radius:99px;
  }
  #stop-head { width:auto; margin:0; padding:9px 18px; flex-shrink:0 }
  .btn-quiet {
    width:auto; margin:0; padding:7px 14px; font-size:11px; font-weight:500;
    background:transparent; color:var(--muted); border-color:var(--hair);
    border-radius:99px; letter-spacing:.08em; text-transform:uppercase;
  }
  .btn-quiet:hover { color:var(--fg); border-color:var(--fg) }
  .btn-quiet.danger { color:var(--bad); border-color:var(--bad) }
  .pill {
    display:inline-block; padding:3px 9px; border-radius:99px;
    font-size:10px; letter-spacing:.04em;
    border:1px solid var(--hair); color:var(--muted); font-weight:500;
  }
  .pill.on { color:var(--ok); border-color:rgba(125,186,138,.35) }
  .pill.off { color:var(--muted) }
  .status-row { display:flex; gap:6px; flex-wrap:wrap; margin-top:12px }
  main { padding:28px 44px 56px; overflow-y:auto; position:relative }
  #live, #curve, #edit { animation:fadein .32s var(--ease) }
  .card {
    background:var(--panel); border:1px solid var(--hair);
    border-radius:2px; padding:18px 20px; margin-bottom:14px; position:relative;
  }
  .card.pass { border-color:rgba(125,186,138,.4) }
  .card.fail { border-color:rgba(224,107,102,.35) }
  .card.escalate { border-color:var(--brass); background:var(--escalate-bg) }
  .card.escalate.fresh { border-color:var(--brass) }
  .card h3 {
    margin:0 0 12px; font-size:15px; font-weight:500;
    display:flex; justify-content:space-between; align-items:center; gap:10px;
    letter-spacing:-.015em;
  }
  .esc-tag {
    font-size:10px; letter-spacing:.14em; text-transform:uppercase;
    color:var(--brass); font-weight:500; border:1px solid var(--brass);
    border-radius:99px; padding:3px 9px; flex-shrink:0;
  }
  .muted { color:var(--muted); font-weight:400; font-size:12px }
  .bar { display:flex; gap:3px; margin:10px 0 12px }
  .seg { height:3px; flex:1; border-radius:99px; background:var(--hair) }
  .seg.p { background:var(--ok) }
  pre {
    background:var(--bg); border:1px solid var(--hair); border-radius:2px;
    padding:14px 16px; overflow-x:auto; font-family:var(--mono);
    font-size:12px; margin:0; white-space:pre-wrap; word-break:break-word;
    line-height:1.55; color:var(--code-fg);
  }
  pre.err { color:var(--err-fg); background:var(--err-bg); border-color:rgba(224,107,102,.35) }
  details summary {
    cursor:pointer; color:var(--muted); font-size:12px;
    margin-bottom:8px; user-select:none; letter-spacing:.02em;
  }
  .code-fold { margin-top:12px }
  .timeline { position:relative; padding-left:28px }
  .timeline::before {
    content:""; position:absolute; left:7px; top:10px; bottom:10px;
    width:1px; background:var(--hair);
  }
  .timeline .card {
    animation:rise .36s var(--ease) both;
  }
  .timeline .card::before {
    content:""; position:absolute; left:-26px; top:22px;
    width:9px; height:9px; border-radius:50%;
    background:var(--hair); border:2px solid var(--bg);
  }
  .timeline .card.pass::before { background:var(--ok) }
  .timeline .card.fail::before { background:var(--bad) }
  .timeline .card.escalate::before { background:var(--brass) }
  .banner {
    padding:14px 18px; border-radius:2px; margin-bottom:16px; font-weight:500;
    animation:rise .32s var(--ease) both;
  }
  .banner.ok { background:var(--ok-wash); color:var(--ok); border:1px solid rgba(125,186,138,.35) }
  .banner.bad { background:var(--bad-wash); color:var(--bad); border:1px solid rgba(224,107,102,.35) }
  html[data-theme="light"] .banner.ok { border-color:rgba(63,143,85,.4) }
  html[data-theme="light"] .banner.bad { border-color:rgba(201,74,69,.4) }
  html[data-theme="light"] .pill.on { border-color:rgba(63,143,85,.45) }
  html[data-theme="light"] .card.pass { border-color:rgba(63,143,85,.5) }
  html[data-theme="light"] .card.fail { border-color:rgba(201,74,69,.45) }
  html[data-theme="light"] pre.err { border-color:rgba(201,74,69,.4) }
  .banner.info { background:var(--panel); color:var(--fg);
                 border:1px solid var(--hair); font-weight:400 }
  .empty {
    text-align:left; padding:min(14vh, 120px) 8px 48px;
    max-width:640px; margin:0;
  }
  .empty .kicker {
    font-size:11px; letter-spacing:.18em; text-transform:uppercase;
    color:var(--brass); margin-bottom:22px; font-weight:500;
  }
  .empty h2 {
    font-family:var(--serif); font-size:clamp(40px, 6vw, 72px); font-weight:400;
    margin:0 0 18px; letter-spacing:-.035em; line-height:.95;
  }
  .empty h2 em { font-style:italic; color:var(--brass) }
  .empty p {
    color:var(--muted); margin:0 0 32px; font-size:16px; line-height:1.5;
    max-width:26em;
  }
  #run-mock { width:auto; display:inline-block; padding:12px 22px; margin:0 }
  .card.waiting { display:flex; align-items:flex-start; gap:14px; margin-bottom:14px }
  .pulse {
    width:8px; height:8px; border-radius:50%; background:var(--brass);
    margin-top:7px; flex-shrink:0; animation:pulse 1.2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:.28 } }
  @keyframes rise {
    from { opacity:0; transform:translateY(12px) }
    to { opacity:1; transform:none }
  }
  @keyframes fadein {
    from { opacity:0 } to { opacity:1 }
  }
  .waiting h3 { margin:0 0 4px; font-family:var(--serif); font-size:22px; font-weight:400 }
  #chart { width:100%; height:230px }
  #heatmap { margin-top:18px }
  .heatmap { margin-top:12px }
  .heatmap:first-child { margin-top:0 }
  .heatmap-label {
    font-size:10px; letter-spacing:.14em; text-transform:uppercase;
    color:var(--muted); font-weight:500; margin-bottom:6px;
  }
  .heat-row { display:flex; align-items:center; gap:8px }
  .heat-n {
    font-family:var(--mono); font-size:10px; color:var(--muted);
    width:1.4em; text-align:right; flex-shrink:0; letter-spacing:.04em;
  }
  .heatmap .bar { flex:1; margin:2px 0 }
  .hist {
    font-size:13px; color:var(--muted); display:flex;
    justify-content:space-between; gap:16px; padding:12px 0;
    border-bottom:1px solid var(--hair);
  }
  .hist span:first-child { color:var(--fg) }
  .card-head { display:flex; justify-content:space-between; align-items:center;
               margin-bottom:12px }
  .card-head h3 { margin:0; font-family:var(--serif); font-size:22px; font-weight:400; font-style:italic }
  .confirm-row { display:flex; flex-wrap:wrap; gap:8px; align-items:center;
                 padding:10px 0 14px; font-size:13px; color:var(--fg) }
  .tabs {
    display:flex; gap:28px; margin-bottom:28px;
    border-bottom:1px solid var(--hair);
  }
  .tab {
    padding:6px 0 14px; cursor:pointer; font-size:13px; color:var(--muted);
    border-bottom:1px solid transparent; margin-bottom:-1px;
    letter-spacing:.04em; transition:color .28s ease, border-color .28s ease;
  }
  .tab:hover { color:var(--fg) }
  .tab.active { color:var(--fg); border-bottom-color:var(--brass) }
  .hide { display:none !important }
  .warn-box {
    background:transparent; border:1px solid var(--hair); color:var(--muted);
    border-radius:2px; padding:16px 18px; font-size:13px; line-height:1.55;
    margin-top:12px;
  }
  .runhead {
    position:sticky; top:-28px; z-index:4; display:flex; align-items:center;
    gap:20px; padding:16px 0 18px; margin:0 0 20px; background:var(--bg);
    border-bottom:1px solid var(--hair);
  }
  .runhead .meta { flex:1; min-width:0; font-size:13px; color:var(--muted) }
  .runhead .meta b { color:var(--fg); font-weight:500; font-size:16px; letter-spacing:-.02em }
  .runhead-score { text-align:right; line-height:1 }
  .runhead-score .num {
    font-family:var(--serif); font-size:52px; color:var(--brass); font-weight:400;
    letter-spacing:-.04em; font-variant-numeric:tabular-nums; line-height:.85;
  }
  .runhead-score .denom { font-size:18px; color:var(--muted); font-weight:400; font-family:var(--serif) }
  .runhead-score .lbl {
    display:block; font-size:10px; letter-spacing:.16em;
    text-transform:uppercase; color:var(--muted); margin-top:8px; font-weight:500;
  }
  ::-webkit-scrollbar { width:10px; height:10px }
  ::-webkit-scrollbar-thumb { background:var(--hair); border-radius:99px }
  ::-webkit-scrollbar-track { background:transparent }
"""

HTML = r"""
<body>
<div class="grain" aria-hidden="true"></div>
<header>
  <div class="brand">
    <h1>Self-improving agent</h1>
    <span class="sub">write &rarr; test &rarr; rewrite</span>
  </div>
  <div class="header-end">
    <span class="clock" id="clock"></span>
    <button type="button" class="theme-toggle" id="theme-toggle"
            aria-label="Dark mode. Switch to light.">
      <span class="theme-thumb">
        <svg class="icon-moon" width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
          <path fill="currentColor" d="M9.15 2.35a5.55 5.55 0 1 0 4.2 9.15 6.2 6.2 0 1 1-4.2-9.15z"/>
          <path fill="currentColor" d="M12.15 2.9l.32.98h1.02l-.82.6.32.98-.84-.6-.82.6.32-.98-.82-.6h1.02z"/>
          <path fill="currentColor" d="M14.55 5.85l.2.62h.66l-.53.38.2.62-.53-.38-.53.38.2-.62-.53-.38h.66z"/>
        </svg>
        <svg class="icon-sun" width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
          <circle cx="8" cy="8" r="2.35" fill="currentColor"/>
          <g stroke="currentColor" stroke-width="1.2" stroke-linecap="round" fill="none">
            <path d="M8 1.6v1.7M8 12.7v1.7M1.6 8h1.7M12.7 8h1.7M3.35 3.35l1.2 1.2M11.45 11.45l1.2 1.2M12.65 3.35l-1.2 1.2M4.55 11.45l-1.2 1.2"/>
          </g>
        </svg>
      </span>
      <span class="theme-caption" aria-hidden="true">
        <span class="cap-dark">Dark<br>mode</span>
        <span class="cap-light">Light<br>mode</span>
      </span>
    </button>
  </div>
</header>

<div class="wrap" id="wrap">
<aside>
  <div class="inspector-full">
    <div class="rail-tools">
      <button type="button" id="inspector-pin" title="Keep inspector open during a run">Pin</button>
      <button type="button" id="inspector-collapse" class="iconbtn" title="Collapse inspector">‹</button>
    </div>

    <section class="isec">
      <h2>Model</h2>
      <label>Backend</label>
      <select id="backend">
        <option value="mock">mock — no model, canned answers</option>
        <option value="mock-stuck">mock-stuck — always 12/15, to test plateau reseeding</option>
        <option value="ollama">ollama — local, free</option>
        <option value="lmstudio">lmstudio — local, LM Studio</option>
        <option value="anthropic">anthropic — paid API</option>
        <option value="groq">groq — free tier, hosted</option>
      </select>
      <div class="status-row" id="envpills"></div>
      <label>Model</label>
      <select id="modelsel" class="hide"></select>
      <input id="model" type="text" placeholder="default">
    </section>

    <section class="isec">
      <h2>Task</h2>
      <label>Task</label>
      <select id="task">
<!--TASK_OPTIONS-->
      </select>
      <div id="customwrap" class="hide">
        <label>Task JSON</label>
        <textarea id="taskjson"></textarea>
      </div>
    </section>

    <section class="isec">
      <h2>Loop</h2>
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
    </section>

    <section class="isec">
      <h2>Safety</h2>
      <label>Where code runs</label>
      <select id="runner">
        <option value="subprocess">subprocess (fast, not a sandbox)</option>
        <option value="docker">docker (sandboxed, no network)</option>
      </select>
      <div class="warn-box" id="sandboxwarn">
        subprocess isolates crashes and infinite loops, but model-written code can
        still touch your files. Switch to docker for anything open-ended.
      </div>
    </section>

    <div class="inspector-actions">
      <button type="button" id="run">Run</button>
      <button type="button" id="stop" class="hide">Stop</button>
    </div>
  </div>

  <div class="inspector-mini" id="inspector-mini">
    <button type="button" id="inspector-expand" class="iconbtn" title="Show settings">›</button>
    <span class="mini-item accent" id="mini-backend">mock</span>
    <span class="mini-item" id="mini-model">—</span>
    <span class="mini-item" id="mini-task">roman</span>
  </div>
</aside>

<main>
  <div class="tabs">
    <div class="tab active" data-tab="live">Live run</div>
    <div class="tab" data-tab="curve">Improvement curve</div>
    <div class="tab" data-tab="edit">Self-edit</div>
  </div>

  <div id="live">
    <div id="runhead" class="runhead hide">
      <div class="meta">
        <div><b id="rh-task">—</b> · <span id="rh-model">—</span></div>
        <div id="rh-attempt">up to 10 attempts</div>
      </div>
      <div class="runhead-score">
        <span class="num" id="rh-score">—</span><span class="denom" id="rh-denom"></span>
        <span class="lbl">best score</span>
      </div>
      <button type="button" id="stop-head" class="hide">Stop</button>
    </div>
    <div id="banner"></div>
    <div id="waiting" class="card waiting hide">
      <div class="pulse"></div>
      <div>
        <h3 id="waiting-msg">Calling the model…</h3>
        <div class="muted" id="waiting-sub">Write a function, run the tests, feed the error back, rewrite.</div>
      </div>
    </div>
    <div class="empty" id="idle">
      <div class="kicker">No run yet</div>
      <h2>Watch the loop <em>climb</em></h2>
      <p>Write a function, run the tests, feed the error back, rewrite.</p>
      <button type="button" id="run-mock">Run mock — no API key</button>
    </div>
    <div class="timeline" id="attempts-out"></div>
  </div>

  <div id="curve" class="hide">
    <div class="card"><svg id="chart" viewBox="0 0 800 230"
         preserveAspectRatio="none"></svg>
      <div id="heatmap"></div></div>
    <div class="card">
      <div class="card-head">
        <h3>Runs</h3>
        <button type="button" id="clearhist" class="btn-quiet">Clear history</button>
      </div>
      <div id="clearconfirm" class="hide confirm-row">
        <span>Delete all logged attempts? This cannot be undone.</span>
        <button type="button" id="clearhist-yes" class="btn-quiet danger">Clear</button>
        <button type="button" id="clearhist-no" class="btn-quiet">Cancel</button>
      </div>
      <div id="histlist"></div>
    </div>
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
            placeholder="show elapsed time on each attempt card"></textarea>
        </div>
        <div style="width:170px">
          <label style="margin-top:0">Target file</label>
          <select id="editfile"><option value="">let it choose</option></select>
        </div>
      </div>
      <div style="display:flex; gap:10px; margin-top:12px">
        <button type="button" id="propose" style="flex:1">Propose a change</button>
        <button type="button" id="restorebtn" class="btn-quiet" style="width:170px">Restore backup…</button>
      </div>
      <select id="backupsel" class="hide" style="margin-top:10px"></select>
    </div>

    <div id="editout"></div>
  </div>
</main>
</div>
"""

JS = r"""
const $ = id => document.getElementById(id);
let es = null, runId = null, env = {
  ollama:{up:false,models:[]}, lmstudio:{up:false,models:[]},
  docker:{up:false,image:false}, anthropic:{key:false}, groq:{key:false},
  selfedit:null, tasks:{}
};
let runState = { attempt:0, max:10, best:0, total:0, model:"", task:"", running:false };

const BACKENDS = [
  ["mock",       "mock — no model, canned answers"],
  ["mock-stuck", "mock-stuck — always 12/15, to test plateau reseeding"],
  ["ollama",     "ollama — local, free"],
  ["lmstudio",   "lmstudio — local, LM Studio"],
  ["anthropic",  "anthropic — paid API"],
  ["groq",       "groq — free tier, hosted"],
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

const THEME_KEY = "sia-theme";
let lastChartRuns = [];

function currentTheme() {
  return document.documentElement.getAttribute("data-theme") === "light"
    ? "light" : "dark";
}

function syncThemeToggle() {
  const btn = $("theme-toggle");
  const light = currentTheme() === "light";
  btn.setAttribute("aria-label", light
    ? "Light mode. Switch to dark."
    : "Dark mode. Switch to light.");
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try { localStorage.setItem(THEME_KEY, theme); } catch (e) {}
  syncThemeToggle();
  if ($("chart")) drawChart(lastChartRuns);
}

$("theme-toggle").onclick = () =>
  setTheme(currentTheme() === "light" ? "dark" : "light");
syncThemeToggle();

function restoreSelect(el, prev) {
  if (prev && [...el.options].some(o => o.value === prev)) el.value = prev;
}

async function loadEnv() {
  const prevBackend = $("backend").value;
  const prevTask = $("task").value;
  const prevModel = $("model").value;
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 8000);
    const r = await fetch("/api/env", { signal: ctl.signal });
    clearTimeout(t);
    if (!r.ok) throw new Error("server returned " + r.status);
    env = Object.assign(env, await r.json());
  } catch (e) {
    // Static dropdowns stay. Do not wipe a selection the user already made.
    return null;
  }
  if (env.tasks && Object.keys(env.tasks).length) {
    $("task").innerHTML =
      Object.entries(env.tasks).map(([k, t]) =>
        `<option value="${esc(k)}">${esc(k)} — ${esc(t.func)}() · ${t.cases} tests</option>`
      ).join("") + `<option value="__custom">custom…</option>`;
    restoreSelect($("task"), prevTask);
  }
  $("backend").innerHTML = BACKENDS
    .map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  restoreSelect($("backend"), prevBackend);
  if (prevModel) $("model").value = prevModel;
  renderPills();
  syncBackend();
  updateMini();
  loadEnvStatus();
  return env;
}

async function loadEnvStatus() {
  const prevModel = currentModel();
  try {
    const r = await fetch("/api/env/status");
    if (!r.ok) return;
    const s = await r.json();
    env.ollama = s.ollama;
    env.lmstudio = s.lmstudio;
    env.docker = s.docker;
    renderPills();
    syncBackend();
    if (prevModel) {
      if (!$("modelsel").classList.contains("hide")) {
        restoreSelect($("modelsel"), prevModel);
      } else {
        $("model").value = prevModel;
      }
    }
    updateMini();
  } catch (e) { /* pills stay unknown */ }
}

function renderPills() {
  const pill = (on, txt) =>
    `<span class="pill ${on ? "on" : "off"}">${on ? "●" : "○"} ${txt}</span>`;
  $("envpills").innerHTML =
    pill(env.ollama && env.ollama.up, "ollama") +
    pill(env.lmstudio && env.lmstudio.up, "lmstudio") +
    pill(env.docker && env.docker.up, "docker") +
    pill(env.anthropic && env.anthropic.key, "anthropic key") +
    pill(env.groq && env.groq.key, "groq key");
}

function modelsFor(b) {
  if (b === "ollama") return (env.ollama && env.ollama.models) || [];
  if (b === "lmstudio") return (env.lmstudio && env.lmstudio.models) || [];
  return [];
}

function syncBackend() {
  const b = $("backend").value;
  const list = modelsFor(b);
  const useList = list.length > 0;
  if (useList) {
    $("modelsel").innerHTML = list
      .map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join("");
  }
  $("modelsel").classList.toggle("hide", !useList);
  $("model").classList.toggle("hide", useList);
  $("model").placeholder =
    b === "anthropic" ? "claude-sonnet-5"
    : b === "groq" ? "llama-3.3-70b-versatile"
    : b === "ollama" ? "qwen2.5-coder:7b"
    : b === "lmstudio" ? "(load a model in LM Studio)" : "n/a";
  updateMini();
}

function currentModel() {
  const b = $("backend").value;
  return modelsFor(b).length ? $("modelsel").value : $("model").value.trim();
}

function updateMini() {
  $("mini-backend").textContent = $("backend").value || "—";
  const m = currentModel();
  $("mini-model").textContent = m ? shortModel(m) : "—";
  const t = $("task").value;
  $("mini-task").textContent = t === "__custom" ? "custom" : (t || "—");
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
  updateMini();
};
$("model").oninput = updateMini;
$("modelsel").onchange = updateMini;
$("runner").onchange = e => {
  const d = e.target.value === "docker";
  $("sandboxwarn").textContent = d
    ? "Code runs in python:3.11-slim with no network, 256MB, read-only disk, as nobody. First run pulls the image."
    : "subprocess isolates crashes and infinite loops, but model-written code can still touch your files. Switch to docker for anything open-ended.";
};

$("inspector-collapse").onclick = () => $("wrap").classList.add("collapsed");
$("inspector-expand").onclick = () => $("wrap").classList.remove("collapsed");
$("inspector-pin").onclick = () => {
  const wrap = $("wrap");
  const pinned = wrap.classList.toggle("pinned");
  $("inspector-pin").classList.toggle("on", pinned);
  $("inspector-pin").textContent = pinned ? "Pinned" : "Pin";
  $("inspector-pin").title = pinned
    ? "Unpin — collapse when a run starts"
    : "Keep inspector open during a run";
  if (pinned) wrap.classList.remove("collapsed");
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

function telemetryLine(ev) {
  const parts = [];
  const ms = ev.elapsed_ms;
  if (ms != null && ms !== "") {
    parts.push(ms >= 1000 ? (ms / 1000).toFixed(1) + "s" : ms + "ms");
  } else if (ev.seconds != null) {
    parts.push(ev.seconds + "s");
  }
  if (ev.tokens != null && ev.tokens !== "") parts.push(ev.tokens + " tok");
  return parts.join(" · ");
}

function bar(passed, total, cases) {
  let s = '<div class="bar">';
  for (let i = 0; i < total; i++) {
    const c = Array.isArray(cases) ? cases[i] : null;
    const ok = c ? !!c.ok : i < passed;
    let title = "";
    if (c) {
      title = (c.ok ? "pass" : "fail") + " · " + (c.input ?? i);
      if (!c.ok && c.error) title += " · " + c.error;
    }
    s += `<div class="seg ${ok ? "p" : ""}"${title ? ` title="${esc(title)}"` : ""}></div>`;
  }
  return s + "</div>";
}

function renderRunhead() {
  $("rh-task").textContent = runState.task || "—";
  $("rh-model").textContent = shortModel(runState.model || "") || "—";
  if (runState.running) {
    $("rh-attempt").textContent = runState.attempt
      ? `attempt ${runState.attempt}/${runState.max}`
      : `up to ${runState.max} attempts`;
  } else if (runState.attempt) {
    $("rh-attempt").textContent = `${runState.attempt} attempts`;
  }
  $("rh-score").textContent = runState.total ? String(runState.best) : "—";
  $("rh-denom").textContent = runState.total ? "/" + runState.total : "";
}

function startRun(forceBackend) {
  if (forceBackend) {
    $("backend").value = forceBackend;
    syncBackend();
  }
  updateMini();
  if (!$("wrap").classList.contains("pinned"))
    $("wrap").classList.add("collapsed");

  $("idle").classList.add("hide");
  $("attempts-out").innerHTML = "";
  $("banner").innerHTML = "";
  $("runhead").classList.remove("hide");
  $("waiting").classList.remove("hide");
  $("waiting-msg").textContent = "Calling the model…";
  $("waiting-sub").textContent = "Write a function, run the tests, feed the error back, rewrite.";
  $("run").disabled = true;
  $("stop").classList.remove("hide");
  $("stop-head").classList.remove("hide");

  runState = {
    attempt: 0,
    max: parseInt($("attempts").value, 10),
    best: 0,
    total: 0,
    model: currentModel() || $("backend").value,
    task: $("task").value === "__custom" ? "custom" : $("task").value,
    running: true,
  };
  renderRunhead();

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
}

$("run").onclick = () => startRun();
$("run-mock").onclick = () => startRun("mock");

function requestStop() {
  if (runId) fetch("/api/cancel?run_id=" + runId);
  $("stop").textContent = "stopping…";
  $("stop-head").textContent = "stopping…";
}
$("stop").onclick = requestStop;
$("stop-head").onclick = requestStop;

function finish() {
  if (es) { es.close(); es = null; }
  $("run").disabled = false;
  $("stop").classList.add("hide");
  $("stop-head").classList.add("hide");
  $("stop").textContent = "Stop";
  $("stop-head").textContent = "Stop";
  $("waiting").classList.add("hide");
  runId = null;
  runState.running = false;
  renderRunhead();
}

function handle(ev) {
  const out = $("attempts-out");

  if (ev.type === "status") {
    $("waiting-msg").textContent = "Calling the model…";
    $("waiting-sub").textContent = ev.message;
  }
  else if (ev.type === "start") {
    runId = ev.run_id;
    runState.model = ev.model;
    runState.task = ev.task;
    runState.total = ev.total;
    runState.max = ev.max_attempts || runState.max;
    renderRunhead();
    updateMini();
    $("waiting-sub").textContent =
      `${ev.model} · ${ev.func_name}() · ${ev.total} tests · ${ev.runner}`;
  }
  else if (ev.type === "attempt") {
    $("waiting").classList.add("hide");
    $("banner").innerHTML = "";
    runState.attempt = ev.attempt;
    runState.total = ev.total;
    if (ev.passed > runState.best) runState.best = ev.passed;
    renderRunhead();
    const ok = ev.passed === ev.total;
    const card = document.createElement("div");
    card.className = "card " + (ok ? "pass" : "fail");
    const codeFold =
      `<details class="code-fold"><summary>Show code</summary>
       <pre>${esc(ev.code)}</pre></details>`;
    card.innerHTML =
      `<h3>Attempt ${ev.attempt}
         <span class="muted">${ev.passed}/${ev.total} passed ·
         ${telemetryLine(ev)}</span></h3>` +
      bar(ev.passed, ev.total, ev.cases) +
      (ev.error && !ok
        ? `<pre class="err">${esc(ev.error)}</pre>` + codeFold
        : codeFold);
    out.appendChild(card);
    card.scrollIntoView({ behavior: "smooth", block: "end" });
  }
  else if (ev.type === "escalate") {
    $("waiting").classList.add("hide");
    const d = document.createElement("div");
    const kind = ev.has_anchor ? "reseed" : "fresh";
    d.className = "card escalate " + kind;
    d.innerHTML = ev.has_anchor
      ? `<h3>Stuck — reseeding from best
         <span class="esc-tag">RESEED</span></h3>
       <div class="muted">level ${ev.level} · temperature ${ev.temperature}</div>
       <div class="muted" style="margin-top:6px">${esc(ev.reason)}. Restarting from a clean context,
         still using the best code so far` +
      (ev.anchor_passed != null ? ` (${ev.anchor_passed} passed)` : "") +
      `.</div>`
      : `<h3>Stuck at 0 — discarding the broken code and starting fresh.
         <span class="esc-tag">FRESH</span></h3>
       <div class="muted">level ${ev.level} · temperature ${ev.temperature}</div>
       <div class="muted" style="margin-top:6px">${esc(ev.reason)}.</div>`;
    out.appendChild(d);
    d.scrollIntoView({ behavior: "smooth", block: "end" });
  }
  else if (ev.type === "done") {
    $("banner").innerHTML = ev.success
      ? `<div class="banner ok">Solved on attempt ${ev.attempts_used} —
         saved as ${esc(ev.solution)}</div>`
      : `<div class="banner bad">Gave up after ${ev.attempts_used} attempts
         (best ${ev.best}/${ev.total})</div>`;
    if (!ev.success && ev.best != null) {
      runState.best = ev.best;
      runState.total = ev.total;
    }
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
  const slice = runs.slice(0, 8);
  drawChart(slice);
  drawHeatmap(slice);
}

function chartToken(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function drawChart(runs) {
  lastChartRuns = runs || [];
  const el = $("chart");
  if (!el) return;
  const W = el.clientWidth || 800, H = 260;
  const PL = 40, PB = 26, PT = 12, PR = 168;
  el.setAttribute("viewBox", `0 0 ${W} ${H}`);
  el.removeAttribute("preserveAspectRatio");

  const grid = chartToken("--hair", "#2c2b26");
  const label = chartToken("--muted", "#8d897c");
  const fg = chartToken("--fg", "#f3efe4");
  const brass = chartToken("--brass", "#c9a05a");
  const ok = chartToken("--ok", "#7dba8a");
  const bad = chartToken("--bad", "#e06b66");
  const light = currentTheme() === "light";
  const MODEL_COLORS = light
    ? [brass, "#5c574c", ok, bad, label, fg]
    : [brass, "#c8c3b4", ok, bad, label, fg];

  const maxX = Math.max(2, ...runs.flatMap(r => r.points.map(p => p.attempt)));
  const x = a => PL + (a - 1) / (maxX - 1) * (W - PL - PR);
  const y = pct => PT + (1 - pct / 100) * (H - PT - PB);

  const DASHES = ["", "7 5", "2 4", "10 4 2 4"];
  const models = [...new Set(runs.map(r => shortModel(r.model)))];
  const tasks = [...new Set(runs.map(r => r.task))];
  const colorOf = m => MODEL_COLORS[models.indexOf(shortModel(m)) % MODEL_COLORS.length];
  const dashOf = t => DASHES[tasks.indexOf(t) % DASHES.length];

  let s = "";
  for (const pct of [0, 25, 50, 75, 100]) {
    s += `<line x1="${PL}" y1="${y(pct)}" x2="${W - PR}" y2="${y(pct)}" stroke="${grid}" stroke-width="1"/>`
       + `<text x="${PL - 7}" y="${y(pct) + 3.5}" fill="${label}" font-size="10" text-anchor="end">${pct}%</text>`;
  }
  for (let a = 1; a <= maxX; a++) {
    s += `<text x="${x(a)}" y="${H - 8}" fill="${label}" font-size="10" text-anchor="middle">${a}</text>`;
  }

  runs.forEach(r => {
    const c = colorOf(r.model), dash = dashOf(r.task);
    const pts = r.points.map(p => `${x(p.attempt)},${y(100 * p.passed / p.total)}`);
    s += `<polyline points="${pts.join(" ")}" fill="none" stroke="${c}" stroke-width="2" stroke-dasharray="${dash}" stroke-linejoin="round"/>`;
    r.points.forEach(p => {
      const solved = p.passed === p.total;
      s += `<circle cx="${x(p.attempt)}" cy="${y(100 * p.passed / p.total)}" r="${solved ? 5 : 3}" fill="${c}"${solved ? ` stroke="${fg}" stroke-width="1.5"` : ''}><title>${esc(shortModel(r.model))} · ${esc(r.task)} · attempt ${p.attempt} · ${p.passed}/${p.total}</title></circle>`;
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
       + `<text x="${lx + 26}" y="${ly + 3.5}" fill="${fg}" font-size="10">${esc(shortModel(r.model))} · ${esc(r.task)}${r.solved_at ? " ✓" : ""}</text>`;
    ly += 18;
  });

  if (!runs.length) {
    s += `<text x="${(W - PR) / 2}" y="${H / 2}" fill="${label}" font-size="13" text-anchor="middle">No runs logged yet</text>`;
  }
  el.innerHTML = s;
}

function drawHeatmap(runs) {
  const el = $("heatmap");
  if (!el) return;
  if (!runs || !runs.length) { el.innerHTML = ""; return; }
  el.innerHTML = runs.map(r => {
    const label = `${esc(shortModel(r.model))} · ${esc(r.task)}`;
    const hasCases = r.points.some(p => Array.isArray(p.cases));
    if (!hasCases) {
      return `<div class="heatmap">
        <div class="heatmap-label">${label}</div>
        <div class="muted">no case log</div>
      </div>`;
    }
    const rows = r.points.map(p => {
      const cases = Array.isArray(p.cases) ? p.cases : null;
      const cells = cases
        ? bar(p.passed, p.total, cases)
        : grayBar(p.total, p.attempt);
      return `<div class="heat-row"><span class="heat-n">${p.attempt}</span>${cells}</div>`;
    }).join("");
    return `<div class="heatmap">
      <div class="heatmap-label">${label}</div>
      ${rows}
    </div>`;
  }).join("");
}

function grayBar(total, attempt) {
  let s = '<div class="bar">';
  for (let i = 0; i < total; i++) {
    s += `<div class="seg" title="attempt ${attempt} · case ${i} · no case log"></div>`;
  }
  return s + "</div>";
}

$("clearhist").onclick = () => $("clearconfirm").classList.remove("hide");
$("clearhist-no").onclick = () => $("clearconfirm").classList.add("hide");
$("clearhist-yes").onclick = async () => {
  try {
    const r = await fetch("/api/history/clear", { method: "POST" });
    const d = await r.json();
    $("clearconfirm").classList.add("hide");
    if (d.error) throw new Error(d.error);
    loadHistory();
  } catch (e) {
    $("clearconfirm").classList.add("hide");
    $("histlist").innerHTML = `<div class="banner bad">${esc(e.message)}</div>`;
  }
};

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
      `<option value="${esc(f.name)}">${esc(f.name)} · ${f.lines} lines</option>`).join("");
  $("backupsel").innerHTML = se.backups.length
    ? se.backups.map(b => `<option value="${esc(b)}">${esc(b)}</option>`).join("")
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
         <button type="button" id="approve" style="flex:1; background:var(--ok);
           color:var(--on-ok); border-color:transparent">Approve and verify</button>
         <button type="button" id="reject" style="flex:1; background:transparent;
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
      return `<span style="color:var(--muted)">${t}</span>`;
    if (l.startsWith("@@"))
      return `<span style="color:var(--action)">${t}</span>`;
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
        row.style.cssText = "padding:5px 0; font-size:13px; border-bottom:1px solid var(--hair)";
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

loadEnv().then(e => { if (e) initSelfEdit(); }).catch(() => {});
"""

PAGE = (
    "<!doctype html>\n<html lang=\"en\">\n<head>\n"
    "<meta charset=\"utf-8\">\n"
    "<script>(function(){try{var t=localStorage.getItem('sia-theme');"
    "if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t)}"
    "catch(e){}})();</script>\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
    "<title>Self-improving agent</title>\n"
    "<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">\n"
    "<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>\n"
    "<link href=\"https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500&family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Instrument+Serif:ital@0;1&display=swap\" rel=\"stylesheet\">\n"
    "<style>\n" + CSS + "\n</style>\n</head>\n"
    + HTML.replace("<!--TASK_OPTIONS-->", _task_options_html())
    + "\n<script>\n" + JS + "\n</script>\n</body>\n</html>\n"
)


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
