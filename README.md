# Self-improving agent

A loop with a scoreboard. The model writes a function, the function runs against
a fixed test suite, failures are fed back verbatim, the model rewrites. The test
suite is the whole reason "improvement" means anything here.

Two ways in: a browser UI, or the command line. Same loop underneath.

## Start here

From the project folder:

```powershell
cd path\to\self-improving-agent
py -3 web_ui.py
```

On Windows, use `py -3`. Plain `python` is often the Microsoft Store stub.

Your browser opens at `http://localhost:8000`. Leave backend on **mock** and
press **Run** — you'll watch it go 2/11 → 9/11 → 11/11. `mock` isn't a model,
it's three canned answers, there so you can confirm everything works before
blaming Ollama for anything.

Then switch the backend to **ollama** (or **lmstudio**) and run it again for real.

The UI has no dependencies — pure standard library, binds to 127.0.0.1 only, so
nothing on your network can reach it. Override the port with `SIA_PORT`; set
`SIA_NO_BROWSER=1` to skip opening a tab.

### What's in it

- **Backend** — mock, mock-stuck, ollama, lmstudio, anthropic, groq. Dots next
  to ollama / docker / API keys show what's reachable right now.
- **Model** — auto-populated from `ollama list` when Ollama is running.
- **Task** — `roman`, `parens`, `csv`, `expr`, `roman_parse`, or `custom…` for
  your own (JSON editor, prefilled with a working example).
- **Where code runs** — subprocess or Docker. See sandboxing below.
- **Max attempts** — raise it for small models; they need more swings.
- **If it plateaus** — reseed from the best attempt so far (the ratchet), or
  keep feeding the same error. Stall limit and temp step are optional; temp
  step defaults to 0 (no heat-up).
- **Live run** — one card per attempt with a pass/fail bar, the exact test
  failures, and the code. Streams in as it happens.
- **Improvement curve** — every run you've ever done, overlaid.
- **Self-edit** — propose a rewrite of the app's own source, approve the diff,
  then verify-or-revert. See below.

## The ratchet

Every retry is anchored on the **best** attempt so far, never on a regression.
The best code is written to `best_<task>_<runid>.py` as soon as it improves.

If the score does not improve for `--stall-limit` attempts (default 3), the
loop **reseeds**: a fresh conversation still showing that best code and the
remaining failures. `--no-escalate` turns reseeding off (for A/B runs).

Temperature stays at 0.2 unless you pass `--temp-step`. Default 0 means no
heat-up — reseeding is the only plateau action. (An earlier version raised
temperature at the same time and could not tell which change mattered.)

## Command line

```powershell
py -3 self_improving_agent.py --backend ollama --model qwen2.5-coder:7b
py -3 self_improving_agent.py --backend ollama --runner docker
py -3 self_improving_agent.py --backend mock --task roman
py -3 self_improving_agent.py --plot
```

```
--backend     anthropic | ollama | lmstudio | groq | mock | mock-stuck
--model       override the default
--runner      subprocess | docker
--task        roman | parens | csv | expr | roman_parse
--task-file   path to custom task JSON
--attempts N  default 10
--stall-limit N   attempts without improvement before reseeding (default 3)
--no-escalate     disable reseeding on plateau
--temp-step F     temperature added per reseed level (default 0: fixed)
--plot            print the curve, write improvement_curve.png
--task-filter     with --plot, one task only
```

`mock-stuck` always returns the same partial CSV parser (12/15) so you can
watch plateau reseeding without a real model.

## Self-edit

`self_edit.py` lets a strong model rewrite one of the app's own source files.
Nothing is written until you approve the diff. Every write is backed up first.
After writing, a verification gate runs (compile, import, a mock roman loop,
and a spare-port UI boot). Any failure reverts the file automatically.
`self_edit.py` itself is not editable by the agent.

```powershell
py -3 self_edit.py --verify-only          # run the gate on the current files
py -3 self_edit.py --list-backups
py -3 self_edit.py --restore NAME
py -3 self_edit.py "add a clear-history button" --backend anthropic
```

The same flow is in the UI's **Self-edit** tab (propose → approve → verify).
Do not use `mock` for this — it cannot write a real file.

## Sandboxing

**subprocess** (default) — runs candidate code with a 10s timeout. Catches
infinite loops and keeps crashes out of the parent. This is isolation, *not* a
sandbox: model-written code can still read your files and reach the network.
Fine for these fixed toy tasks whose code you can read.

**docker** — the real thing. Candidate code runs in `python:3.11-slim` with:

- `--network=none` — no exfiltration, no downloads
- `--read-only` filesystem, with a 16MB tmpfs for scratch
- 256MB memory, 1 CPU, 64 process cap
- runs as `nobody`, `no-new-privileges`
- killed on timeout

Code and test cases go in over stdin and a JSON verdict comes back on stdout, so
there are no volume mounts and no Windows path translation to get wrong. Start
Docker Desktop first; the first run pulls a ~45MB image.

Use Docker the moment you point this at problems you haven't read the code for.

## Setup

Optional packages (backends + the CLI chart). Core app is stdlib:

```powershell
py -3 -m pip install -r requirements.txt
ollama pull qwen2.5-coder:7b
```

`requirements.txt` lists `openai` (Ollama, Groq, LM Studio), `anthropic`,
`matplotlib`, and `pytest`. Install only what you use.

Model by VRAM: ≥8GB → `qwen2.5-coder:7b`, ≥16GB → `:14b`, CPU-only → `:3b`.

Environment variables:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:GROQ_API_KEY = "gsk_..."
$env:OLLAMA_HOST_URL = "http://localhost:11434/v1"     # default
$env:LMSTUDIO_HOST_URL = "http://localhost:1234/v1"    # default
$env:SIA_PORT = "8000"                                 # UI listen port
$env:SIA_NO_BROWSER = "1"                              # don't open a tab
```

Set keys before launching `web_ui.py` — the UI reads them from its own process.

## Tests

Stdlib-only checks plus a mock roman loop. No paid APIs.

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m pytest
```

## The comparison worth running

Run the same task on a strong API model and on a local 7B, then open the curve
tab. The strong model will often solve Roman numerals on attempt 1, which makes
the loop look like a formality. The 7B usually fails once or twice first — and
that's the only time you can actually see the error-feedback step doing work.

## Your own problem

Pick `custom…` in the task dropdown, or write a JSON file:

```json
{
  "name": "fizz",
  "func_name": "fizzbuzz",
  "spec": "Write fizzbuzz(n) returning 'Fizz', 'Buzz', 'FizzBuzz' or str(n).",
  "cases": [[[3], "Fizz"], [[5], "Buzz"], [[15], "FizzBuzz"], [[7], "7"]]
}
```

Each case is `[[args...], expected]`, compared with `==`. Writing the cases *is*
the work — a vague spec with three cases lets the model pass by cheating. The
scoreboard is only as honest as you make it.

## Files it writes

- `attempts.jsonl` — one line per attempt: run id, model, tests passed, error,
  full code. Append-only, so runs accumulate and stay comparable.
- `solution_<task>_<runid>.py` — the winning function.
- `best_<task>_<runid>.py` — the best attempt so far, even if the run never
  fully solves the task.
- `improvement_curve.png` — from `--plot`.
- `backups/` — timestamped copies of `*.py` taken before a self-edit.

## What's handled

Small models are messier than the API, and the harness expects it: prose wrapped
around code, stray or unlabeled fences, multiple code blocks, syntax errors, a
function defined under the wrong name, exceptions at call time, infinite loops,
and out-of-memory in the container all produce a clean message that gets fed
back to the model rather than crashing the run.
