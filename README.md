# Self-improving agent

A loop with a scoreboard. The model writes a function, the function runs against
a fixed test suite, failures are fed back verbatim, the model rewrites. The test
suite is the whole reason "improvement" means anything here.

Two ways in: a browser UI, or the command line. Same loop underneath.

## Start here

Put both files in the same folder, then:

```powershell
cd C:\Users\danie\selfimprove
python web_ui.py
```

Your browser opens at `http://localhost:8000`. Leave backend on **mock** and
press **Run** — you'll watch it go 2/11 → 9/11 → 11/11. `mock` isn't a model,
it's three canned answers, there so you can confirm everything works before
blaming Ollama for anything.

Then switch the backend to **ollama** and run it again for real.

The UI has no dependencies — pure standard library, binds to 127.0.0.1 only, so
nothing on your network can reach it.

### What's in it

- **Backend** — mock, ollama, anthropic, groq. Dots next to each show what's
  actually reachable right now, so you're not guessing.
- **Model** — auto-populated from `ollama list` when Ollama is running.
- **Task** — three built-ins, or `custom…` for your own (JSON editor, prefilled
  with a working example).
- **Where code runs** — subprocess or Docker. See sandboxing below.
- **Max attempts** — raise it for small models; they need more swings.
- **Live run** — one card per attempt with a pass/fail bar, the exact test
  failures, and the code. Streams in as it happens.
- **Improvement curve** — every run you've ever done, overlaid.

## Command line

```powershell
python self_improving_agent.py --backend ollama --model qwen2.5-coder:7b
python self_improving_agent.py --backend ollama --runner docker
python self_improving_agent.py --plot
```

```
--backend  anthropic | ollama | groq | mock
--model    override the default
--runner   subprocess | docker
--task     roman | parens | roman_parse
--task-file  path to custom task JSON
--attempts N   default 10
--plot     print the curve, write improvement_curve.png
```

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

```powershell
python -m pip install openai matplotlib     # ollama/groq + the CLI chart
python -m pip install anthropic             # only for the paid backend
ollama pull qwen2.5-coder:7b
```

Model by VRAM: ≥8GB → `qwen2.5-coder:7b`, ≥16GB → `:14b`, CPU-only → `:3b`.

Keys, if you use those backends:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:GROQ_API_KEY = "gsk_..."
```

Set them before launching `web_ui.py` — the UI reads them from its own process.

## The comparison worth running

Run the same task on Sonnet and on the 7B, then open the curve tab. Sonnet will
almost certainly solve Roman numerals on attempt 1, which makes the loop look
like a formality. The 7B usually fails once or twice first — and that's the only
time you can actually see the error-feedback step doing work.

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
- `improvement_curve.png` — from `--plot`.

## What's handled

Small models are messier than the API, and the harness expects it: prose wrapped
around code, stray or unlabeled fences, multiple code blocks, syntax errors, a
function defined under the wrong name, exceptions at call time, infinite loops,
and out-of-memory in the container all produce a clean message that gets fed
back to the model rather than crashing the run.
