#!/usr/bin/env bash
# Idempotent bootstrap for the self-improving-agent dev environment.
# The app, UI, and self-edit are pure standard library; the venv only carries
# the optional backend SDKs (openai/anthropic), matplotlib for --plot, and
# pytest for the test suite.
set -euo pipefail

# The core loop runs on Python's stdlib, but `python3 -m venv` needs ensurepip,
# which Debian/Ubuntu split into a separate package.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3-venv
fi

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
