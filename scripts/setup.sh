#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
npm ci --prefix apps/web
npm run build --prefix apps/web
printf '%s\n' 'Ready. Run: bash scripts/run.sh (synthetic) or bash scripts/run.sh data/processed/task-512 (real graph).'
