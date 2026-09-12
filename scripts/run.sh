#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export FLYPAINT_GRAPH="${1:-${FLYPAINT_GRAPH:-}}"
exec .venv/bin/python -m uvicorn services.trainer.app:app_factory --factory --host 127.0.0.1 --port 8000
