#!/usr/bin/env bash
# Run with a Python environment containing the dependencies; set PYTHON_BIN to choose the interpreter.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/train_eval_drugs.py" "$@"
