#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/train_eval_multidose.py" "$@"
