#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONTROL_PATH="${SCRIPT_DIR}/data/controls/ctl_vehicle_default164_ctrl1.h5ad"
CUSTOM_CONTROL=0
for arg in "$@"; do
    case "$arg" in
        --control_path|--control_path=*) CUSTOM_CONTROL=1 ;;
    esac
done

if [[ "$CUSTOM_CONTROL" == 0 && ! -f "$CONTROL_PATH" ]]; then
    "$PYTHON_BIN" "${SCRIPT_DIR}/tools/prepare_data.py" restore
fi

exec "$PYTHON_BIN" "${SCRIPT_DIR}/recommend_drugs.py" "$@"
