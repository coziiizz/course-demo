#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="dl-lab5-check"
ENV_FILE="$ROOT_DIR/envs/lab5-check.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found in PATH." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  conda env create -f "$ENV_FILE"
fi

conda run --no-capture-output -n "$ENV_NAME" \
  python "$ROOT_DIR/scripts/lab5_checker.py" "$@"
