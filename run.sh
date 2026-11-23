#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Set up the application first: python -m venv .venv && .venv/bin/pip install -e ." >&2
  exit 1
fi
exec "$ROOT/.venv/bin/python" -m plate_ocr_review.app "$@"
