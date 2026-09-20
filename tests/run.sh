#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${MACQWEN_PYTHON:-$ROOT/.venv/bin/python}"
exec "$PYTHON" -m macqwen.testsuite "$@"
