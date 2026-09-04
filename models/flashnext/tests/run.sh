#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHON="${MACQWEN_FLASHNEXT_PYTHON:-$ROOT/.venv/bin/python}"
exec "$PYTHON" -m models.flashnext.tests "$@"
