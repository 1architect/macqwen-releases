#!/bin/zsh
# Run Qwen3.8-Flash-Next from SSD on Apple Silicon.
PYTHON_BIN="${FLASHNEXT_PYTHON:-python3}"
exec caffeinate -i "$PYTHON_BIN" -u "$(dirname "$0")/chat.py" "$@"
