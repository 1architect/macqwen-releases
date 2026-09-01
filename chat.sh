#!/bin/bash
# One chat interface. macqwen/cli.py selects the model's Python environment.
if [[ -f "$HOME/.frankenstein/env" ]]; then
  set -a; source "$HOME/.frankenstein/env"; set +a
fi
exec /usr/bin/python3 "$(dirname "$0")/macqwen/cli.py" "$@"
