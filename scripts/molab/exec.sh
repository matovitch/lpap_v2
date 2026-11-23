#!/usr/bin/env bash
# Run Python in the paired molab kernel scratchpad (marimo-pair execute-code).
# Usage: exec.sh -c 'print(1)' | exec.sh file.py | exec.sh <<'PY' ... PY
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/lib.sh"
molab_require_env
molab_guard_busy
execute="${MARIMO_PAIR_SCRIPTS:-$HOME/.cursor/skills/marimo-pair/scripts}/execute-code.sh"
[[ -x "$execute" ]] || { echo "missing $execute (npx skills add marimo-team/marimo-pair)" >&2; exit 1; }
args=(--url "$MOLAB_URL" --token "$MOLAB_TOKEN")
[[ -n "${MOLAB_SESSION:-}" ]] && args+=(--session "$MOLAB_SESSION")
exec "$execute" "${args[@]}" "$@"
