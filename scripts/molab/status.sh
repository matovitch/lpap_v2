#!/usr/bin/env bash
# Run heartbeat (state, step, metrics) via the server file API; safe while training.
# Usage: status.sh [--json]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/lib.sh"
molab_require_env
status="$(molab_read_file "$LPAP_REMOTE_ROOT/status.json")" || { echo "no status yet" >&2; exit 1; }
if [[ "${1:-}" == "--json" ]]; then echo "$status"; exit 0; fi
jq -r '"\(.run) state=\(.state) step=\(.step)/\(.target) heartbeat_age=\((now - .heartbeat) | floor)s",
       (.metrics | to_entries[] | "  \(.key)=\(.value)")' <<<"$status"
