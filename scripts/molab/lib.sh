# Shared helpers for the molab pairing scripts. Source, do not execute.
#
# Env: MOLAB_URL, MOLAB_TOKEN (or MARIMO_TOKEN)
#      MARIMO_PAIR_SCRIPTS  marimo-pair scripts dir (default ~/.cursor/skills/marimo-pair/scripts;
#                           install with: npx skills add marimo-team/marimo-pair)
#      LPAP_REMOTE_ROOT     run root on the molab box (default /marimo/lpap)

MOLAB_TOKEN="${MOLAB_TOKEN:-${MARIMO_TOKEN:-}}"
LPAP_REMOTE_ROOT="${LPAP_REMOTE_ROOT:-/marimo/lpap}"

molab_require_env() {
  [[ -n "${MOLAB_URL:-}" ]] || { echo "set MOLAB_URL to the paired notebook URL" >&2; return 1; }
  [[ -n "$MOLAB_TOKEN" ]] || { echo "set MOLAB_TOKEN (or MARIMO_TOKEN)" >&2; return 1; }
}

# Read a file of the molab box through the server file API: never touches the kernel.
molab_read_file() {
  curl -sf -m 15 -X POST "${MOLAB_URL%/}/api/files/file_details" \
    -H "Authorization: Bearer ${MOLAB_TOKEN}" -H "Content-Type: application/json" \
    -d "$(jq -n --arg p "$1" '{path: $p}')" | jq -er '.contents // empty'
}

# Refuse kernel execs while a run cell is training: the exec would queue behind
# it, and abandoning the exec interrupts the run. MOLAB_EXEC_FORCE=1 overrides.
molab_guard_busy() {
  [[ "${MOLAB_EXEC_FORCE:-0}" == "1" ]] && return 0
  local status age
  status="$(molab_read_file "$LPAP_REMOTE_ROOT/status.json" 2>/dev/null)" || return 0
  [[ "$(jq -r .state <<<"$status")" == "running" ]] || return 0
  age="$(jq -r 'now - .heartbeat | floor' <<<"$status")"
  (( age < ${MOLAB_BUSY_STALE_S:-1800} )) || return 0
  echo "run '$(jq -r .run <<<"$status")' is training (heartbeat ${age}s ago); use status.sh." >&2
  return 1
}
