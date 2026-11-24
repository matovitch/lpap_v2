#!/usr/bin/env bash
# Wait (locally) until molab run LAST_RUN stops running, or the heartbeat goes stale.
# Sends one Pushover message from secrets.toml, then exits. Read-only on molab (file API).
# Usage: MOLAB_URL=... MOLAB_TOKEN=... watch.sh LAST_RUN [poll_seconds] [stale_seconds]
set -uo pipefail
last_run=${1:?last run name}
poll=${2:-300}
stale=${3:-1200}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)

notify() {
    python3 - "$1" "$2" "$root/secrets.toml" <<'EOF'
import sys, tomllib, urllib.parse, urllib.request
title, message, path = sys.argv[1:]
try:
    push = tomllib.load(open(path, "rb")).get("pushover", {})
    data = urllib.parse.urlencode({"token": push["token"], "user": push["user"],
                                   "title": title, "message": message}).encode()
    urllib.request.urlopen("https://api.pushover.net/1/messages.json", data=data, timeout=20)
except Exception as error:
    print(f"pushover failed: {error}", file=sys.stderr)
EOF
}

misses=0
while true; do
    # molab file API can take 30s+ under load; a short timeout turns slowness into false alarms
    status=$(MOLAB_HTTP_TIMEOUT=${MOLAB_HTTP_TIMEOUT:-60} "$here/status.sh" --json 2>/dev/null)
    if [ -z "$status" ]; then
        misses=$((misses + 1))
        echo "$(date +%H:%M) no status (miss $misses)"
        if [ "$misses" -ge 6 ]; then
            notify "lpap watch: molab unreachable" "status.sh failed $misses times; session may be gone."
            echo "unreachable"; exit 2
        fi
        sleep "$poll"; continue
    fi
    misses=0
    read -r run state step age < <(python3 -c '
import json, sys, time
d = json.loads(sys.argv[1])
age = time.time() - d.get("heartbeat", time.time()) if isinstance(d.get("heartbeat"), (int, float)) else 0
print(d.get("run"), d.get("state"), d.get("step"), int(age))' "$status")
    echo "$(date +%H:%M) $run $state step=$step heartbeat_age=${age}s"
    if [ "$run" = "$last_run" ] && [ "$state" != "running" ]; then
        notify "lpap: study done" "$run $state at step $step. Wake Copilot to read results."
        echo "done"; exit 0
    fi
    if [ "$state" = "running" ] && [ "$age" -gt "$stale" ]; then
        notify "lpap watch: stalled" "$run heartbeat ${age}s old at step $step."
        echo "stalled"; exit 3
    fi
    if [ "$state" = "failed" ]; then
        notify "lpap watch: failed" "$run failed at step $step."
        echo "failed"; exit 4
    fi
    sleep "$poll"
done
