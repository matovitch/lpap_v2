#!/usr/bin/env bash
# Export secrets.toml (repo root, gitignored) into the paired kernel's environment.
# Nothing is written to disk on molab. Re-run after every kernel restart.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
secrets="${LPAP_SECRETS_TOML:-$here/../../secrets.toml}"
[[ -f "$secrets" ]] || { echo "missing $secrets (copy secrets.toml.example)" >&2; exit 1; }
payload="$(python3 - "$secrets" <<'PY'
import base64, json, sys, tomllib
data = tomllib.load(open(sys.argv[1], "rb"))
pairs = {"HF_TOKEN": data.get("huggingface", {}).get("token"),
         "PUSHOVER_TOKEN": data.get("pushover", {}).get("token"),
         "PUSHOVER_USER": data.get("pushover", {}).get("user")}
env = {key: value.strip() for key, value in pairs.items() if value and value.strip()}
print(base64.b64encode(json.dumps(env).encode()).decode())
PY
)"
"$here/exec.sh" -c "
import base64, json, os
_env = json.loads(base64.b64decode('${payload}'))
os.environ.update(_env)
print('injected', ' '.join(sorted(_env)))
"
