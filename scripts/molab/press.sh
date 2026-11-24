#!/usr/bin/env bash
# Press a notebook UI element like a human click (fire-and-forget).
# The dependent cells run as a normal notebook run, not inside the scratch
# request, so long runs survive this client exiting.
# Usage: press.sh <element_global_name> [json_value] [cell_name]   (value defaults to true)
# cell_name: also run that cell in the same request. Needed when the notebook runtime is lazy (a press then
# only marks dependents stale and run_buttons reset).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
name="${1:?usage: press.sh <element> [json_value]}"
value="${2:-true}"
cell="${3:-}"
[[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || { echo "invalid element name: $name" >&2; exit 1; }
python3 -c 'import json,sys; json.loads(sys.argv[1])' "$value" || { echo "value must be JSON" >&2; exit 1; }
"$here/exec.sh" <<PY
import json
import marimo._code_mode as cm
from marimo._code_mode._context import UIElementId, UpdateUIElementCommand

async with cm.get_context() as ctx:
    element = ctx.globals.get("$name")
    if element is None or not hasattr(element, "_id"):
        raise LookupError("no UI element named '$name'")
    ctx.enqueue_command(UpdateUIElementCommand(object_ids=[UIElementId(element._id)], values=[json.loads('''$value''')]))
    if "$cell":
        ctx.run_cell(next(c.id for c in ctx.cells if c.name == "$cell"))
print("queued $name =", '''$value''')
PY
