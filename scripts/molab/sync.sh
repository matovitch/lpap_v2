#!/usr/bin/env bash
# Push notebooks/molab.py cell code into the paired notebook (match by position,
# edit only changed cells). With --reinstall: force-reinstall lpap from git main
# (push first) and re-run the install cell, which purges loaded lpap modules
# and re-runs every dependent cell. Cell count must match; otherwise add or
# remove cells by hand first. Edited cells are re-run (run cells stay gated by their buttons).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/lib.sh"
molab_require_env
molab_guard_busy
reinstall=0
[[ "${1:-}" == "--reinstall" ]] && reinstall=1
cells="$(cd "$here/../.." && pixi run -q python - <<'PY'
import base64, importlib.util, json
spec = importlib.util.spec_from_file_location("nb", "notebooks/molab.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cells = [{"code": d.code, "name": d.name} for d in module.app._cell_manager.cell_data()]
print(base64.b64encode(json.dumps(cells).encode()).decode())
PY
)"
"$here/exec.sh" -c "
import base64, json, os
import marimo._code_mode as cm
_local = json.loads(base64.b64decode('${cells}'))
async with cm.get_context() as ctx:
    _remote = list(ctx.cells)
    if len(_remote) != len(_local):
        raise RuntimeError(f'cell count differs: notebook {len(_remote)}, repo {len(_local)}')
    _edited = [t.id for c, t in zip(_local, _remote) if t.code != c['code']]
    for _cell, _target in zip(_local, _remote):
        if _target.id in _edited:
            ctx.edit_cell(_target.id, _cell['code'])
            print('edited', _target.id, _target.name)
if _edited:
    async with cm.get_context() as ctx:
        for _id in _edited:
            ctx.run_cell(_id)
if ${reinstall}:
    os.environ['LPAP_FORCE_REINSTALL'] = '1'
    async with cm.get_context() as ctx:
        ctx.run_cell(next(c.id for c in ctx.cells if c.name == 'install'))
"
