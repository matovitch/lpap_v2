#!/usr/bin/env bash
# Push notebooks/molab.py cell code into the paired notebook (match by position,
# edit only changed cells). With --reinstall: force-reinstall lpap from git main
# (push first) and re-run the install cell, which purges loaded lpap modules
# and re-runs every dependent cell. Named cells missing from the notebook are
# created in place; removals and new unnamed cells are done by hand. Edited and
# created cells are re-run (run cells stay gated by their buttons).
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
    _names = {c.name for c in _remote}
    # Named repo cells missing from the notebook are created in place; others pair up by position.
    _plan, _next = [], 0
    for _cell in _local:
        if _cell['name'] != '_' and _cell['name'] not in _names:
            _plan.append((_cell, None))
        else:
            _plan.append((_cell, _remote[_next] if _next < len(_remote) else None))
            _next += 1
    if _next != len(_remote) or any(t is None and c['name'] == '_' for c, t in _plan):
        raise RuntimeError(f'cell layout differs: notebook {len(_remote)}, repo {len(_local)}; fix by hand')
    _run, _previous = [], None
    for _cell, _target in _plan:
        if _target is None:
            _previous = ctx.create_cell(_cell['code'], after=_previous, name=_cell['name'])
            _run.append(_previous)
            print('created', _previous, _cell['name'])
            continue
        if _target.code != _cell['code']:
            ctx.edit_cell(_target.id, _cell['code'])
            _run.append(_target.id)
            print('edited', _target.id, _target.name)
        _previous = _target.id
if _run:
    async with cm.get_context() as ctx:
        for _id in _run:
            ctx.run_cell(_id)
if ${reinstall}:
    os.environ['LPAP_FORCE_REINSTALL'] = '1'
    async with cm.get_context() as ctx:
        ctx.run_cell(next(c.id for c in ctx.cells if c.name == 'install'))
"
