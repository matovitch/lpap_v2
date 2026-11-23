# Agent guide

- `FOUNDATION.md` is the design authority. Code that departs from it needs the
  doc updated in the same change, after agreement with the user.
- Environment: `pixi run test`, `pixi run lint`, `pixi run format`.
- molab: never start background processes (nohup, `&`, threads that outlive a
  cell, detached subprocesses). The account has been banned for it before.
  All training runs in a visible notebook cell with a step target and a
  wall-clock budget.
- Pairing: `scripts/molab/exec.sh` (kernel scratchpad), `press.sh <button>` to
  start a run like a human click (never `set_ui_value` for long work),
  `status.sh` to monitor. Never exec into the kernel while a run cell trains
  (`exec.sh` refuses based on `status.json`). Prefer Pushover over polling.
- After editing `notebooks/molab.py` or `src/`, run `scripts/molab/sync.sh
  [--reinstall]`: it edits and re-runs changed cells positionally.
- Keep big files out of the molab workspace (`/marimo`): run root and data in
  `/tmp/lpap`, persistence on HF, derived data in memory.
- Secrets reach molab only through `inject-secrets.sh` (kernel env, no files).
- Every §3 behaviour (gradient routing) has a test in `tests/test_codec.py`.
  A change to `codec.py` must keep these tests passing or update them together
  with FOUNDATION §3.
- One test module per source module. jaxtyping shapes on tensor APIs.
- No backward compatibility: regenerate artifacts when a schema changes.
- Commits carry `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.
