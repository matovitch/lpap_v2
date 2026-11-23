# LPAP v2

Image autoencoder whose latent code is the exact output of LPAP (Linear
Probing Amplitude Packing) applied to a flow-matched energy. This is a
from-scratch reboot of `lpap-molab` (v3).

**Read [FOUNDATION.md](FOUNDATION.md) first.** It is the authority for the
module contracts, gradient routing, losses and training stages.

## Layout

| Path | Content |
| --- | --- |
| `src/lpap/geometry.py` | pair spec, lane table `G`, collision sets |
| `src/lpap/lpap.py`, `lpap_triton.py` | exact LPAP (torch reference + Triton kernel) |
| `src/lpap/data.py` | images, Hilbert order, 95/5 split |
| `src/lpap/flow.py` | bidirectional flow field, Euler integration, prior |
| `src/lpap/surrogate.py`, `decoder.py`, `transformer.py` | per-pair models |
| `src/lpap/codec.py` | §3 gradient routing (frontend ST, split probe ST) |
| `src/lpap/losses.py` | §4 losses |
| `src/lpap/runtime/` | strict config loader, checkpoints, HF artifact store, SQLite log, Pushover, foreground run loop |
| `src/lpap/stages/` | one module per training stage (flow so far) |
| `src/lpap/configs/` | packaged TOML configs |
| `notebooks/molab.py` | the single marimo notebook |
| `scripts/molab/` | pairing helpers: `exec.sh`, `press.sh`, `status.sh`, `inject-secrets.sh`, `sync.sh` (push `notebooks/molab.py` cell edits, `--reinstall` lpap) |
| `scripts/review.py` | pull run logs and galleries from the HF repo, compare runs |

## Usage

```bash
pixi run test       # pytest (Triton parity tests need CUDA)
pixi run lint       # ruff check + format check
pixi run format
pixi run notebook   # local marimo preview
```

Remote training runs in the molab notebook only, in foreground cells. There
are no background processes (see FOUNDATION §7).

## molab

1. Open `notebooks/molab.py` on molab with a GPU runtime and pair
   (`MOLAB_URL`, `MOLAB_TOKEN`; marimo-pair skill installed on the host).
2. `scripts/molab/inject-secrets.sh` exports `secrets.toml` (copy of
   `secrets.toml.example`) into the kernel: `HF_TOKEN` (write access to the
   artifacts repo of `src/lpap/configs/storage.toml`), `PUSHOVER_*`.
3. Re-run the `session` cell, press **Load images**, then a stage's train
   button (`scripts/molab/press.sh flow_button` from the host).
4. While a run cell trains, monitor with `scripts/molab/status.sh` only: any
   kernel exec would queue behind the cell, and abandoning it interrupts it.

Run artifacts live under `<root>/runs/<run>/{last.pt,best.pt,log.sqlite}`
(root `/marimo/lpap` on molab) and in the public HF model repo, one commit
per upload, with `gallery.png` when the stage writes one. Review locally with
`pixi run python scripts/review.py <run-prefix> [metric ...]`.
