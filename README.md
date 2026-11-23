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
| `src/lpap/runtime/` | strict config loader (checkpoint, hub, log, run loop to come) |
| `src/lpap/configs/` | packaged TOML configs |
| `notebooks/molab.py` | the single marimo notebook |
| `docs/` | extra documentation |

## Usage

```bash
pixi run test       # pytest (Triton parity tests need CUDA)
pixi run lint       # ruff check + format check
pixi run format
pixi run notebook   # local marimo preview
```

Remote training runs in the molab notebook only, in foreground cells. There
are no background processes (see FOUNDATION §7).
