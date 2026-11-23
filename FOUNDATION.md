# LPAP v2 — founding document

Status: **v0. All sections settled in a first pass; under review.**

Prior art: `lpap-molab` branch `lpap-v3-priority1` (v3 code + v4 redesign in
`HANDOVER.md`, contracts in `doc/stage-io-losses.md`, operator in `doc/lpap.md`).
This repo is a from-scratch rewrite; v3 code is ported only where listed in §9.

---

## 0. Purpose, scope, non-goals

Decisions:
- Multi-pair from day one. The code is generic over a list of pairs
  `(C, k_max, permutation_seed)` sharing one length-`N` energy.
- Initial pair set: the v3 trio `c128_k16`, `c256_k24`, `c512_k32` (`N = 1024`).
- A **lossless pair** `N = C` (so `P = 1`) must be supported by every module
  and loss (degenerate probe dimension, trivial collision sets).
- Goal of the model: remap the input into a geometry that is **progressively
  compressible under LPAP**. The lossless pair is a real, jointly trained pair:
  it forces the remap to stay information-preserving while the lossy pairs
  shape compressibility.
- Data unchanged: 32×32 grayscale, `/255`, Hilbert order, `N = 1024`.
- Cut from v3:
  - the final exact-LPAP decoder finetune stage;
  - the script/CLI pipeline runner (everything runs from notebooks).
- Success criterion: simpler code, fewer options, easier to debug, one
  consistent organisation (e.g. a single docs folder, not `doc/` + `docs/`).
- With no exact finetune, inference uses exact LPAP straight out of the AE
  stage. The AE forward is therefore the exact code (§3.1), so there is no
  ST-vs-exact gap to close.

## 1. Notation and invariants

Sizes. Energy length `N = 1024`. A pair `r` is `(C, k_max, permutation_seed)`
with `N % C == 0`, `P = N / C` (probe count), `1 <= k_max <= C`.

| Pair | C | P | k_max | Local classes `k_max·P` | Permutation |
| --- | ---: | ---: | ---: | ---: | --- |
| `c128_k16` | 128 | 8 | 16 | 128 | `randperm(N)`, seed 123 |
| `c256_k24` | 256 | 4 | 24 | 96 | `randperm(N)`, seed 256 |
| `c512_k32` | 512 | 2 | 32 | 64 | `randperm(N)`, seed 512 |
| `lossless` | 1024 | 1 | 1 | 1 | identity |

Terms (one name each, used in code and docs):
- **energy** `x ∈ ℝ^{B×N}`: the flow latent at `t = 0`, Hilbert order.
- **lane table** `G_r ∈ ℕ^{C×P}`: `G[b, p]` is the energy index at lane `b`,
  probe `p`. Built once on CPU from the seeded permutation `π` with a
  **row-major** fold: `G = π.reshape(C, P)`. It is the single source of truth
  for folding, collision sets and scattering; no separate fold/unfold code.
- **tokens** `x[..., G] ∈ ℝ^{B×C×P}`.
- **bucket** `b ∈ [0, C)`, **probe** `p ∈ [0, P)`, **dib** `d ∈ [0, k_max)`
  (distance to initial bucket). Lane read by bucket `b` at dib `d`:
  `(b − d) mod C`.
- **local class** `c = d·P + p ∈ [0, k_max·P)`.
- **collision set** `S(b, d) = G[(b − d) mod C, :]` (`P` energy indices).
  Source index of a kept value: `s(b, d, p) = G[(b − d) mod C, p]`.
- **code** of a pair: per bucket, signed **amplitude** `a_b` and dib `d_b`.
  The probe `p_b` is *not* part of the code.

Exact LPAP operator: v3 semantics unchanged (see `lpap-molab/doc/lpap.md`):
rolls `k = 0..k_max−1`, per lane pick `argmax_p |work|` (lowest index on
ties), take when the bucket is empty or `|cand| >= |bucket|`, swap the
displaced value back with a dib offset, signed values compared by `|x|`.
Outputs per bucket: `a*`, `d*`, `p*`, `valid`; source `i* = s(b, d*, p*)`.

Invariants (each one is a test):
1. `G_r` is a bijection onto `[0, N)`; stored in checkpoints and validated on load.
2. `0 <= d* < k_max`; `i*` values are distinct within a sample.
3. The decoder input never carries probe information.
4. Lossless pair: LPAP is the identity (`a* = x[G]`, `d* = 0`).
5. Nesting across pairs is *not* enforced; it is expected to emerge from the
   approximate top-C property.

Decisions:
- Row-major fold by default. Column-major only differs by how `G` is built;
  with a random `π` both are random lane partitions (same statistics, same
  speed). The fold matters only for an identity permutation (contiguous
  Hilbert patches vs strided samples) — parked in §9.

## 2. Module contracts

Forward contracts only: shapes, meaning, owned parameters. A marker
**[§3]** flags a tensor whose backward differs from plain autograd; §3 is
the authority for those. `B` = batch, all floats fp32 unless stated.

### 2.1 Data

| | |
| --- | --- |
| Output | image `I ∈ [0,1]^{B×1×N}`, Hilbert order |
| Params | none |

Display unflattens with the inverse Hilbert map. Model I/O stays in Hilbert order.

### 2.2 Bidirectional flow `v_θ(x, t)`

| | |
| --- | --- |
| Input | `x ∈ ℝ^{B×1×N}`, `t ∈ [−1, 1]^B` |
| Output | velocity `ℝ^{B×1×N}` |
| Params | `DilatedConvFlow1d` (v3 architecture: width 128, sinusoidal time embedding, 2 × dilations 1..128, GroupNorm + FiLM, zero-init head) |

- `t ∈ [−1, 0]`: image → energy; `t ∈ [0, 1]`: energy → image.
- Prior at `t = 0`: signed log-normal, `σ = 1`, `scale = 0.2`.
- Integration: Euler midpoint, fp32 even under autocast.
- In AE training the pretrained field is loaded into two independent copies:
  `encoder = Euler_4(v_i2e, I, −1 → 0)` gives the energy `x ∈ ℝ^{B×N}`;
  `renderer = Euler_4(v_e2i, ê, 0 → 1)` gives `Î ∈ ℝ^{B×1×N}`.

### 2.3 Exact LPAP (per pair, no params)

| | |
| --- | --- |
| Input | tokens `x[..., G] ∈ ℝ^{B×C×P}` |
| Output | `a* ∈ ℝ^{B×C}`, `d* ∈ ℕ^{B×C}`, `p* ∈ ℕ^{B×C}`, `i* ∈ ℕ^{B×C}` |

Pure function, run on detached tokens (no autograd). Torch reference plus
optional Triton kernel with bit-identical outputs.
Derived: oracle local class `c* = d*·P + p*`.

### 2.4 Surrogate (per lossy pair)

| | |
| --- | --- |
| Input | tokens `∈ ℝ^{B×C×P}` |
| Output | local-class logits `z ∈ ℝ^{B×C×(k_max·P)}` |
| Params | `Linear(P→256)`, 12 transformer blocks (8 heads, circular RoPE over `C`), LayerNorm, `Linear(256→k_max·P)` |

- Input normalisation: tokens are divided by a per-sample RMS
  `ρ = sqrt(mean(x²) + ε)` before the input linear (`ρ` live, §3.4).
  Raw tokens are still used for amplitudes.
- Attention mask: bucket `i` attends to `j` iff `(i − j) mod C < k_max`
  (diagonal included, `True` = may attend).
- Meaning: `softmax(z_b)` predicts which candidate (dib, probe) exact LPAP
  keeps in bucket `b`.

### 2.5 Frontend (per lossy pair, no params)

Turns the exact code into decoder inputs.

| | |
| --- | --- |
| Input | tokens `∈ ℝ^{B×C×P}`, exact LPAP outputs, optional surrogate logits `z` |
| Output | amplitude `a ∈ ℝ^{B×C}`, dib one-hot `w ∈ {0,1}^{B×C×k_max}`, dib `d ∈ ℕ^{B×C}` |

- The forward is always the exact code: `a_b = x[i*_b]`, `w_b = one_hot(d*_b)`,
  `d_b = d*_b`.
- With `z` (AE stage) the same forward values carry a surrogate backward
  **[§3: categorical ST]**. Without `z`, `a` keeps only its direct gradient
  at the winning coordinate and `w` is a constant (the amplitude-only
  baseline of §3.4; in the teacher decoder stage the tokens are detached
  anyway).
- The probe of the chosen candidate is **not** exported (invariant 3).

### 2.6 Decoder (per lossy pair)

| | |
| --- | --- |
| Input | `a ∈ ℝ^{B×C}`, `w ∈ ℝ^{B×C×k_max}` |
| Output | probe logits `ℓ ∈ ℝ^{B×C×P}` |
| Params | amplitude `Linear(1→256)`, source-lane `Embedding(C, 256)`, dib `Linear(k_max→256, no bias)`, bucket position `Parameter(C, 256)`, 8 blocks (8 heads, circular RoPE, no mask), LayerNorm, `Linear(256→P)` |

Token `b`: `h_b = W_a a_b + Σ_d w_{b,d} E[(b−d) mod C] + W_d w_b + pos_b`.
Buckets fed from the same lane share `E[lane]`, so they can coordinate and
pick distinct probes. Meaning: `softmax(ℓ_b)` is the distribution over the
`P` members of `S(b, d_b)`. The amplitude passes through unchanged.

### 2.7 Scatter / decoded energy (per pair, no params)

| | |
| --- | --- |
| Input | `a`, `w`, `ℓ` |
| Output | decoded energy `ê ∈ ℝ^{B×N}` |

Forward: one spike per bucket, `ê = scatter_add(a_b → s(b, d_b, argmax ℓ_b))`,
zero elsewhere (collisions sum) **[§3: probe ST]**.
Lossless pair: `ê = x` (identity, no surrogate or decoder).

### 2.8 Pair bundle

A lossy pair owns `{G, surrogate, decoder}`; the lossless pair owns `{G}`.
The flow copies are shared by all pairs. Each pair yields one `ê^{(r)}` and
one `Î^{(r)}`.

## 3. Gradient routing

Authority for every backward that is not plain autograd. Applies to the AE
stage; teacher stages (§5) use plain autograd on detached inputs.

### 3.1 Principles

1. **The forward is always the real code.** Selection in the AE forward is
   exact LPAP, so the decoder and renderer only ever see codes that inference
   will produce. ST-vs-exact gap is zero by construction; no finetune stage.
2. **The surrogate is a differentiable model of LPAP, not a policy.** Its
   parameters are trained only by the surrogate CE. Reconstruction gradients
   pass *through* it (to tokens, hence the encoder) without updating it.
3. **Use the true gradient whenever one exists.** ST surrogates only replace
   the gradient of a discrete choice (argmax), never that of a value.
4. Targets, labels and loss weights are always detached.

### 3.2 Frontend (per lossy pair)

Visible candidates: `V ∈ ℝ^{B×C×k_max·P}`,
`V[b, d·P + p] = x[s(b, d, p)] = tokens[(b − d) mod C, p]`.

```
z      = surrogate_frozen(tokens)        # params detached, tokens live (3.4)
q      = softmax(z)                      # (B, C, k_max·P)
y      = one_hot(c*)                     # exact LPAP choice, no grad
p_st   = y + (q − q.detach())            # forward == y exactly (also bf16)
a      = (p_st · V).sum(-1)              # (B, C)
w      = p_st.reshape(B, C, k_max, P).sum(-1)   # (B, C, k_max)
d      = d*                              # integer, no grad
```

| Output | Forward | Backward |
| --- | --- | --- |
| `a_b` | `x[i*_b]` (= `a*_b`) | to `V` only at the winner `c*_b` (true gradient, no smear); to `z` via `V_c · ∂q_c` |
| `w_b` | `one_hot(d*_b)` | to `z` via the per-dib marginal of `q` |

### 3.3 Scatter: probe ST, split form (per lossy pair)

```
p̂      = argmax ℓ                         # decoder choice
h      = scatter_add(a_b → s(b, d_b, p̂_b))                          # live in a
u      = scatter_add(w_{b,d} · softmax(ℓ_b)_p · a_b.detach() → s(b, d, p))
ê      = h + (u − u.detach())             # forward == h exactly
```

| Receiver | Gradient |
| --- | --- |
| `a_b` | `∂L/∂ê` at the chosen site only (its true gradient) |
| `ℓ_b` | through `softmax(ℓ_b)`: rewards sites where the loss wants mass `a_b` |
| `w_b` | `Σ_p softmax(ℓ_b)_p a_b ∂L/∂ê[s(b,d,p)]`: rewards dibs whose collision set the loss prefers, then flows to `z` → tokens |

Rejected: v3 `ê = h.detach() + (u − u.detach())`, which routes the
amplitude gradient through the soft probe mix (`Σ_p σ_p g_p` instead of `g_p̂`).

### 3.4 Surrogate: two calls, two roles

- **CE call:** `z_ce = surrogate(tokens.detach())`. Params live, tokens
  detached: the CE trains the surrogate only, never the encoder.
- **Frontend call:** `z = functional_call(surrogate, params.detach(), tokens)`.
  Tokens live, params detached: reconstruction gradients reach the encoder
  through the surrogate but do not update it.
- RMS input normalisation `ρ` is live (plain autograd). LPAP is
  scale-invariant; with `ε > 0` the normalised surrogate is only
  approximately so (exact away from `x ≈ 0`).
- Cost: two surrogate forwards per pair per step (one without param grads).
- Hypothesis to validate: a surrogate trained only by CE has a *useful*
  Jacobian w.r.t. tokens (CE accuracy alone does not guarantee it; a
  saturated softmax gives no selection gradient). Baseline for comparison:
  amplitude-only backward (no `z` path, `a` and `w` carry only the direct
  `V` gradient).

### 3.5 Other edges (plain autograd)

- Encoder → energy → tokens (`x[..., G]` gather).
- Renderer `ê → Î`.
- Lossless pair: `ê = x`. The image loss reaches the encoder directly. This
  is the information-preservation pressure.
- Frozen modules: `requires_grad = False` **and** `.eval()`, enforced by one
  helper after every `.train()` call.

### 3.6 Where each gradient goes (AE stage)

| Source of gradient | encoder `v_i2e` | renderer `v_e2i` | surrogate | decoder |
| --- | :-: | :-: | :-: | :-: |
| Image loss, lossy pair | ✓ (via `a` and via `z`/`w`) | ✓ | ✗ | ✓ (via `ℓ`) |
| Image loss, lossless pair | ✓ | ✓ | — | — |
| Surrogate CE | ✗ | ✗ | ✓ | ✗ |
| Probe CE | ✗ | ✗ | ✗ | ✓ |
| Signed mass | ✓ | ✗ | ✗ | ✗ |

## 4. Losses

All means are over the batch and the listed axes. `|a*|` weights are
always detached. `r` ranges over pairs; "lossy" excludes the lossless pair.

### 4.1 Flow matching (prerequisite stage)

v3 bridge, unchanged. Image `I`, independent prior draw `e`, half chosen with
probability ½, `u ~ Beta(0.1, 0.1)` squashed into `(ε, 1 − ε)`, `t = −1 + u`
(left half) or `t = u` (right half), `α = 1 − |t|`:

```
x_t = (1 − α) I + α e,    v* = sign-of-half · (e − I)     # e − I for t ≤ 0, I − e for t > 0
L_flow = mean ‖v_θ(x_t, t) − v*‖²
```

Trains: the single bidirectional field.

What the loss can and cannot show (measured on the v3 bridge, see
`stages/flow.py` validation):

- **Irreducible floor.** The coupling is independent, so near the image end
  (`α → 0`) the target `e − I` is unpredictable: loss ≥ `Var(e) = 0.04·e² ≈
  0.30`. Near the prior end the floor is about the pixel variance (≈ 0.06).
- **Sign ambiguity at `t → 0`.** The target flips sign between `t = 0⁻` and
  `0⁺`; with time embedding frequencies ≤ 1 the field cannot separate
  `|t| ≲ 10⁻³` early in training, and `Beta(0.1, 0.1)` puts about a third of
  the samples there. Integrators never query `|t| < 1/(2·steps)`.
- **Velocity error vs round trip.** Lower flow-matching error does not give
  better few-step round trips: as the field converges its paths get more
  curved and Euler round trips degrade (4 steps loses the image mean first).
  Knobs `flow.time_scale` and `time_concentration` lower the velocity error
  but worsen round trips, so they keep the v3 values (1.0, 0.1).

Validation therefore reports, on fixed images, draws and times: the Beta-weighted
loss, `fm_euler` (loss at the Euler midpoints the integrators use), per-band
losses at fixed `α`, round-trip L2 at 4 and 16 steps, energy and sample
statistics, plus a gallery (images, energies, round trips, samples).

### 4.2 Surrogate CE (teacher stage and AE)

```
CE_surr = Σ_{n,b} |a*_{n,b}| · CE(z_{n,b}, c*_{n,b}) / max(Σ_{n,b} |a*_{n,b}|, ε)
```

Weighted CEs reduce over the whole batch (`n`, `b`) with a clamped
denominator, so all-zero samples contribute 0 instead of NaN.

Trains: the surrogate only (tokens detached, §3.4). Logged: accuracy and
`|a*|`-weighted accuracy of `argmax z = c*`.

### 4.3 Decoder probe CE (teacher stage and AE)

Decoder inputs are exact codes, so the oracle probe `p*` is always a valid label.

```
CE_probe = Σ_{n,b} |a*_{n,b}| · CE(ℓ_{n,b}, p*_{n,b}) / max(Σ_{n,b} |a*_{n,b}|, ε)
```

Trains: the decoder only. In AE the CE is computed on
`decoder(a.detach(), w.detach())`, so it never reaches the encoder or the
surrogate path. This needs a second decoder forward. The image-loss forward
keeps `(a, w)` live, because "make the code decodable" is a signal the encoder
needs. Logged: accuracy and weighted accuracy of `p̂ = p*`.

### 4.4 Image reconstruction (AE)

```
L2^{(r)} = mean (Î^{(r)} − I)²,     L_img = mean_r L2^{(r)}     # equal weights, lossless included
```

Trains: renderer, decoder (`ℓ` via §3.3), encoder (via `a`, and via `w`/`z`
through the frozen surrogate).

### 4.5 Signed mass (AE)

On the shared energy `x`, `τ = 0.1`:

```
m± = mean relu(±x)
L_signed = ((m+ − m−)/τ)² + (relu(τ − m+)/τ)² + (relu(τ − m−)/τ)²
```

Trains: encoder only. It is the only constraint on the energy distribution:
the energy is otherwise free to move away from the flow prior (the goal is to
learn a new, LPAP-compressible distribution).

### 4.6 AE total

```
L_AE = L_img + λ_s mean_lossy CE_surr + λ_d mean_lossy CE_probe + λ_m L_signed
```

Starting values: `λ_s = 0.05`, `λ_m = 0.01` (v3); `λ_d` to be set from a
magnitude probe at step 0.

`mean_lossy` terms are 0 when the pair list has no lossy pair.

Dropped from v3: energy L1. Under exact forward its target is reached exactly
when `p̂ = p*` (what `CE_probe` trains), but it also constrained encoder
amplitudes; dropping it is a deliberate removal of that constraint, to
revisit as an ablation if needed. Also dropped:
soft-scatter L1 and the adaptive CE schedule in the teacher decoder stage,
the dib-match mask.

## 5. Training stages

Stages run one at a time, each from its own notebook (§7), and are inspected
before the next one starts. There are no automatic gates and no
end-to-end pipeline runner.

| # | Stage | Data | Trains | Frozen / absent | Loss | Budget (start) |
| - | --- | --- | --- | --- | --- | --- |
| 1 | Flow | images + prior draws | flow field (EMA exported) | — | §4.1 | 20k steps, batch 256, lr 3e-4 (warmup + cosine), EMA 0.999 |
| 2 | Energy bank | images | — (no params) | flow | — | 16-step i2e encode of every image |
| 3 | Surrogates (per lossy pair) | bank rows | surrogate | — | §4.2 | v3 budgets (~960k samples per pair) |
| 4 | Decoders (per lossy pair) | bank rows, exact codes | decoder | — (no surrogate) | §4.3 | v3 budgets |
| 5 | AE (all pairs) | images | encoder, renderer, decoders; surrogates by CE only | — | §4.6 | ~120k steps, batch 64 (try 128) |

Notes:
- Stage 4 needs only `G` and exact LPAP; it does not load a surrogate. Pair
  identity (`G`) is checked when stage 5 assembles the bundle.
- Stage 5: both flow copies start from stage 1. Surrogates and decoders start
  from stages 3 and 4. Inference = exact LPAP + decoder + renderer, which
  is exactly the stage-5 forward.
- Budgets are starting points; runs are monitored live and stopped by hand.

## 6. Metrics and evaluation

- Fixed deterministic 95/5 train/validation split of the image set: a pure
  function of the image count and a fixed seed (`data.split_indices`). The
  energy bank keeps image order, so its rows are split the same way.
- AE best checkpoint: lowest validation **mean image L2 over all pairs**
  (lossless included).
- Lean logging to start. Per step and per validation:
  - every loss term (per pair where applicable);
  - per-pair image L2;
  - surrogate accuracy and `|a*|`-weighted accuracy;
  - decoder probe accuracy and `|a*|`-weighted accuracy.
- More metrics (compressibility fraction, gradient norms, energy stats) are
  added only when a question needs them.
- Galleries: a stage may write `gallery.png` after each validation; it is
  shown live in the notebook and uploaded with the checkpoint, so humans and
  agents review the same image (`scripts/review.py` pulls runs from the hub).
- Train metrics are averaged over each log window.

## 7. Execution platform (pixi + molab, notebook-first)

- **Local:** pixi for env, tests, lint and a local CPU notebook preview.
- **Remote:** a single molab (marimo) notebook, so there is one pairing
  prompt. It is organised in sections: a setup cell, then one section per
  stage (§5), each with one run cell. All stage logic lives in the package;
  cells only pick a config, call it and display progress.
- **No background processes.** Training runs in a visible cell and stops on
  a step target or a wall-clock budget. A kernel interrupt
  (`KeyboardInterrupt`) is caught: save the checkpoint, upload it, exit
  cleanly. Resuming means re-running the cell. This covers the 12 h session
  limit and runs that span several sessions.
- **Install on molab:** pip install the package from a git ref. Configs ship
  inside the package, so nothing needs syncing.
- **Config:** TOML files inside the package (`importlib.resources`), loaded
  by one strict generic loader into frozen dataclasses (unknown or missing
  key = error). The notebook may override fields. The resolved config is
  stored in every checkpoint.
- **Artifacts:** a public Hugging Face model repo (`configs/storage.toml`). Checkpoint and log are
  uploaded together in one atomic commit. History is squashed when it grows.
  `last` = weights + optimizer + RNG + step; `best` = weights only.
- **Logs:** SQLite per run (v3 format idea), uploaded with the checkpoint.
- **Notifications:** Pushover at the end of a run (target, budget or interrupt).
- **Agent:** free to pair and act by default; guidelines get tightened only
  if a problem shows up.

## 8. Repo layout and engineering rules

```
lpap_v2/
  pixi.toml  pyproject.toml  README.md  AGENTS.md  FOUNDATION.md
  docs/                      # the only docs folder
  notebooks/molab.py         # the single marimo notebook (§7)
  src/lpap/
    geometry.py   # Pair, lane table G, collision sets s(b,d,p)
    lpap.py       # exact operator, torch reference
    lpap_triton.py
    data.py       # images, Hilbert, 95/5 split, energy bank
    flow.py       # field, Euler integration
    surrogate.py
    decoder.py
    codec.py      # §3: frontend ST, split probe ST, two-role surrogate call
    losses.py     # §4
    stages/       # flow, bank, surrogate, decoder, ae: one train loop each
    runtime/      # config loader, checkpoint, hub, sqlite log, notify, run loop
    configs/*.toml
  tests/          # one test module per src module
```

Rules:
- Every §3 behaviour is tested in `tests/test_codec.py`: forward equals the
  exact code bit-for-bit (fp32 and bf16); with `z` held constant, the direct
  `V` gradient of `a` is nonzero only at the winning coordinate (the `z` path
  is tested separately); surrogate params receive no reconstruction
  gradient; CEs never reach the encoder; split probe ST amplitude gradient
  is `g_p̂`; `w` receives gradient through `u`.
- jaxtyping shape annotations on tensor APIs.
- Ruff lint + format; pytest.
- Precision: bf16 autocast for the transformers and the flow training
  forward (`torch.compile`d, ~3× throughput); flow integration, flow
  validation and LPAP in fp32.
- No backward compatibility: artifacts are regenerated when a schema changes.

## 9. Carry-over from v3 and parking lot

Ported from `lpap-molab` (`lpap-v3-priority1`), adapted to the lane table:
- exact LPAP torch reference and Triton kernel (`ops.py`, `triton_ops.py`),
  with their parity tests;
- Hilbert curve and image dataset fetch;
- `DilatedConvFlow1d` and Euler integration;
- transformer block with circular RoPE and boolean mask.

Rewritten: everything else (frontend, decoder, losses, training loops,
runtime, notebook).

Parking lot (not in the first version):
- Identity permutation with row-major (contiguous Hilbert patches) vs
  column-major lanes.
- Extra metrics: compressibility fraction `Σa*²/Σx²` per pair, gradient
  norms, energy statistics, galleries.
- Re-probing model sizes; `torch.compile`; AE batch 128.
- Exploration noise on the surrogate (Gumbel) if the ST gradient proves too weak.
- LPAP-regularised recurrent states (v3 README "future experiments").
