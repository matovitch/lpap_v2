# EXPERIMENTS

Lab journal for the AE stage. [FOUNDATION.md](FOUNDATION.md) says *what* the system is; this file says what we
tried, why, and what we learned. Each study gets its hypothesis, arms and decision rule **before** it runs;
results are appended, never rewritten. Negative results stay.

## Protocol

- **Arms** live in [`src/lpap/experiments.py`](src/lpap/experiments.py). Each arm is one fresh AE run from the
  stage 1/3/4 teachers, in its own HF run folder (`matovitch/lpap-v2-runs/<run>/log.sqlite`).
- **Execution**: the notebook `study_train` cell runs a study's arms in sequence, in the foreground.
- **Rungs (successive halving)**: every arm of a study shares one LR schedule (`run.steps`) and stops at its
  rung (`run.stop_at`). Promoting an arm means raising its `stop_at` in a commit, and the run resumes. We
  promote any arm that is not clearly worse; when unsure, spend the compute.
- **Seeds** (`seed` = data order + noise; same teachers): at least 2 per arm at rung 1 and 3 for finalists.
  A paired gap on one seed is preliminary, even when large: paired runs share initialization and data order,
  so trajectory luck can create a gap. Call an effect only when it exceeds the seed spread of both arms and
  has the same sign on a second seed.
- **Metrics** (validation, fixed 256 images, noise-free forward):
  - primary: `image_l2` (mean over pairs) and `image_l2_<pair>`;
  - compressibility: `kept_share_<pair>`, `probe_weighted_accuracy_<pair>`;
  - geometry: `stage1_corr` (Pearson correlation of the energy with the frozen stage-1 energy; 1 = has not
    moved), `support_entropy_<pair>`, `usage_kl`, `energy_top10`;
  - gathering footprint (Jacobian rows `∂x_i/∂input`; live metric uses 16 sites × 4 validation images):
    - `gather_area` = participation ratio, about the number of input samples one energy coordinate uses;
    - `gather_span` = rms distance of those inputs along the 1-D signal;
    - `*_stage1` = the frozen stage-1 values, as reference.

    It tests whether the encoder learns local, JPEG-like transforms of patches (area > 1) and whether those
    patches grow. The live estimate is noisy; use `scripts/footprints.py` (64 images × 128 sites) for
    quantitative footprint claims, reporting distributions (at least median and spread), not only means;
  - routing: `grad_<term>_<module>`.
- **Galleries**: use the same fixed 8 images and row order for every comparison. Show each run's energy and
  kept coefficients divided by that run's energy RMS, with a shared ±4 RMS color scale; include c128 and c512
  renders and their L2s. Save and link one gallery per rung being interpreted.
- **Compute**: about 3 steps/s at batch 64 (RTX PRO 6000 Blackwell, molab), so a 5k rung takes about 25 min per arm.
  molab is free: when in doubt, add seeds rather than guess.
- **molab etiquette** (an account ban was appealed once): use a fresh session for each long batch, size the
  batch to finish well before the 12h session limit (target ≤ 6h), leave a break between batches, and never
  chain a new batch automatically. Debug locally first.

## Reference: baseline `ae` (35k schedule)

Plain AE (FOUNDATION §4.6). λ_m = 0.01 until step 15k, then 0. Stopped at about 23k by choice. It is a
reference curve only: its schedule (35k) and signed-mass history differ from the study arms. Step 15k:
image L2 0.0075, kept_share c128/c256/c512 = 0.88/0.95/0.99. The gallery at 17k shows the energy is a blotchy,
low-frequency signed copy of the image: a local, one-scale concentration near the stage-1 start ("sand heaps").

## S1: annealed relative noise (4 encoder steps)

**Question.** Is the baseline a local optimum near the stage-1 energy that early noise lets the encoder escape?

**Intervention.** During training only, LPAP and the renderer see `x̃ = x + σ(step)·rms(x).detach()·ε` for
every pair, including lossless. σ decays linearly from σ0 to 0 over 2,500 steps. Noise relative to rms(x)
cannot be beaten by inflating the energy. The intended pressure: small heaps drown, so tall and separated
heaps pay off, and noisy LPAP choices early give exploration.

**Arms.** σ0 ∈ {0 (control), 0.1, 0.3, 1.0}, × seeds {2987, 2988}. Schedule 15k, rung 1 = 5k. The decay is
fixed at 2.5k, so every arm has 2.5k noise-free steps before rung 1.

**Predictions** (written before running):
- H1: noise lowers `stage1_corr` in proportion to σ0. If it does not, the noise is not moving anything.
- H2: at rung 1, some σ0 > 0 has lower `image_l2` than control, beyond the seed spread. The likeliest gain is
  at c128 (largest drop).
- H0 (null): the gap closes after the decay. That would mean the basin is not the bottleneck (or 4 steps
  cannot express a different one; see S3).
- H3 (Camille): if a global basis is out of reach in this landscape, the encoder may still find a local one,
  like JPEG's per-tile DCT. More noise then makes the patches grow: `gather_area` / `gather_span` increase
  with σ0, and larger patches go with better c128 (the strongest compression).

**Decision rule.**
- Promote control and every σ0 arm not worse than control beyond the seed spread to 15k, and add seed 2989
  for the finalists.
- If no arm separates at 15k, S1 is negative for 4 steps. Record it, then run S3 before discarding noise.

**Results, rung 1 (5k, 2 seeds, 16:25).** Mean ± half the seed gap. There were no restarts in any of the 8 runs.

| σ0 | image_l2 ×1e3 | c128 | c512 | lossless | kept c128 | probe acc c128 | stage1_corr | 4k→5k slope |
|---|---|---|---|---|---|---|---|---|
| 0 | 10.98 ± 0.06 | 23.08 | 5.65 | 0.99 | 0.755 | 0.486 | 0.51 | −4.7% |
| 0.1 | **10.75 ± 0.07** | 22.59 | 5.51 | 0.92 | 0.757 | 0.487 | 0.52 | −4.5% |
| 0.3 | **10.76 ± 0.19** | 22.55 | 5.50 | 1.06 | 0.745 | 0.498 | 0.49 | −3.7% |
| 1 | 11.09 ± 0.17 | 23.21 | 5.56 | 1.53 | 0.689 | **0.540** | 0.60 | **−8.0%** |

- **H2 (weak positive):** σ0 0.1 and 0.3 are about 2% below control at every pair. That is beyond the control
  seed spread (±0.6%) but small in size. σ0 1 is behind (by about 1%, within its spread) and still descending
  twice as fast: it trades early loss for something not yet visible.
- **H1 rejected as stated:** `stage1_corr` is not monotone in σ0. σ0 1 stays *closest* to stage 1 (0.60). Strong
  early noise seems to slow the drift away from stage 1, not speed it up.
- **σ0 1 is a different regime:** it has the lowest kept_share c128 (0.69 vs 0.76, i.e. less concentration) but the
  best probe accuracy (0.54). Its energy is more spread yet more predictable.
- **Footprints (offline, 64 images × 128 sites, `footprints_s1_all.png`):** area medians are 21–25 for control,
  22–28 for σ0 0.1, 25–29 for σ0 0.3 and 20.5–22 for σ0 1. The seed spread within an arm is as large as the gaps
  between arms, so **H3 is not supported at 5k**. σ0 1 is the clear outlier in *shape*: it is the most uniform
  (cv 0.41–0.46 vs 0.51–0.61), has the widest span (median 18.5–19.5 vs 13–21) and few small patches (p10 13–14.5).
  Both seeds agree, so this is a real effect of strong noise. Whether it persists or is only lag is H3-alt's question at 15k.
- **Decision (pre-registered rule):** no arm is worse than control beyond the seed spread, so **all 4 arms go to
  15k**. That is about 4h per seed batch, run as studies `s1-r2-s2988` then `s1-r2-s2987` in separate molab
  sessions with a break between. Seed 2989 is added only for the finalists after 15k.

*Earlier interim notes (kept for the record):*

*Interim check, 14:15, seed 2987 only (control, σ0 0.3, σ0 1 done):*
- **Sanity checks pass:** `noise_sigma` decays linearly to 0 at 2,500, each arm has the right config, there was one session per run (no restarts) and `grad_norm` is unremarkable.
- **Validation `image_l2` at 5k:** 0.01104 / 0.01095 / 0.01092. The gaps are under 1%, which is probably within the seed spread.
- **σ0 1:** it was far behind at 2.5k (0.0141 vs 0.0126) and caught up by 5k, so its curve is still steeper. It also has the best c512 L2 (0.00553 vs 0.00569) and probe weighted accuracy (c128: 0.55 vs 0.47). This trend makes it the arm to watch at 15k.
- **H1:** `stage1_corr` is not monotone in σ0. At 5k, σ0 1 is the *highest* (0.59 vs 0.53 control).
- **Measurement caveat for H3:** `gather_area` uses only 4 images × 16 sites and jumps by about 30% between validations (σ0 0.3: 23.8 → 36.6). It cannot settle H3 as it stands. Plan: recompute it offline on the rung checkpoints with more images and sites, and report the median as well as the mean. The live metric is kept unchanged during S1 so all arms are measured the same way.
- *Offline distribution (`scripts/footprints.py`, 64 images × 128 sites, fp32, local GPU):* the area mean, median,
  p10 and p90 come out as 7.1 / 6.6 / 5.3 / 9.1 for stage 1, 24.9 / 21.3 / 11.3 / 43.9 for the control, 29.8 / 25.0 / 10.8 / 56.0
  for σ0 0.3, and 24.6 / 22.2 / 14.5 / 37.7 for σ0 1. Stage 1 is narrow (cv 0.26). The AE gives a broad, right-skewed spread
  (cv 0.4–0.6): coordinates differ, with some gathering more than 2 rows of the image (> 64 samples). σ0 1 is the
  *most uniform* (cv 0.41) and the widest in span (median 18.5 vs 14.5). σ0 0.3 has the heaviest tail. Seed 2988 is
  needed before any H3 reading.
- *H3-alt (Camille): footprints depend on training progress, not on noise.* Area grows as the AE trains (stage 1
  is AE step 0), and noise only delays that growth, so the end-state distributions match. Test: compare
  histograms at 5k and 15k for every arm, and plot area against `image_l2` instead of step. If H3-alt holds, the
  noise arms at 15k look like the control at 15k and the area-vs-loss curves overlap. The 5k checkpoints are kept
  locally (`/tmp/s1/ckpt5k`) because resuming to 15k overwrites `last.pt`.

*Interim check, 15:40, 6 of 8 arms at 5k:* validation `image_l2` is 0.01104 / 0.01092 for the control (seed spread
≈ 1%), 0.01095 / 0.01058 for σ0 0.3 (mean 2% below control), 0.01082 for σ0 0.1, and 0.01092 for σ0 1. Every arm is
within about 2× the seed spread, with no outlier and no restart. σ0 0.3 leads weakly, and σ0 1 keeps the best probe accuracy.

*Rung 2 results, seed 2988 at 15k (22:36).* Validation `image_l2` ×1e-3 at 5k / 7.5k / 10k / 12.5k / 15k:

| σ0 | 5k | 7.5k | 10k | 12.5k | 15k | vs control | c128 | c512 | probe wacc c128 | mean abs energy at 15k |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 10.92 | 9.96 | 9.38 | 9.16 | 9.05 | — | 19.04 | 4.64 | 0.595 | 0.23 |
| 0.1 | 10.68 | 9.63 | 9.12 | 8.88 | 8.75 | −3.3% | 18.28 | 4.56 | 0.626 | 0.23 |
| 0.3 | 10.58 | 9.56 | 9.13 | 8.83 | 8.70 | −3.9% | 18.37 | 4.43 | 0.619 | 0.32 |
| 1 | 11.26 | 9.41 | 8.72 | 8.36 | **8.15** | **−10.0%** | 17.74 | **3.78** | **0.655** | 0.69 |

- σ0 1 was the *worst* arm at 5k, passed the control by 7.5k, and its lead keeps growing (−7% at 10k, −9% at 12.5k).
  A −10% gap is about 3× the 0.1/0.3 gaps, so trajectory luck is a much less plausible reading for this arm.
  σ0 0.1 and 0.3 remain within what luck might explain.
- The energy is still inflated at 15k (0.69 vs 0.23), even though the noise ended at 2.5k.
- Offline footprints (64 images × 128 sites), area mean / median / cv and span median, 5k → 15k:
  control 30 / 25 / 0.61 → 41 / 33 / 0.60, span 17 → 23; σ0 0.1 32 → 53; σ0 0.3 34 → 51;
  **σ0 1 23 / 20.5 / 0.46 → 26 / 21 / 0.61, span 19.5 → 14**. The other arms' footprints grow by ~50% with
  training. σ0 1's stay small and become *more local* (span shrinks), while it has the best loss.
- **H3-alt (footprint follows convergence) is rejected on this seed:** the best-converged arm has the smallest
  footprints, so area is not a function of loss. **H5's "more uniform" part does not persist** (cv back to 0.61); what persists is
  "small, local footprints". The original H3 (noise grows footprints) is also contradicted for σ0 1.
- Gallery (each run divided by its own rms): the σ0 1 energy is sparse and follows edges, and its c128 survivors sit on object
  contours. Its c128 renders are made of fewer, larger flat regions that follow object boundaries. The other three arms still look alike.
- **Gate:** passed (all three arms beat control by more than 1%), so `s1-r2-s2987` runs next. It is the decisive test for σ0 1.

*Observation, 21:05, seed 2988 (not pre-registered): noise inflates the energy.* Train mean |energy| is about
0.25 in the control throughout. It rises to 0.47 (σ0 0.3) and 1.31 (σ0 1) while the noise is on, then decays
once σ reaches 0 (σ0 1: 1.03 at 5k). The noise is relative (scaled by a detached rms), so inflating doesn't
reduce it. Inflation is therefore a side effect, not an escape: it is probably driven by the renderer or the surrogate. In the 5k gallery, with
each run divided by its own rms, the control, σ0 0.1 and σ0 0.3 show the same scattered texture-like layout.
σ0 1 is a different solution: the energy follows object edges and structure, and its c128 survivors cluster
on objects. Its 8-image render L2 is no better at 5k. The 15k check is whether this layout persists and pays off.

*Rung 2 results, seed 2987 at 15k (2026-11-25, 09:40; study `s1-r2-s2987`, control and σ0 1 only).* Validation
`image_l2` ×1e-3 at 5k / 7.5k / 10k / 12.5k / 15k:

| σ0 | 5k | 7.5k | 10k | 12.5k | 15k | vs control | c128 | c512 | probe wacc c128 | energy std at 15k |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 11.04 | 9.96 | 9.37 | 9.10 | 9.01 | — | 19.01 | 4.63 | 0.592 | 0.41 |
| 1 | 10.92 | 9.23 | 8.68 | 8.33 | **8.12** | **−9.9%** | 17.66 | **3.75** | **0.650** | 1.13 |

- **σ0 1 is confirmed on two seeds.** Paired gap at 15k: −10.0% (seed 2988) and −9.9% (seed 2987). The control seed
  spread at 15k is 0.4% (9.05 vs 9.01), so the gap is about 25× the spread. c512 (−19%), c128 (−7%) and probe
  weighted accuracy (+0.06) improve the same way on both seeds. The trajectories differ only before 7.5k: σ0 1 is
  worst at 5k on seed 2988 and slightly ahead on seed 2987.
- Energy is inflated on this seed too (std 1.13 vs 0.41). The energy-scale check is still open.
- Offline footprints (64 images × 128 sites), area mean / median / cv and span median, 5k → 15k:
  control 24.9 / 21.3 / 0.54 → 43.0 / 36.0 / 0.57, span 14.5 → 20.0;
  **σ0 1 24.6 / 22.2 / 0.41 → 25.8 / 21.1 / 0.56, span 18.5 → 13.3**. Same pattern as seed 2988: the control's
  footprints grow by ~70%, while σ0 1's stay at the same area and become more local. **The H3-alt rejection and the
  "small, local footprints" reading now hold on both seeds.**
- Gallery (standard format): σ0 1's c128 survivors sit on object contours (the insect, the figure, the dog), and its
  c128/c256 renders keep the object shapes. The control's survivors are scattered and texture-like. This is the
  same qualitative split as seed 2988.
- **S1 conclusion:** a strong early exploration noise (σ0 1, relative, 2.5k linear decay) finds a different and better
  energy layout, reproducibly. σ0 0.1 and 0.3 were not confirmed (seed 2987 kept at 5k by design). This motivates S2:
  get the same exploration without a hand-tuned schedule.

## Post-S1 follow-ups (done)

- ~~**S1 confirmation: seed 2987 to 15k**~~ **Done 2026-11-25:** σ0 1 confirmed (−9.9%, see the seed 2987 results).
  Ran `s1-r2-s2987` with control and σ0 1 only, in one molab session.
  - **Gate** (fixed before the 15k results): if no σ0 arm of `s1-r2-s2988` beat control by more than 1% at 15k,
    seed 2987 would be skipped. The gate passed (see the rung 2 results).
  - *Caveat (Camille, 20:56): trajectory luck.* Paired arms share init and data order, then drift apart, so a
    single-seed gap is weak evidence. The sign must repeat on seed 2987. The control seed spread at 15k may also
    exceed the 1% measured at 5k.
- ~~**Energy-scale check (code reading, no compute).**~~ **Done 2026-11-25** (code reading + 15k `s2988` checkpoint measurements):
  - *Invariant / equivariant:* exact LPAP (no absolute thresholds; `amplitudes` linear, discrete indices invariant),
    `Surrogate` (`tokens / rms` with `rms_eps = 1e-6`, verified to 4 decimals at 0.33× and 3×), `weighted_ce`
    (scale cancels in weights), `usage_kl`, `residual_share`, `gather_footprint`, `stage1_corr`, `energy_top10`.
    Gradient clipping (`max_grad_norm = 1.0`) never triggers (0/600 logged steps).
  - *Scale-sensitive:*
    1. **`Decoder.forward`** has **no RMS normalization on `amplitudes`** (unlike `Surrogate`), and at `lr = 2.8e-5`
       the Stage-4 input weights barely move (`amp.w` norm 8.42 vs 8.34). At 3× energy scale, `‖amp_wx‖ / ‖lane‖`
       jumps from 0.45 → 1.39 (`c128_k16`) and 0.25 → 0.77 (`c512_k32`) before the first `LayerNorm`.
    2. **`FlowField.input`** in the renderer (`Conv1d(1, 128, 1)` into `GroupNorm(1, 128)`) has virtually unchanged
       `weight` (`0.630`) and `bias` (`0.600`), so `wx_rms / b_rms` jumps from 0.41 → 1.22 (`c128_k16`) and 0.40 → 1.24
       (`c512_k32`), raising the input signal variance fraction inside `block[0]` ~4× (from ~15% to ~60%). Meanwhile
       the Euler skip forces ~2.8× larger renderer velocities to cancel bigger spikes back into `[0, 1]`.
    3. **`signed_mass`** scales as $\alpha^2$ above `tau = 0.1`, but `signed_mass_weight = 0.0` (inactive in S1).
    4. **AdamW `weight_decay = 0.01`** (PyTorch default) is why `sig1`'s energy slowly relaxes once noise ends
       (`std` 1.84 at 2k → 1.52 at 5k → 1.21 at 15k).
  - *Why noise inflated the energy:* in `_forward`, `rms = energy.detach().square().mean(-1, keepdim=True).sqrt()`.
    Because `rms` is **detached**, the renderer damps its gain by $\sim 1/(1 + \sigma^2)$ to attenuate the noise
    (`l2_lossless` spikes to 40.0 at step 25 and drops to 13.4 by step 250 *before* energy inflates), while the
    encoder's gradient ignores the detached noise-variance cost and pushes `|x|` up to restore image contrast
    (`mean |x|` 0.29 at step 250 → 1.30 at step 1000). With $\sigma > 0$, renderer and encoder have no joint
    equilibrium until $\sigma \to 0$. Under an anytime noise floor ($\sigma_\infty > 0$), this would inflate energy
    indefinitely — **fix implemented:** `add_relative_noise` keeps `rms` live (`.add(1e-12).sqrt()`) so encoder and
    renderer optimize the same expected noisy objective. Verified locally (500 steps, batch 4, σ0 = 1, 5060):
    detached `rms` reversed and began inflating after step 250 (`val_e_std` 0.357 → 0.329 → 0.337 → 0.397), while
    live `rms` settled cleanly at the joint equilibrium (`val_e_std` 0.357 → 0.285 → 0.221 → 0.226, lower lossless L2).
  - *Verdict:* the basin change is genuine exploration (footprints and RMS-normalized galleries are strictly
    scale-invariant, and scaling `sig1`'s energy down 0.33× into its frozen `c128` decoder still edges out control:
    `probe_ce` 1.080 vs 1.098), **but** the unnormalized `Decoder` and `Renderer` inputs mean part of the −10% /
    +0.06 magnitude could be a scale confound. Fixing `rms` stops noise-driven inflation going forward.

## S2: anytime training (Schedule-Free AdamW + gradient-driven noise)

**Question.** Can we remove both step-count dependencies (cosine LR down to `run.steps`, linear noise decay over
`noise_decay_steps`) so training is an unbounded process truncated by compute, while matching or beating S1's
`σ0 = 1` basin without energy inflation?

**Intervention.**
1. **Optimizer:** Schedule-Free AdamW (`schedulefree.AdamWScheduleFree`, Defazio et al. 2024) with
   `lr = 2.8e-5`, `warmup_steps = 500`, `weight_decay = 0.01` (matching PyTorch `AdamW`'s default;
   `AdamWScheduleFree` defaults to `0.0`). `optimizer.train()` runs during `train_step`; `optimizer.eval()` runs
   before `validate`, `footprint`, `gradient_norms`, `gallery`, `state_dict`, and `weights` so evaluations and
   checkpoints always use the averaged iterate $x_t$.
2. **Live-`rms` relative noise:** `x̃ = x + σ_t · rms(x) · ε` with `rms` live (fixing S1's detached-`rms`
   energy inflation).
3. **Gradient-driven noise controller (`NoiseController`):** observes clipped encoder gradients $g_t$ strictly
   *before* `optimizer.step()` (because `AdamWScheduleFree.step()` divides `p.grad` in-place). Half-life $h$ in
   steps sets $\beta = 2^{-1/h}$; two knobs ($\sigma_{\max} = 1.0$, shared $h = 500 = \text{warmup\_steps}$),
   nothing depends on `run.steps`:
   - **`grad_norm` (primary, Camille's proposal):** $n_t = \beta n_{t-1} + (1 - \beta)\|g_t\|$ (zero-init, no
     bias correction so step-0 spikes do not freeze the peak), $\sigma_t = \sigma_{\max} \cdot n_t / \max_{s \le t} n_s$.
   - **`grad_snr` (alternative):** $m_t = \beta m_{t-1} + (1 - \beta) g_t$,
     $n_t = \beta n_{t-1} + (1 - \beta)\|g_t\|$, $\sigma_t = \sigma_{\max} \cdot \|m_t\| / n_t \in [0, \sigma_{\max}]$
     (bias correction cancels identically; theoretical stochastic floor is $\sigma_{\max}\sqrt{\ln 2 / (2h)}$).

**Pre-launch feedback-loop probe (1k steps, batch 64, RTX PRO 6000, `h = 250`, `σ_max = 1.0`, seed 2988):**
- Both controllers were stable (no wild oscillation) and confirmed the live-`rms` fix at full batch size:
  zero energy inflation (`e_std`: 0.357 → 0.210 on `grad_norm` even at $\sigma \approx 1.0$ for 1,000 steps;
  0.357 → 0.314 on `grad_snr`).
- `grad_norm (h = 250)`: $\sigma$ dipped to `0.883` during warmup (step 400) and stayed `0.996–1.000` through
  step 1000 while the encoder reorganized (`s1_corr = 0.657`, compact footprint `area = 13.3, span = 10.6`,
  cumulative dose $\sum \sigma = 966$).
- `grad_snr (h = 250)`: decayed smoothly (`1.00 → 0.65 → 0.28 → 0.09 → 0.041` at step 1000), closely matching
  the random-walk floor formula $\sqrt{\ln 2 / 500} = 0.0372$. At `h = 250`, its cumulative dose was only `378`
  (comparable to S1's weak `σ0 = 0.3` arm, with wider footprint `area = 16.9, span = 15.8`).
- **Choice of shared $h = 500$:** matching `warmup_steps = 500` doubles `grad_snr`'s early exploration window
  past warmup (dose ~750–900) and lowers its late random-walk floor to $\sqrt{\ln 2 / 1000} = 0.026$, while
  keeping `grad_norm` near `1.0` during initial reorganization before decaying as $\|g_t\|$ declines, with one
  shared knob across both arms.

**Arms (`s2-s2988`, then `s2-s2987` confirmation; 3 × 15k from scratch ≈ 4.5h per session):**
- `s2-sf-control-s<seed>`: schedule-free AdamW, no noise (`σ = 0`).
- `s2-sf-gnorm-s<seed>`: schedule-free AdamW + `grad_norm` (`σ_max = 1.0, h = 500`).
- `s2-sf-gsnr-s<seed>`: schedule-free AdamW + `grad_snr` (`σ_max = 1.0, h = 500`).
- External reference: S1 15k cosine arms (`s1-control`, `s1-sig1`, noting S1 `sig1` used detached `rms`).

**Predictions** (written before running):
- **H6:** `sf-control` matches S1 cosine `control` (`image_l2` 9.01–9.04 at 15k) within or better than the seed
  spread (~0.4%), confirming the schedule-free swap alone is neutral or positive.
- **H7:** `sf-gnorm` beats `sf-control` by more than the seed spread at 15k and reproduces `σ0 = 1`'s signature
  (sparse, edge-aligned energy; compact, local offline footprints) without inflating `energy_std` or using any
  step-count schedule. Caveat to measure: if its late $\sigma_t$ floor stays high (~0.4–0.5), noisy late training
  could hurt clean-eval `image_l2` even if it reaches the right basin.
- **H7′:** `sf-gsnr` beats `sf-control` by more than the seed spread at 15k and reproduces the `σ0 = 1`
  signature while decaying near its ~0.026 stochastic floor for clean late refinement.

**Decision rule.**
- Confirm the winning noise controller (or both, if close) plus `sf-control` on seed 2987 (`s2-s2987`), plus
  offline footprints (`scripts/footprints.py`) and RMS-normalized 8-image galleries.
- If `sf-gnorm` reaches the compact basin (footprints/gallery) but loses on `image_l2` strictly because of a
  high late $\sigma$ floor, record that explicitly before deciding whether `grad_snr` becomes the default.
- If neither `sf-gnorm` nor `sf-gsnr` reproduces S1 `sig1`'s −10% gain while `e_std` stays ~0.35, that also
  directly tests how much of S1 `sig1`'s −10% came from the detached-`rms` scale inflation vs pure exploration.

*Results, seed 2988 at 15k (2026-11-25, 17:42; study `s2-s2988`, RTX PRO 6000).* Validation `image_l2` ×1e-3
at 5k / 7.5k / 10k / 12.5k / 15k, with S1 seed-2988 cosine arms as reference:

| Run | 5k | 7.5k | 10k | 12.5k | 15k | vs `sf-control` | c128 | c256 | c512 | lossless | probe wacc c128 | `σ` @ 15k | cum. dose |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `s1-control-s2988` (ref) | 10.92 | 9.96 | 9.38 | 9.16 | 9.05 | (−11.0%) | 19.04 | 11.75 | 4.64 | 0.78 | 0.595 | 0 | 0 |
| `s1-sig1-s2988` (ref, detached rms) | 11.26 | 9.41 | 8.72 | 8.36 | 8.15 | (−19.9%) | 17.74 | 10.15 | 3.78 | 0.92 | 0.655 | 0 | 1,250 |
| `s2-sf-control-s2988` | 12.21 | 11.49 | 11.04 | 10.56 | **10.17** | — | 21.53 | 13.02 | 5.24 | 0.87 | 0.518 | 0 | 0 |
| `s2-sf-gnorm-s2988` | 19.78 | 20.40 | 25.15 | 20.58 | **17.36** | +70.7% | 30.48 | 20.20 | 10.39 | 8.37 | 0.462 | 0.648 | 13,321 |
| `s2-sf-gsnr-s2988` | 12.49 | 11.85 | 11.35 | 10.85 | **10.44** | +2.7% | 21.98 | 13.51 | 5.39 | 0.90 | 0.507 | 0.0195 | 739 |

Offline gathering footprints (64 images × 128 sites, fp32) and full validation energy stats at 15k:

| Run | area mean / med / cv | span mean / med | energy rms | kurtosis | `s1_corr` | `|s1|` corr |
|---|---|---|---:|---:|---:|---:|
| stage 1 | 7.1 / 6.6 / 0.26 | 5.5 / 5.0 | 0.370 | 7.30 | +1.000 | +1.000 |
| `s1-control-s2988` | 41.3 / 33.4 / 0.60 | 25.4 / 22.8 | 0.396 | 14.55 | +0.401 | +0.094 |
| `s1-sig1-s2988` | 26.2 / 20.9 / 0.61 | 18.0 / 13.8 | 1.211 | 17.92 | +0.455 | +0.205 |
| `s2-sf-control-s2988` | 29.8 / 24.6 / 0.61 | 16.3 / 14.3 | 0.453 | 10.47 | +0.473 | +0.122 |
| `s2-sf-gnorm-s2988` | **19.4 / 15.0 / 0.68** | **16.5 / 13.8** | **0.403** | 8.42 | +0.587 | +0.287 |
| `s2-sf-gsnr-s2988` | 26.3 / 21.5 / 0.63 | 18.0 / 15.2 | 0.444 | 10.04 | +0.481 | +0.134 |

- **Live-`rms` fix confirmed at 15k:** zero energy inflation on all three S2 arms (`rms` 0.40–0.45 vs `s1-sig1`'s
  1.211), even under `sf-gnorm`'s 13,321 cumulative noise dose.
- **H6 rejected (`sf-control` lags `s1-control` by +12.4%):** reusing cosine AdamW's peak `lr = 2.8e-5` left
  Schedule-Free moving at roughly half speed from step 500 onward (`17.24` vs `16.37`). At 15k, `sf-control`
  (`l2 = 10.17`, `area = 29.8`, `span = 16.3`, `kurt = 10.47`) matches `s1-control` around step ~6.5k and is still
  falling steeply (`−0.39e-3` per 2.5k vs `−0.11e-3` on `s1-control`). Schedule-Free AdamW needs its own higher LR.
- **H7 rejected on `image_l2`, but the pre-registered caveat occurred verbatim on `sf-gnorm`:**
  - *Why `grad_norm` never annealed:* `‖g_t‖` is the norm of a *stochastic mini-batch* gradient, not the expected
    gradient; even in noiseless `sf-control`, `encoder_grad_norm` only falls 46% over 15k (`0.217 → 0.117`), so
    `n_t / n_max` has a structural floor of ~0.5–0.7. Closed-loop noise added a positive feedback loop: `egn`
    spiked from `0.093` (step 4k) to `0.388` (step 10k), keeping `σ` pinned at `0.84–1.00` through step 11k and
    still `0.65` at 15k. Because the renderer/decoders trained on 65–100% noise the whole time, clean renders
    are heavily blurred and low-contrast (`lossless = 8.37e-3`, `c128 = 30.48e-3`).
  - *Gallery & footprints:* despite the blurred renders, `sf-gnorm`'s RMS-normalized energy and `c128` survivors
    are strikingly clean, edge-aligned, and coherent across all 8 images (sharper object contours than `sf-control`
    and `sf-gsnr`), with the most compact, peaked area distribution of any run (`area 19.4 / 15.0`, `span 16.5 / 13.8`,
    `rms = 0.403`). This confirms **heavy energy noise produces the compact edge-aligned basin without energy
    inflation**, but `grad_norm` cannot turn the noise off to let the decoder/renderer refine cleanly.
- **H7′ rejected (`sf-gsnr` +2.7% vs `sf-control`):** `grad_snr` solved the stochastic-floor problem completely
  (`σ`: `0.52 @ 500 → 0.12 @ 1k → 0.024 @ 2.5k → 0.019` steady floor, zero feedback lock, clean `signed_mass = +0.129`),
  **but** its early dose was too small and too brief while Schedule-Free was moving slowly (`dose = 445` by step 1k,
  `507` by 2.5k vs `s1-sig1`'s `1,250`). In the gallery and footprint histograms, `sf-gsnr` stays in the same broad,
  scattered-dot basin as `sf-control` (`area` flat plateau, `kurt = 10.04` vs `10.47`, `s1_corr = +0.481` vs `+0.473`).
- **Decision:** skip `s2-s2987` confirmation — all three hypotheses failed cleanly on seed 2988 with unambiguous
  mechanistic causes, so replicating the current S2 settings would waste ~4.5h of GPU time. Two cleanly separable
  questions remain before an S2b anytime retry:
  1. **Schedule-Free LR:** find the LR (e.g. 2×–5× `2.8e-5`) where `sf-control` matches or beats cosine `s1-control`
     at 15k (and verify `s1-sig1` with live `rms` on cosine AdamW to isolate pure exploration gain from S1's scale
     inflation).
  2. **Noise turn-off with sufficient early dose:** `grad_snr` turns off cleanly where `grad_norm` cannot, but
     decayed before delivering S1's ~1,250 dose (and sits on a `~0.02` random-walk floor). Options: larger `h`
     and/or `σ_max` with floor subtraction `max(0, snr - c·√(ln 2 / (2h)))`, or gating a higher initial `σ` by
     `grad_snr` once SF's LR is tuned.

## S2a: live-`rms` baseline isolation + Schedule-Free LR calibration

**Questions.**
1. How much of S1 `σ0 = 1`'s −10.0% gain on seed 2988 (`8.15` vs `9.05`) comes from pure exploration vs S1's
   detached-`rms` energy inflation (`rms = 1.21` vs `0.40`)?
2. What peak LR lets noiseless Schedule-Free AdamW (`σ = 0`) match or beat cosine AdamW (`9.05` at 15k)
   without instability?

**Local pre-flight probe (100 steps, batch 4, warmup 20, RTX 5060, seed 2988):**
- All four Schedule-Free LRs were stable with lower peak `grad_norm` at higher LR and monotonically lower
  100-step `val_l2`: `adamw 1×` (`2.8e-5`) `23.79` (`max_gn 2.15`), `sf 1×` `23.68` (`2.61`), `sf 3×` (`8.4e-5`)
  `22.32` (`1.86`), `sf 6×` (`1.68e-4`) `20.31` (`2.05`), `sf 10×` (`2.8e-4`) `20.15` (`1.45`).

**Arms (`s2a-s2988`, seed 2988, 4 × 15k ≈ 5.5h in one molab session):**
- `s2a-live-sig1-s2988`: exact S1 `sig1` recipe (AdamW cosine, `lr = 2.8e-5`, linear `σ0 = 1.0` over 2.5k),
  now with live `rms`. Compared directly against `s1-control-s2988` (`9.05`) and `s1-sig1-s2988` (`8.15`).
- `s2a-sf-lr3x-s2988`: Schedule-Free AdamW, `lr = 8.4e-5` (3×), `warmup_steps = 500`, `σ = 0`.
- `s2a-sf-lr6x-s2988`: Schedule-Free AdamW, `lr = 1.68e-4` (6×), `warmup_steps = 500`, `σ = 0`.
- `s2a-sf-lr10x-s2988`: Schedule-Free AdamW, `lr = 2.8e-4` (10×), `warmup_steps = 500`, `σ = 0`.
  (Paired with `s2-sf-control-s2988` at 1× = `2.8e-5` → `10.17`.)

**Predictions** (written before running):
- **H8 (exploration without inflation):** `s2a-live-sig1-s2988` keeps `energy_rms ~ 0.40–0.45`, reproduces the
  compact edge-aligned basin (footprints & gallery), and beats `s1-control-s2988` (`9.05`) by > 2% (5× seed
  spread), though possibly less than detached-`rms` `s1-sig1`'s −10.0% if scale inflation contributed to the
  unnormalized decoder/renderer gain.
- **H9 (Schedule-Free LR):** at least one of `{3×, 6×, 10×}` matches or beats `s1-control-s2988` (`9.05`) at
  15k cleanly (`lossless ≤ 0.85e-3`, no divergence), giving the calibrated anytime baseline for S2b (anytime
  noise retry). Also watch whether high-LR SF alone escapes the scattered-dot basin (footprints/gallery) —
  answering the planned "stuck vs slow" LR control simultaneously.

**Decision rule.**
- Pick the best stable Schedule-Free LR (lowest 15k `image_l2` with clean `lossless` and monotone trajectory)
  as the S2b anytime optimizer baseline.
- If `s2a-live-sig1` keeps ≥ ~4% gain over `s1-control` with compact footprints, proceed to S2b (re-tuning
  anytime `grad_snr` dose + floor subtraction on top of the calibrated SF LR). If `s2a-live-sig1` collapses
  back near `s1-control` despite compact footprints, inspect the unnormalized `Decoder`/`Renderer` input scale
  before adding more exploration noise.

*Results, seed 2988 at 15k (2026-11-26, 00:00; study `s2a-s2988`, RTX PRO 6000).* Validation `image_l2` ×1e-3
at 5k / 7.5k / 10k / 12.5k / 15k, with S1 cosine and S2 `sf 1×` arms as reference:

| Run | 5k | 7.5k | 10k | 12.5k | 15k | vs `s1-ctrl` | c128 | c256 | c512 | lossless | `kept128` | `pwacc128` | `ukl` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `s1-control-s2988` (ref) | 10.92 | 9.96 | 9.38 | 9.16 | 9.05 | — | 19.04 | 11.75 | 4.64 | 0.78 | 0.820 | 0.595 | 0.179 |
| `s1-sig1-s2988` (ref, detached rms) | 11.26 | 9.41 | 8.72 | 8.36 | 8.15 | −10.0% | 17.74 | 10.15 | 3.78 | 0.92 | 0.819 | 0.655 | 0.250 |
| `s2-sf-control-s2988` (ref, SF 1×) | 12.21 | 11.49 | 11.04 | 10.56 | 10.17 | +12.4% | 21.53 | 13.02 | 5.24 | 0.87 | 0.770 | 0.518 | 0.088 |
| `s2a-live-sig1-s2988` (live rms) | 12.39 | 11.36 | 10.08 | 9.65 | **9.40** | +3.9% | 19.22 | 12.04 | 5.07 | 1.28 | 0.753 | 0.630 | 0.143 |
| `s2a-sf-lr3x-s2988` (SF 3×) | 10.30 | 9.44 | 8.78 | 8.38 | **8.09** | −10.6% | 17.10 | 10.44 | 4.12 | 0.69 | 0.865 | 0.671 | 0.201 |
| `s2a-sf-lr6x-s2988` (SF 6×) | 8.44 | 7.53 | 6.94 | 6.57 | **6.33** | **−30.1%** | **13.65** | **8.12** | **3.05** | 0.50 | **0.898** | 0.782 | 0.435 |
| `s2a-sf-lr10x-s2988` (SF 10×) | 7.95 | 7.33 | 6.94 | 6.71 | **6.49** | **−28.3%** | 13.95 | 8.33 | 3.19 | **0.47** | 0.868 | **0.804** | 0.330 |

Independent fp32 forensic audit (uncompiled exact LPAP on `valA` and 256 completely unseen validation images
`valB`), offline gathering footprints (64 images × 128 sites, fp32), and relative weight displacement from
Stage 1/3/4 init (`‖W − W₀‖ / ‖W₀‖`) at 15k:

| Run | `valB` L2 (`c128`) | energy rms | `s1_corr` | `maxP` (`>50%` / `>80%`) | `bEnt` / `bMaxP` | `area` mean / med / cv | `span` mean / med | `dW` enc / ren / dec / sur |
|---|---:|---:|---:|---:|---|---|---|---:|
| stage 1 | — | 0.370 | +1.000 | — | — | 7.1 / 6.6 / 0.26 | 5.5 / 5.0 | 0 / 0 / 0 / 0 |
| `s1-control-s2988` | 8.94 (19.15) | 0.396 | +0.406 | 0.38 (0 / 0) | 0.738 / 0.180 | 41.3 / 33.4 / 0.60 | 25.4 / 22.8 | 5.4% / 9.7% / 6.3% / 7.6% |
| `s1-sig1-s2988` | 7.99 (17.50) | 1.211 | +0.465 | 0.60 (4 / 0) | 0.730 / 0.198 | 26.2 / 20.9 / 0.61 | 18.0 / 13.8 | 6.9% / 10.2% / 6.4% / 8.0% |
| `s2a-live-sig1-s2988` | 9.32 (19.22) | 0.391 | +0.491 | 0.40 (0 / 0) | 0.738 / 0.180 | 32.1 / 28.5 / 0.46 | 22.9 / 21.5 | 5.5% / 9.4% / 6.2% / 7.8% |
| `s2a-sf-lr3x-s2988` | 7.83 (16.81) | 0.450 | +0.377 | 0.27 (0 / 0) | 0.738 / 0.181 | 40.5 / 32.8 / 0.60 | 20.1 / 18.0 | 9.9% / 17.4% / 11.4% / 15.1% |
| `s2a-sf-lr6x-s2988` | **6.03 (13.15)** | 0.538 | +0.317 | 0.72 (4 / 0) | 0.969* / 0.203 | **29.5 / 25.4 / 0.50** | **13.8 / 12.3** | 15.0% / 24.4% / 16.0% / 22.3% |
| `s2a-sf-lr10x-s2988` | **6.14 (13.33)** | 0.513 | +0.281 | **0.38 (0 / 0)** | 0.965* / 0.214 | 69.2 / 63.4 / 0.40 | 28.6 / 27.0 | 20.9% / 31.7% / 20.3% / 30.2% |

*\*Note on `bEnt`: `s2a-sf-lr6x` and `s2a-sf-lr10x` rows report exact per-bucket `code.probes` normalized entropy
(`0.969 / 0.965`, `bMaxP = 0.203 / 0.214`); on `s2a-sf-lr6x`, the same metric measured on `sources % 16` gave
`0.727` vs `0.738` on `s1-control` — no fixed-support collapse under either definition.*

- **H8 rejected (`s2a-live-sig1-s2988`: `9.40`, +3.9% vs `s1-control`):** fixing detached-`rms` energy inflation
  kept `rms = 0.391` (vs `s1-sig1`'s `1.211`) and erased *all* of S1 `sig1`'s −10.0% gain, leaving `s2a-live-sig1`
  behind `s1-control` across every pair (`lossless = 1.28` vs `0.78`, `kept128 = 0.753` vs `0.820`). It still
  retained modestly tighter footprints (`area 32.1`, `span 22.9` vs `41.3 / 25.4`) and clean edge-aligned energy
  in the gallery, strongly confirming the energy-scale audit hypothesis: S1 `sig1`'s −10% gain at `lr = 2.8e-5`
  was largely an indirect input-gain boost from 3× `rms` inflation into the unnormalized `Decoder`/`Renderer`,
  while pure noise at a slow LR spent 2.5k of the 15k cosine budget without catching up.
- **H9 strongly confirmed (`s2a-sf-lr6x`: `6.33`, −30.1%; `s2a-sf-lr10x`: `6.49`, −28.3%; `s2a-sf-lr3x`: `8.09`,
  −10.6%):**
  - *Real, not a bug:* independent fp32 exact-LPAP matches logged `valA` to 3 digits and generalizes even better
    to unseen `valB` (`6.03` for `6×`, `6.14` for `10×`); 8-image galleries show visibly sharper `c128` (`15.6e-3`
    / `15.4e-3`) and `c256` (`8.9e-3` / `9.0e-3`) renders across all images; `rms` stays `0.51–0.54`, parameter
    norms are unchanged, `support_entropy = 0.979 / 0.975`, and a constant per-bucket majority prior only gets
    `bMaxP = 20.3% / 21.4%` while the decoder reaches `pwacc128 = 78.2% / 80.4%` (`swacc128 = 96.3% / 96.5%`).
  - *Why it happened ("slow", not "stuck"):* at `lr = 2.8e-5`, the baseline AE barely moved off initialization
    (`enc_dW = 5.4%`, `dec_dW = 6.3%`). Raising LR under Schedule-Free (`3× → 6× → 10×`) let all four modules
    co-adapt (`15–32%` relative weight displacement), answering the planned "stuck vs slow" control decisively.
  - *Crossover and two distinct coding regimes (`lr6x` vs `lr10x`):* `lr10x` led early by −9.6% at step 3k
    (`8.92` vs `9.86`), tied `lr6x` at step 10k (`6.94` vs `6.94`), and finished +2.5% behind at 15k (`6.49` vs
    `6.33`). Crucially, the two arms landed in different representation basins with similar L2:
    - **`sf-lr6x` (local, ultra-sparse):** footprint peaked at step ~4.5k (`area 43, span 25.5`) then contracted
      steadily to `area med = 25.4, span med = 12.3` (tightest span of any run), concentrating `89.8%` of energy
      into `c128` (`top10 = 87.5%`), with modestly higher `usage_kl = 0.435` (`maxP = 0.72`, 4 sites >50%).
    - **`sf-lr10x` (global, ultra-uniform):** footprint expanded to `area med = 63.4, span med = 27.0` (peaking at
      step 11k `87.3 / 35.8` and only beginning to contract over the last 4k steps), with lower sparsity
      (`kept128 = 86.8%`, `top10 = 83.8%`) but the highest `pwacc128` (`0.804`), lowest `lossless` (`0.47e-3`),
      lower `usage_kl` (`0.330`), and zero sites >50% (`maxP = 0.38`, `0` dead sites).
  - Both `sf-lr6x` and `sf-lr10x` are still improving steeply at 15k (`6.94 → 6.33` and `6.94 → 6.49` over the
    final 5k steps) with no plateau in sight.
- **Open questions for next study (not yet pre-registered):**
  1. Separate LR from optimizer: test cosine AdamW at `6×` (`1.68e-4`) alongside a seed-2987 confirmation of
     `sf-lr6x`.
  2. Longer horizon / anytime resume: continue `sf-lr6x` (and/or `sf-lr10x`, whose footprint just turned downward)
     to 30k–45k from the 15k checkpoints.
  3. Re-evaluate exploration noise (if needed at all) only at the calibrated LR, and watch `usage_kl` on `lr6x`.

## Planned (not pre-registered yet)

- **S3: encoder steps × exploration.** A 2 × 2 design: encoder 4 / 8 steps (renderer 4) × no exploration / the
  best exploration from S1 or S2. It tests whether escaping the basin needs a more expressive encoder.
- **Learning-rate control.** LR × 3, to tell "stuck" from "slow". With schedule-free, this becomes a one-knob
  sensitivity check.

**Dropped or deferred (with reasons):**
- *S2 as long noise* (a dose ladder σ0 0.5 / 1 / 2 with a 10k decay, decided 19:05). Dropped at 23:30 (Camille):
  it only tunes a hand-set, budget-dependent schedule, which works against a generic algorithm.
- *A pull toward the stage-1 energy.* Rejected: stage 1 is only an init and has no claim to the target property
  (progressive compressibility under LPAP). Kept as a caveat only: σ0 1 ends up closest to stage 1
  (`stage1_corr` 0.47), so its gain may be init-dependent.
- *Random LPAP pair per step* (Camille's original "swap" idea, instead of three in parallel). It is stationary
  and cheaper per step. Deferred: not a priority. A continuous random budget is ruled out because the surrogates
  and decoders are per-pair.
- *Batch-size schedule.* It is another budget-dependent schedule, and it injects noise into the weights rather
  than the energy.
