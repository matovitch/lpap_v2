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

## Planned (not pre-registered yet)

- **S1 confirmation: seed 2987 to 15k** (`s1-r2-s2987`), one molab session.
  - **Gate** (fixed before the 15k results): if no σ0 arm of `s1-r2-s2988` beat control by more than 1% at 15k,
    seed 2987 would be skipped. The gate passed (see the rung 2 results).
  - *Caveat (Camille, 20:56): trajectory luck.* Paired arms share init and data order, then drift apart, so a
    single-seed gap is weak evidence. The sign must repeat on seed 2987. The control seed spread at 15k may also
    exceed the 1% measured at 5k.
- **Energy-scale check (code reading, no compute).** σ0 1's energy is about 3× the control's at 15k. List every
  place where the absolute scale of the energy matters (renderer input, surrogate and probe logits, `usage_kl`,
  `signed_mass_tau`, LPAP thresholds). Then decide whether part of the −10% could come from scale, not from
  exploration.
- **S2: anytime training (draft, to settle before pre-registration).** *Principle (Camille, 23:17):* training
  should be an unbounded process that the compute budget truncates. Nothing should depend on the planned number
  of steps. Two parts of today's setup break this: the cosine LR down to `run.steps`, and a noise decay given
  in steps.
  - **Optimizer:** schedule-free AdamW (Defazio et al., 2024, `schedulefree` package). It keeps the warmup,
    has no decay, and its averaged iterate is evaluable at any step. It needs `optimizer.train()` / `.eval()`
    switching around validation and checkpoints. Rungs then become plain truncations of a single run.
  - **Noise driven by gradients instead of steps:**
    - g is the encoder gradient; β is set by a half-life *h*, in steps. Two knobs: σ_max (relative, starting
      at S1's 1.0) and *h*. Nothing else refers to the step count.
    - **Primary form (Camille's proposal): gradient-norm EMA.** σ_t = σ_max · n_t / max_{s≤t} n_s, where
      n_t = EMA_β(‖g‖). The running max makes it unit-free without a step count. The noise follows how hard
      the encoder is still being pushed. Risk: with stochastic gradients ‖g‖ does not go to 0, so σ may settle
      at a floor instead of vanishing. That is acceptable for an anytime process, but we have to measure it.
    - **Alternative form: gradient signal-to-noise ratio.** σ_t = σ_max · ρ_t, where
      ρ_t = ‖EMA_β(g)‖ / EMA_β(‖g‖) ∈ [0, 1]. ρ is high while gradients agree (early) and falls as they cancel
      (near a basin), so it can go to 0. Its feedback is negative: more noise means noisier gradients, so ρ
      falls and the noise falls with it.
    - Both feedback loops (noise → gradient → noise) must be checked on the local GPU before molab: logged
      σ_t trace, no oscillation, no collapse to 0 in the first few hundred steps.
  - **Arms (seed 2988 first, then 2987; 3 × 15k from scratch ≈ 4.5h, one session):**
    - (a) schedule-free, no noise;
    - (b) schedule-free + gradient-norm-EMA noise (primary);
    - (c) schedule-free + gradient-SNR noise (alternative);
    - reference: the S1 15k cosine arms (control and σ0 1).
  - **Draft hypotheses:**
    - H6: (a) matches the cosine control at equal steps within the seed spread, so the optimizer swap alone is
      neutral or better.
    - H7: (b) beats (a) by more than the seed spread. It also reproduces σ0 1's signature (sparse, edge-aligned
      energy; small, local footprints), with no step count anywhere in its configuration.
    - H7′: same as H7 for (c). If only one of (b) and (c) holds, it becomes the default.
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
