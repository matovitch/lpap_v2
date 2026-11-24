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
  A difference is only called when it exceeds the seed spread of both arms involved.
- **Metrics** (validation, fixed 256 images, noise-free forward):
  - primary: `image_l2` (mean over pairs) and `image_l2_<pair>`;
  - compressibility: `kept_share_<pair>`, `probe_weighted_accuracy_<pair>`;
  - geometry: `stage1_corr` (Pearson correlation of the energy with the frozen stage-1 energy; 1 = has not
    moved), `support_entropy_<pair>`, `usage_kl`, `energy_top10`;
  - gathering footprint (Jacobian rows `∂x_i/∂input`, 16 sites × 4 validation images):
    - `gather_area` = participation ratio, about the number of input samples one energy coordinate uses;
    - `gather_span` = rms distance of those inputs along the 1-D signal;
    - `*_stage1` = the frozen stage-1 values, as reference.

    It tests whether the encoder learns local, JPEG-like transforms of patches (area > 1) and whether those
    patches grow;
  - routing: `grad_<term>_<module>`.
- **Compute**: about 2 steps/s at batch 64 (H100-class, molab), so a 5k rung takes about 45 min per arm.
  molab is free: when in doubt, add seeds rather than guess.

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

**Results.** _pending (rung 1 needs both seeds)._

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

## Planned (not pre-registered yet)

- **S2: noise schedule.** Only if S1 shows an effect: decay length (1k / 2.5k / 7.5k) at the best σ0.
- **S3: encoder steps × noise.** A 2 × 2 design (encoder 4 / 8 steps; renderer 4) × (no noise / best or
  middle σ0) to test whether annealing needs a more expressive encoder (interaction).
- **Learning-rate control.** lr × 3, to tell "stuck" from "slow".
