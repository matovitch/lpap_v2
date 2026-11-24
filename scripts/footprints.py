"""Distribution of encoder gathering footprints (per energy coordinate), from saved AE checkpoints.

Usage: pixi run python scripts/footprints.py OUT.png RUN[:FILE] ...   (FILE defaults to last.pt)
Each (image, site) gives one ``area`` (participation ratio of the Jacobian row, ≈ input samples used) and one
``span`` (rms distance along the 1-D signal). The stage-1 flow is included as reference. fp32, no compile.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lpap.data import ImageSet, load_images
from lpap.flow import integrate
from lpap.geometry import PairGeometry
from lpap.runtime import checkpoint
from lpap.runtime.config import load, load_model
from lpap.runtime.hub import ArtifactStore
from lpap.stages import ae
from lpap.stages.flow import load_field

IMAGES = 64
SITE_STRIDE = 8  # every 8th energy coordinate
CHUNK = 16  # Jacobian rows per backward (8 GB GPU, fp32)


def sites(encode, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    n = images.shape[-1]
    coords = torch.arange(SITE_STRIDE // 2, n, SITE_STRIDE, device=images.device)
    areas, spans = [], []
    for image in images:
        for start in range(0, len(coords), CHUNK):
            area, span = ae.footprint_sites(encode, image[None], coords[start : start + CHUNK])
            areas.append(area.cpu())
            spans.append(span.cpu())
    return torch.cat(areas).numpy(), torch.cat(spans).numpy()


def main(out: Path, specs: list[str]) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    geos = {s.name: PairGeometry(s, model.value_count).to(dev) for s in model.pairs}
    image_set = ImageSet(load_images("data/images_32x32_gray.pt"), dev)
    store = ArtifactStore(Path("/tmp/lpap_footprints/store"), "matovitch/lpap-v2-runs")
    config = load(ae.AEStageConfig, "ae", {"compile": False, "bf16": False})
    live_metric = ae.gather_footprint  # the task's own 64-row stage-1 probe does not fit in 8 GB; skip it
    ae.gather_footprint = lambda *args, **kwargs: (torch.zeros(()), torch.zeros(()))
    task = ae.AETask(model, config, geos, image_set, store)
    ae.gather_footprint = live_metric
    images = torch.cat(task.validation_images)[:IMAGES]
    steps = config.integration_steps

    stage1 = load_field(model, store, dev).eval()
    results = {"stage 1": sites(lambda x: integrate(stage1, x, t0=-1.0, t1=0.0, steps=steps).squeeze(1), images)}
    del stage1
    for spec in specs:
        run, _, filename = spec.partition(":")
        state = checkpoint.load(store.fetch(run, filename or "last.pt"))
        weights = state["weights"] if "weights" in state else state["task"]
        task.modules.load_state_dict(weights["modules"])
        task.modules.eval()
        results[f"{run} @{state['step']}"] = sites(task._encode, images)

    print(f"{'':32s} {'area: mean  med   p10   p90   cv':>36s}   {'span: mean  med   p10   p90':>30s}")
    for label, (area, span) in results.items():
        qa, qs = np.percentile(area, [50, 10, 90]), np.percentile(span, [50, 10, 90])
        print(
            f"{label:32s} {area.mean():11.1f} {qa[0]:5.1f} {qa[1]:5.1f} {qa[2]:5.1f} {area.std() / area.mean():5.2f}"
            f"   {span.mean():10.1f} {qs[0]:5.1f} {qs[1]:5.1f} {qs[2]:5.1f}"
        )

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5))
    area_bins = np.logspace(0, np.log10(max(a.max() for a, _ in results.values()) + 1), 60)
    span_bins = np.logspace(-1, np.log10(max(s.max() for _, s in results.values()) + 1), 60)
    for label, (area, span) in results.items():
        left.hist(area, bins=area_bins, histtype="step", density=True, label=label, linewidth=1.4)
        right.hist(span, bins=span_bins, histtype="step", density=True, label=label, linewidth=1.4)
    left.set(xscale="log", xlabel="area (input samples gathered, participation ratio)", ylabel="density")
    right.set(xscale="log", xlabel="span (rms distance along the 1-D signal, samples)")
    for axis in (left, right):
        axis.axvline(32, color="grey", linestyle=":", linewidth=0.8)  # one image row
    left.legend(fontsize=8)
    figure.suptitle(f"Gathering footprints: {len(images)} images × every {SITE_STRIDE}th energy coordinate")
    figure.tight_layout()
    figure.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
