import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # LPAP v2

    One notebook, one section per stage (FOUNDATION §5, §7). Cells only pick a
    config, call the package and display progress. Training cells run in the
    foreground until their step target or wall-clock budget; interrupting the
    kernel saves and uploads, and re-running the cell resumes.

    ## Setup
    """)
    return


@app.cell
def install():
    # On molab: install the package from a git ref when missing (or LPAP_FORCE_REINSTALL=1).
    # Locally pixi already provides it as an editable install.
    import importlib.util
    import os
    import subprocess
    import sys

    LPAP_GIT_REF = "main"
    _source = f"git+https://github.com/matovitch/lpap_v2@{LPAP_GIT_REF}"
    if importlib.util.find_spec("lpap") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _source], check=True)
    elif os.environ.get("LPAP_FORCE_REINSTALL") == "1":
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", _source], check=True
        )
        for _module in [name for name in sys.modules if name == "lpap" or name.startswith("lpap.")]:
            del sys.modules[_module]
    return (os,)


@app.cell
def session(os):
    from pathlib import Path

    import torch

    from lpap.geometry import PairGeometry
    from lpap.runtime.config import load, load_model, load_storage
    from lpap.runtime.hub import ArtifactStore
    from lpap.runtime.notify import enabled as notify_enabled

    # molab persists only small source files in /marimo: data and runs live in /tmp, HF keeps the runs.
    root = Path(os.environ.get("LPAP_ROOT", "/tmp/lpap" if Path("/marimo").is_dir() else "lpap_root"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = load_model()
    storage = load_storage()
    store = ArtifactStore(root, storage.artifacts_repo if os.environ.get("HF_TOKEN") else None)
    geometries = {spec.name: PairGeometry(spec, model_config.value_count).to(device) for spec in model_config.pairs}
    environment = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "root": str(root),
        "artifacts": storage.artifacts_repo if store.remote else "local only (no HF_TOKEN)",
        "pushover": notify_enabled(),
        "pairs": list(geometries),
    }
    environment
    return device, geometries, load, model_config, root, storage, store, torch


@app.cell
def smoke(device, geometries, torch):
    # Exact code and §3 forward on random energies, for every pair.
    from lpap.codec import decoded_energy, frontend
    from lpap.lpap import exact_code

    smoke_results = {}
    _energy = torch.randn(4, 1024, device=device)
    for _name, _geometry in geometries.items():
        _code = exact_code(_energy, _geometry)
        if _geometry.is_lossless:
            smoke_results[_name] = bool(torch.equal(_code.amplitudes, _energy))
            continue
        _inputs = frontend(_geometry.fold(_energy), _code, _geometry)
        _oracle = torch.nn.functional.one_hot(_code.probes, _geometry.probe_count).float()
        _decoded = decoded_energy(_inputs, _oracle, _geometry)
        smoke_results[_name] = bool(torch.equal(_decoded.gather(1, _code.sources), _code.amplitudes))
    smoke_results
    return


@app.cell
def _(mo):
    images_button = mo.ui.run_button(label="Load images")
    images_button
    return (images_button,)


@app.cell
def images(device, images_button, mo, root, storage):
    mo.stop(not images_button.value, mo.md("_Press **Load images** (downloads 1.2 GB on first use)._"))
    from lpap.data import ImageSet, ensure_images, load_images

    image_set = ImageSet(
        load_images(ensure_images(root, bucket=storage.images_bucket, remote=storage.images_remote)), device
    )
    {"images": image_set.images.shape[0], "train": image_set.train.numel(), "validation": image_set.validation.numel()}
    return (image_set,)


@app.cell
def _(mo):
    import time as _time

    def progress_view(title):
        """Progress callback for ``runtime.run``: redraws the cell output at most every 5 s."""
        last = {"time": 0.0, "train": {}, "validation": {}, "gallery": None}

        def update(event):
            last.update({key: value for key, value in event.items() if key in ("train", "validation", "gallery")})
            now = _time.monotonic()
            if now - last["time"] < 5.0 and event["step"] != event["target"]:
                return
            last["time"] = now
            rows = [
                f"| {split} | {name} | {value:.5g} |"
                for split in ("train", "validation")
                for name, value in last[split].items()
            ]
            views = [
                mo.md(f"**{title}**: step {event['step']} / {event['target']}, best {event['best']:.5g}"),
                mo.md("| split | metric | value |\n| --- | --- | --- |\n" + "\n".join(rows)),
            ]
            if last["gallery"] is not None:
                views.append(mo.image(last["gallery"].read_bytes()))
            mo.output.replace(mo.vstack(views))

        return update

    return (progress_view,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Flow

    Bidirectional field on images and signed log-normal prior draws, plus a
    4-step round-trip loss. Best checkpoint = lowest validation
    `objective` (`fm_euler + w · round_trip_l2_4`).
    """)
    return


@app.cell
def flow_config(load, mo):
    from lpap.runtime.config import to_dict
    from lpap.stages import flow as flow_stage

    flow_overrides = {}
    flow_cfg = load(flow_stage.FlowStageConfig, "flow", flow_overrides)
    flow_button = mo.ui.run_button(label="Train flow")
    mo.vstack([mo.tree(to_dict(flow_cfg)), flow_button])
    return flow_button, flow_cfg, flow_stage


@app.cell
def flow_train(flow_button, flow_cfg, flow_stage, image_set, mo, model_config, progress_view, store):
    mo.stop(not flow_button.value, mo.md("_Press **Train flow**. Interrupt the kernel to save and stop._"))
    flow_result = flow_stage.train(model_config, flow_cfg, image_set, store, progress_view("flow"))
    flow_result
    return


@app.cell
def flow_sweep_config(mo):
    # Experiments: each variant trains its own run ``flow-sweep-<name>`` (stage 1 keeps ``flow``).
    flow_sweep_steps = 5000
    flow_sweep = {
        "rtbf16": ({}, {"round_trip_bf16": True}),
        "rtbf16_e2": ({}, {"round_trip_bf16": True, "round_trip_every": 2}),
        "rtbf16_e4": ({}, {"round_trip_bf16": True, "round_trip_every": 4}),
    }
    flow_sweep_button = mo.ui.run_button(label="Run flow sweep")
    mo.vstack(
        [mo.md(f"**Flow sweep**: {flow_sweep_steps} steps per variant: {', '.join(flow_sweep)}"), flow_sweep_button]
    )
    return flow_sweep, flow_sweep_button, flow_sweep_steps


@app.cell
def flow_sweep_run(
    flow_stage, flow_sweep, flow_sweep_button, flow_sweep_steps, image_set, load, load_model, mo, progress_view, store
):
    mo.stop(not flow_sweep_button.value, mo.md("_Press **Run flow sweep**. Interrupt the kernel to stop it._"))
    flow_sweep_results = {}
    for _name, (_model_overrides, _stage_overrides) in flow_sweep.items():
        _config = load(flow_stage.FlowStageConfig, "flow", {**_stage_overrides, "run.steps": flow_sweep_steps})
        _result = flow_stage.train(
            load_model(_model_overrides),
            _config,
            image_set,
            store,
            progress_view(f"sweep {_name}"),
            name=f"flow-sweep-{_name}",
        )
        flow_sweep_results[_name] = _result.last_validation
        if _result.reason == "interrupt":
            break
    _metrics = list(next(iter(flow_sweep_results.values())))
    mo.md(
        "| variant | "
        + " | ".join(_metrics)
        + " |\n|"
        + " --- |" * (len(_metrics) + 1)
        + "\n"
        + "\n".join(
            f"| {_name} | " + " | ".join(f"{_values.get(_m, float('nan')):.4g}" for _m in _metrics) + " |"
            for _name, _values in flow_sweep_results.items()
        )
    )
    return (flow_sweep_results,)


@app.cell
def bank_config(load, mo):
    from lpap.stages import bank as bank_stage

    bank_cfg = load(bank_stage.BankStageConfig, "bank", {})
    bank_button = mo.ui.run_button(label="Build energy bank")
    mo.vstack(
        [
            mo.md("""
    ## 2. Energy bank

    4-step i2e encode of every image with `flow/best.pt`, kept in GPU memory as
    float16 and never written to disk (recomputed each session).
    """),
            bank_cfg,
            bank_button,
        ]
    )
    return bank_button, bank_cfg, bank_stage


@app.cell
def bank(bank_button, bank_cfg, bank_stage, image_set, mo, model_config, store):
    mo.stop(not bank_button.value, mo.md("_Press **Build energy bank** (a few minutes)._"))
    with mo.status.progress_bar(total=image_set.images.shape[0], title="Encoding images") as _bar:
        _done = [0]

        def _progress(done, total):
            _bar.update(increment=done - _done[0])
            _done[0] = done

        energy_bank, bank_stats = bank_stage.build(model_config, bank_cfg, image_set, store, _progress)
    bank_stats
    return (energy_bank,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Surrogates
    _Not implemented yet._

    ## 4. Decoders
    _Not implemented yet._

    ## 5. Autoencoder
    _Not implemented yet._
    """)
    return


if __name__ == "__main__":
    app.run()
