import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # LPAP v2

        One notebook, one section per stage (FOUNDATION §5, §7). Cells only pick a
        config, call the package and display progress. Training cells run in the
        foreground until their step target or wall-clock budget; interrupting the
        kernel saves and uploads, and re-running the cell resumes.

        ## Setup
        """
    )
    return


@app.cell
def _():
    # On molab: install the package from a git ref. Locally pixi already provides it.
    import importlib.util
    import subprocess
    import sys

    LPAP_GIT_REF = "main"
    if importlib.util.find_spec("lpap") is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", f"git+https://github.com/matovitch/lpap_v2@{LPAP_GIT_REF}"],
            check=True,
        )
    return


@app.cell
def _():
    import torch

    from lpap.geometry import PairGeometry
    from lpap.runtime.config import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = load_model()
    geometries = {spec.name: PairGeometry(spec, model_config.value_count).to(device) for spec in model_config.pairs}
    environment = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "pairs": list(geometries),
    }
    environment
    return device, geometries, torch


@app.cell
def _(device, geometries, torch):
    # Smoke check: exact code and §3 forward on random energies, for every pair.
    from lpap.codec import decoded_energy, frontend
    from lpap.lpap import exact_code

    smoke = {}
    energy = torch.randn(4, 1024, device=device)
    for name, geometry in geometries.items():
        code = exact_code(energy, geometry)
        if geometry.is_lossless:
            smoke[name] = bool(torch.equal(code.amplitudes, energy))
            continue
        inputs = frontend(geometry.fold(energy), code, geometry)
        oracle = torch.nn.functional.one_hot(code.probes, geometry.probe_count).float()
        decoded = decoded_energy(inputs, oracle, geometry)
        smoke[name] = bool(torch.equal(decoded.gather(1, code.sources), code.amplitudes))
    smoke
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 1. Flow
        _Not implemented yet._

        ## 2. Energy bank
        _Not implemented yet._

        ## 3. Surrogates
        _Not implemented yet._

        ## 4. Decoders
        _Not implemented yet._

        ## 5. Autoencoder
        _Not implemented yet._
        """
    )
    return


if __name__ == "__main__":
    app.run()
