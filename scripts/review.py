"""Pull run logs and galleries from the HF artifacts repo and compare runs (never touches molab).

Usage: pixi run python scripts/review.py RUN_PREFIX [METRIC ...]
Writes to lpap_root/review/<run>/{log.sqlite,gallery.png}; prints the last
validation of every run under the prefix, plus the step of each run's best.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from lpap.runtime.config import load_storage

ROOT = Path("lpap_root/review")


def last_validation(path: Path) -> tuple[int, dict[str, float]]:
    with sqlite3.connect(path) as db:
        step = db.execute("SELECT MAX(step) FROM metrics WHERE split = 'validation'").fetchone()[0]
        rows = db.execute("SELECT name, value FROM metrics WHERE split = 'validation' AND step = ?", (step,))
        return step, dict(rows.fetchall())


def main(prefix: str, metrics: list[str]) -> None:
    patterns = [f"{prefix}*/log.sqlite", f"{prefix}*/gallery.png"]
    snapshot_download(load_storage().artifacts_repo, allow_patterns=patterns, local_dir=ROOT)
    runs = sorted(path.parent for path in ROOT.glob(f"{prefix}*/log.sqlite"))
    table = {run.name: last_validation(run / "log.sqlite") for run in runs}
    names = metrics or sorted({name for _, values in table.values() for name in values})
    print("run\tstep\t" + "\t".join(names))
    for run, (step, values) in table.items():
        print(f"{run}\t{step}\t" + "\t".join(f"{values.get(name, float('nan')):.4g}" for name in names))
    print("\n".join(str(run / "gallery.png") for run in runs if (run / "gallery.png").exists()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
