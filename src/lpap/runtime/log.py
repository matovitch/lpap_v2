"""Per-run SQLite metric log (one file per run, uploaded with the checkpoint)."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session INTEGER PRIMARY KEY AUTOINCREMENT,
    started REAL NOT NULL,
    start_step INTEGER NOT NULL,
    config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    session INTEGER NOT NULL,
    step INTEGER NOT NULL,
    split TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS metrics_by_name ON metrics (split, name, step);
"""


class MetricLog:
    """A session is one execution of a run cell; resuming opens a new session.

    Rows after the resume step come from a run whose checkpoint was lost, so
    they are dropped to keep the log consistent with the checkpoint.
    """

    def __init__(self, path: Path, *, start_step: int, config: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        with self.connection:
            self.connection.executescript(SCHEMA)
            self.connection.execute("DELETE FROM metrics WHERE step > ?", (start_step,))
            cursor = self.connection.execute(
                "INSERT INTO sessions (started, start_step, config_json) VALUES (?, ?, ?)",
                (time.time(), start_step, json.dumps(config, default=str)),
            )
        self.session = cursor.lastrowid

    def write(self, step: int, split: str, metrics: Mapping[str, float]) -> None:
        with self.connection:
            self.connection.executemany(
                "INSERT INTO metrics (session, step, split, name, value) VALUES (?, ?, ?, ?, ?)",
                [(self.session, step, split, name, float(value)) for name, value in metrics.items()],
            )

    def series(self, split: str, name: str) -> list[tuple[int, float]]:
        return self.connection.execute(
            "SELECT step, value FROM metrics WHERE split = ? AND name = ? ORDER BY step", (split, name)
        ).fetchall()

    def close(self) -> None:
        self.connection.close()


def read_series(path: Path, split: str, name: str) -> list[tuple[int, float]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return connection.execute(
            "SELECT step, value FROM metrics WHERE split = ? AND name = ? ORDER BY step", (split, name)
        ).fetchall()
    finally:
        connection.close()
