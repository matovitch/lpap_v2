"""Run artifacts on a private Hugging Face model repo (FOUNDATION §7).

Remote layout: ``<run>/last.pt``, ``<run>/best.pt``, ``<run>/log.sqlite``.
Files pushed together land in one commit, so a checkpoint and its log never
disagree remotely. Locally the same layout lives under ``<root>/runs``.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError


class ArtifactStore:
    def __init__(self, root: Path, repo_id: str | None, *, token: str | None = None) -> None:
        """``repo_id = None`` keeps everything local (tests, offline runs)."""
        self.root = Path(root)
        self.repo_id = repo_id
        self.api = HfApi(token=token or os.environ.get("HF_TOKEN")) if repo_id else None
        self._repo_ready = False

    @property
    def remote(self) -> bool:
        return self.api is not None

    def run_dir(self, run: str) -> Path:
        return self.root / "runs" / run

    def local(self, run: str, filename: str) -> Path:
        return self.run_dir(run) / filename

    def fetch(self, run: str, filename: str) -> Path | None:
        """Local file if present, else download it; ``None`` when it exists nowhere."""
        path = self.local(run, filename)
        if path.is_file() or not self.remote:
            return path if path.is_file() else None
        try:
            downloaded = self.api.hf_hub_download(self.repo_id, f"{run}/{filename}", local_dir=self.root / "runs")
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None
        return Path(downloaded)

    def push(self, run: str, filenames: list[str], message: str) -> bool:
        """Upload ``filenames`` of ``run`` in one commit. ``False`` when local only."""
        if not self.remote:
            return False
        self._ensure_repo()
        operations = [
            CommitOperationAdd(path_in_repo=f"{run}/{name}", path_or_fileobj=str(self.local(run, name)))
            for name in filenames
            if self.local(run, name).is_file()
        ]
        if operations:
            self.api.create_commit(self.repo_id, operations=operations, commit_message=message)
        return bool(operations)

    def squash(self) -> None:
        """Drop the history of old checkpoints (run by hand when the repo grows)."""
        self._ensure_repo()
        self.api.super_squash_history(self.repo_id)

    def _ensure_repo(self) -> None:
        if not self._repo_ready:
            self.api.create_repo(self.repo_id, private=True, exist_ok=True)
            self._repo_ready = True
