"""Checkpoint persistence utilities for `!deepthink_continue <id>`."""

from __future__ import annotations

import json
import re
from pathlib import Path

from config import CHECKPOINT_DIR, CHECKPOINT_SCHEMA_VERSION
from models import DeepThinkCheckpoint

_RESUME_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,128}$")


class CheckpointStoreError(ValueError):
    """Raised when resume id or checkpoint payload is invalid."""


def is_valid_resume_id(resume_id: str) -> bool:
    """Return True when resume id matches the allowed pattern."""
    return bool(_RESUME_ID_RE.fullmatch(resume_id or ""))


class CheckpointStore:
    """File-backed checkpoint storage."""

    def __init__(self, base_dir: str | Path = CHECKPOINT_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, resume_id: str) -> Path:
        if not is_valid_resume_id(resume_id):
            raise CheckpointStoreError(
                "invalid resume id: expected [A-Za-z0-9_-]{6,128}"
            )
        path = (self.base_dir / f"{resume_id}.json").resolve()
        base = self.base_dir.resolve()
        if base not in path.parents:
            raise CheckpointStoreError("invalid resume id path")
        return path

    def exists(self, resume_id: str) -> bool:
        path = self._path_for(resume_id)
        fb_path = path.with_name(f"{path.stem}_fallback.json")
        return fb_path.exists() or path.exists()

    def create(
        self, resume_id: str, schema_version: int = CHECKPOINT_SCHEMA_VERSION
    ) -> DeepThinkCheckpoint:
        if not is_valid_resume_id(resume_id):
            raise CheckpointStoreError(
                "invalid resume id: expected [A-Za-z0-9_-]{6,128}"
            )
        return DeepThinkCheckpoint(
            schema_version=schema_version,
            resume_id=resume_id,
        )

    def load(self, resume_id: str) -> DeepThinkCheckpoint:
        path = self._path_for(resume_id)
        fb_path = path.with_name(f"{path.stem}_fallback.json")
        
        target_path = fb_path if fb_path.exists() else path
        if not target_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {resume_id}")
        data = json.loads(target_path.read_text(encoding="utf-8"))
        return DeepThinkCheckpoint(**data)

    def save(self, checkpoint: DeepThinkCheckpoint) -> None:
        path = self._path_for(checkpoint.resume_id)
        tmp_path = path.with_suffix(".json.tmp")
        payload = checkpoint.model_dump(mode="json")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            tmp_path.replace(path)
        except OSError:
            fb_path = path.with_name(f"{path.stem}_fallback.json")
            tmp_path.replace(fb_path)
