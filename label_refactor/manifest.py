"""Manifest helpers for structured dispersion datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable


def hash_parameters(params: Dict[str, Iterable[float]]) -> str:
    payload = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def write_manifest_entry(path: Path, sample_id: str, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"id": sample_id, **metadata}
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f)
        f.write("\n")


def validate_manifest(path: Path) -> None:
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            sample_id = data["id"]
            if sample_id in seen:
                raise ValueError(f"Duplicate sample id detected: {sample_id}")
            seen.add(sample_id)
