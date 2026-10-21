"""Atomic structured JSON storage for image-match and planogram audit results."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def save_audit_result(results_dir: Path, audit_id: str, payload: dict[str, Any]) -> Path:
    """Write one JSON audit file atomically and return its path."""
    if not audit_id or any(character not in "0123456789abcdef-" for character in audit_id.lower()):
        raise ValueError("audit_id must be a UUID-like hexadecimal identifier")
    root = Path(results_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = (root / f"{audit_id}.json").resolve()
    if destination.parent != root:
        raise ValueError("Audit result path escaped the configured results directory")
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=root, delete=False, suffix=".tmp") as stream:
            temporary_path = stream.name
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return destination
