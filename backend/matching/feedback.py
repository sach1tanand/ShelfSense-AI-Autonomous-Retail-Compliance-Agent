"""Operator review feedback: verdicts on SKU matches, extra reference crops, and measured accuracy.

A reviewer marks a detected product's match as correct, wrong (with the right SKU), or not a
product. Each verdict is stored in data/feedback/feedback.json. Correct and corrected verdicts
save the crop as an extra reference image for that SKU, so the next analysis can recognise the
real shelf packaging instead of relying only on the catalogue photo. "Not a product" crops are
kept as negative examples: a crop most similar to one of them is never accepted as a SKU.

The verdicts also measure the matcher on real photos: for any threshold and margin, how many of
the reviewed crops the matcher would have accepted, and how many of those were right.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

NOT_PRODUCT_SKU = "__not_product__"
DECISIONS = {"correct", "wrong", "not_product"}
_lock = threading.Lock()


def feedback_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "feedback"


def _feedback_file(data_dir: Path) -> Path:
    return feedback_dir(data_dir) / "feedback.json"


def load_feedback(data_dir: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(_feedback_file(data_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    records = payload.get("records") if isinstance(payload, dict) else None
    return [record for record in records or [] if isinstance(record, dict)]


def _write_feedback(data_dir: Path, records: list[dict[str, Any]]) -> None:
    path = _feedback_file(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as stream:
            temporary = stream.name
            json.dump({"version": 1, "records": records}, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def record_feedback(data_dir: Path, record: dict[str, Any], crop: Image.Image | None) -> dict[str, Any]:
    """Store one verdict (replacing an earlier verdict on the same detection) and its reference crop."""
    if record.get("decision") not in DECISIONS:
        raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
    record = {**record, "id": uuid.uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat()}
    references = feedback_dir(data_dir) / "references"
    if crop is not None and record.get("use_as_reference", True):
        references.mkdir(parents=True, exist_ok=True)
        name = f"{record['id']}.jpg"
        crop.convert("RGB").save(references / name, format="JPEG", quality=92)
        record["reference_image"] = name
    with _lock:
        records = load_feedback(data_dir)
        replaced = [
            old for old in records
            if old.get("report_id") == record.get("report_id") and old.get("detection_index") == record.get("detection_index")
        ]
        for old in replaced:
            if old.get("reference_image"):
                (references / old["reference_image"]).unlink(missing_ok=True)
        records = [old for old in records if old not in replaced] + [record]
        _write_feedback(data_dir, records)
    return record


def load_feedback_references(data_dir: Path, catalogue_skus: set[str]) -> list[dict[str, Any]]:
    """Reference crops from verdicts, as {"sku", "name", "image_path", "key", "source"}.

    Verdicts for SKUs no longer in the active catalogue are skipped; negative examples use
    NOT_PRODUCT_SKU.
    """
    root = (feedback_dir(data_dir) / "references").resolve()
    references = []
    for record in load_feedback(data_dir):
        name = record.get("reference_image")
        if not name:
            continue
        path = (root / Path(str(name)).name).resolve()
        if path.parent != root or not path.is_file():
            continue
        if record.get("decision") == "not_product":
            sku, label = NOT_PRODUCT_SKU, "Not a product (reviewed)"
        else:
            sku = str(record.get("true_sku") or "")
            if sku not in catalogue_skus:
                continue
            label = record.get("product_name", "")
        references.append({"sku": sku, "name": label, "image_path": path, "key": f"feedback:{record['id']}",
                           "source": "review"})
    return references


def feedback_accuracy(records: list[dict[str, Any]], threshold: float, margin: float) -> dict[str, Any]:
    """How the matcher's top guess fared on reviewed crops at a score threshold.

    Only verdicts recorded with the model's top guess and scores count. "accepted" crops are
    those whose best SKU score is above the threshold; they are correct when the top guess equals
    the reviewer's SKU (a crop marked "not a product" is never a correct acceptance). ``margin``
    remains in the signature for compatibility; it is reported as ambiguity context, not a gate.
    """
    scored = [
        record for record in records
        if record.get("model_top_similarity") is not None and record.get("model_top_margin") is not None
        and record.get("model_top_sku")
    ]
    accepted = [record for record in scored
                if record["model_top_similarity"] > threshold]
    correct = sum(record.get("decision") != "not_product" and record.get("true_sku") == record["model_top_sku"]
                  for record in accepted)
    in_catalogue = [record for record in scored if record.get("decision") != "not_product"]
    found = sum(record.get("true_sku") == record["model_top_sku"] for record in accepted
                if record.get("decision") != "not_product")
    return {
        "reviewed": len(scored),
        "accepted": len(accepted),
        "correct": correct,
        "precision": round(correct / len(accepted), 4) if accepted else None,
        "coverage": round(found / len(in_catalogue), 4) if in_catalogue else None,
    }
