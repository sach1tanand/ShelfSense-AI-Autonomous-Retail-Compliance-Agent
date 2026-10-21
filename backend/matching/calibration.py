"""Calibration utilities for labelled crop validation; never assumes raw CLIP scores are probabilities."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def calibrate_threshold(
    validation_records: list[dict[str, Any]],
    *,
    target_precision: float = 0.95,
) -> dict[str, Any]:
    """Choose score and margin cutoffs maximizing coverage at a labelled precision target.

    Each validation record must contain `expected_sku` and ranked `candidates` with
    `sku` and `similarity`. The returned values are empirical for this validation
    set and should be checked on a separate held-out set before deployment.
    """
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be in (0, 1]")
    usable = [record for record in validation_records if record.get("candidates")]
    if not usable:
        raise ValueError("No labelled validation records have candidate scores")

    margins = []
    scores = []
    for record in usable:
        candidates = record["candidates"]
        scores.append(float(candidates[0]["similarity"]))
        second = float(candidates[1]["similarity"]) if len(candidates) > 1 else -1.0
        margins.append(float(candidates[0]["similarity"]) - second)

    score_options = sorted(set(scores))
    margin_options = sorted(set(margins))
    best: tuple[int, float, float, int, int] | None = None
    for threshold in score_options:
        for min_margin in margin_options:
            accepted = [
                record for record in usable
                if float(record["candidates"][0]["similarity"]) >= threshold
                and (float(record["candidates"][0]["similarity"]) - (float(record["candidates"][1]["similarity"]) if len(record["candidates"]) > 1 else -1.0)) >= min_margin
            ]
            if not accepted:
                continue
            correct = sum(record["candidates"][0]["sku"] == record["expected_sku"] for record in accepted)
            precision = correct / len(accepted)
            if precision < target_precision:
                continue
            option = (len(accepted), threshold, min_margin, correct, len(accepted))
            if best is None or option[0] > best[0] or (option[0] == best[0] and option[1] < best[1]):
                best = option

    top1_correct = sum(record["candidates"][0]["sku"] == record["expected_sku"] for record in usable)
    failure_cases = [
        {
            "expected_sku": record["expected_sku"],
            "top_sku": record["candidates"][0]["sku"],
            "top_similarity": float(record["candidates"][0]["similarity"]),
            "top_margin": margins[index],
        }
        for index, record in enumerate(usable)
        if record["candidates"][0]["sku"] != record["expected_sku"]
    ]
    if best is None:
        return {
            "status": "uncalibrated",
            "target_precision": target_precision,
            "validation_count": len(usable),
            "top1_accuracy": top1_correct / len(usable),
            "accepted_count": 0,
            "coverage": 0.0,
            "accepted_precision": None,
            "threshold": None,
            "minimum_margin": None,
            "failure_cases": failure_cases,
            "warning": "No threshold met the requested precision on labelled validation records.",
        }

    accepted_count, threshold, min_margin, correct_count, _ = best
    return {
        "status": "calibrated_on_validation_set",
        "target_precision": target_precision,
        "validation_count": len(usable),
        "top1_accuracy": top1_correct / len(usable),
        "accepted_count": accepted_count,
        "coverage": accepted_count / len(usable),
        "accepted_precision": correct_count / accepted_count,
        "accepted_recall": correct_count / max(top1_correct, 1),
        "threshold": threshold,
        "minimum_margin": min_margin,
        "failure_cases": failure_cases,
        "warning": "Empirical validation-set calibration is not a guarantee; validate on a held-out labelled set.",
    }


def save_calibration(path: Path, calibration: dict[str, Any], model_id: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**calibration, "model_id": model_id}
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as stream:
            temporary_path = stream.name
            json.dump(payload, stream, indent=2)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
