"""Calibration utilities for labelled crop validation; never assumes raw CLIP scores are probabilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

CALIBRATED_STATUS = "calibrated_on_validation_set"
RECALIBRATE_HINT = "`python -m tools.calibrate_matching`"


def file_fingerprint(path: str | Path) -> dict[str, Any] | None:
    """Name, size and SHA-256 of a model or data file (None when it does not exist)."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"file": path.name, "size": path.stat().st_size, "sha256": digest.hexdigest()}


def gallery_fingerprint(references: list[dict[str, Any]]) -> str:
    """Order-independent SHA-256 of the reference gallery (SKU plus reference-image bytes)."""
    digest = hashlib.sha256()
    for reference in sorted(references, key=lambda item: str(item["sku"])):
        digest.update(str(reference["sku"]).encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(Path(reference["image_path"]).read_bytes()).digest())
    return digest.hexdigest()


def load_runtime_calibration(
    path: Path,
    model_id: str,
    *,
    detector_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read and validate the acceptance calibration for the active matcher.

    Returns {"state", "threshold", "minimum_margin", "issues", "payload"}. The state is:

    * ``valid``: compatible with the active matcher and detector; thresholds applied.
    * ``stale``: thresholds are well formed and for this matcher, but provenance (detector
      checkpoint) differs or was not recorded. Thresholds are applied and the audit is told to
      recalibrate; the cosine scale of a matcher does not change with the detector, but the
      crop distribution the precision target was measured on does.
    * ``missing`` / ``malformed`` / ``uncalibrated`` / ``model_mismatch``: no threshold is
      applied, so every SKU candidate stays advisory (fail closed).

    Each issue is {"code", "message"} so reports and logs can state the exact fix.
    """
    issues: list[dict[str, str]] = []

    def closed(state: str, code: str, message: str, payload: dict | None = None) -> dict[str, Any]:
        if RECALIBRATE_HINT not in message:
            message = f"{message} No SKU is accepted until you recalibrate with {RECALIBRATE_HINT}."
        issues.append({"code": code, "message": message})
        return {"state": state, "threshold": None, "minimum_margin": None, "issues": issues, "payload": payload or {}}

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return closed("missing", "calibration_missing",
                      f"No calibration file at {path.name}.")
    except (OSError, json.JSONDecodeError) as error:
        return closed("malformed", "calibration_unreadable", f"Calibration file {path.name} is unreadable: {error}")
    if not isinstance(payload, dict):
        return closed("malformed", "calibration_not_object", f"Calibration file {path.name} is not a JSON object.")
    if payload.get("status") != CALIBRATED_STATUS:
        return closed("uncalibrated", "calibration_status",
                      f"Calibration status is {payload.get('status')!r}; no threshold met the precision target.", payload)
    calibrated_model = payload.get("model_id")
    if calibrated_model != model_id:
        return closed("model_mismatch", "calibration_model_mismatch",
                      f"Calibration was made for matcher {calibrated_model!r} but the active matcher is {model_id!r}.", payload)
    try:
        threshold = float(payload.get("threshold"))
        margin = float(payload.get("minimum_margin"))
    except (TypeError, ValueError):
        return closed("malformed", "calibration_values_invalid",
                      "Calibration threshold/minimum_margin are missing or not numbers.", payload)
    if not (math.isfinite(threshold) and -1.0 <= threshold <= 1.0) or not (math.isfinite(margin) and margin >= 0):
        return closed("malformed", "calibration_values_out_of_range",
                      f"Calibration threshold {threshold} must be in [-1, 1] and margin {margin} must be >= 0.", payload)

    state = "valid"
    recorded_detector = payload.get("detector_model")
    if detector_fingerprint is not None:
        if not isinstance(recorded_detector, dict) or not recorded_detector.get("sha256"):
            state = "stale"
            issues.append({"code": "calibration_detector_unrecorded",
                           "message": "Calibration does not record which detector produced its crops; "
                                      "recalibrate with `python -m tools.calibrate_matching`."})
        elif recorded_detector.get("sha256") != detector_fingerprint.get("sha256"):
            state = "stale"
            issues.append({"code": "calibration_detector_changed",
                           "message": f"Calibration crops came from detector {recorded_detector.get('file')!r}, "
                                      f"but the active detector is {detector_fingerprint.get('file')!r}; "
                                      "recalibrate with `python -m tools.calibrate_matching`."})
    return {"state": state, "threshold": threshold, "minimum_margin": margin, "issues": issues, "payload": payload}


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
    write_json_atomic(path, {**calibration, "model_id": model_id})


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as stream:
            temporary_path = stream.name
            json.dump(payload, stream, indent=2)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
