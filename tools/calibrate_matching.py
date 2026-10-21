"""Calibrate the SKU-match acceptance threshold on the validation split of the test set.

Runs the real detector on data/test_set/validation scenes, labels each detected crop from
ground truth (IoU >= 0.5), and ranks it against a gallery that excludes the holdout
products. Holdout products and boxes matching no product count as "unknown" and must be
rejected. The (threshold, margin) pair with the most correct acceptances at >= 98%
precision is written to data/clip_calibration.json, which the matching pipeline reads
when its model_id matches.

Usage:  python -m tools.calibrate_matching
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

from PIL import Image

from backend.matching.recognizer import ProductRecognizer
from config.settings import DATA_DIR, DETECTION_CONFIDENCE, MATCHER_MODEL_ID
from src.detector import get_detector

TEST_SET_DIR = DATA_DIR / "test_set"
TARGET_PRECISION = 0.98


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def label_detections(detections: list[dict], facings: list[dict]) -> list[str | None]:
    """Ground-truth SKU per detection (None = matches no product facing)."""
    labels = []
    for detection in detections:
        best = max(facings, key=lambda facing: iou(detection["box"], facing["box"]), default=None)
        labels.append(best["sku"] if best and iou(detection["box"], best["box"]) >= 0.5 else None)
    return labels


def collect_scores(model_id: str = MATCHER_MODEL_ID, confidence: float = DETECTION_CONFIDENCE) -> list[dict]:
    truth = json.loads((TEST_SET_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    holdout = set(truth["split"]["holdout_unknown_skus"])
    with (DATA_DIR / "catalogue.csv").open(encoding="utf-8-sig", newline="") as stream:
        catalogue = [
            {**row, "name": row.get("name") or row.get("product_name", "")}
            for row in csv.DictReader(stream) if row["sku"] not in holdout
        ]
    recognizer = ProductRecognizer(data_dir=DATA_DIR, model_id=model_id, acceptance_threshold=None, top_k=2)
    detector = get_detector()
    samples = []
    for scene in truth["validation"]:
        image_path = TEST_SET_DIR / scene["image"]
        detections = detector.detect(image_path, confidence)
        labels = label_detections(detections, scene["facings"])
        with Image.open(image_path) as opened:
            matches = recognizer.match_detections(opened.convert("RGB"), detections, catalogue)
        for match, label in zip(matches, labels):
            if not match.get("candidates"):
                continue
            samples.append({
                "truth": label if label and label not in holdout else "unknown",
                "top_sku": match["candidates"][0]["sku"],
                "similarity": match["top_similarity"],
                "margin": match["top_margin"],
            })
    return samples


def choose_threshold(samples: list[dict], target_precision: float = TARGET_PRECISION) -> dict:
    known = sum(sample["truth"] != "unknown" for sample in samples)
    unknown = len(samples) - known
    best = None
    for t in range(30, 100):
        threshold = t / 100
        for m in range(0, 41):
            margin = m / 200
            accepted = [s for s in samples if s["similarity"] >= threshold and s["margin"] >= margin]
            if not accepted:
                continue
            correct = sum(s["top_sku"] == s["truth"] for s in accepted)
            precision = correct / len(accepted)
            if precision < target_precision:
                continue
            key = (correct, threshold, margin)
            if best is None or key > best[0]:
                best = (key, {
                    "threshold": threshold, "minimum_margin": margin, "accepted": len(accepted),
                    "correct": correct, "precision": round(precision, 4),
                    "known_coverage": round(correct / known, 4) if known else None,
                    "unknown_rejection": round(
                        1 - sum(s["truth"] == "unknown" for s in accepted) / unknown, 4) if unknown else None,
                })
    if best is None:
        raise SystemExit(f"No threshold reaches {target_precision:.0%} precision on the validation crops.")
    return {**best[1], "validation_crops": len(samples), "known_crops": known, "unknown_crops": unknown}


def main() -> None:
    samples = collect_scores()
    result = choose_threshold(samples)
    payload = {
        "status": "calibrated_on_validation_set",
        "model_id": MATCHER_MODEL_ID,
        "threshold": result["threshold"],
        "minimum_margin": result["minimum_margin"],
        "target_precision": TARGET_PRECISION,
        "detection_confidence": DETECTION_CONFIDENCE,
        "metrics": result,
        "validation_source": "data/test_set/validation (synthetic shelves, products disjoint from the test split)",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = DATA_DIR / "clip_calibration.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
