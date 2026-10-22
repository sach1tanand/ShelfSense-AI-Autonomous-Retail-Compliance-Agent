"""Calibrate the SKU-match acceptance threshold on the validation split of the test set.

Runs the real detector on data/test_set/validation scenes, labels each detected crop from
ground truth with one-to-one IoU matching (IoU >= 0.5; a second box on an already-labelled
facing is a duplicate and is excluded), and ranks it against a gallery that excludes the holdout
products. Holdout products and boxes matching no product count as "unknown" and must be
rejected. The (threshold, margin) pair with the most correct acceptances at >= 98% precision is
written to data/clip_calibration.json together with its provenance (matcher, detector checkpoint
hash, gallery hash, ground-truth hash, sample counts). The matching pipeline applies it only for
the same matcher and reports it as stale when the detector checkpoint differs.

The validation scenes are synthetic. A threshold that reaches the precision target on them is
not evidence of the same precision on real store photos.

Usage:  python -m tools.calibrate_matching [--dry-run] [--output PATH]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from backend.matching.calibration import CALIBRATED_STATUS, file_fingerprint, gallery_fingerprint, write_json_atomic
from backend.matching.catalog import load_catalogue_references
from backend.matching.recognizer import ProductRecognizer
from config.settings import DATA_DIR, DETECTION_CONFIDENCE, MATCHER_MODEL_ID, MODEL_PATH
from src.detector import get_detector

TEST_SET_DIR = DATA_DIR / "test_set"
TARGET_PRECISION = 0.98
MIN_ACCEPTED_FOR_TARGET = 100  # below this many accepted crops a 98% precision estimate is too noisy
MIN_UNKNOWN_CROPS = 30


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_one_to_one(boxes: list[list[float]], truths: list[list[float]], threshold: float = 0.5) -> dict[int, int]:
    """Greedy highest-IoU-first one-to-one assignment {box_index: truth_index} at IoU >= threshold."""
    pairs = sorted(
        ((iou(box, truth), box_index, truth_index)
         for box_index, box in enumerate(boxes) for truth_index, truth in enumerate(truths)),
        reverse=True,
    )
    assigned: dict[int, int] = {}
    used_truths: set[int] = set()
    for overlap, box_index, truth_index in pairs:
        if overlap < threshold:
            break
        if box_index in assigned or truth_index in used_truths:
            continue
        assigned[box_index] = truth_index
        used_truths.add(truth_index)
    return assigned


def label_detections(detections: list[dict], facings: list[dict]) -> list[str | None]:
    """Ground-truth SKU per detection: a SKU, None (background) or "duplicate".

    "duplicate" marks a box that overlaps an already-labelled facing at IoU >= 0.5; counting it
    as a second correct sample would inflate the precision estimate.
    """
    boxes = [detection["box"] for detection in detections]
    assigned = match_one_to_one(boxes, [facing["box"] for facing in facings])
    labels: list[str | None] = []
    for index, box in enumerate(boxes):
        if index in assigned:
            labels.append(facings[assigned[index]]["sku"])
        elif any(iou(box, facing["box"]) >= 0.5 for facing in facings):
            labels.append("duplicate")
        else:
            labels.append(None)
    return labels


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_test_set(truth: dict) -> list[str]:
    """Problems that make the validation split unusable for calibration (empty list = usable)."""
    problems = []
    if truth.get("planogram_sha256") and truth["planogram_sha256"] != file_sha256(DATA_DIR / "planogram.csv"):
        problems.append("data/planogram.csv changed after the test set was built; run `python -m tools.build_test_set`.")
    split = truth.get("split", {})
    groups = {
        "validation": set(split.get("validation_skus", [])),
        "holdout": set(split.get("holdout_unknown_skus", [])),
        "planogram": set(split.get("test_planogram_skus", [])),
        "distractor": set(split.get("test_distractor_skus", [])),
    }
    with (DATA_DIR / "planogram.csv").open(encoding="utf-8-sig", newline="") as stream:
        live_planogram = {row.get("expected_sku", "") for row in csv.DictReader(stream)} - {""}
    if live_planogram != groups["planogram"]:
        problems.append("The test set's planogram SKUs differ from data/planogram.csv; run `python -m tools.build_test_set`.")
    groups["live planogram"] = live_planogram
    names = list(groups)
    for position, left in enumerate(names):
        for right in names[position + 1:]:
            if {left, right} == {"planogram", "live planogram"}:
                continue
            overlap = groups[left] & groups[right]
            if overlap:
                problems.append(f"SKU split overlap between {left} and {right}: {sorted(overlap)}")
    return problems


def collect_scores(model_id: str = MATCHER_MODEL_ID, confidence: float = DETECTION_CONFIDENCE) -> tuple[list[dict], dict]:
    truth = json.loads((TEST_SET_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    holdout = set(truth["split"]["holdout_unknown_skus"])
    with (DATA_DIR / "catalogue.csv").open(encoding="utf-8-sig", newline="") as stream:
        catalogue = [
            {**row, "name": row.get("name") or row.get("product_name", "")}
            for row in csv.DictReader(stream) if row["sku"] not in holdout
        ]
    references = load_catalogue_references(catalogue, DATA_DIR)
    if holdout & {reference["sku"] for reference in references}:
        raise SystemExit("Holdout products leaked into the calibration gallery.")
    # Reviewed shelf crops are excluded: they are not part of the catalogue gallery being calibrated.
    recognizer = ProductRecognizer(data_dir=DATA_DIR, model_id=model_id, acceptance_threshold=None, top_k=2,
                                   use_feedback=False)
    detector = get_detector()
    samples = []
    counts = {"detections": 0, "duplicates": 0, "known": 0, "background": 0, "holdout": 0, "facings": 0}
    for scene in truth["validation"]:
        image_path = TEST_SET_DIR / scene["image"]
        detections = detector.detect(image_path, confidence)
        labels = label_detections(detections, scene["facings"])
        counts["detections"] += len(detections)
        counts["facings"] += len(scene["facings"])
        with Image.open(image_path) as opened:
            matches = recognizer.match_detections(opened.convert("RGB"), detections, catalogue)
        for match, label in zip(matches, labels):
            if label == "duplicate":
                counts["duplicates"] += 1
                continue
            if not match.get("candidates"):
                continue
            counts["background" if label is None else "holdout" if label in holdout else "known"] += 1
            samples.append({
                "truth": label if label and label not in holdout else "unknown",
                "top_sku": match["candidates"][0]["sku"],
                "similarity": match["top_similarity"],
                "margin": match["top_margin"],
            })
    provenance = {
        "gallery_sha256": gallery_fingerprint(references),
        "gallery_size": len(references),
        "holdout_excluded": len(holdout),
        "ground_truth_sha256": file_sha256(TEST_SET_DIR / "ground_truth.json"),
        "sample_counts": counts,
    }
    return samples, provenance


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float | None:
    """95% Wilson lower bound of a binomial proportion (None without samples)."""
    if not total:
        return None
    phat = successes / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    spread = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))
    return round((centre - spread) / denominator, 4)


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
                accepted_unknown = sum(s["truth"] == "unknown" for s in accepted)
                best = (key, {
                    "threshold": threshold, "minimum_margin": margin, "accepted": len(accepted),
                    "correct": correct, "precision": round(precision, 4),
                    "precision_wilson_lower_95": wilson_lower_bound(correct, len(accepted)),
                    "known_coverage": round(correct / known, 4) if known else None,
                    "unknown_accepted": accepted_unknown,
                    "unknown_rejection": round(1 - accepted_unknown / unknown, 4) if unknown else None,
                })
    if best is None:
        raise SystemExit(f"No threshold reaches {target_precision:.0%} precision on the validation crops.")
    result = {**best[1], "validation_crops": len(samples), "known_crops": known, "unknown_crops": unknown}
    warnings = []
    if result["accepted"] < MIN_ACCEPTED_FOR_TARGET:
        warnings.append(f"Only {result['accepted']} accepted crops: too few to establish {target_precision:.0%} precision.")
    if unknown < MIN_UNKNOWN_CROPS:
        warnings.append(f"Only {unknown} unknown-product crops: the unknown-rejection rate is a rough estimate.")
    if result["precision_wilson_lower_95"] is not None and result["precision_wilson_lower_95"] < target_precision:
        warnings.append(
            f"The 95% lower bound of the measured precision is {result['precision_wilson_lower_95']:.3f}, below the "
            f"{target_precision:.0%} target; the target is met on this sample, not demonstrated statistically."
        )
    result["warnings"] = warnings
    result["curve"] = threshold_curve(samples, result["minimum_margin"])
    return result


def threshold_curve(samples: list[dict], margin: float) -> list[dict]:
    """Precision, catalogue coverage and unknown rejection per threshold at a fixed margin.

    Lets an operator see what a non-calibrated threshold costs on the same validation crops.
    """
    known = sum(sample["truth"] != "unknown" for sample in samples)
    unknown = len(samples) - known
    curve = []
    for t in range(30, 96):
        threshold = t / 100
        accepted = [s for s in samples if s["similarity"] >= threshold and s["margin"] >= margin]
        correct = sum(s["top_sku"] == s["truth"] for s in accepted)
        accepted_unknown = sum(s["truth"] == "unknown" for s in accepted)
        curve.append({
            "threshold": threshold, "accepted": len(accepted),
            "precision": round(correct / len(accepted), 4) if accepted else None,
            "known_coverage": round(correct / known, 4) if known else None,
            "unknown_rejection": round(1 - accepted_unknown / unknown, 4) if unknown else None,
        })
    return curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print the calibration without writing it")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "clip_calibration.json")
    args = parser.parse_args()

    truth = json.loads((TEST_SET_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    problems = check_test_set(truth)
    if problems:
        raise SystemExit("Test set is not usable for calibration:\n- " + "\n- ".join(problems))
    detector = file_fingerprint(MODEL_PATH)
    if detector is None:
        raise SystemExit(f"Detector checkpoint not found: {MODEL_PATH}")
    samples, provenance = collect_scores()
    result = choose_threshold(samples)
    payload = {
        "status": CALIBRATED_STATUS,
        "model_id": MATCHER_MODEL_ID,
        "threshold": result["threshold"],
        "minimum_margin": result["minimum_margin"],
        "target_precision": TARGET_PRECISION,
        "detection_confidence": DETECTION_CONFIDENCE,
        "detector_model": detector,
        "metrics": {key: value for key, value in result.items() if key != "curve"},
        "curve": result["curve"],
        "provenance": provenance,
        "validation_source": "data/test_set/validation (synthetic shelves, products disjoint from the test split)",
        "limitations": [
            "Calibrated on synthetic shelf renders of catalogue photos; real-store precision is not measured.",
            "Unknown-product rejection is measured on holdout catalogue products and background boxes only.",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(payload, indent=2))
    if args.dry_run:
        return
    if args.output.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = args.output.with_name(f"{args.output.stem}.{stamp}.bak.json")
        shutil.copy2(args.output, backup)
        print(f"Previous calibration backed up to {backup}")
    write_json_atomic(args.output, payload)


if __name__ == "__main__":
    main()
