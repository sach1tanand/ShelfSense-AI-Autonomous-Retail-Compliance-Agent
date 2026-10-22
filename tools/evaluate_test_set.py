"""Run every labelled synthetic test scene through the real audit workflow and score it.

Each scene goes through POST /api/analyze (real detector, SKU matcher, slot mapping, OCR,
planogram comparison) and is compared with data/test_set/ground_truth.json:

* detection recall / precision: one-to-one matching of detections to ground-truth facings at
  IoU >= 0.5 (highest IoU first). A second box on an already-matched facing is a duplicate, a
  box on no facing is a false positive, and an unmatched facing is a miss.
* SKU precision / coverage: accepted SKUs that are correct for their matched facing / facings
  correctly identified. An accepted duplicate or false-positive box counts as a wrong acceptance.
* slot status accuracy: report slot status equals the expected status. A missing slot row, an
  unexpected extra row, or a failed/malformed audit counts as a failure, never as skipped.
* price tag recall and price status accuracy (tags matched by position, not just by value).

All numbers are for SYNTHETIC scenes. They are not real-store accuracy.

By default reports go to a temporary folder. With --save they are stored in the normal
report history (store name "Test Set Store") so the Reports page charts show them.

Usage:  python -m tools.evaluate_test_set [--save] [--confidence 0.35]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from config.settings import TEST_SET_DIR
from tools.calibrate_matching import check_test_set, iou, match_one_to_one


def _valid_box(box) -> bool:
    try:
        return len(box) == 4 and float(box[2]) > float(box[0]) and float(box[3]) > float(box[1])
    except (TypeError, ValueError):
        return False


def failed_scene(scene: dict, reason: str) -> dict:
    """A scene whose audit failed or was malformed: every check counts as failed."""
    slots = [{"slot": f"R{slot['row']}S{slot['slot']}", "expected": slot["expected_status"], "actual": None, "ok": False}
             for slot in scene["slots"]]
    prices = [{"sku": sku, "expected": status, "actual": None, "ok": False}
              for sku, status in scene["expected_price_status"].items()]
    return {
        "image": scene["image"], "error": reason, "facings": len(scene["facings"]), "detections": 0,
        "recalled": 0, "duplicates": 0, "false_positives": 0, "missed": len(scene["facings"]),
        "accepted": 0, "accepted_correct": 0, "identified_correct": 0, "slots": slots, "unexpected_slot_rows": [],
        "price_tags": len(scene["price_tags"]), "price_tags_read": 0, "prices": prices,
        "compliance_rate": None, "report_id": None,
    }


def score_scene(scene: dict, report: dict) -> dict:
    if not isinstance(report, dict) or not isinstance(report.get("detections"), list) \
            or not isinstance(report.get("compliance"), list):
        return failed_scene(scene, "Malformed report: detections/compliance missing.")
    detections = [
        detection for detection in report["detections"]
        if isinstance(detection, dict) and not detection.get("duplicate_detection") and _valid_box(detection.get("box"))
    ]
    facings = scene["facings"]
    assigned = match_one_to_one([detection["box"] for detection in detections], [facing["box"] for facing in facings])
    duplicates = false_positives = 0
    for index, detection in enumerate(detections):
        if index in assigned:
            continue
        if any(iou(detection["box"], facing["box"]) >= 0.5 for facing in facings):
            duplicates += 1
        else:
            false_positives += 1
    identified_correct = sum(detections[index].get("matched_sku") == facings[truth]["sku"] for index, truth in assigned.items())
    accepted = [index for index, detection in enumerate(detections) if detection.get("matched_sku")]
    accepted_correct = sum(
        index in assigned and detections[index]["matched_sku"] == facings[assigned[index]]["sku"] for index in accepted
    )

    expected_keys = {(int(slot["row"]), int(slot["slot"])) for slot in scene["slots"]}
    statuses: dict[tuple[int, int], str] = {}
    unexpected_rows = []
    for row in report["compliance"]:
        try:
            key = (int(row["row"]), int(row.get("slot", row.get("position"))))
        except (KeyError, TypeError, ValueError):
            unexpected_rows.append(str(row)[:80])
            continue
        if key not in expected_keys or key in statuses:
            unexpected_rows.append(f"R{key[0]}S{key[1]}: {row.get('status')}")
            continue
        statuses[key] = row.get("status")
    slot_rows = []
    for slot in scene["slots"]:
        actual = statuses.get((slot["row"], slot["slot"]))
        slot_rows.append({"slot": f"R{slot['row']}S{slot['slot']}", "expected": slot["expected_status"], "actual": actual,
                          "ok": actual is not None and actual == slot["expected_status"]})
    for extra in unexpected_rows:
        slot_rows.append({"slot": extra, "expected": None, "actual": "unexpected row", "ok": False})

    tags = [tag for tag in report.get("price_audit", {}).get("tags", []) if isinstance(tag, dict)]
    tag_hits = 0
    used_tags: set[int] = set()
    for truth_tag in scene["price_tags"]:
        tx1, ty1, tx2, ty2 = truth_tag["box"]
        for index, tag in enumerate(tags):
            price, box = tag.get("price"), tag.get("box")
            if index in used_tags or price is None or not _valid_box(box):
                continue
            center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if tx1 <= center_x <= tx2 and ty1 <= center_y <= ty2 and abs(float(price) - truth_tag["price"]) < 0.01:
                used_tags.add(index)
                tag_hits += 1
                break
    price_status = {row["sku"]: row.get("status") for row in report.get("price_results", []) if isinstance(row, dict)}
    price_rows = [{"sku": sku, "expected": status, "actual": price_status.get(sku), "ok": price_status.get(sku) == status}
                  for sku, status in scene["expected_price_status"].items()]
    return {
        "image": scene["image"],
        "error": None,
        "facings": len(facings),
        "detections": len(detections),
        "recalled": len(assigned),
        "duplicates": duplicates,
        "false_positives": false_positives,
        "missed": len(facings) - len(assigned),
        "accepted": len(accepted),
        "accepted_correct": accepted_correct,
        "identified_correct": identified_correct,
        "slots": slot_rows,
        "unexpected_slot_rows": unexpected_rows,
        "price_tags": len(scene["price_tags"]),
        "price_tags_read": tag_hits,
        "prices": price_rows,
        "compliance_rate": report.get("compliance_rate"),
        "report_id": report.get("report_id"),
    }


def evaluate(save: bool = False, confidence: float | None = None) -> dict:
    import app as web_app

    truth = json.loads((TEST_SET_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    problems = check_test_set(truth)
    if problems:
        raise SystemExit("Test set is stale or inconsistent:\n- " + "\n- ".join(problems))
    client = web_app.app.test_client()
    results = []
    with tempfile.TemporaryDirectory() as temp:
        patches = [] if save else [patch.object(web_app, "AUDIT_RESULTS_DIR", Path(temp)), patch.object(web_app, "UPLOAD_DIR", Path(temp))]
        for active in patches:
            active.start()
        try:
            for scene in truth["test"]:
                form = {"test_image": Path(scene["image"]).name, "store": "Test Set Store",
                        "planogram": "SNACKS_V1 (test set)", "price_check": "on"}
                if confidence is not None:
                    form["confidence_threshold"] = str(confidence)
                response = client.post("/api/analyze", data=form, content_type="multipart/form-data")
                payload = response.get_json(silent=True) or {}
                if response.status_code != 201 or payload.get("report_id") not in web_app._reports:
                    results.append(failed_scene(scene, f"HTTP {response.status_code}: {payload.get('error', 'no report')}"))
                    continue
                report = web_app._reports[payload["report_id"]]
                results.append(score_scene(scene, report))
                if not save:
                    web_app._reports.pop(payload["report_id"], None)
        finally:
            for active in patches:
                active.stop()

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    facings = sum(result["facings"] for result in results)
    slots = [row for result in results for row in result["slots"]]
    prices = [row for result in results for row in result["prices"]]
    summary = {
        "data": "synthetic test scenes (not real-store accuracy)",
        "scenes": len(results),
        "failed_scenes": sum(bool(result["error"]) for result in results),
        "detection_recall": ratio(sum(r["recalled"] for r in results), facings),
        "detection_precision": ratio(sum(r["recalled"] for r in results), sum(r["detections"] for r in results)),
        "duplicate_boxes": sum(r["duplicates"] for r in results),
        "false_positive_boxes": sum(r["false_positives"] for r in results),
        "sku_precision": ratio(sum(r["accepted_correct"] for r in results), sum(r["accepted"] for r in results)),
        "sku_coverage": ratio(sum(r["identified_correct"] for r in results), facings),
        "slot_status_accuracy": ratio(sum(row["ok"] for row in slots), len(slots)),
        "unexpected_slot_rows": sum(len(r["unexpected_slot_rows"]) for r in results),
        "price_tag_recall": ratio(sum(r["price_tags_read"] for r in results), sum(r["price_tags"] for r in results)),
        "price_status_accuracy": ratio(sum(row["ok"] for row in prices), len(prices)),
        "saved_to_history": save,
    }
    return {"summary": summary, "scenes": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", action="store_true", help="store the audits in the normal report history")
    parser.add_argument("--confidence", type=float, default=None, help="detector confidence threshold")
    args = parser.parse_args()
    outcome = evaluate(save=args.save, confidence=args.confidence)
    for scene in outcome["scenes"]:
        if scene["error"]:
            print(f"{scene['image']}: FAILED - {scene['error']}")
        wrong = [f"{row['slot']}: expected {row['expected']}, got {row['actual']}" for row in scene["slots"] if not row["ok"]]
        wrong += [f"price {row['sku']}: expected {row['expected']}, got {row['actual']}" for row in scene["prices"] if not row["ok"]]
        print(f"{scene['image']}: recall {scene['recalled']}/{scene['facings']} (dup {scene['duplicates']}, "
              f"fp {scene['false_positives']}), SKU ok {scene['identified_correct']}, "
              f"tags {scene['price_tags_read']}/{scene['price_tags']}, compliance {scene['compliance_rate']}%")
        for line in wrong:
            print("   -", line)
    print(json.dumps(outcome["summary"], indent=2))


if __name__ == "__main__":
    main()
