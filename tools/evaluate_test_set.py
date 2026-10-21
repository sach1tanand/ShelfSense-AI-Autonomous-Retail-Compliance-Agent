"""Run every labelled test scene through the real audit workflow and score it.

Each scene goes through POST /api/analyze (real detector, SKU matcher, slot mapping, OCR,
planogram comparison) and is compared with data/test_set/ground_truth.json:

* detection recall: ground-truth facings overlapped (IoU >= 0.5) by a detection
* SKU precision / coverage: accepted SKU matches that are correct / facings correctly identified
* slot status accuracy: report slot status equals the expected status
* price tag recall and price status accuracy

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
from tools.calibrate_matching import iou


def score_scene(scene: dict, report: dict) -> dict:
    detections = report["detections"]
    recalled = identified_correct = 0
    for facing in scene["facings"]:
        best = max(detections, key=lambda detection: iou(detection["box"], facing["box"]), default=None)
        if best and iou(best["box"], facing["box"]) >= 0.5:
            recalled += 1
            identified_correct += best.get("matched_sku") == facing["sku"]
    accepted = [detection for detection in detections if detection.get("matched_sku")]
    accepted_correct = 0
    for detection in accepted:
        best = max(scene["facings"], key=lambda facing: iou(detection["box"], facing["box"]), default=None)
        accepted_correct += bool(best and iou(detection["box"], best["box"]) >= 0.5 and best["sku"] == detection["matched_sku"])

    statuses = {(int(row["row"]), int(row["slot"])): row["status"] for row in report["compliance"]}
    slot_rows = []
    for slot in scene["slots"]:
        actual = statuses.get((slot["row"], slot["slot"]))
        slot_rows.append({"slot": f"R{slot['row']}S{slot['slot']}", "expected": slot["expected_status"], "actual": actual,
                          "ok": actual == slot["expected_status"]})

    tags_read = {(tag.get("slot"), round(tag["price"], 2)) for tag in report.get("price_audit", {}).get("tags", [])}
    tag_hits = sum((f"R{tag['row']}S{tag['slot']}", round(tag["price"], 2)) in tags_read for tag in scene["price_tags"])
    price_status = {row["sku"]: row["status"] for row in report.get("price_results", [])}
    price_rows = [{"sku": sku, "expected": status, "actual": price_status.get(sku), "ok": price_status.get(sku) == status}
                  for sku, status in scene["expected_price_status"].items()]
    return {
        "image": scene["image"],
        "facings": len(scene["facings"]),
        "recalled": recalled,
        "accepted": len(accepted),
        "accepted_correct": accepted_correct,
        "identified_correct": identified_correct,
        "slots": slot_rows,
        "price_tags": len(scene["price_tags"]),
        "price_tags_read": tag_hits,
        "prices": price_rows,
        "compliance_rate": report.get("compliance_rate"),
        "report_id": report.get("report_id"),
    }


def evaluate(save: bool = False, confidence: float | None = None) -> dict:
    import app as web_app

    truth = json.loads((TEST_SET_DIR / "ground_truth.json").read_text(encoding="utf-8"))
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
                if response.status_code != 201:
                    raise SystemExit(f"{scene['image']}: {response.status_code} {response.json}")
                report = web_app._reports[response.json["report_id"]]
                results.append(score_scene(scene, report))
                if not save:
                    web_app._reports.pop(response.json["report_id"], None)
        finally:
            for active in patches:
                active.stop()

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    facings = sum(result["facings"] for result in results)
    slots = [row for result in results for row in result["slots"]]
    prices = [row for result in results for row in result["prices"]]
    summary = {
        "scenes": len(results),
        "detection_recall": ratio(sum(r["recalled"] for r in results), facings),
        "sku_precision": ratio(sum(r["accepted_correct"] for r in results), sum(r["accepted"] for r in results)),
        "sku_coverage": ratio(sum(r["identified_correct"] for r in results), facings),
        "slot_status_accuracy": ratio(sum(row["ok"] for row in slots), len(slots)),
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
        wrong = [f"{row['slot']}: expected {row['expected']}, got {row['actual']}" for row in scene["slots"] if not row["ok"]]
        wrong += [f"price {row['sku']}: expected {row['expected']}, got {row['actual']}" for row in scene["prices"] if not row["ok"]]
        print(f"{scene['image']}: recall {scene['recalled']}/{scene['facings']}, SKU ok {scene['identified_correct']}, "
              f"tags {scene['price_tags_read']}/{scene['price_tags']}, compliance {scene['compliance_rate']}%")
        for line in wrong:
            print("   -", line)
    print(json.dumps(outcome["summary"], indent=2))


if __name__ == "__main__":
    main()
