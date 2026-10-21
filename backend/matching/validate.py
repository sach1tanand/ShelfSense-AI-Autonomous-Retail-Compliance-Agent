"""Run CLIP scoring on a manually labelled crop manifest and calibrate acceptance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .calibration import calibrate_threshold, save_calibration
from .recognizer import ProductRecognizer


def score_labelled_manifest(
    manifest_path: Path,
    catalogue_path: Path,
    data_dir: Path,
    *,
    model_id: str = "openai/clip-vit-base-patch32",
) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        if manifest_path.suffix.lower() == ".csv":
            records = list(csv.DictReader(stream))
        else:
            records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("Validation manifest must be a JSON list or CSV rows")
    with Path(catalogue_path).open(encoding="utf-8-sig", newline="") as stream:
        catalogue_rows = list(csv.DictReader(stream))

    products = []
    for row in catalogue_rows:
        name = row.get("name") or row.get("product_name", "")
        products.append({
            **row,
            "name": name,
            "model_label": row.get("model_label") or name.casefold(),
            "active": (row.get("active") or row.get("status", "active")).casefold() in {"true", "active", "1", "yes"} and "true" or "false",
        })

    data_dir = Path(data_dir).resolve()
    recognizer = ProductRecognizer(data_dir=data_dir, model_id=model_id, acceptance_threshold=None)
    scored = []
    for record in records:
        image_path = Path(str(record.get("image_path", "")))
        if not image_path.is_absolute():
            image_path = (manifest_path.parent / image_path).resolve()
        box = record.get("box")
        if isinstance(box, str):
            box = json.loads(box)
        detection = {
            "label": "labelled-validation-crop",
            "box": box,
            "image_width": record.get("image_width"),
            "image_height": record.get("image_height"),
        }
        match = recognizer.match_detections_from_path(image_path, detection, products)
        scored.append({
            "image_path": str(image_path),
            "expected_sku": str(record.get("expected_sku", "")).strip(),
            "candidates": match.get("candidates", []),
            "status": match.get("status"),
        })
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON/CSV rows: image_path, box, image_width, image_height, expected_sku")
    parser.add_argument("--catalogue", type=Path, default=Path("data/catalogue.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=Path("data/clip_calibration.json"))
    args = parser.parse_args()

    scored = score_labelled_manifest(args.manifest, args.catalogue, args.data_dir, model_id=args.model_id)
    calibration = calibrate_threshold(scored, target_precision=args.target_precision)
    save_calibration(args.output, calibration, args.model_id)
    print(json.dumps({"calibration": calibration, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
