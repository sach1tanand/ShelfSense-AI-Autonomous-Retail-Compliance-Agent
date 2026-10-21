"""Evaluate the external sku110k-yolov8n.pt baseline detector against ShelfSense AI."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Set env before importing config
MODEL_FILE = ROOT_DIR / "models" / "external" / "sku110k-yolov8n.pt"
os.environ["SHELFSENSE_MODEL"] = str(MODEL_FILE)
os.environ["SHELFSENSE_CONFIDENCE"] = "0.35"

import cv2
from PIL import Image
from ultralytics import YOLO

import config.settings
config.settings.MODEL_PATH = str(MODEL_FILE)

import src.detector
src.detector.MODEL_PATH = str(MODEL_FILE)

from backend.matching.encoder import PerceptualFeatureEncoder
from backend.matching.pipeline import ProductMatchingPipeline
from backend.matching.shelf_mapper import CoordinateShelfMapper
from src.detector import detect_products
from src.planogram import evaluate_planogram


def run_baseline_evaluation() -> dict:
    image_path = ROOT_DIR / "data" / "samples" / "real_supermarket_shelf.jpg"
    annotated_path = ROOT_DIR / "data" / "samples" / "annotated_shelf_sku110k.jpg"

    assert MODEL_FILE.is_file(), f"Model not found at {MODEL_FILE}"
    assert image_path.is_file(), f"Image not found at {image_path}"

    # 1. Direct Ultralytics inference
    model = YOLO(str(MODEL_FILE))
    results = model.predict(
        source=str(image_path),
        conf=0.35,
        iou=0.55,
        imgsz=640,
        verbose=False,
    )
    res = results[0]
    boxes = res.boxes
    confs = [float(b.conf.item()) for b in boxes]
    classes = [int(b.cls.item()) for b in boxes]
    class_names = [res.names[c] for c in classes]

    # Save annotated image
    annotated = res.plot(line_width=2, font_size=1)
    cv2.imwrite(str(annotated_path), annotated)

    # 2. Test via existing detector adapter
    adapter_detections = detect_products(image_path, confidence=0.35)

    # 3. Test generic 'object' class in src/planogram.py
    with open(ROOT_DIR / "data" / "catalogue.csv", "r", encoding="utf-8") as f:
        catalogue = list(csv.DictReader(f))
    with open(ROOT_DIR / "data" / "planogram.csv", "r", encoding="utf-8") as f:
        planogram = list(csv.DictReader(f))

    plano_audit = evaluate_planogram(planogram, catalogue, adapter_detections)
    unresolved_count = sum(
        1 for d in plano_audit["detections"]
        if d.get("status") == "Needs Review" and "does not resolve to a unique catalogue SKU" in d.get("detail", "")
    )

    # 4. Test in matching pipeline (crop -> embed -> rank)
    pipeline = ProductMatchingPipeline(
        data_dir=ROOT_DIR / "data",
        encoder=PerceptualFeatureEncoder(),
        slot_mapper=CoordinateShelfMapper(trusted=False),
        acceptance_threshold=None,
    )
    matching_audit = pipeline.run(image_path, adapter_detections, catalogue, planogram)

    summary = {
        "model_file": str(MODEL_FILE),
        "image_file": str(image_path),
        "annotated_output": str(annotated_path),
        "total_detections": len(boxes),
        "classes_detected": list(set(class_names)),
        "confidence_min": round(min(confs), 4) if confs else 0.0,
        "confidence_max": round(max(confs), 4) if confs else 0.0,
        "confidence_mean": round(sum(confs) / len(confs), 4) if confs else 0.0,
        "top_5_confidences": [round(c, 4) for c in sorted(confs, reverse=True)[:5]],
        "adapter_detections_count": len(adapter_detections),
        "planogram_unresolved_detections": unresolved_count,
        "matching_crops_processed": len(matching_audit["matches"]),
        "matching_findings_generated": len(matching_audit["findings"]),
    }
    return summary


if __name__ == "__main__":
    out = run_baseline_evaluation()
    for k, v in out.items():
        print(f"{k}: {v}")
