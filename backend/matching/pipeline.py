"""Orchestrate CLIP recognition, modular slot mapping, comparison, and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .planogram_compare import compare_planogram
from .recognizer import MatchingUnavailable, ProductRecognizer
from .shelf_mapper import CoordinateShelfMapper


class ProductMatchingPipeline:
    def __init__(
        self,
        *,
        data_dir: Path,
        cache_path: Path | None = None,
        calibration_path: Path | None = None,
        model_id: str = "openai/clip-vit-base-patch32",
        encoder: Any = None,
        slot_mapper: Any = None,
        acceptance_threshold: float | None = None,
        minimum_margin: float = 0.0,
        top_k: int = 5,
    ):
        self.data_dir = Path(data_dir)
        self.calibration_path = Path(calibration_path or self.data_dir / "clip_calibration.json")
        calibration = self._read_calibration()
        if acceptance_threshold is None and calibration.get("status") == "calibrated_on_validation_set" and calibration.get("model_id") == model_id:
            value = calibration.get("threshold")
            acceptance_threshold = float(value) if value is not None else None
            minimum_margin = float(calibration.get("minimum_margin", minimum_margin))
        self.recognizer = ProductRecognizer(
            data_dir=self.data_dir,
            cache_path=cache_path or self.data_dir / "cache" / "clip_catalogue_embeddings.json",
            encoder=encoder,
            model_id=model_id,
            acceptance_threshold=acceptance_threshold,
            minimum_margin=minimum_margin,
            top_k=top_k,
        )
        self.slot_mapper = slot_mapper or CoordinateShelfMapper()

    def _read_calibration(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"status": "uncalibrated"}
        return payload if isinstance(payload, dict) else {"status": "uncalibrated"}

    def run(
        self,
        image_path: Path,
        detections: list[dict[str, Any]],
        catalogue: list[dict[str, Any]],
        planogram: list[dict[str, Any]],
        slot_mapper: Any = None,
    ) -> dict[str, Any]:
        slot_mapper = slot_mapper or self.slot_mapper
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        try:
            matches = self.recognizer.match_detections(image, detections, catalogue)
            for index, match in enumerate(matches):
                match["detection_index"] = int(detections[index].get("_source_detection_index", index))
            matching_error = None
        except MatchingUnavailable as error:
            matches = [
                {
                    "detection_index": int(detection.get("_source_detection_index", index)),
                    "detector_label": str(detection.get("label", "")),
                    "detector_confidence": detection.get("confidence"),
                    "box": detection.get("box"),
                    "image_width": detection.get("image_width", image.width),
                    "image_height": detection.get("image_height", image.height),
                    "candidates": [],
                    "top_similarity": None,
                    "top_margin": None,
                    "accepted_sku": None,
                    "status": "Unknown / Needs Review",
                    "detail": str(error),
                    "similarity_is_probability": False,
                }
                for index, detection in enumerate(detections)
            ]
            matching_error = str(error)

        comparison = compare_planogram(planogram, matches, slot_mapper)
        slots = [finding for finding in comparison["findings"] if "slot" in finding]
        unmatched = [finding for finding in comparison["findings"] if "slot" not in finding]
        mapping_trusted = bool(getattr(slot_mapper, "trusted", False))
        assumptions = [
            "Equal-grid mapping estimates horizontal row bands and left-to-right slots from bbox centers.",
            "The operator confirmed the image follows the calibrated fixed-camera shelf layout."
            if mapping_trusted else
            "This mapping is not perspective-aware or calibrated against shelf boundaries.",
            "SKU acceptance requires a labelled validation calibration file; otherwise all candidates remain Needs Review.",
            "A product with no mapped detection is Needs Review, not Missing, because occlusion/model failure cannot be ruled out.",
        ]
        return {
            "matches": matches,
            "findings": comparison["findings"],
            "slot_findings": slots,
            "unmatched_findings": unmatched,
            "slot_mapping": {
                "id": comparison["slot_mapping_id"],
                "trusted": mapping_trusted,
                "assumptions": assumptions,
            },
            "calibration": {
                "path": str(self.calibration_path),
                "model_id": self.recognizer.model_id,
                "acceptance_threshold": self.recognizer.acceptance_threshold,
                "minimum_margin": self.recognizer.minimum_margin,
                "calibration_status": self._read_calibration().get("status", "uncalibrated"),
            },
            "matching_error": matching_error,
            "empty_slot_count": sum(not str(row.get("expected_sku", row.get("sku", ""))).strip() for row in planogram),
            "detection_count": len(detections),
        }
