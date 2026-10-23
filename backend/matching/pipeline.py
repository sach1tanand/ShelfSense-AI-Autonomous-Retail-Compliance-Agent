"""Orchestrate image-embedding SKU recognition, modular slot mapping, comparison, and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from config.settings import SKU_MATCH_THRESHOLD

from .calibration import load_runtime_calibration
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
        acceptance_threshold: float | None = SKU_MATCH_THRESHOLD,
        minimum_margin: float = 0.0,
        top_k: int = 5,
        detector_fingerprint: dict[str, Any] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.calibration_path = Path(calibration_path or self.data_dir / "clip_calibration.json")
        self.calibration = load_runtime_calibration(
            self.calibration_path, model_id, detector_fingerprint=detector_fingerprint,
        )
        if acceptance_threshold is None:
            acceptance_threshold = SKU_MATCH_THRESHOLD
        self.recognizer = ProductRecognizer(
            data_dir=self.data_dir,
            cache_path=cache_path or self.data_dir / "cache" / "clip_catalogue_embeddings.json",
            encoder=encoder,
            model_id=model_id,
            acceptance_threshold=acceptance_threshold,
            minimum_margin=minimum_margin,
            top_k=top_k,
            calibration_state=self.calibration["state"],
        )
        self.slot_mapper = slot_mapper or CoordinateShelfMapper()

    def run(
        self,
        image_path: Path,
        detections: list[dict[str, Any]],
        catalogue: list[dict[str, Any]],
        planogram: list[dict[str, Any]],
        slot_mapper: Any = None,
        acceptance_threshold: float | None = None,
        minimum_margin: float | None = None,
    ) -> dict[str, Any]:
        """Match, map and compare. Run-level settings override constructor defaults; the
        validation calibration remains available as reference metadata."""
        slot_mapper = slot_mapper or self.slot_mapper
        threshold = self.recognizer.acceptance_threshold if acceptance_threshold is None else float(acceptance_threshold)
        margin = self.recognizer.minimum_margin if minimum_margin is None else float(minimum_margin)
        threshold_source = "operator" if acceptance_threshold is not None else "configured_default"
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        try:
            matches = self.recognizer.match_detections(
                image, detections, catalogue, acceptance_threshold=threshold, minimum_margin=margin,
            )
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
                    "reason_code": "matcher_unavailable",
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
            "SKU identity uses the configured best-match similarity threshold; validation calibration is reference metadata.",
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
                "path": self.calibration_path.name,
                "model_id": self.recognizer.model_id,
                "acceptance_threshold": threshold,
                "minimum_margin": margin,
                "threshold_source": threshold_source,
                "calibrated_threshold": self.calibration["threshold"],
                "calibrated_minimum_margin": self.calibration["minimum_margin"],
                "calibration_status": "explicit" if self.calibration["state"] == "explicit"
                else self.calibration["payload"].get("status", "uncalibrated"),
                "state": self.calibration["state"],
                "issues": self.calibration["issues"],
                "detector_model": self.calibration["payload"].get("detector_model"),
                "metrics": self.calibration["payload"].get("metrics"),
                "created_at": self.calibration["payload"].get("created_at"),
            },
            "matching_error": matching_error,
            "empty_slot_count": sum(not str(row.get("expected_sku", row.get("sku", ""))).strip() for row in planogram),
            "detection_count": len(detections),
        }
