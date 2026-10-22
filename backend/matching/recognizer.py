"""Bounding-box cropping and calibrated CLIP SKU candidate matching."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image

from .catalog import CatalogueEmbeddingCache, load_catalogue_references
from .encoder import ClipEncoderUnavailable, ImageEncoder, create_encoder
from .feedback import NOT_PRODUCT_SKU, load_feedback_references

_DEFAULT = object()


class MatchingUnavailable(RuntimeError):
    """Raised when SKU recognition cannot run due to missing model/runtime data."""


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embeddings must have the same non-zero dimension")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def crop_detection(image: Image.Image, detection: dict[str, Any]) -> Image.Image | None:
    """Crop a detector xyxy box after scaling it to source-image dimensions."""
    box = detection.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
        reported_width = float(detection.get("image_width", image.width))
        reported_height = float(detection.get("image_height", image.height))
    except (TypeError, ValueError):
        return None
    if reported_width <= 0 or reported_height <= 0:
        return None
    scale_x, scale_y = image.width / reported_width, image.height / reported_height
    left = max(0, min(image.width, round(x1 * scale_x)))
    top = max(0, min(image.height, round(y1 * scale_y)))
    right = max(0, min(image.width, round(x2 * scale_x)))
    bottom = max(0, min(image.height, round(y2 * scale_y)))
    if right <= left or bottom <= top:
        return None
    return image.crop((left, top, right, bottom)).convert("RGB")


class ProductRecognizer:
    """Top-k image-to-image SKU recognizer with fail-closed calibrated thresholds.

    An absent acceptance threshold means no prediction is accepted. Candidate
    rankings remain available for review, but a raw CLIP cosine value is not a
    calibrated probability and cannot be used as a guarantee of correctness.
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        cache_path: Path | None = None,
        encoder: ImageEncoder | None = None,
        model_id: str = "openai/clip-vit-base-patch32",
        acceptance_threshold: float | None = None,
        minimum_margin: float = 0.0,
        top_k: int = 5,
        calibration_state: str | None = None,
        use_feedback: bool = True,
    ):
        if acceptance_threshold is not None and not -1.0 <= acceptance_threshold <= 1.0:
            raise ValueError("acceptance_threshold must be within [-1, 1]")
        if minimum_margin < 0:
            raise ValueError("minimum_margin must be non-negative")
        self.data_dir = Path(data_dir)
        self.cache = CatalogueEmbeddingCache(cache_path or self.data_dir / "cache" / "clip_catalogue_embeddings.json")
        self.encoder = encoder
        self.model_id = model_id
        self.acceptance_threshold = acceptance_threshold
        self.minimum_margin = minimum_margin
        self.top_k = max(1, top_k)
        # "valid", "stale", "missing", ... (see calibration.load_runtime_calibration); None = set directly.
        self.calibration_state = calibration_state
        # Reviewed shelf crops extend the gallery; calibration turns this off to measure the catalogue alone.
        self.use_feedback = use_feedback

    def _get_encoder(self) -> ImageEncoder:
        if self.encoder is None:
            try:
                self.encoder = create_encoder(self.model_id)
            except ClipEncoderUnavailable as error:
                raise MatchingUnavailable(str(error)) from error
        return self.encoder

    def match_detections(
        self,
        image: Image.Image,
        detections: list[dict[str, Any]],
        catalogue: list[dict[str, Any]],
        *,
        acceptance_threshold: float | None | object = _DEFAULT,
        minimum_margin: float | object = _DEFAULT,
    ) -> list[dict[str, Any]]:
        """Rank each crop against every reference image and accept the best SKU when it clears
        the threshold and margin. Thresholds passed here apply to this call only, so concurrent
        audits with different operator settings do not interfere."""
        threshold = self.acceptance_threshold if acceptance_threshold is _DEFAULT else acceptance_threshold
        margin_required = self.minimum_margin if minimum_margin is _DEFAULT else float(minimum_margin)
        if threshold is not None and not -1.0 <= float(threshold) <= 1.0:
            raise ValueError("acceptance_threshold must be within [-1, 1]")
        if margin_required < 0:
            raise ValueError("minimum_margin must be non-negative")
        encoder = self._get_encoder()
        references = load_catalogue_references(catalogue, self.data_dir)
        if not references:
            return [
                self._needs_review(detection, "No valid catalogue reference images are available.",
                                   "no_catalogue_references", index, threshold, margin_required)
                for index, detection in enumerate(detections)
            ]
        if self.use_feedback:
            references += load_feedback_references(self.data_dir, {str(reference["sku"]) for reference in references})

        reference_embeddings, cache_stats = self.cache.get_embeddings(references, encoder)
        name_by_sku: dict[str, str] = {}
        for reference in references:
            name_by_sku.setdefault(str(reference["sku"]), reference.get("name", ""))
        prepared: list[tuple[dict[str, Any], Image.Image]] = []
        results: list[dict[str, Any] | None] = [None] * len(detections)
        for index, detection in enumerate(detections):
            crop = crop_detection(image, detection)
            if crop is None:
                results[index] = self._needs_review(
                    detection, "Detection bounding box is missing or invalid; crop could not be made.", "invalid_crop",
                    index, threshold, margin_required,
                )
            else:
                prepared.append(({"index": index, "detection": detection}, crop))

        if prepared:
            crop_vectors = encoder.encode_images([crop for _, crop in prepared])
            if len(crop_vectors) != len(prepared):
                raise MatchingUnavailable("Image encoder returned an unexpected number of crop embeddings")
            for (metadata, crop), vector in zip(prepared, crop_vectors):
                # A SKU scores the best similarity over all of its reference images.
                best: dict[str, tuple[float, str]] = {}
                for reference in references:
                    embedding = reference_embeddings.get(str(reference.get("key") or reference["sku"]))
                    if embedding is None:
                        continue
                    score = cosine_similarity(vector, embedding)
                    sku = str(reference["sku"])
                    if sku not in best or score > best[sku][0]:
                        best[sku] = (score, reference.get("source", "catalogue"))
                ranked = sorted(best.items(), key=lambda pair: pair[1][0], reverse=True)
                candidates = [
                    {
                        "sku": sku,
                        "product": name_by_sku.get(sku, ""),
                        "similarity": round(float(score), 6),
                        "reference": source,
                    }
                    for sku, (score, source) in ranked[: self.top_k]
                ]
                top_score = ranked[0][1][0] if ranked else -1.0
                second_score = ranked[1][1][0] if len(ranked) > 1 else -1.0
                margin = top_score - second_score
                top_is_negative = bool(ranked) and ranked[0][0] == NOT_PRODUCT_SKU
                accepted = (
                    bool(ranked)
                    and not top_is_negative
                    and threshold is not None
                    and top_score >= threshold
                    and margin >= margin_required
                )
                if accepted:
                    top_sku = ranked[0][0]
                    status = "Matched"
                    reason_code = "accepted"
                    detail = (f"Accepted: similarity {top_score:.3f} >= threshold {threshold:.3f} and margin "
                              f"{margin:.3f} >= {margin_required:.3f}"
                              + (" (matched a reviewed reference photo)." if ranked[0][1][1] == "review" else "."))
                else:
                    top_sku = None
                    status = "Unknown / Needs Review"
                    if not ranked:
                        reason_code = "no_candidates"
                        detail = "No catalogue candidates could be scored for this crop."
                    elif top_is_negative:
                        reason_code = "known_non_product"
                        detail = "This crop looks most like an object a reviewer marked as not a product."
                    elif threshold is None:
                        reason_code = "no_calibration"
                        detail = "No validation-calibrated acceptance threshold is configured; candidates are advisory only."
                    elif top_score < threshold:
                        reason_code = "below_threshold"
                        detail = (
                            f"Top cosine similarity {top_score:.3f} is below the acceptance threshold "
                            f"{threshold:.3f}; the product may not be in the catalogue."
                        )
                    else:
                        reason_code = "low_margin"
                        detail = (
                            f"Top-two candidate margin {margin:.3f} is below the required margin "
                            f"{margin_required:.3f}; two catalogue products look alike."
                        )
                results[metadata["index"]] = {
                    "detection_index": metadata["index"],
                    "detector_label": str(metadata["detection"].get("label", "")),
                    "detector_confidence": metadata["detection"].get("confidence"),
                    "box": metadata["detection"].get("box"),
                    "image_width": metadata["detection"].get("image_width", image.width),
                    "image_height": metadata["detection"].get("image_height", image.height),
                    "candidates": candidates,
                    "top_similarity": round(float(top_score), 6) if ranked else None,
                    "top_margin": round(float(margin), 6) if ranked else None,
                    "crop_width": crop.width,
                    "crop_height": crop.height,
                    "accepted_sku": top_sku,
                    "status": status,
                    "reason_code": reason_code,
                    "detail": detail,
                    "threshold": threshold,
                    "minimum_margin": margin_required,
                    "calibration_state": self.calibration_state,
                    "similarity_is_probability": False,
                    "catalogue_cache": cache_stats,
                }
        return [result for result in results if result is not None]

    def match_detections_from_path(
        self,
        image_path: Path,
        detection: dict[str, Any],
        catalogue: list[dict[str, Any]],
    ) -> dict[str, Any]:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        match = self.match_detections(image, [detection], catalogue)
        return match[0] if match else self._needs_review(
            detection, "No match result was produced.", "no_result", 0, self.acceptance_threshold, self.minimum_margin,
        )

    def _needs_review(self, detection: dict[str, Any], detail: str, reason_code: str, index: int,
                      threshold: float | None, margin: float) -> dict[str, Any]:
        return {
            "detection_index": index,
            "detector_label": str(detection.get("label", "")),
            "detector_confidence": detection.get("confidence"),
            "box": detection.get("box"),
            "image_width": detection.get("image_width"),
            "image_height": detection.get("image_height"),
            "candidates": [],
            "top_similarity": None,
            "top_margin": None,
            "crop_width": None,
            "crop_height": None,
            "accepted_sku": None,
            "status": "Unknown / Needs Review",
            "reason_code": reason_code,
            "detail": detail,
            "threshold": threshold,
            "minimum_margin": margin,
            "calibration_state": self.calibration_state,
            "similarity_is_probability": False,
        }
