"""Bounding-box cropping and calibrated CLIP SKU candidate matching."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image

from .catalog import CatalogueEmbeddingCache, load_catalogue_references
from .encoder import ClipEncoderUnavailable, ImageEncoder, TransformersClipEncoder


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

    def _get_encoder(self) -> ImageEncoder:
        if self.encoder is None:
            try:
                self.encoder = TransformersClipEncoder(self.model_id)
            except ClipEncoderUnavailable as error:
                raise MatchingUnavailable(str(error)) from error
        return self.encoder

    def match_detections(
        self,
        image: Image.Image,
        detections: list[dict[str, Any]],
        catalogue: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        encoder = self._get_encoder()
        references = load_catalogue_references(catalogue, self.data_dir)
        if not references:
            return [self._needs_review(detection, "No valid catalogue reference images are available.") for detection in detections]

        catalogue_embeddings, cache_stats = self.cache.get_embeddings(references, encoder)
        reference_by_sku = {str(reference["sku"]): reference for reference in references}
        prepared: list[tuple[dict[str, Any], Image.Image]] = []
        results: list[dict[str, Any] | None] = [None] * len(detections)
        for index, detection in enumerate(detections):
            crop = crop_detection(image, detection)
            if crop is None:
                results[index] = self._needs_review(detection, "Detection bounding box is missing or invalid; crop could not be made.")
            else:
                prepared.append(({"index": index, "detection": detection}, crop))

        if prepared:
            crop_vectors = encoder.encode_images([crop for _, crop in prepared])
            if len(crop_vectors) != len(prepared):
                raise MatchingUnavailable("CLIP encoder returned an unexpected number of crop embeddings")
            for (metadata, _crop), vector in zip(prepared, crop_vectors):
                ranked = sorted(
                    (
                        (sku, cosine_similarity(vector, embedding))
                        for sku, embedding in catalogue_embeddings.items()
                    ),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
                candidates = [
                    {
                        "sku": sku,
                        "product": reference_by_sku[sku].get("name", ""),
                        "similarity": round(float(score), 6),
                    }
                    for sku, score in ranked[: self.top_k]
                ]
                top_score = ranked[0][1] if ranked else -1.0
                second_score = ranked[1][1] if len(ranked) > 1 else -1.0
                margin = top_score - second_score
                accepted = (
                    bool(ranked)
                    and self.acceptance_threshold is not None
                    and top_score >= self.acceptance_threshold
                    and margin >= self.minimum_margin
                )
                if accepted:
                    top_sku = ranked[0][0]
                    status = "Matched"
                    detail = "Accepted using configured calibrated threshold and candidate margin."
                else:
                    top_sku = None
                    status = "Unknown / Needs Review"
                    if self.acceptance_threshold is None:
                        detail = "No validation-calibrated acceptance threshold is configured; candidates are advisory only."
                    elif top_score < self.acceptance_threshold:
                        detail = "Top cosine similarity is below the configured acceptance threshold."
                    else:
                        detail = "Top-two candidate margin is below the configured ambiguity margin."
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
                    "accepted_sku": top_sku,
                    "status": status,
                    "detail": detail,
                    "threshold": self.acceptance_threshold,
                    "minimum_margin": self.minimum_margin,
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
        return match[0] if match else self._needs_review(detection, "No match result was produced.")

    @staticmethod
    def _needs_review(detection: dict[str, Any], detail: str) -> dict[str, Any]:
        return {
            "detector_label": str(detection.get("label", "")),
            "detector_confidence": detection.get("confidence"),
            "box": detection.get("box"),
            "image_width": detection.get("image_width"),
            "image_height": detection.get("image_height"),
            "candidates": [],
            "top_similarity": None,
            "top_margin": None,
            "accepted_sku": None,
            "status": "Unknown / Needs Review",
            "detail": detail,
            "similarity_is_probability": False,
        }
