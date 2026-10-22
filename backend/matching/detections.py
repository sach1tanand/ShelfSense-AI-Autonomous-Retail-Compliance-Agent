"""Utilities for suppressing duplicate detector boxes without losing source evidence."""

from __future__ import annotations

from typing import Any


def _xyxy(detection: dict[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        box = tuple(float(value) for value in detection.get("box", []))
    except (TypeError, ValueError):
        return None
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (area_left + area_right - intersection)


def suppress_duplicate_detections(
    detections: list[dict[str, Any]], iou_threshold: float = 0.82,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Return confidence-ranked unique boxes and {duplicate_index: kept_index} references.

    Only same detector-class boxes with very high overlap are suppressed. Product identity
    is deliberately not used: separate adjacent same-SKU facings must remain countable.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be within (0, 1]")
    ranked = sorted(
        range(len(detections)),
        key=lambda index: (float(detections[index].get("confidence") or 0), -index),
        reverse=True,
    )
    kept: list[int] = []
    duplicates: dict[int, int] = {}
    for index in ranked:
        box = _xyxy(detections[index])
        matched = None
        if box:
            class_id = detections[index].get("class_id", detections[index].get("label"))
            for kept_index in kept:
                kept_class = detections[kept_index].get("class_id", detections[kept_index].get("label"))
                kept_box = _xyxy(detections[kept_index])
                if class_id == kept_class and kept_box and _iou(box, kept_box) >= iou_threshold:
                    matched = kept_index
                    break
        if matched is None:
            kept.append(index)
        else:
            duplicates[index] = matched
    return [detections[index] for index in sorted(kept)], duplicates
