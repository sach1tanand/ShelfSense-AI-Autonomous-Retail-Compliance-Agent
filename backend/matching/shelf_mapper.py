"""Modular bounding-box-to-planogram slot mapping.

The default mapper uses equal vertical row bands and equal horizontal slot bands.
This is only an approximation: shelf-region calibration or perspective-aware
mapping should replace it for operationally trustworthy slot conclusions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


class CoordinateShelfMapper:
    """Map normalized bbox centers to equal-grid planogram rows and slots."""

    mapping_id = "equal-grid-v1"
    trusted = False

    def __init__(self, boundary_margin: float = 0.06, trusted: bool = False):
        if not 0 <= boundary_margin < 0.5:
            raise ValueError("boundary_margin must be in [0, 0.5)")
        self.boundary_margin = boundary_margin
        # Only a calibrated deployment should set trusted=True; untrusted mappings stay proposals.
        self.trusted = trusted

    @staticmethod
    def cluster_detections_by_row(detections: list[dict[str, Any]], gap_ratio: float = 0.5) -> list[list[dict[str, Any]]]:
        """Group boxes into shelf rows by vertical centre, top-to-bottom, each row left-to-right.

        A new row starts when the centre-y gap exceeds gap_ratio x the median box height.
        """
        boxed = []
        for detection in detections:
            try:
                x1, y1, x2, y2 = [float(value) for value in detection.get("box")]
            except (TypeError, ValueError):
                continue
            if x2 > x1 and y2 > y1:
                boxed.append(((y1 + y2) / 2, (x1 + x2) / 2, y2 - y1, detection))
        if not boxed:
            return []
        heights = sorted(item[2] for item in boxed)
        threshold = heights[len(heights) // 2] * gap_ratio
        boxed.sort(key=lambda item: item[0])
        rows: list[list[tuple]] = [[boxed[0]]]
        for item in boxed[1:]:
            row_center = sum(member[0] for member in rows[-1]) / len(rows[-1])
            if item[0] - row_center > threshold:
                rows.append([item])
            else:
                rows[-1].append(item)
        return [[member[3] for member in sorted(row, key=lambda member: member[1])] for row in rows]

    def map_detection(self, detection: dict[str, Any], planogram: list[dict[str, Any]]) -> dict[str, Any]:
        box = detection.get("box")
        try:
            image_width = float(detection.get("image_width", 0))
            image_height = float(detection.get("image_height", 0))
            x1, y1, x2, y2 = [float(value) for value in box]
        except (TypeError, ValueError):
            return self._review("Missing or invalid bounding box/image dimensions.")
        if image_width <= 0 or image_height <= 0 or x2 <= x1 or y2 <= y1:
            return self._review("Bounding box or image dimensions are invalid.")

        grouped: dict[tuple[str, str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for row in planogram:
            key = (
                str(row.get("store_id", "")),
                str(row.get("planogram_id", "")),
                str(row.get("physical_shelf", row.get("shelf", ""))),
            )
            grouped[key][str(row.get("row", ""))].append(row)
        if len(grouped) != 1 or not grouped:
            return self._review("The image-to-planogram mapper requires exactly one configured shelf group.")

        shelf_key, rows_by_number = next(iter(grouped.items()))
        ordered_rows = sorted(rows_by_number, key=lambda value: (int(value) if value.isdigit() else 10**9, value))
        center_x = ((x1 + x2) / 2) / image_width
        center_y = ((y1 + y2) / 2) / image_height
        row_position = center_y * len(ordered_rows)
        row_index = min(int(row_position), len(ordered_rows) - 1)
        row_fraction = row_position - row_index
        if row_fraction < self.boundary_margin or 1 - row_fraction < self.boundary_margin:
            return self._review("Detection center is too close to a row-band boundary.")

        row_number = ordered_rows[row_index]
        row_slots = sorted(rows_by_number[row_number], key=lambda row: int(str(row.get("slot", 0))))
        slot_position = center_x * len(row_slots)
        slot_index = min(int(slot_position), len(row_slots) - 1)
        slot_fraction = slot_position - slot_index
        if slot_fraction < self.boundary_margin or 1 - slot_fraction < self.boundary_margin:
            return self._review("Detection center is too close to a slot-band boundary.")

        target = row_slots[slot_index]
        identity = (
            shelf_key[0], shelf_key[1], shelf_key[2],
            str(target.get("row", "")), str(target.get("slot", "")),
        )
        return {
            "slot_identity": identity,
            "status": "Mapped (assumption-based)",
            "mapping_id": self.mapping_id,
            "confidence": round(min(slot_fraction, 1 - slot_fraction, row_fraction, 1 - row_fraction), 4),
            "assumptions": [
                "Image is upright and shows one complete shelf face.",
                "Rows are equally spaced horizontal bands from top to bottom.",
                "Slots are equally spaced left-to-right within each row.",
                "No shelf-region calibration or perspective correction is applied.",
            ],
        }

    @staticmethod
    def _review(reason: str) -> dict[str, Any]:
        return {
            "slot_identity": None,
            "status": "Needs Review",
            "mapping_id": "equal-grid-v1",
            "confidence": 0.0,
            "assumptions": [],
            "detail": reason,
        }
