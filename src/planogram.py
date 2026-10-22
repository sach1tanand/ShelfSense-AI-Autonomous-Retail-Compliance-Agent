"""Planogram validation and conservative detection-to-slot auditing.

Bounding boxes identify image regions, not planogram slots. Without an injected
slot_mapper that can confidently map each detection to a physical slot, expected
slots remain Needs Review rather than being reported as confirmed missing or
compliant.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, TypedDict


SlotMapper = Callable[[dict, list[dict]], tuple[str, str, str, str, str] | None]
"""Mapper contract; return None when physical slot mapping is uncertain."""


def _normalize_label(value: object) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(value or "").casefold()).split())


def _slot_identity(row: dict) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("store_id", "")).strip(),
        str(row.get("planogram_id", "")).strip(),
        str(row.get("physical_shelf", row.get("shelf", ""))).strip(),
        str(row.get("row", "")).strip(),
        str(row.get("slot", row.get("position", ""))).strip(),
    )


def validate_planogram(planogram: list[dict], catalogue: list[dict]) -> list[str]:
    """Validate physical slot uniqueness, SKU references, and facings/empty-slot rules."""
    errors: list[str] = []
    valid_skus = {str(item.get("sku", "")).strip() for item in catalogue if str(item.get("sku", "")).strip()}
    seen_slots: set[tuple[str, str, str, str, str]] = set()
    required_fields = ("store_id", "planogram_id", "shelf", "row", "slot")

    for index, row in enumerate(planogram, start=1):
        missing_fields = [field for field in required_fields if not str(row.get(field, "")).strip()]
        if missing_fields:
            errors.append(f"Row {index}: missing required fields: {', '.join(missing_fields)}")
            continue
        identity = _slot_identity(row)
        if identity in seen_slots:
            errors.append(f"Row {index}: duplicate slot assignment for {'/'.join(identity)}")
        seen_slots.add(identity)

        for field in ("row", "slot"):
            try:
                if int(str(row[field])) < 1:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"Row {index}: {field} must be a positive whole number")

        sku = str(row.get("expected_sku", row.get("sku", ""))).strip()
        raw_facings = row.get("facing_count", row.get("expected_facings", ""))
        try:
            facings = int(str(raw_facings))
            if facings < 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"Row {index}: facing_count must be a non-negative whole number")
            continue

        if sku:
            if sku not in valid_skus:
                errors.append(f"Row {index}: unknown expected_sku {sku!r}")
            if facings < 1:
                errors.append(f"Row {index}: assigned products require facing_count >= 1")
        elif facings != 0:
            errors.append(f"Row {index}: intentionally empty slots must have facing_count 0")

    return errors


def _catalogue_label_index(catalogue: list[dict]) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {}
    for product in catalogue:
        sku = str(product.get("sku", "")).strip()
        if not sku:
            continue
        for label in (product.get("model_label"), product.get("name"), product.get("product_name")):
            normalized = _normalize_label(label)
            if normalized:
                labels.setdefault(normalized, set()).add(sku)
    return labels


def evaluate_planogram(
    planogram: list[dict],
    catalogue: list[dict],
    detections: list[dict],
    slot_mapper: Callable[[dict, list[dict]], tuple[str, str, str, str, str] | None] | None = None,
) -> dict:
    """Return slot findings, unplaced detection findings, and validation errors.

    When provided, slot_mapper must map a detection to (store_id, planogram_id,
    shelf, row, slot), returning None on uncertainty. No bounding-box-only
    mapper is supplied because boxes do not establish physical shelf positions.
    """
    validation_errors = validate_planogram(planogram, catalogue)
    products_by_sku = {str(item.get("sku", "")).strip(): item for item in catalogue if str(item.get("sku", "")).strip()}
    label_index = _catalogue_label_index(catalogue)
    expected_skus = {
        str(row.get("expected_sku", row.get("sku", ""))).strip()
        for row in planogram
        if str(row.get("expected_sku", row.get("sku", ""))).strip()
    }

    slots: list[dict] = []
    mapped_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    mapped_products: dict[tuple[str, str, str, str, str], list[str]] = {}
    detection_findings: list[dict] = []
    unmapped_expected_counts: Counter[str] = Counter()

    for detection in detections:
        label = str(detection.get("label", ""))
        explicit_sku = str(detection.get("sku", detection.get("barcode", ""))).strip()
        if explicit_sku in products_by_sku:
            matches = {explicit_sku}
        else:
            matches = label_index.get(_normalize_label(label), set())
        if len(matches) != 1:
            reason = "Detector label does not resolve to a unique catalogue SKU."
            detection_findings.append({
                "product": label or "Unidentified detection",
                "sku": "",
                "status": "Needs Review",
                "detail": reason,
                "detection": detection,
            })
            continue

        sku = next(iter(matches))
        if sku not in expected_skus and slot_mapper is None:
            product = products_by_sku[sku]
            detection_findings.append({
                "product": product.get("name", label),
                "sku": sku,
                "status": "Unexpected Product",
                "detail": "This uniquely identified catalogue SKU is not assigned anywhere in the active planogram.",
                "detection": detection,
            })
            continue

        if slot_mapper is None:
            unmapped_expected_counts[sku] += 1
            continue

        identity = slot_mapper(detection, planogram)
        known_identities = {_slot_identity(row) for row in planogram}
        if identity not in known_identities:
            detection_findings.append({
                "product": products_by_sku[sku].get("name", label),
                "sku": sku,
                "status": "Needs Review",
                "detail": "No confident planogram slot mapping was available for this detection.",
                "detection": detection,
            })
            continue
        mapped_counts[identity] += 1
        mapped_products.setdefault(identity, []).append(sku)

    for row in planogram:
        identity = _slot_identity(row)
        sku = str(row.get("expected_sku", row.get("sku", ""))).strip()
        facings_raw = row.get("facing_count", row.get("expected_facings", 0))
        try:
            expected_facings = int(str(facings_raw))
        except (TypeError, ValueError):
            expected_facings = 0
        product = products_by_sku.get(sku, {})
        row_number = str(row.get("row", "")).strip()
        slot_number = str(row.get("slot", row.get("position", ""))).strip()
        base = {
            "store_id": identity[0],
            "planogram_id": identity[1],
            "shelf": identity[2],
            "row": row_number,
            "position": slot_number,
            "slot": slot_number,
            "sku": sku,
            "product": product.get("name", "") if sku else "Empty slot",
            "expected": expected_facings,
            "observed": None,
            "status": "Needs Review",
            "detail": "Exact product-to-slot mapping is not configured.",
        }

        if not sku:
            if slot_mapper is None:
                base.update(status="Empty Slot", detail="Intentionally unassigned planogram slot.")
            elif mapped_counts[identity]:
                observed_sku = mapped_products[identity][0]
                base.update(
                    sku=observed_sku,
                    product=products_by_sku[observed_sku].get("name", "Unknown product"),
                    observed=mapped_counts[identity],
                    status="Unexpected Product",
                    detail="A product was confidently mapped to an intentionally empty slot.",
                )
            else:
                base.update(status="Empty Slot", detail="Intentionally unassigned planogram slot; no detection was mapped here.")
        elif slot_mapper is None:
            base["detail"] = (
                f"{unmapped_expected_counts[sku]} matching detection(s) found in the image, but bounding boxes do not establish this slot."
                if unmapped_expected_counts[sku]
                else "No matching detection was observed, but slot mapping is unavailable; absence is not confirmed."
            )
        else:
            observed_skus = mapped_products.get(identity, [])
            observed = len(observed_skus)
            base["observed"] = observed
            if observed == 0:
                base.update(status="Missing Expected Product", detail="No product was mapped to this assigned slot.")
            elif any(observed_sku != sku for observed_sku in observed_skus):
                base.update(status="Misplaced", detail="A different catalogue product was mapped to this assigned slot.")
            elif observed < expected_facings:
                base.update(status="Missing Expected Product", detail=f"Expected {expected_facings} facings; mapped {observed}.")
            elif observed > expected_facings:
                base.update(status="Unexpected Product", detail=f"Expected {expected_facings} facings; mapped {observed}.")
            else:
                base.update(status="Compliant", detail="Mapped product and facing count match the planogram.")
        slots.append(base)

    return {
        "slots": slots,
        "detections": detection_findings,
        "validation_errors": validation_errors,
        "slot_mapping_available": slot_mapper is not None,
        "empty_slot_count": sum(not str(row.get("expected_sku", row.get("sku", ""))).strip() for row in planogram),
    }
