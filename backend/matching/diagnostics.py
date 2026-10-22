"""Per-detection and per-scene trace of why audit results are confirmed or left for review.

Every detection gets one record that follows it through detection, crop, SKU recognition,
slot mapping, planogram comparison and price verification, ending in a machine-readable
reason code. The scene summary names the dominant causes of Needs Review so an operator can
tell "the products are not in the catalogue" apart from "the layout was not confirmed" or
"the calibration is stale".
"""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

DIAGNOSTICS_VERSION = 1

SLOT_STATUS_CODES = {
    "Compliant": "compliant",
    "Misplaced": "different_product_mapped",
    "Unexpected Product": "unexpected_product",
    "Empty Slot": "empty_slot",
    "Missing Expected Product": "void_confirms_absence",
}

REASON_EXPLANATIONS = {
    "accepted": "SKU accepted at the configured similarity threshold and margin.",
    "below_threshold": "Best catalogue similarity is below the threshold; the product is probably not in the catalogue.",
    "known_non_product": "Looks like an object a reviewer marked as not a product.",
    "low_margin": "Two catalogue products are too similar to choose between.",
    "no_calibration": "No valid calibration is loaded, so no SKU can be accepted.",
    "invalid_crop": "The detection box is invalid, so no crop could be recognised.",
    "no_catalogue_references": "The catalogue has no usable reference images.",
    "matcher_unavailable": "The SKU matcher could not run.",
    "no_match_result": "No recognition result was produced for this detection.",
    "duplicate_box": "Overlapping duplicate box; excluded from counts and matching.",
    "slot_unmapped": "The SKU is known but its box could not be placed in a planogram slot.",
    "layout_untrusted": "Slot mapping is an unconfirmed equal-grid proposal for this image.",
    "compliant": "Identified SKU, slot and facing count agree with the planogram.",
    "different_product_mapped": "A different identified product occupies this planogram slot.",
    "unexpected_product": "An identified product is in a slot or count the planogram does not expect.",
    "empty_slot": "Intentionally empty planogram slot.",
    "void_confirms_absence": "A void detection confirms the expected product is absent.",
    "void_confirms_facing_shortfall": "A void detection confirms fewer facings than expected.",
    "void_conflicts_with_products": "Identified products and a void detection contradict each other.",
    "absence_not_confirmed": "Fewer products than expected, but nothing confirms the gap.",
    "unidentified_objects_in_slot": "Objects in this slot could not be identified.",
    "insufficient_evidence": "Not enough evidence to confirm this slot.",
}


def _slot_code(slot: dict[str, Any], layout_trusted: bool) -> str:
    if slot.get("reason_code"):
        return str(slot["reason_code"])
    status = str(slot.get("status", "Needs Review"))
    if status == "Needs Review":
        return "layout_untrusted" if not layout_trusted else "insufficient_evidence"
    return SLOT_STATUS_CODES.get(status, "insufficient_evidence")


def build_pipeline_diagnostics(
    *,
    detections: list[dict[str, Any]],
    matching_audit: dict[str, Any],
    compliance: list[dict[str, Any]],
    price_results: list[dict[str, Any]],
    price_audit: dict[str, Any],
    layout_trusted: bool,
    layout_validation: dict[str, Any],
    catalogue_size: int,
    planogram: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return {"scene", "detections", "slots", "reason_counts"}.

    Compliance rows without a reason_code are stamped with the derived one so saved reports
    carry it too. Nothing else is modified.
    """
    matches_by_index = {int(match.get("detection_index", -1)): match for match in matching_audit.get("matches", [])}
    slots_by_key = {(str(slot.get("row", "")), str(slot.get("slot", slot.get("position", "")))): slot for slot in compliance}
    expected_by_key = {
        (str(row.get("row", "")), str(row.get("slot", row.get("position", "")))): row for row in planogram
    }
    price_by_sku = {str(row.get("sku", "")): row for row in price_results}
    mapping_id = (matching_audit.get("slot_mapping") or {}).get("id", "equal-grid-v1")

    slot_records = []
    for slot in compliance:
        code = _slot_code(slot, layout_trusted)
        slot.setdefault("reason_code", code)
        slot_records.append({
            "slot": f"R{slot.get('row')}S{slot.get('slot', slot.get('position'))}",
            "expected_sku": slot.get("sku", ""), "expected_facings": slot.get("expected"),
            "observed_facings": slot.get("observed"), "status": slot.get("status"), "reason_code": code,
            "explanation": slot.get("detail") or REASON_EXPLANATIONS.get(code, ""),
        })

    records = []
    for detection in detections:
        index = int(detection.get("detection_index", len(records)))
        match = matches_by_index.get(index, {})
        identity = detection.get("slot_identity")
        slot_key = (str(identity[3]), str(identity[4])) if identity else None
        expected = expected_by_key.get(slot_key, {}) if slot_key else {}
        slot = slots_by_key.get(slot_key) if slot_key else None
        accepted_sku = match.get("accepted_sku") or ""
        if detection.get("duplicate_detection"):
            identity_status, placement_status, code = "Duplicate", "Excluded", "duplicate_box"
        elif not accepted_sku:
            identity_status, placement_status = "Unidentified", "Needs Review"
            code = match.get("reason_code") or ("matcher_unavailable" if matching_audit.get("matching_error") else "no_match_result")
        elif not identity:
            identity_status, placement_status, code = "Identified", "Needs Review", "slot_unmapped"
        elif not layout_trusted:
            identity_status, placement_status, code = "Identified", "Needs Review", "layout_untrusted"
        else:
            identity_status = "Identified"
            placement_status = slot.get("status", "Needs Review") if slot else "Needs Review"
            code = _slot_code(slot, layout_trusted) if slot else "insufficient_evidence"
        price_row = price_by_sku.get(accepted_sku) if accepted_sku else None
        records.append({
            "detection_index": index,
            "detector": {
                "label": detection.get("label"), "confidence": detection.get("confidence"),
                "box": detection.get("box"), "duplicate_of": detection.get("duplicate_of"),
            },
            "crop": {
                "valid": match.get("reason_code") != "invalid_crop" and not detection.get("duplicate_detection"),
                "width": match.get("crop_width"), "height": match.get("crop_height"),
            },
            "recognition": {
                "candidates": [
                    {"sku": item.get("sku"), "similarity": item.get("similarity")}
                    for item in (match.get("candidates") or [])[:3]
                ],
                "top_similarity": match.get("top_similarity"), "top_margin": match.get("top_margin"),
                "threshold": match.get("threshold"), "minimum_margin": match.get("minimum_margin"),
                "calibration_state": match.get("calibration_state"),
                "reason_code": match.get("reason_code"), "accepted_sku": accepted_sku or None,
            },
            "mapping": {
                "slot": f"R{slot_key[0]}S{slot_key[1]}" if slot_key else None, "method": mapping_id,
                "trusted": bool(layout_trusted and identity), "status": detection.get("slot_mapping_status"),
                "detail": detection.get("slot_mapping_detail") or "",
            },
            "planogram": {
                "expected_sku": expected.get("sku") or expected.get("expected_sku") or None,
                "expected_facings": expected.get("expected_facings") or expected.get("facing_count"),
            },
            "identity_status": identity_status,
            "placement_status": placement_status,
            "price_status": (price_row or {}).get("status", "Unverified") if accepted_sku else "Not applicable",
            "reason_code": code,
            "explanation": REASON_EXPLANATIONS.get(code, match.get("detail", "")),
        })

    active = [record for record in records if record["identity_status"] != "Duplicate"]
    accepted = [record for record in active if record["identity_status"] == "Identified"]
    tops = [record["recognition"]["top_similarity"] for record in active
            if record["recognition"]["top_similarity"] is not None]
    calibration = matching_audit.get("calibration") or {}
    threshold = calibration.get("acceptance_threshold")
    issues: list[dict[str, str]] = []
    if matching_audit.get("matching_error"):
        issues.append({"code": "matcher_unavailable", "message": f"SKU matching did not run: {matching_audit['matching_error']}"})
    for issue in calibration.get("issues", []):
        issues.append({"code": issue["code"], "message": issue["message"]})
    calibrated = calibration.get("calibrated_threshold")
    if (calibration.get("threshold_source") == "operator" and threshold is not None and calibrated is not None
            and threshold < calibrated):
        issues.append({
            "code": "operator_threshold_below_calibrated",
            "message": (
                f"The similarity threshold was set to {threshold:.2f}, below the calibrated {calibrated:.2f} (the value "
                "that kept validation precision at 98%). More products are identified, and more of them can be "
                "wrong: confirm or correct identifications in the review queue."
            ),
        })
    if catalogue_size == 0:
        issues.append({"code": "empty_catalogue", "message": "The active catalogue is empty, so no product can be identified."})
    elif active and threshold is not None and not matching_audit.get("matching_error"):
        below = sum(record["reason_code"] == "below_threshold" for record in active)
        if not accepted:
            issues.append({
                "code": "no_catalogue_matches",
                "message": (
                    f"None of the {len(active)} detected products matched a catalogue product: {below} were below the "
                    f"similarity threshold {threshold:.2f} (median best similarity "
                    f"{median(tops):.2f}). The photographed products are most likely not in the {catalogue_size}-product "
                    "catalogue, or this image does not show the configured planogram; add reference images for these "
                    "products before expecting confirmed results."
                ) if tops else f"None of the {len(active)} detected products could be scored against the catalogue.",
            })
        elif len(accepted) / len(active) < 0.25:
            issues.append({
                "code": "few_catalogue_matches",
                "message": (f"Only {len(accepted)} of {len(active)} detected products matched a catalogue product; "
                            f"{below} were below the similarity threshold, which usually means they are not in the catalogue."),
            })
    if price_audit.get("requested"):
        tags = price_audit.get("tags", [])
        if price_audit.get("tag_source") == "full_image_ocr":
            issues.append({"code": "price_tags_from_full_image_ocr",
                           "message": f"{price_audit.get('tag_source_reason', '')} Prices were read by whole-image OCR "
                                      f"instead ({len(tags)} price reading(s)).".strip()})
        elif tags and not any(tag.get("ocr_status") == "read" for tag in tags):
            issues.append({"code": "price_tags_unread",
                           "message": f"{len(tags)} price tag(s) were detected but no price could be read from them; "
                                      "prices remain Unverified."})

    slot_reasons = Counter(record["reason_code"] for record in slot_records)
    review_slots = [record for record in slot_records if record["status"] == "Needs Review"]
    review_causes = Counter(record["reason_code"] for record in review_slots)
    return {
        "version": DIAGNOSTICS_VERSION,
        "scene": {
            "detections": len(records), "unique_detections": len(active), "identified": len(accepted),
            "median_top_similarity": round(median(tops), 4) if tops else None,
            "acceptance_threshold": threshold, "minimum_margin": calibration.get("minimum_margin"),
            "calibration_state": calibration.get("state"),
            "layout_trusted": bool(layout_trusted), "layout_detail": layout_validation.get("detail", ""),
            "slots_needing_review": len(review_slots), "slots": len(slot_records),
            "needs_review_causes": [
                {"reason_code": code, "slots": count, "explanation": REASON_EXPLANATIONS.get(code, "")}
                for code, count in review_causes.most_common()
            ],
            "issues": issues,
        },
        "detections": records,
        "slots": slot_records,
        "reason_counts": {
            "detections": dict(Counter(record["reason_code"] for record in records)),
            "slots": dict(slot_reasons),
        },
    }
