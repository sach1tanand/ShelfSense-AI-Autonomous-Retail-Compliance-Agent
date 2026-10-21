"""Compare accepted crop-to-SKU matches and mapped slots with planogram expectations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from src.planogram import _slot_identity


def compare_planogram(
    planogram: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    slot_mapper: Any,
) -> dict[str, Any]:
    """Produce auditable findings without turning non-detections into confirmed misses.

    A slot with no accepted mapped detection is Needs Review, not Missing, because
    occlusion and detector failures are not observable from a single image. Correct
    requires accepted SKU identity, confident coordinate mapping, and count equality.
    """
    by_identity: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    unassigned: list[dict[str, Any]] = []
    for match in matches:
        match = dict(match)
        mapper_result = slot_mapper.map_detection(match, planogram)
        match["slot_mapping"] = mapper_result
        identity = mapper_result.get("slot_identity")
        accepted_sku = match.get("accepted_sku")
        trusted_mapping = bool(getattr(slot_mapper, "trusted", False))
        if identity is None or not trusted_mapping or not accepted_sku:
            match["status"] = "Needs Review"
            if not accepted_sku:
                match["detail"] = match.get("detail", "SKU match is uncertain.")
            elif identity is not None and not trusted_mapping:
                match["proposed_slot"] = list(identity)
                match["detail"] = "A coordinate-based slot is proposed, but this mapper is not calibrated/trusted; review the suggested slot manually."
            else:
                match["detail"] = mapper_result.get("detail", "Shelf slot mapping is uncertain.")
            unassigned.append(match)
            continue
        by_identity[tuple(str(value) for value in identity)].append(match)

    findings: list[dict[str, Any]] = []
    expected_skus = {
        str(row.get("expected_sku", row.get("sku", ""))).strip()
        for row in planogram
        if str(row.get("expected_sku", row.get("sku", ""))).strip()
    }
    for planogram_row in planogram:
        identity = _slot_identity(planogram_row)
        identity = tuple(str(value) for value in identity)
        expected_sku = str(planogram_row.get("expected_sku", planogram_row.get("sku", ""))).strip()
        try:
            expected_facings = int(str(planogram_row.get("facing_count", planogram_row.get("expected_facings", 0))))
        except (TypeError, ValueError):
            expected_facings = 0
        detections = by_identity.get(identity, [])
        known = [item for item in detections if item.get("accepted_sku")]
        uncertain = [item for item in detections if not item.get("accepted_sku")]

        base = {
            "store_id": identity[0], "planogram_id": identity[1], "shelf": identity[2],
            "row": identity[3], "slot": identity[4], "expected_sku": expected_sku,
            "expected_facings": expected_facings,
            "observed_skus": [item["accepted_sku"] for item in known],
            "observed_count": len(known),
        }
        if not expected_sku:
            if known:
                for item in known:
                    findings.append({
                        **base,
                        "status": "Unexpected",
                        "sku": item["accepted_sku"],
                        "product": item["candidates"][0].get("product", "") if item.get("candidates") else "",
                        "detail": "A confidently matched product maps to an intentionally empty planogram slot.",
                        "detection_index": item.get("detection_index"),
                    })
            else:
                findings.append({**base, "status": "Empty Slot", "sku": "", "product": "", "detail": "Intentionally empty planogram slot."})
            continue

        if not known:
            findings.append({
                **base,
                "status": "Needs Review",
                "sku": expected_sku,
                "product": "",
                "detail": "No accepted product was mapped here. This is not reported Missing because occlusion or detector failure cannot be ruled out.",
                "uncertain_detections": [item.get("detection_index") for item in uncertain],
            })
            continue

        matching = [item for item in known if item["accepted_sku"] == expected_sku]
        wrong = [item for item in known if item["accepted_sku"] != expected_sku]
        if wrong:
            for item in wrong:
                findings.append({
                    **base,
                    "status": "Misplaced",
                    "sku": item["accepted_sku"],
                    "product": (item.get("candidates") or [{}])[0].get("product", ""),
                    "detail": f"SKU {item['accepted_sku']} is confidently mapped to a slot expecting {expected_sku}.",
                    "detection_index": item.get("detection_index"),
                    "slot_mapping": item.get("slot_mapping", {}),
                })
            if not matching:
                findings.append({
                    **base,
                    "status": "Missing",
                    "sku": expected_sku,
                    "product": "",
                    "detail": "No accepted crop of the expected SKU was mapped to this slot while another accepted SKU was present. Occlusion may still hide the expected product.",
                    "uncertainty": "Expected product may be occluded behind the visible product.",
                })
        if matching:
            observed_count = len(matching)
            if observed_count == expected_facings and not uncertain:
                status = "Correct"
                detail = "Accepted SKU, mapped slot, and detected object count match expected facing count."
            else:
                status = "Needs Review"
                detail = (
                    f"SKU matches, but {observed_count} product crop(s) were detected for {expected_facings} expected facing(s); "
                    "single-image object counts may be affected by occlusion or detector misses."
                )
            for item in matching:
                findings.append({
                    **base,
                    "status": status,
                    "sku": expected_sku,
                    "product": (item.get("candidates") or [{}])[0].get("product", ""),
                    "detail": detail,
                    "detection_index": item.get("detection_index"),
                    "slot_mapping": item.get("slot_mapping", {}),
                })
        elif not wrong:
            findings.append({
                **base,
                "status": "Missing",
                "sku": expected_sku,
                "product": "",
                "detail": "A different accepted SKU occupies the mapped slot; expected SKU is not detected there.",
            })

    for item in unassigned:
        sku = item.get("accepted_sku")
        status = "Unexpected" if sku and sku not in expected_skus else "Needs Review"
        findings.append({
            "status": status,
            "sku": sku or "",
            "product": (item.get("candidates") or [{}])[0].get("product", item.get("detector_label", "")),
            "detail": item.get("detail", "SKU or slot mapping is uncertain."),
            "detection_index": item.get("detection_index"),
            "candidates": item.get("candidates", []),
            "slot_mapping": item.get("slot_mapping", {}),
        })

    return {
        "findings": findings,
        "matches": matches,
        "slot_mapping_id": getattr(slot_mapper, "mapping_id", type(slot_mapper).__name__),
        "slot_mapping_trusted": bool(getattr(slot_mapper, "trusted", False)),
        "uncertainty_policy": "No accepted detection at an expected slot is Needs Review, not Missing; occlusion/model failure cannot be ruled out.",
    }
