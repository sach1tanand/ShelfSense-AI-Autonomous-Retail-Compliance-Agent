"""Reconstruct a reviewable detected-shelf grid from product and void boxes."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .shelf_mapper import CoordinateShelfMapper


def _box(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        values = tuple(float(value) for value in item.get("box", ()))
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _reliable_sku(item: dict[str, Any]) -> str:
    sku = str(item.get("sku") or "").strip()
    status = str(item.get("match_status") or "").casefold()
    if not sku or any(marker in status for marker in ("unknown", "review", "unmatched", "uncertain")):
        return ""
    return sku


def _adjacent(left: dict[str, Any], right: dict[str, Any], max_gap_ratio: float, min_vertical_overlap: float) -> bool:
    a, b = _box(left), _box(right)
    if not a or not b:
        return False
    if (a[0] + a[2]) / 2 > (b[0] + b[2]) / 2:
        a, b = b, a
    width = max(1.0, min(a[2] - a[0], b[2] - b[0]))
    horizontal_gap = b[0] - a[2]
    if horizontal_gap > width * max_gap_ratio or horizontal_gap < -width * 0.2:
        return False
    smaller_height = max(1.0, min(a[3] - a[1], b[3] - b[1]))
    vertical_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1])) / smaller_height
    return vertical_overlap >= min_vertical_overlap


def _same_or_unmapped_slot(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Do not collapse products that the mapper placed in distinct expected cells."""
    left_identity = left.get("slot_identity")
    right_identity = right.get("slot_identity")
    if left_identity and right_identity:
        return tuple(str(value) for value in left_identity) == tuple(str(value) for value in right_identity)
    return True


def _group_product_cells(
    row_items: list[dict[str, Any]], max_gap_ratio: float, min_vertical_overlap: float,
) -> list[dict[str, Any]]:
    """Group consecutive accepted same-SKU facings without altering source detections."""
    ordered = sorted(row_items, key=lambda item: ((_box(item) or (0,))[0] + (_box(item) or (0, 0, 0, 0))[2]) / 2)
    cells: list[dict[str, Any]] = []
    for item in ordered:
        if cells and item["kind"] == "product" and cells[-1]["kind"] == "product":
            previous = cells[-1]
            same_sku = bool(_reliable_sku(item)) and _reliable_sku(item) == previous.get("sku")
            references = previous.get("individual_detections", [])
            previous_detection = references[-1] if references else previous
            if (same_sku and _same_or_unmapped_slot(previous_detection, item)
                    and _adjacent(previous_detection, item, max_gap_ratio, min_vertical_overlap)):
                box = _box(item)
                old_box = _box(previous)
                previous["box"] = [min(old_box[0], box[0]), min(old_box[1], box[1]),
                                   max(old_box[2], box[2]), max(old_box[3], box[3])]
                previous["individual_detections"].append({
                    "detection_index": item["detection_index"], "box": list(item["box"]),
                    "confidence": item.get("confidence"), "slot_identity": item.get("slot_identity"),
                })
                previous["detection_indices"].append(item["detection_index"])
                previous["facing_count"] += 1
                confidences = [reference["confidence"] for reference in previous["individual_detections"]
                               if reference.get("confidence") is not None]
                previous["confidence"] = sum(confidences) / len(confidences) if confidences else None
                previous["group_status"] = f"{previous['facing_count']} facings · one product cell"
                identities = {
                    tuple(str(value) for value in reference["slot_identity"])
                    for reference in previous["individual_detections"] if reference.get("slot_identity")
                }
                if len(identities) > 1:
                    previous["slot_identity"] = None
                    previous["slot_mapping_status"] = "Group spans multiple mapped slots; review required"
                previous["price_tags"].extend(item.get("price_tags", []))
                continue
        cell = dict(item)
        cell["price_tags"] = list(item.get("price_tags", []))
        if item["kind"] == "product":
            reliable_sku = _reliable_sku(item)
            cell["sku"] = reliable_sku
            cell["identified"] = bool(reliable_sku)
            cell["facing_count"] = 1
            cell["detection_indices"] = [item["detection_index"]]
            cell["individual_detections"] = [{
                "detection_index": item["detection_index"], "box": list(item["box"]),
                "confidence": item.get("confidence"), "slot_identity": item.get("slot_identity"),
            }]
            cell["group_status"] = "Identified single facing" if reliable_sku else "Needs Review: SKU identity uncertain"
        else:
            cell["facing_count"] = 0
        cells.append(cell)
    # Unknown identity cannot be grouped. Flag touching unknown neighbours for manual review.
    for index, cell in enumerate(cells):
        if cell.get("kind") != "product" or cell.get("sku"):
            continue
        neighbours = cells[max(0, index - 1):index] + cells[index + 1:index + 2]
        if any(neighbor.get("kind") == "product" and _adjacent(cell, neighbor, max_gap_ratio, min_vertical_overlap)
               for neighbor in neighbours):
            cell["group_status"] = "Needs Review: adjacent identity is uncertain"
    return cells


def _associate_price_tags(
    products: list[dict[str, Any]], tags: list[dict[str, Any]], image_height: int,
) -> list[dict[str, Any]]:
    """Associate tags one-to-one using same-row horizontal overlap and below-product geometry."""
    pending = []
    for tag_index, original in enumerate(tags):
        tag = {**original, "tag_index": tag_index, "associated_detection_index": None,
               "association_status": "Needs Review"}
        tag_box = _box(tag)
        if not tag_box:
            pending.append((tag, []))
            continue
        candidates = []
        tag_center = (tag_box[0] + tag_box[2]) / 2
        for product in products:
            product_box = _box(product)
            if not product_box:
                continue
            gap = tag_box[1] - product_box[3]
            horizontal_overlap = max(0.0, min(tag_box[2], product_box[2]) - max(tag_box[0], product_box[0]))
            width = max(1.0, min(tag_box[2] - tag_box[0], product_box[2] - product_box[0]))
            if gap < -0.02 * image_height or gap > 0.22 * image_height or horizontal_overlap / width < 0.2:
                continue
            product_center = (product_box[0] + product_box[2]) / 2
            score = max(0.0, gap) / max(1, image_height) + abs(tag_center - product_center) / max(1, product_box[2] - product_box[0]) * 0.05
            candidates.append((score, product))
        candidates.sort(key=lambda pair: pair[0])
        pending.append((tag, candidates))

    # Resolve all tags together so two labels cannot silently attach to the same product.
    proposals = []
    for tag, candidates in pending:
        if not candidates:
            continue
        best_score, product = candidates[0]
        separated = len(candidates) == 1 or candidates[1][0] - best_score >= 0.025
        if separated:
            proposals.append((best_score, tag, product))
    used_products: set[int] = set()
    used_tags: set[int] = set()
    for _score, tag, product in sorted(proposals, key=lambda item: item[0]):
        product_index = int(product["detection_index"])
        if product_index in used_products or tag["tag_index"] in used_tags:
            continue
        tag["associated_detection_index"] = product_index
        tag["association_status"] = "Associated by unique shelf-row geometry"
        used_products.add(product_index)
        used_tags.add(tag["tag_index"])
    return [tag for tag, _candidates in pending]


def reconstruct_detected_planogram(
    detections: list[dict[str, Any]],
    voids: list[dict[str, Any]],
    price_tags: list[dict[str, Any]],
    catalogue: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    grouping_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Cluster actual boxes into image rows and left-to-right observed cells.

    Geometry is a proposal only: without shelf-boundary calibration, row and column
    locations do not assert equivalence with saved planogram slots.
    """
    products_by_sku = {str(item.get("sku", "")): item for item in catalogue}
    observed = []
    for index, detection in enumerate(detections):
        sku = str(detection.get("matched_sku") or detection.get("sku") or "")
        observed.append({
            **detection, "kind": "product", "detection_index": index,
            "sku": sku, "product": products_by_sku.get(sku, {}).get("name") if sku else "Unidentified",
            "identified": bool(sku) and not detection.get("duplicate_detection"),
            "cell_status": (
                "Duplicate detection suppressed" if detection.get("duplicate_detection")
                else "Detected product" if sku else "Unidentified · Needs Review"
            ),
        })
    associated_tags = _associate_price_tags(observed, price_tags, image_height)
    tags_by_detection: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for tag in associated_tags:
        if tag.get("associated_detection_index") is not None:
            tags_by_detection[int(tag["associated_detection_index"])].append(tag)
    for item in observed:
        item["price_tags"] = tags_by_detection.get(int(item["detection_index"]), [])

    for index, void in enumerate(voids):
        observed.append({
            **void, "kind": "void", "void_index": index, "sku": "", "product": "Possible void",
            "identified": False, "cell_status": "VOID / EMPTY SLOT · Needs Review", "price_tags": [],
        })

    boxed = [item for item in observed if _box(item) and not item.get("duplicate_detection")]
    unassigned = [item for item in observed if not _box(item)]
    rows = CoordinateShelfMapper.cluster_detections_by_row(boxed, gap_ratio=0.65)
    grouping_config = grouping_config or {}
    max_gap_ratio = float(grouping_config.get("max_gap_ratio", 0.25))
    min_vertical_overlap = float(grouping_config.get("min_vertical_overlap", 0.6))
    max_gap_ratio = min(1.0, max(0.0, max_gap_ratio))
    min_vertical_overlap = min(1.0, max(0.0, min_vertical_overlap))
    detected_rows = []
    duplicate_indices: set[int] = set()
    for row_index, row_items in enumerate(rows, start=1):
        cells = []
        deduplicated = []
        for item in sorted(row_items, key=lambda value: ((_box(value) or (0,))[0] + (_box(value) or (0, 0, 0, 0))[2]) / 2):
            box = _box(item)
            duplicate_of = None
            for cell in deduplicated:
                other_box = _box(cell)
                if item["kind"] == cell["kind"] and other_box and _iou(box, other_box) >= 0.82:
                    duplicate_of = cell.get("detection_index", cell.get("void_index"))
                    break
            if duplicate_of is not None:
                item["duplicate_of"] = duplicate_of
                if item["kind"] == "product":
                    duplicate_indices.add(int(item["detection_index"]))
                continue
            deduplicated.append(item)
        cells = _group_product_cells(deduplicated, max_gap_ratio, min_vertical_overlap)
        for column, cell in enumerate(cells, start=1):
            cell["column"] = column
            cell["group_bbox"] = list(cell["box"])
        detected_rows.append({"row": row_index, "slots": cells, "slot_count": len(cells), "status": "Needs Review"})

    return {
        "rows": detected_rows,
        "unassigned": unassigned,
        "items": observed,
        "price_tags": associated_tags,
        "row_count": len(detected_rows),
        "slot_count": sum(len(row["slots"]) for row in detected_rows),
        "product_facing_count": sum(
            cell.get("facing_count", 0) for row in detected_rows for cell in row["slots"]
            if cell.get("kind") == "product"
        ),
        "duplicate_detection_indices": sorted(duplicate_indices),
        "image_width": image_width,
        "image_height": image_height,
        "grouping": {"max_gap_ratio": max_gap_ratio, "min_vertical_overlap": min_vertical_overlap},
        "geometry_status": "Needs Review",
        "geometry_method": "Detected product and void boxes clustered by vertical center, then ordered by horizontal center.",
        "limitations": [
            "No shelf-edge class is available in the configured detector models.",
            "Detected row and column boundaries are estimates from boxes, not calibrated shelf geometry.",
            "A void detection indicates a possible empty area; it does not prove a specific planogram slot is empty.",
        ],
    }


def compare_expected_to_detected(
    planogram: list[dict[str, Any]],
    detected: dict[str, Any],
    planogram_audit: dict[str, Any],
    price_results: list[dict[str, Any]],
    layout_trusted: bool,
    catalogue: list[dict[str, Any]] | None = None,
    matching_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine saved slot findings with a non-authoritative geometric proposal."""
    expected_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for slot in planogram:
        expected_rows[(str(slot.get("physical_shelf", slot.get("shelf", ""))), str(slot.get("row", "1")))].append(slot)
    catalogue_by_sku = {str(item.get("sku", "")): item for item in (catalogue or [])}
    expected_groups = [
        {
            "shelf": key[0], "row": key[1],
            "slots": [
                {
                    **slot,
                    "expected_product": (catalogue_by_sku.get(str(slot.get("sku", ""))) or {}).get("name", ""),
                    "expected_price": (
                        ((catalogue_by_sku.get(str(slot.get("sku", ""))) or {}).get("price") or slot.get("min_price"))
                        if slot.get("sku") else None
                    ),
                }
                for slot in sorted(value, key=lambda item: int(item.get("position", 0)))
            ],
        }
        for key, value in sorted(expected_rows.items(), key=lambda item: (item[0][0].casefold(), int(item[0][1]) if item[0][1].isdigit() else 10**9))
    ]
    audit_by_slot = {
        (str(item.get("shelf", "")), str(item.get("row", "")), str(item.get("position", item.get("slot", "")))): item
        for item in planogram_audit.get("slots", [])
    }
    comparison_by_slot: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    status_map = {
        "Correct": "MATCH", "Misplaced": "WRONG PRODUCT / MISPLACED",
        "Unexpected": "UNEXPECTED PRODUCT", "Empty Slot": "EXPECTED EMPTY SLOT",
    }
    for finding in (matching_audit or {}).get("findings", []):
        if finding.get("slot") is None:
            continue
        status = status_map.get(finding.get("status"))
        if status:
            identity = (str(finding.get("shelf", "")), str(finding.get("row", "")), str(finding.get("slot", "")))
            comparison_by_slot[identity].append(finding)
    uncertain_slots = set()
    for match in (matching_audit or {}).get("matches", []):
        if match.get("accepted_sku"):
            continue
        identity = (match.get("slot_mapping") or {}).get("slot_identity")
        if identity:
            uncertain_slots.add(tuple(str(value) for value in identity))
    price_by_sku = {str(item.get("sku", "")): item for item in price_results}
    detected_rows = detected.get("rows", [])
    expected_row_count = len(expected_groups)
    findings = []
    for row_index, expected_row in enumerate(expected_groups):
        detected_row = detected_rows[row_index] if row_index < len(detected_rows) else None
        expected_slots = expected_row["slots"]
        actual_slots = detected_row["slots"] if detected_row else []
        assigned_actual: set[tuple[str, Any]] = set()
        for slot_index, expected in enumerate(expected_slots):
            expected_sku = str(expected.get("sku", ""))
            expected_price_raw = (price_by_sku.get(expected_sku) or {}).get("expected_price") if expected_sku else None
            if expected_sku and expected_price_raw is None:
                expected_price_raw = (catalogue_by_sku.get(expected_sku) or {}).get("price") or expected.get("min_price")
            candidate = None
            if detected_row and actual_slots:
                if layout_trusted:
                    identity = (
                        str(expected.get("store_id", "")), str(expected.get("planogram_id", "")),
                        str(expected.get("physical_shelf", "")), str(expected.get("row", "")),
                        str(expected.get("position", "")),
                    )
                    candidate = next((item for item in actual_slots
                                      if tuple(str(value) for value in (item.get("slot_identity") or [])) == identity), None)
                else:
                    # Keep expected slot centers fixed; grouping must never compress the grid and
                    # shift every later detected product into an earlier expected position.
                    image_width = max(1.0, float(detected.get("image_width") or 1.0))
                    target_x = ((slot_index + 0.5) / max(1, len(expected_slots))) * image_width
                    possible = [item for item in actual_slots if _box(item)]
                    if possible:
                        nearest = min(possible, key=lambda item: abs((_box(item)[0] + _box(item)[2]) / 2 - target_x))
                        slot_width = image_width / max(1, len(expected_slots))
                        if abs((_box(nearest)[0] + _box(nearest)[2]) / 2 - target_x) <= slot_width * 0.5:
                            candidate = nearest
                candidate_index = ((candidate["kind"], candidate.get("detection_index", candidate.get("void_index")))
                                   if candidate else None)
                if candidate_index in assigned_actual:
                    candidate = None
                elif candidate_index is not None:
                    assigned_actual.add(candidate_index)
            audit = audit_by_slot.get((str(expected.get("physical_shelf", "")), str(expected.get("row", "")), str(expected.get("position", ""))))
            confirmed_status = None
            accepted_findings = comparison_by_slot.get((str(expected.get("physical_shelf", "")), str(expected.get("row", "")), str(expected.get("position", ""))), [])
            additional_statuses = []
            accepted_statuses = {status_map.get(item.get("status")) for item in accepted_findings}
            accepted_statuses.discard(None)
            expected_identity = (
                str(expected.get("store_id", "")), str(expected.get("planogram_id", "")),
                str(expected.get("physical_shelf", "")), str(expected.get("row", "")),
                str(expected.get("position", "")),
            )
            if expected_identity in uncertain_slots:
                accepted_statuses.discard("EXPECTED EMPTY SLOT")
            # The slot audit is the authoritative result (it also drives the score). When it is left
            # for review (unidentified objects, void/product contradiction, unconfirmed absence), a
            # per-detection finding or a void cell must not confirm a different status here.
            audit_needs_review = bool(layout_trusted and audit and audit.get("status") == "Needs Review")
            if audit_needs_review:
                accepted_statuses = set()
            if layout_trusted and accepted_statuses:
                priority = ["WRONG PRODUCT / MISPLACED", "UNEXPECTED PRODUCT", "MATCH", "EXPECTED EMPTY SLOT"]
                confirmed_status = next(status for status in priority if status in accepted_statuses)
                additional_statuses = [status for status in priority if status in accepted_statuses and status != confirmed_status]
                audit = next((item for item in accepted_findings if status_map.get(item.get("status")) == confirmed_status), audit)
            elif layout_trusted and audit and audit.get("status") == "Empty Slot" and not expected_sku:
                # An unassigned planogram cell is expected empty; this says nothing about a visible void.
                confirmed_status = "EXPECTED EMPTY SLOT"
            elif layout_trusted and audit and audit.get("status") in {"Misplaced", "Unexpected Product", "Missing Expected Product"}:
                # Misplaced/Unexpected require a mapped product; under the trusted layout the slot audit
                # reports Missing only when a void detection confirms the gap.
                mapped_status = {
                    "Misplaced": "WRONG PRODUCT / MISPLACED", "Unexpected Product": "UNEXPECTED PRODUCT",
                    "Missing Expected Product": "MISSING PRODUCT",
                }.get(audit.get("status"))
                if mapped_status:
                    confirmed_status = mapped_status
            if layout_trusted and not audit_needs_review and not confirmed_status and candidate and candidate.get("kind") == "void":
                confirmed_status = "MISSING PRODUCT" if expected_sku else "VOID / EMPTY SLOT"
                audit = {"detail": "A void-model detection maps to this slot under the trusted layout."}
            suggested_status = "Needs Review"
            if candidate:
                if candidate["kind"] == "void":
                    suggested_status = "MISSING PRODUCT" if expected_sku else "VOID / EMPTY SLOT"
                elif not expected_sku:
                    suggested_status = "UNEXPECTED PRODUCT" if candidate.get("identified") else "Needs Review"
                elif not candidate.get("identified"):
                    suggested_status = "Needs Review"
                elif candidate.get("sku") == expected_sku:
                    suggested_status = "MATCH"
                else:
                    suggested_status = "WRONG PRODUCT / MISPLACED"
            status = confirmed_status or "Needs Review"
            if confirmed_status == "EXPECTED EMPTY SLOT" and candidate and candidate.get("kind") == "void":
                status = "VOID / EMPTY SLOT"
            tags = candidate.get("price_tags", []) if candidate else []
            tag = tags[0] if len(tags) == 1 else None
            observed_price = tag.get("price") if tag else None
            identity_matches = bool(candidate and candidate.get("slot_identity") and tuple(
                str(value) for value in candidate["slot_identity"]
            ) == (
                str(expected.get("store_id", "")), str(expected.get("planogram_id", "")),
                str(expected.get("physical_shelf", "")), str(expected.get("row", "")),
                str(expected.get("position", "")),
            ))
            price_link_confirmed = bool(
                layout_trusted and identity_matches and candidate and candidate.get("kind") == "product"
                and candidate.get("sku") == expected_sku
            )
            if tag and price_link_confirmed and tag.get("association_status") == "Associated by unique shelf-row geometry" and tag.get("ocr_status") == "read" and expected_price_raw is not None:
                try:
                    price_status = "PRICE MATCH" if float(observed_price) == float(expected_price_raw) else "PRICE MISMATCH"
                except (TypeError, ValueError):
                    price_status = "PRICE UNREADABLE"
            elif tag and price_link_confirmed and tag.get("ocr_status") in {"unreadable", "ambiguous", "error", "unavailable"}:
                price_status = "PRICE UNREADABLE"
            elif tag:
                price_status = "Needs Review"
            else:
                price_status = "Needs Review"
            findings.append({
                "shelf": expected_row["shelf"], "row": expected_row["row"],
                "slot": str(expected.get("position", "")), "expected_sku": expected_sku,
                "expected_product": (
                    expected.get("product", {}).get("name", "") if isinstance(expected.get("product"), dict)
                    else (catalogue_by_sku.get(expected_sku) or {}).get("name", "")
                ),
                "expected_price": expected_price_raw, "detected_product": candidate.get("product") if candidate else "",
                "detected_sku": candidate.get("sku", "") if candidate else "",
                "detection_index": candidate.get("detection_index") if candidate else None,
                "void_index": candidate.get("void_index") if candidate else None,
                "box": candidate.get("box") if candidate else None,
                "confidence": candidate.get("confidence") if candidate else None,
                "facing_count": candidate.get("facing_count", 0) if candidate else 0,
                "detection_indices": candidate.get("detection_indices", []) if candidate else [],
                "individual_detections": candidate.get("individual_detections", []) if candidate else [],
                "status": status, "suggested_status": suggested_status,
                "additional_statuses": additional_statuses,
                "price_status": price_status, "observed_price": observed_price,
                "price_tag": tag, "reason": (audit or {}).get("detail") or "Geometric alignment is a proposal; no calibrated slot mapping confirms this comparison.",
                "confirmed": confirmed_status is not None,
            })
        # Extra cells are reported as unassigned comparison items; no expected slot is fabricated.
        for candidate in actual_slots:
            candidate_index = (candidate["kind"], candidate.get("detection_index", candidate.get("void_index")))
            if candidate_index in assigned_actual:
                continue
            suggested = "UNEXPECTED PRODUCT" if candidate.get("identified") else ("VOID / EMPTY SLOT" if candidate["kind"] == "void" else "Needs Review")
            findings.append({
                "shelf": expected_row["shelf"], "row": expected_row["row"], "slot": None,
                "expected_sku": "", "expected_product": "", "expected_price": None,
                "detected_product": candidate.get("product", ""), "detected_sku": candidate.get("sku", ""),
                "detection_index": candidate.get("detection_index"), "void_index": candidate.get("void_index"),
                "box": candidate.get("box"), "confidence": candidate.get("confidence"),
                "status": "Needs Review", "suggested_status": suggested, "price_status": "Needs Review",
                "observed_price": None, "price_tag": None,
                "reason": "Detected cell has no one-to-one expected slot alignment.", "confirmed": False,
            })
    expected_skus = {str(slot.get("sku", "")) for slot in planogram if slot.get("sku")}
    for extra_row in detected_rows[expected_row_count:]:
        for candidate in extra_row["slots"]:
            suggested = (
                "UNEXPECTED PRODUCT" if candidate.get("identified") and candidate.get("sku") not in expected_skus
                else "VOID / EMPTY SLOT" if candidate.get("kind") == "void"
                else "Needs Review"
            )
            findings.append({
                "shelf": "", "row": str(extra_row["row"]), "slot": None,
                "expected_sku": "", "expected_product": "", "expected_price": None,
                "detected_product": candidate.get("product", ""), "detected_sku": candidate.get("sku", ""),
                "detection_index": candidate.get("detection_index"), "void_index": candidate.get("void_index"),
                "box": candidate.get("box"), "confidence": candidate.get("confidence"),
                "status": "Needs Review", "suggested_status": suggested, "price_status": "Needs Review",
                "observed_price": None, "price_tag": None,
                "reason": "Detected row has no corresponding expected row; row alignment needs review.", "confirmed": False,
            })
    # SKU recognition answers whether a catalogue product is visible. Slot
    # confirmation separately requires a trusted layout mapping. Preserve both
    # facts so the UI can avoid calling a recognized product unidentified while
    # keeping its placement unconfirmed.
    identified_skus = {
        str(item.get("sku") or "")
        for item in detected.get("items", [])
        if item.get("identified") and item.get("sku")
    }
    for item in findings:
        expected_sku = str(item.get("expected_sku") or "")
        item["expected_sku_detected"] = bool(expected_sku and expected_sku in identified_skus)
        item["identity_status"] = (
            "Detected" if item["expected_sku_detected"] else "Not confirmed"
        )

    summary = Counter()
    price_summary = Counter()
    for item in findings:
        if item["confirmed"]:
            for status in [item["status"], *item.get("additional_statuses", [])]:
                summary[status] += 1
        else:
            summary["Needs Review"] += 1
        if item["price_status"] != "Needs Review":
            price_summary[item["price_status"]] += 1
    tags = detected.get("price_tags", [])
    unreadable_tags = {
        tag.get("tag_index", index)
        for index, tag in enumerate(tags)
        if tag.get("ocr_status") in {"unreadable", "ambiguous", "error", "unavailable"}
    }
    price_review_tags = {
        tag.get("tag_index", index)
        for index, tag in enumerate(tags)
        if tag.get("association_status") == "Needs Review"
        or tag.get("ocr_status") in {"needs_review", "not_requested"}
    }
    return {
        "rows": expected_groups,
        "findings": findings,
        "summary": {
            "matches": summary["MATCH"], "missing_products": summary["MISSING PRODUCT"],
            "wrong_or_misplaced": summary["WRONG PRODUCT / MISPLACED"],
            "voids": summary["VOID / EMPTY SLOT"], "expected_empty_slots": summary["EXPECTED EMPTY SLOT"],
            "unexpected_products": summary["UNEXPECTED PRODUCT"],
            "needs_review": summary["Needs Review"], "price_matches": price_summary["PRICE MATCH"],
            "expected_skus_detected": sum(
                1 for item in findings if item.get("expected_sku_detected")
            ),
            "price_mismatches": price_summary["PRICE MISMATCH"],
            "price_unreadable": max(price_summary["PRICE UNREADABLE"], len(unreadable_tags)),
            "price_needs_review": len(price_review_tags),
        },
        "slot_mapping_trusted": bool(layout_trusted),
        "alignment_policy": "Confirmed statuses require the existing trusted slot mapping; geometric row and column alignment remains a proposal.",
    }
