"""ShelfSense AI web application."""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import platform
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, flash, jsonify, make_response, redirect, render_template, request, send_file, url_for
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from backend.matching.pipeline import ProductMatchingPipeline
from backend.matching.shelf_mapper import CoordinateShelfMapper
from backend.matching.storage import save_audit_result
from config.settings import (
    ALLOWED_EXTENSIONS,
    AUDIT_RESULTS_DIR,
    DATA_DIR,
    DETECTION_CONFIDENCE,
    MAX_CONTENT_LENGTH,
    MATCHER_MODEL_ID,
    MAX_IMAGE_PIXELS,
    MODEL_PATH,
    ROOT_DIR,
    SAMPLE_IMAGE_PATH,
    SECRET_KEY,
    TEST_SET_DIR,
    UPLOAD_DIR,
)
from src.detector import DetectorUnavailable, annotate_image, detect_products, detector_status, get_detector
from src.ocr import OcrUnavailable, read_price_tags
from src.planogram import evaluate_planogram, validate_planogram
from src.price_checker import check_prices
from src.report import build_report

logging.basicConfig(level=os.getenv("SHELFSENSE_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("shelfsense")

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_reports: dict[str, dict] = {}
_detect_results: dict[str, dict] = {}
_matching_pipelines: dict[tuple[str, str], ProductMatchingPipeline] = {}
REPORT_SCHEMA_VERSION = 2
CATALOGUE_PATH = DATA_DIR / "catalogue.csv"
SETTINGS_PATH = ROOT_DIR / "config" / "user_settings.json"
CATALOGUE_FIELDS = ["sku", "name", "category", "brand", "pack_size", "price", "model_label", "active"]
DEMO_REPORT_ID = "demo-shelf-report"


def _valid_price(value: str) -> bool:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(price) and price >= 0


REPORT_EXPORT_FIELDS = [
    "record_type", "created_at", "store_name", "planogram_name", "store_id", "planogram_id", "image_name",
    "compliance_rate", "detection_count", "compliant_count", "product", "sku",
    "shelf", "row", "slot", "position", "expected_facings", "observed_facings", "status",
    "expected_price", "observed_price", "confidence", "bounding_box",
    "accepted_sku", "top_similarity", "top_margin", "candidates_json", "slot_mapping_json",
]


def _report_export_rows(report: dict) -> list[dict]:
    metadata = {
        key: report.get(key, "")
        for key in (
            "created_at", "store_name", "planogram_name", "store_id", "planogram_id", "image_name",
            "compliance_rate", "detection_count", "compliant_count",
        )
    }
    rows = []
    for item in report.get("compliance", []):
        rows.append({
            **metadata,
            "record_type": "compliance",
            "product": item.get("product", ""),
            "sku": item.get("sku", ""),
            "shelf": item.get("shelf", ""),
            "row": item.get("row", ""),
            "slot": item.get("slot", item.get("position", "")),
            "position": item.get("position", ""),
            "expected_facings": item.get("expected", ""),
            "observed_facings": item.get("observed", ""),
            "status": item.get("status", ""),
        })
    for item in report.get("price_results", []):
        rows.append({
            **metadata,
            "record_type": "price_check",
            "product": item.get("product", ""),
            "sku": item.get("sku", ""),
            "expected_price": item.get("expected_price", ""),
            "observed_price": item.get("observed_price", ""),
            "status": item.get("status", ""),
        })
    for item in report.get("detections", []):
        rows.append({
            **metadata,
            "record_type": "detection",
            "product": item.get("label", ""),
            "confidence": item.get("confidence", ""),
            "bounding_box": json.dumps(item.get("box", [])),
        })
    for item in report.get("matching_audit", {}).get("matches", []):
        rows.append({
            **metadata,
            "record_type": "sku_match",
            "product": item.get("detector_label", ""),
            "status": item.get("status", "Needs Review"),
            "accepted_sku": item.get("accepted_sku") or "",
            "top_similarity": item.get("top_similarity") if item.get("top_similarity") is not None else "",
            "top_margin": item.get("top_margin") if item.get("top_margin") is not None else "",
            "candidates_json": json.dumps(item.get("candidates", []), ensure_ascii=False),
            "slot_mapping_json": json.dumps(item.get("slot_mapping", {}), ensure_ascii=False),
        })
    return rows


def _get_matching_pipeline() -> ProductMatchingPipeline | None:
    """Return the SKU-matching pipeline for the configured matcher, or None when disabled."""
    if MATCHER_MODEL_ID.casefold() == "none":
        return None
    key = (str(DATA_DIR), MATCHER_MODEL_ID)
    if key not in _matching_pipelines:
        _matching_pipelines[key] = ProductMatchingPipeline(
            data_dir=DATA_DIR,
            model_id=MATCHER_MODEL_ID,
            top_k=int(os.getenv("SHELFSENSE_MATCH_TOP_K", "5")),
        )
    return _matching_pipelines[key]


def _load_settings() -> dict:
    defaults = {
        "application_name": "ShelfSense AI",
        "default_store": "Local Store",
        "default_planogram": "Current planogram",
        "default_catalogue": "Retail Catalogue",
        "detection_confidence": DETECTION_CONFIDENCE,
        "price_confidence": 0.5,
        "minimum_product_height": 30,
        "price_verification": True,
        "shelf_gap_detection": True,
        "low_stock_alerts": True,
        "low_stock_threshold": 2,
    }
    if SETTINGS_PATH.exists():
        try:
            defaults.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


def _save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CATALOGUE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _read_catalogue() -> list[dict]:
    raw_rows = _read_csv(CATALOGUE_PATH)
    products = []
    for raw in raw_rows:
        name = raw.get("name") or raw.get("product_name", "")
        products.append({
            **raw,
            "sku": raw.get("sku", ""),
            "name": name,
            "category": raw.get("category", ""),
            "brand": raw.get("brand", ""),
            "pack_size": raw.get("pack_size", ""),
            "price": raw.get("price") or raw.get("expected_price", ""),
            "model_label": raw.get("model_label") or name.casefold(),
            "active": (raw.get("active") or raw.get("status", "active")).casefold() in {"true", "active", "1", "yes"} and "true" or "false",
            "_source": raw,
        })
    return products


def _save_catalogue(products: list[dict]) -> None:
    with CATALOGUE_PATH.open(encoding="utf-8-sig", newline="") as csv_file:
        fields = next(csv.reader(csv_file), [])
    if not fields:
        fields = CATALOGUE_FIELDS
    existing = {product["sku"].casefold(): product for product in products}
    with CATALOGUE_PATH.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for product in existing.values():
            row = dict(product.get("_source", {}))
            mapping = {
                "sku": ("sku",),
                "name": ("name", "product_name"),
                "category": ("category",),
                "brand": ("brand",),
                "pack_size": ("pack_size",),
                "price": ("price", "expected_price"),
                "model_label": ("model_label",),
                "active": ("active", "status"),
            }
            for canonical, candidates in mapping.items():
                target = next((field for field in candidates if field in fields), None)
                if target:
                    value = product.get(canonical, "")
                    if canonical == "active" and target == "status":
                        value = "Active" if str(value).casefold() == "true" else "Inactive"
                    row[target] = value
            writer.writerow(row)


def _read_planogram() -> list[dict]:
    rows = _read_csv(DATA_DIR / "planogram.csv")
    normalized = []
    for raw in rows:
        row_number = raw.get("row") or re.search(r"\d+", str(raw.get("shelf", "")))
        if isinstance(row_number, re.Match):
            row_number = row_number.group(0)
        row_number = str(row_number or "1")
        shelf_name = f"Row {row_number}" if raw.get("row") else raw.get("shelf", "Shelf 1")
        slot_number = raw.get("slot") or raw.get("position", "1")
        expected_sku = raw.get("expected_sku") or raw.get("sku", "")
        facing_count = raw.get("facing_count") or raw.get("expected_facings", "0")
        normalized.append({
            **raw,
            "store_id": raw.get("store_id") or "STORE001",
            "planogram_id": raw.get("planogram_id") or "DEFAULT",
            "shelf": shelf_name,
            "physical_shelf": raw.get("shelf", "S1"),
            "row": row_number,
            "slot": slot_number,
            "position": slot_number,
            "expected_sku": expected_sku,
            "sku": expected_sku,
            "facing_count": facing_count,
            "expected_facings": facing_count,
            "min_price": raw.get("min_price", ""),
            "_source": raw,
            "_shelf_id": raw.get("shelf", "S1"),
            "_row_number": row_number,
        })
    return normalized


def _save_planogram(rows: list[dict]) -> None:
    path = DATA_DIR / "planogram.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as csv_file:
            fields = next(csv.reader(csv_file), [])
            existing_rows = list(csv.DictReader(csv_file))
    else:
        fields, existing_rows = [], []
    if not fields:
        fields = ["shelf", "position", "sku", "expected_facings", "min_price"]
    baseline = existing_rows[0] if existing_rows else {}
    legacy_schema = "expected_sku" in fields or "facing_count" in fields
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for planogram_row in rows:
            source = dict(planogram_row.get("_source", {}))
            source_shelf = planogram_row.get("_shelf_id") or baseline.get("shelf", "S1")
            row_match = re.search(r"\d+", str(planogram_row.get("shelf", "")))
            row_value = planogram_row.get("_row_number") or source.get("row") or (row_match.group(0) if row_match else "1")
            values = {
                "shelf": source_shelf if legacy_schema else planogram_row.get("shelf", source_shelf),
                "row": row_value,
                "slot": planogram_row.get("position", "1"),
                "expected_sku": planogram_row.get("sku", ""),
                "facing_count": planogram_row.get("facing_count", planogram_row.get("expected_facings", "0")),
                "position": planogram_row.get("position", "1"),
                "sku": planogram_row.get("sku", ""),
                "expected_facings": planogram_row.get("expected_facings", planogram_row.get("facing_count", "0")),
                "min_price": planogram_row.get("min_price", ""),
                "store_id": source.get("store_id", baseline.get("store_id", "STORE001")),
                "planogram_id": source.get("planogram_id", baseline.get("planogram_id", "DEFAULT")),
            }
            for field in fields:
                if field in values:
                    source[field] = values[field]
            writer.writerow(source)


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


class AuditError(Exception):
    """A user-facing audit failure; endpoint is where the HTML flow redirects."""

    def __init__(self, message: str, endpoint: str = "upload_page", status: int = 400):
        super().__init__(message)
        self.message = message
        self.endpoint = endpoint
        self.status = status


def _uploaded_path(name: str | None) -> Path | None:
    """Resolve a stored upload filename, refusing anything outside UPLOAD_DIR."""
    if not name:
        return None
    path = (UPLOAD_DIR / Path(name).name).resolve()
    if path.parent != UPLOAD_DIR.resolve() or not path.is_file():
        return None
    return path


def _report_image_path(report: dict | None, key: str = "image_file") -> Path | None:
    if not report:
        return None
    path = _uploaded_path(report.get(key))
    if path is None and key == "image_file" and report.get("image_path"):
        legacy = Path(report["image_path"])
        if legacy.is_file() and legacy.parent.resolve() == UPLOAD_DIR.resolve():
            path = legacy
    return path


def _store_image(source, display_name: str) -> tuple[Path, int, int]:
    """Validate an image (file stream or path) and store an EXIF-upright JPEG copy in UPLOAD_DIR."""
    data = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as opened:
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise AuditError("That image has too many pixels to analyze. Use an image under 40 megapixels.")
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except AuditError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
        raise AuditError("The uploaded file is not a readable PNG, JPG, or WEBP image.") from error
    if image.width < 32 or image.height < 32:
        raise AuditError("That image is too small to analyze (minimum 32 x 32 pixels).")
    stored_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(display_name).stem or 'shelf'}.jpg"
    image.save(stored_path, format="JPEG", quality=92)
    return stored_path, image.width, image.height


def _parse_confidence(form) -> float:
    default = float(_load_settings()["detection_confidence"])
    try:
        confidence = float(form.get("confidence_threshold", default))
    except (TypeError, ValueError):
        confidence = default
    if not 0.1 <= confidence <= 0.95:
        raise AuditError("Detection confidence must be between 0.10 and 0.95.")
    return confidence


def _incoming_image(files, form) -> tuple[object, str]:
    image = files.get("shelf_image")
    if image is not None and image.filename:
        if not _allowed_file(image.filename):
            raise AuditError("Upload a PNG, JPG, JPEG, or WEBP image.")
        return image.stream, secure_filename(image.filename) or "shelf.jpg"
    test_image = form.get("test_image")
    if test_image:
        path = _test_images().get(test_image)
        if path is None:
            raise AuditError("That test-set image does not exist. Run: python -m tools.build_test_set")
        return path, path.name
    if form.get("use_example") and SAMPLE_IMAGE_PATH.is_file():
        return SAMPLE_IMAGE_PATH, SAMPLE_IMAGE_PATH.name
    raise AuditError("Choose a shelf image to analyze.")


def _test_images() -> dict[str, Path]:
    """Labelled synthetic test-set scenes (data/test_set/test), by file name."""
    directory = TEST_SET_DIR / "test"
    return {path.name: path for path in sorted(directory.glob("*.jpg"))} if directory.is_dir() else {}


def _identify_detections(detections: list[dict], matches: list[dict], planogram: list[dict], slot_mapper) -> list[dict]:
    """Attach the calibrated SKU match and the mapped slot (if any) to each detection."""
    identified = []
    for index, detection in enumerate(detections):
        match = matches[index] if index < len(matches) else {}
        mapping = slot_mapper.map_detection(detection, planogram)
        identity = mapping.get("slot_identity")
        identified.append({
            **detection,
            "sku": match.get("accepted_sku") or "",
            "match_status": match.get("status", "Unknown / Needs Review"),
            "slot_identity": list(identity) if identity else None,
        })
    return identified


def _downgrade_slots_with_unidentified(slots: list[dict], identified: list[dict]) -> None:
    """A slot holding objects the matcher could not identify cannot be confirmed either way."""
    unresolved: dict[tuple, int] = {}
    for detection in identified:
        if not detection["sku"] and detection["slot_identity"]:
            key = tuple(detection["slot_identity"])
            unresolved[key] = unresolved.get(key, 0) + 1
    for slot in slots:
        key = (slot["store_id"], slot["planogram_id"], slot["shelf"], slot["row"], slot["slot"])
        if unresolved.get(key) and slot["status"] != "Empty Slot":
            slot["detail"] = (
                f"{unresolved[key]} object(s) in this slot could not be identified above the calibrated match "
                f"threshold, so the slot is not confirmed (was: {slot['status']})."
            )
            slot["status"] = "Needs Review"
        elif unresolved.get(key):
            slot.update(status="Needs Review", detail=f"{unresolved[key]} unidentified object(s) in an intentionally empty slot.")


def _assign_price_tags(tags: list[dict], planogram: list[dict], identified: list[dict], slot_mapper,
                       image_width: int, image_height: int) -> list[dict]:
    """Link each OCR price tag to a SKU: by calibrated slot, else by the identified product directly above."""
    slots_by_identity = {
        (row["store_id"], row["planogram_id"], row["physical_shelf"], str(row["row"]), str(row["slot"])): row
        for row in planogram
    }
    for tag in tags:
        tag.update(sku="", slot=None, association="unassigned")
        if slot_mapper.trusted:
            mapping = slot_mapper.map_detection(
                {"box": tag["box"], "image_width": image_width, "image_height": image_height}, planogram)
            row = slots_by_identity.get(tuple(mapping["slot_identity"])) if mapping.get("slot_identity") else None
            if row and row.get("sku"):
                tag.update(sku=row["sku"], slot=f"R{row['row']}S{row['slot']}", association="calibrated slot")
            continue
        center_x, tag_top = (tag["box"][0] + tag["box"][2]) / 2, tag["box"][1]
        above = [
            detection for detection in identified
            if detection["sku"] and detection["box"][0] <= center_x <= detection["box"][2]
            and -0.02 * image_height <= tag_top - detection["box"][3] <= 0.25 * image_height
        ]
        if above:
            nearest = min(above, key=lambda detection: tag_top - detection["box"][3])
            tag.update(sku=nearest["sku"], association="nearest identified product above (heuristic)")
    return tags


def _price_results(planogram_products: list[dict], tags: list[dict]) -> list[dict]:
    observed: dict[str, list[float]] = {}
    for tag in tags:
        if tag.get("sku"):
            observed.setdefault(tag["sku"], []).append(tag["price"])
    results = check_prices(planogram_products, {sku: max(set(values), key=values.count) for sku, values in observed.items()})
    for row in results:
        values = observed.get(row["sku"], [])
        row["tag_count"] = len(values)
        if len(set(values)) > 1:
            row["detail"] = f"Conflicting tag prices read: {sorted(set(values))}"
        methods = {tag["association"] for tag in tags if tag.get("sku") == row["sku"]}
        row["association"] = ", ".join(sorted(methods))
    return results


def _facing_summary(planogram: list[dict], detections: list[dict]) -> dict:
    """Class-agnostic comparison of detected object counts against planogram facings."""
    expected_by_row: dict[str, int] = {}
    for slot in planogram:
        if slot.get("sku"):
            expected_by_row[str(slot.get("row", "1"))] = expected_by_row.get(str(slot.get("row", "1")), 0) + int(slot.get("expected_facings") or 0)
    ordered_rows = sorted(expected_by_row, key=lambda value: (int(value) if value.isdigit() else 10**9, value))
    detected_rows = [len(row) for row in CoordinateShelfMapper.cluster_detections_by_row(detections)]
    expected_total = sum(expected_by_row.values())
    return {
        "expected_facings": expected_total,
        "detected_objects": len(detections),
        "difference": len(detections) - expected_total,
        "expected_by_row": [{"row": row, "facings": expected_by_row[row]} for row in ordered_rows],
        "detected_by_row": detected_rows,
        "note": (
            "Object counts are class-agnostic estimates from one image. They are not SKU-level stock counts: "
            "occlusion, depth, duplicate or missed boxes, and products outside the planogram all affect them."
        ),
    }


def _audit_limitations(model_info: dict, matching_audit: dict, layout_trusted: bool, price_audit: dict) -> list[str]:
    limitations = []
    classes = model_info.get("class_names") or []
    if model_info.get("class_agnostic"):
        limitations.append(
            f"The detector ({model_info.get('model_file')}) has a single class ('{classes[0]}'): it locates products; "
            "SKU identity comes only from catalogue image matching."
        )
    calibration = matching_audit.get("calibration", {})
    if matching_audit.get("matching_error"):
        limitations.append(f"SKU matching is unavailable: {matching_audit['matching_error']}")
    elif calibration.get("calibration_status") != "calibrated_on_validation_set":
        limitations.append("SKU candidates are advisory: no validation-calibrated acceptance threshold is configured.")
    else:
        accepted = sum(bool(match.get("accepted_sku")) for match in matching_audit.get("matches", []))
        limitations.append(
            f"SKU matching ({calibration.get('model_id')}) accepted {accepted} of {len(matching_audit.get('matches', []))} "
            f"crops at the calibrated threshold {calibration.get('acceptance_threshold')} / margin "
            f"{calibration.get('minimum_margin')}; the rest stay Needs Review. Calibration used synthetic shelves, "
            "so accuracy on real photos may be lower."
        )
    if layout_trusted:
        limitations.append(
            "Slot mapping is trusted because the image was declared to follow the calibrated fixed-camera layout "
            "(planogram rows and slots as equal bands). Results are wrong if that declaration is wrong."
        )
    else:
        limitations.append("Slot mapping uses an uncalibrated equal-grid approximation, so slot findings remain Needs Review.")
    if not price_audit.get("requested"):
        limitations.append("Price check was not requested for this audit.")
    elif price_audit.get("error"):
        limitations.append(f"Price OCR is unavailable: {price_audit['error']}")
    else:
        assigned = sum(bool(tag.get("sku")) for tag in price_audit.get("tags", []))
        limitations.append(
            f"OCR ({price_audit.get('engine')}) read {len(price_audit.get('tags', []))} price tag(s); {assigned} were linked "
            "to a SKU. Prices without a linked tag remain Unverified."
        )
    return limitations


def _run_audit(files, form) -> tuple[str, dict]:
    """Image upload -> detection -> SKU matching -> slot mapping -> planogram comparison -> saved report."""
    settings = _load_settings()
    source, display_name = _incoming_image(files, form)
    confidence = _parse_confidence(form)

    all_catalogue = _read_catalogue()
    catalogue = [product for product in all_catalogue if product.get("active", "true").casefold() == "true"]
    complete_planogram = _read_planogram()
    planogram_errors = validate_planogram(complete_planogram, all_catalogue)
    if planogram_errors:
        raise AuditError("Planogram validation failed: " + " ".join(planogram_errors[:3]), endpoint="planogram_page", status=422)
    active_skus = {product["sku"] for product in catalogue}
    planogram = [slot for slot in complete_planogram if not slot.get("sku") or slot["sku"] in active_skus]

    stored_path, image_width, image_height = _store_image(source, display_name)
    started = time.perf_counter()
    try:
        detections = detect_products(stored_path, confidence=confidence)
    except DetectorUnavailable as error:
        stored_path.unlink(missing_ok=True)
        raise AuditError(str(error), status=503) from error
    except Exception as error:
        stored_path.unlink(missing_ok=True)
        logger.exception("Shelf image analysis failed")
        raise AuditError("Image analysis failed unexpectedly. Check the model and image, then try again.", status=500) from error
    inference_ms = round((time.perf_counter() - started) * 1000, 1)

    report_id = uuid.uuid4().hex
    layout_trusted = bool(form.get("calibrated_layout") or form.get("test_image"))
    slot_mapper = CoordinateShelfMapper(trusted=layout_trusted)
    pipeline = _get_matching_pipeline()
    if pipeline is None:
        matching_audit = {"matches": [], "findings": [], "matching_error": "SKU matching is disabled (SHELFSENSE_MATCHER=none)."}
    else:
        try:
            matching_audit = pipeline.run(stored_path, detections, catalogue, planogram, slot_mapper=slot_mapper)
        except Exception as error:
            logger.exception("SKU matching pipeline failed for report %s", report_id)
            matching_audit = {"matches": [], "findings": [], "matching_error": f"Matching pipeline error: {error}"}
    identified = _identify_detections(detections, matching_audit.get("matches", []), planogram, slot_mapper)
    names_by_sku = {product["sku"]: product.get("name") or f"SKU {product['sku']}" for product in all_catalogue}
    for detection, enriched in zip(detections, identified):
        # The model class ("object") stays in "label"; the catalogue identity is kept separately.
        detection["matched_sku"] = enriched["sku"]
        detection["product_name"] = names_by_sku.get(enriched["sku"], "") if enriched["sku"] else ""

    annotated_name = None
    try:
        annotated_name = annotate_image(stored_path, detections, UPLOAD_DIR / f"{report_id}_annotated.jpg").name
    except Exception:
        logger.exception("Could not draw annotated image for report %s", report_id)

    price_audit: dict = {"requested": bool(form.get("price_check")) and bool(settings.get("price_verification", True)), "tags": []}
    if price_audit["requested"]:
        price_audit["engine"] = "rapidocr-onnxruntime"
        try:
            tags = read_price_tags(stored_path, float(settings.get("price_confidence", 0.5)))
            price_audit["tags"] = _assign_price_tags(tags, planogram, identified, slot_mapper, image_width, image_height)
        except OcrUnavailable as error:
            price_audit["error"] = str(error)
        except Exception as error:
            logger.exception("Price OCR failed for report %s", report_id)
            price_audit["error"] = f"OCR failed: {error}"

    try:
        planogram_audit = evaluate_planogram(
            planogram, all_catalogue, identified,
            slot_mapper=(lambda detection, _rows: tuple(detection["slot_identity"]) if detection["slot_identity"] else None)
            if layout_trusted else None,
        )
        if layout_trusted:
            _downgrade_slots_with_unidentified(planogram_audit["slots"], identified)
        compliance = planogram_audit["slots"]
        planogram_skus = {slot["sku"] for slot in planogram if slot.get("sku")}
        price_results = (
            _price_results([product for product in catalogue if product["sku"] in planogram_skus], price_audit["tags"])
            if price_audit["requested"] else []
        )
        report = build_report(
            compliance=compliance,
            price_results=price_results,
            detections=detections,
            image_name=display_name,
            store_name=form.get("store", settings["default_store"]).strip() or settings["default_store"],
        )
    except Exception as error:
        stored_path.unlink(missing_ok=True)
        logger.exception("Shelf audit report generation failed")
        raise AuditError("The audit could not be completed. Check the catalogue and planogram, then try again.", status=500) from error

    model_info = {key: value for key, value in detector_status().items() if key != "model_path"}
    report.update(
        report_id=report_id,
        schema_version=REPORT_SCHEMA_VERSION,
        created_at_iso=datetime.now(timezone.utc).isoformat(),
        planogram_name=form.get("planogram", settings["default_planogram"]).strip(),
        confidence_threshold=confidence,
        planogram_audit=planogram_audit,
        matching_audit=matching_audit,
        image_file=stored_path.name,
        image_path=str(stored_path.resolve()),
        annotated_image=annotated_name,
        image_width=image_width,
        image_height=image_height,
        inference_ms=inference_ms,
        model=model_info,
        facing_summary=_facing_summary(planogram, detections),
        layout_trusted=layout_trusted,
        price_audit=price_audit,
        identified_count=sum(bool(detection["sku"]) for detection in identified),
        limitations=_audit_limitations(model_info, matching_audit, layout_trusted, price_audit),
    )
    try:
        save_audit_result(AUDIT_RESULTS_DIR, report_id, {key: value for key, value in report.items() if key != "image_path"})
        report["persisted"] = True
    except (OSError, ValueError, TypeError):
        logger.exception("Could not persist audit report %s", report_id)
        report["persisted"] = False
    _reports[report_id] = report
    logger.info("Audit %s: %d detections in %.0f ms (conf %.2f)", report_id, len(detections), inference_ms, confidence)
    return report_id, report


def _normalize_saved_report(report_id: str, report: dict) -> dict:
    """Fill fields missing from older saved reports so every page can render them."""
    report.setdefault("report_id", report_id)
    report.setdefault("image_name", "unknown image")
    report.setdefault("created_at", "")
    report.setdefault("compliance", [])
    report.setdefault("detections", [])
    report.setdefault("price_results", [])
    report.setdefault("detection_count", len(report["detections"]))
    report.setdefault("compliant_count", sum(row.get("status") == "Compliant" for row in report["compliance"]))
    report.setdefault("compliance_rate", None)
    report.setdefault("price_unverified", sum(row.get("status") == "Unverified" for row in report["price_results"]))
    report.setdefault("planogram_audit", {
        "slots": report["compliance"], "detections": [], "validation_errors": [],
        "slot_mapping_available": False,
        "empty_slot_count": sum(row.get("status") == "Empty Slot" for row in report["compliance"]),
    })
    report["persisted"] = True
    return report


def _load_saved_reports(results_dir: Path | None = None) -> dict[str, dict]:
    """Load persisted JSON audit reports, oldest first, skipping unreadable files."""
    directory = Path(results_dir or AUDIT_RESULTS_DIR)
    loaded = []
    for path in directory.glob("*.json") if directory.is_dir() else []:
        if not re.fullmatch(r"[0-9a-f-]+", path.stem):
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable audit report %s", path.name)
            continue
        if not isinstance(report, dict):
            continue
        sort_key = report.get("created_at_iso") or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        loaded.append((sort_key, path.stem, _normalize_saved_report(path.stem, report)))
    return {report_id: report for _, report_id, report in sorted(loaded, key=lambda item: item[0])}


_reports.update(_load_saved_reports())


@app.get("/")
def dashboard():
    catalogue = _read_catalogue()
    planogram = _read_planogram()
    latest_report_id, latest_report = next(reversed(_reports.items()), (None, None))
    products_by_sku = {product["sku"]: product for product in catalogue}
    shelf_one = next(iter(dict.fromkeys(slot["shelf"] for slot in planogram)), None)
    expected_shelf = [
        {**slot, "product": products_by_sku.get(slot["sku"])}
        for slot in planogram
        if slot["shelf"] == shelf_one
    ]
    latest_issues = []
    if latest_report:
        for row in latest_report["compliance"]:
            if row["status"] in {"Missing Expected Product", "Misplaced", "Unexpected Product"}:
                latest_issues.append(
                    {
                        **row,
                        "issue": row["status"],
                        "detail": row.get("detail", "Confirmed planogram finding."),
                    }
                )
        if latest_report.get("planogram_audit", {}).get("slot_mapping_available") is False:
            latest_issues.append({
                "product": "Planogram slots",
                "sku": "",
                "issue": "Needs Review",
                "detail": "Bounding boxes do not identify exact shelf slots; slot-level findings are unconfirmed.",
            })
        detection_findings = latest_report.get("planogram_audit", {}).get("detections", [])
        unidentified = [finding for finding in detection_findings if not finding.get("sku")]
        if unidentified:
            latest_issues.append({
                "product": f"{len(unidentified)} detected objects",
                "sku": "",
                "issue": "Needs Review",
                "detail": "Could not be matched to a catalogue product above the calibrated threshold.",
                "shelf": "",
                "position": "",
            })
        latest_issues.extend(
            {**finding, "issue": finding.get("status", "Needs Review"), "shelf": "", "position": ""}
            for finding in detection_findings
            if finding.get("sku")
        )
        latest_issues.extend(
            {**row, "issue": "Price mismatch", "detail": f"Shelf tag ₹{row['observed_price']:.2f}; catalogue ₹{row['expected_price']:.2f}."}
            for row in latest_report["price_results"]
            if row["status"] == "Mismatch" and row["observed_price"] is not None
        )
    return render_template(
        "dashboard.html",
        catalogue=catalogue,
        planogram=planogram,
        latest_report=latest_report,
        latest_report_id=latest_report_id,
        latest_issues=latest_issues,
        expected_shelf=expected_shelf,
        expected_total=sum(int(slot["expected_facings"]) for slot in planogram),
        active_product_count=sum(product.get("active", "true").lower() == "true" for product in catalogue),
        model_configured=bool(MODEL_PATH and Path(MODEL_PATH).is_file()),
        product_names={product["sku"]: product.get("name") or f"SKU {product['sku']}" for product in catalogue},
        latest_image_available=_report_image_path(latest_report) is not None,
        latest_annotated_available=_report_image_path(latest_report, "annotated_image") is not None,
        active_page="dashboard",
    )


@app.get("/reports/<report_id>/image")
def report_image(report_id: str):
    report = _reports.get(report_id)
    if report is None and report_id == DEMO_REPORT_ID:
        if SAMPLE_IMAGE_PATH.is_file():
            return send_file(SAMPLE_IMAGE_PATH, mimetype="image/jpeg", max_age=3600)
        abort(404)
    image_path = _report_image_path(report)
    if image_path is None:
        abort(404)
    suffix = image_path.suffix.lower()
    mime_type = "image/svg+xml" if suffix == ".svg" else "image/jpeg"
    return send_file(io.BytesIO(image_path.read_bytes()), mimetype=mime_type, download_name=image_path.name, max_age=0)


@app.get("/reports/<report_id>/annotated")
def report_annotated_image(report_id: str):
    image_path = _report_image_path(_reports.get(report_id), "annotated_image")
    if image_path is None:
        abort(404)
    return send_file(io.BytesIO(image_path.read_bytes()), mimetype="image/jpeg", download_name=image_path.name, max_age=0)


@app.get("/upload")
def upload_page():
    example = None
    if SAMPLE_IMAGE_PATH.is_file():
        with Image.open(SAMPLE_IMAGE_PATH) as sample:
            example = {"name": SAMPLE_IMAGE_PATH.name, "size_kb": round(SAMPLE_IMAGE_PATH.stat().st_size / 1024),
                       "width": sample.width, "height": sample.height}
    return render_template(
        "upload.html",
        page_title="Upload & Analyze",
        active_page="upload",
        settings=_load_settings(),
        example=example,
        test_images=list(_test_images()),
    )


@app.get("/planogram")
def planogram_page():
    catalogue = _read_catalogue()
    planogram = _read_planogram()
    products_by_sku = {product["sku"]: product for product in catalogue}
    shelf_rows: dict[str, list[dict]] = {}
    for slot in planogram:
        shelf_rows.setdefault(slot["shelf"], []).append(
            {**slot, "product": products_by_sku.get(slot["sku"])}
        )
    total_slots = len(planogram)
    assigned_slots = sum(bool(slot.get("sku") and products_by_sku.get(slot["sku"])) for slot in planogram)
    planogram_errors = validate_planogram(planogram, catalogue)
    return render_template(
        "planogram.html",
        shelf_rows=shelf_rows,
        products=[product for product in catalogue if product.get("active", "true").casefold() == "true"],
        total_slots=total_slots,
        assigned_slots=assigned_slots,
        empty_slots=total_slots - assigned_slots,
        planogram_errors=planogram_errors,
        page_title="Planogram",
        active_page="planogram",
    )


@app.post("/planogram/row/add")
def add_planogram_row():
    shelf = request.form.get("shelf", "").strip()
    if not shelf:
        flash("Enter a shelf row name.", "error")
        return redirect(url_for("planogram_page"))
    planogram_path = DATA_DIR / "planogram.csv"
    rows = _read_planogram()
    if any(row["shelf"].casefold() == shelf.casefold() for row in rows):
        flash("That shelf row already exists.", "error")
        return redirect(url_for("planogram_page"))
    row_numbers = [int(match.group(0)) for row in rows if (match := re.search(r"\d+", str(row.get("shelf", ""))))]
    next_row_number = max(row_numbers, default=0) + 1
    source_shelf_id = rows[0].get("_shelf_id", "S1") if rows else "S1"
    rows.extend(
        {"shelf": shelf, "position": str(position), "slot": str(position), "sku": "", "expected_sku": "", "expected_facings": "0", "facing_count": "0", "min_price": "", "_shelf_id": source_shelf_id, "_row_number": next_row_number}
        for position in range(1, 6)
    )
    _save_planogram(rows)
    flash(f"{shelf} was added with five empty slots.", "success")
    return redirect(url_for("planogram_page"))


@app.post("/planogram/slot/add")
def assign_planogram_slot():
    shelf = request.form.get("shelf", "").strip()
    sku = request.form.get("sku", "").strip()
    try:
        position = int(request.form.get("position", ""))
        facings = int(request.form.get("expected_facings", "1"))
    except ValueError:
        flash("Slot position and expected facings must be whole numbers.", "error")
        return redirect(url_for("planogram_page"))
    if not shelf or position < 1 or facings < 1:
        flash("Choose a shelf and enter positive slot and facing counts.", "error")
        return redirect(url_for("planogram_page"))

    catalogue = _read_catalogue()
    product = next(
        (item for item in catalogue if item["sku"].casefold() == sku.casefold() and item.get("active", "true").casefold() == "true"),
        None,
    )
    if product is None:
        flash("Choose an active catalogue product.", "error")
        return redirect(url_for("planogram_page"))

    rows = _read_planogram()
    slot = next((row for row in rows if row["shelf"].casefold() == shelf.casefold() and row["position"] == str(position)), None)
    if slot is None:
        row_match = re.search(r"\d+", shelf)
        slot = {
            "shelf": shelf,
            "position": str(position),
            "sku": "",
            "expected_facings": "0",
            "facing_count": "0",
            "expected_sku": "",
            "min_price": "",
            "_shelf_id": rows[0].get("_shelf_id", "S1") if rows else "S1",
            "_row_number": row_match.group(0) if row_match else "1",
        }
        rows.append(slot)
    elif slot.get("sku"):
        flash("That slot is already assigned. Clear it before assigning another product.", "error")
        return redirect(url_for("planogram_page"))
    slot["sku"] = product["sku"]
    slot["expected_sku"] = product["sku"]
    slot["expected_facings"] = str(facings)
    slot["facing_count"] = str(facings)
    slot["min_price"] = product["price"]
    rows.sort(key=lambda row: (row["shelf"].casefold(), int(row["position"])))
    _save_planogram(rows)
    flash(f"{product['name']} was assigned to {shelf}, slot {position}.", "success")
    return redirect(url_for("planogram_page"))


@app.post("/planogram/clear")
def clear_planogram():
    _save_planogram([])
    flash("Planogram cleared.", "success")
    return redirect(url_for("planogram_page"))


@app.post("/planogram/save")
def save_planogram():
    rows = _read_planogram()
    if not rows:
        flash("The planogram is empty. Add a row or product before saving.", "error")
        return redirect(url_for("planogram_page"))
    errors = validate_planogram(rows, _read_catalogue())
    if errors:
        flash("Planogram validation failed: " + " ".join(errors[:3]), "error")
        return redirect(url_for("planogram_page"))
    flash(f"Planogram saved with {len(rows)} slots.", "success")
    return redirect(url_for("planogram_page"))


@app.post("/analyze")
def analyze():
    try:
        report_id, report = _run_audit(request.files, request.form)
    except AuditError as error:
        flash(error.message, "error")
        return redirect(url_for(error.endpoint))
    if not report.get("persisted"):
        flash("The audit finished but could not be saved to disk; it will be lost when the server restarts.", "error")
    return redirect(url_for("report_detail", report_id=report_id))


@app.get("/detect")
def detect_page():
    return render_template(
        "detect.html",
        settings=_load_settings(),
        model=detector_status(),
        page_title="Run Product Detection",
        active_page="upload",
    )


@app.post("/detect")
def detect():
    """Detection-only run: boxes and timings without a planogram audit or saved report."""
    try:
        source, display_name = _incoming_image(request.files, request.form)
        confidence = _parse_confidence(request.form)
        stored_path, _, _ = _store_image(source, display_name)
    except AuditError as error:
        flash(error.message, "error")
        return redirect(url_for("detect_page"))
    started = time.perf_counter()
    try:
        detections = detect_products(stored_path, confidence=confidence)
    except DetectorUnavailable as error:
        stored_path.unlink(missing_ok=True)
        flash(str(error), "error")
        return redirect(url_for("detect_page"))
    except Exception:
        stored_path.unlink(missing_ok=True)
        logger.exception("Detection-only run failed")
        flash("Image analysis failed unexpectedly. Check the model and image, then try again.", "error")
        return redirect(url_for("detect_page"))
    detect_time = time.perf_counter() - started
    result_id = uuid.uuid4().hex
    annotated = annotate_image(stored_path, detections, UPLOAD_DIR / f"{result_id}_detect.jpg")
    stored_path.unlink(missing_ok=True)
    _detect_results[result_id] = {
        "image_name": display_name, "detections": detections, "detect_time": detect_time,
        "confidence": confidence, "annotated_image": annotated.name, "model": detector_status(),
    }
    return render_template("detect_result.html", result=_detect_results[result_id], result_id=result_id,
                           page_title="Detection Result", active_page="upload")


@app.get("/detect/<result_id>/image")
def detect_image(result_id: str):
    result = _detect_results.get(result_id)
    path = _uploaded_path(result.get("annotated_image")) if result else None
    if path is None:
        abort(404)
    return send_file(io.BytesIO(path.read_bytes()), mimetype="image/jpeg", download_name=path.name, max_age=0)


@app.get("/catalogue")
def catalogue_page():
    products = _read_catalogue()
    return render_template(
        "catalogue.html",
        products=products,
        categories=sorted({product["category"] for product in products if product.get("category")}),
        page_title="Product Catalogue",
        active_page="catalogue",
    )


@app.get("/catalogue/image/<barcode>")
def catalogue_product_image(barcode: str):
    product = next(
        (item for item in _read_catalogue() if item.get("barcode") == barcode),
        None,
    )
    if product is None or not product.get("image_local_path"):
        abort(404)
    image_path = (DATA_DIR / product["image_local_path"]).resolve()
    image_directory = (DATA_DIR / "product_images").resolve()
    if image_path.parent != image_directory or not image_path.is_file():
        abort(404)
    suffix = image_path.suffix.lower()
    mimetype = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(suffix)
    if mimetype is None:
        abort(404)
    return send_file(image_path, mimetype=mimetype, conditional=True, max_age=3600)


@app.post("/catalogue/add")
def add_product():
    values = {field: request.form.get(field, "").strip() for field in CATALOGUE_FIELDS}
    values["active"] = values["active"] or "true"
    if not all(values[field] for field in ("sku", "name", "category", "price", "model_label")):
        flash("SKU, product name, category, price, and model label are required.", "error")
        return redirect(url_for("catalogue_page"))
    if not _valid_price(values["price"]):
        flash("Enter a valid numeric price.", "error")
        return redirect(url_for("catalogue_page"))
    products = _read_catalogue()
    if any(product["sku"].casefold() == values["sku"].casefold() for product in products):
        flash("That SKU already exists in the catalogue.", "error")
        return redirect(url_for("catalogue_page"))
    products.append(values)
    _save_catalogue(products)
    flash(f"{values['name']} was added to the catalogue.", "success")
    return redirect(url_for("catalogue_page"))


@app.post("/catalogue/<sku>/edit")
def edit_product(sku: str):
    products = _read_catalogue()
    product = next((item for item in products if item["sku"].casefold() == sku.casefold()), None)
    if product is None:
        abort(404)
    for field in CATALOGUE_FIELDS:
        if field != "sku":
            product[field] = request.form.get(field, product.get(field, "")).strip()
    if not all(product.get(field, "") for field in ("name", "category", "price", "model_label")):
        flash("Product name, category, price, and model label are required.", "error")
        return redirect(url_for("catalogue_page"))
    if not _valid_price(product["price"]):
        flash("Enter a valid numeric price.", "error")
        return redirect(url_for("catalogue_page"))
    product["active"] = "true" if request.form.get("active") == "on" else "false"
    _save_catalogue(products)
    flash(f"{product['name']} was updated.", "success")
    return redirect(url_for("catalogue_page"))


@app.post("/catalogue/import")
def import_catalogue():
    upload = request.files.get("catalogue_file")
    if upload is None or not upload.filename:
        flash("Choose a CSV file to import.", "error")
        return redirect(url_for("catalogue_page"))
    try:
        imported = list(csv.DictReader(io.StringIO(upload.stream.read().decode("utf-8-sig"))))
        required = {"sku", "name", "category", "price", "model_label"}
        if not imported or not required.issubset(imported[0]):
            raise ValueError("CSV must include sku, name, category, price, and model_label columns.")
        existing = {product["sku"].casefold(): product for product in _read_catalogue()}
        for product in imported:
            if not all(product.get(field, "").strip() for field in required):
                raise ValueError("Every imported row must include SKU, name, category, price, and model label.")
            product["active"] = product.get("active") or "true"
            existing[product["sku"].casefold()] = {field: product.get(field, "") for field in CATALOGUE_FIELDS}
            if not _valid_price(product["price"]):
                raise ValueError(f"Invalid price for SKU {product['sku']}.")
        _save_catalogue(list(existing.values()))
    except (UnicodeDecodeError, csv.Error, ValueError, KeyError) as error:
        flash(f"Catalogue import failed: {error}", "error")
        return redirect(url_for("catalogue_page"))
    flash(f"Imported {len(imported)} catalogue products.", "success")
    return redirect(url_for("catalogue_page"))


@app.get("/reports")
def reports_page():
    reports = list(reversed(list(_reports.items())))
    trend_reports = [(report_id, report) for report_id, report in reports if report.get("compliance_rate") is not None]
    confirmed_statuses = {"Compliant", "Misplaced", "Missing Expected Product", "Unexpected Product"}
    total_placements = sum(
        row.get("status") in confirmed_statuses
        for _, report in reports
        for row in report.get("compliance", [])
    )
    compliant = sum(report["compliant_count"] for _, report in reports)
    issue_count = sum(
        row.get("status") in {"Misplaced", "Missing Expected Product", "Unexpected Product"}
        for _, report in reports
        for row in report.get("compliance", [])
    )
    mismatch_count = sum(
        row["status"] == "Mismatch"
        for _, report in reports
        for row in report["price_results"]
    )
    by_store: dict[str, dict] = {}
    for report_id, report in reports:
        store_name = report.get("store_name", "Local Store")
        entry = by_store.setdefault(store_name, {"store": store_name, "audits": 0, "compliance_total": 0, "placements": 0, "issues": 0})
        entry["audits"] += 1
        entry["compliance_total"] += report["compliant_count"]
        entry["placements"] += sum(row.get("status") in confirmed_statuses for row in report.get("compliance", []))
        entry["issues"] += sum(row.get("status") in {"Misplaced", "Missing Expected Product", "Unexpected Product"} for row in report.get("compliance", []))
    store_summaries = [
        {**store, "compliance_rate": round(store["compliance_total"] / store["placements"] * 100) if store["placements"] else None}
        for store in by_store.values()
    ]
    status_counts = {
        status: sum(
            row["status"] == status
            for _, report in reports
            for row in report["compliance"]
        )
        for status in ("Compliant", "Misplaced", "Missing Expected Product", "Unexpected Product", "Needs Review", "Empty Slot")
    }
    catalogue_by_sku = {product["sku"]: product for product in _read_catalogue()}
    category_totals: dict[str, dict[str, int]] = {}
    issue_products: dict[str, int] = {}
    price_issues_by_store: dict[str, int] = {}
    issue_type_counts = {"Misplaced": 0, "Missing": 0, "Unexpected Product": 0, "Needs Review": 0}
    issue_trends = []
    for _, report in reports:
        store_name = report.get("store_name", "Local Store")
        trend_counts = {"Missing": 0, "Misplaced": 0, "Unexpected Product": 0, "Needs Review": 0, "Price mismatch": 0}
        for row in report["compliance"]:
            if row.get("status") in {"Empty Slot", "Intentionally Empty"}:
                continue
            product = catalogue_by_sku.get(row.get("sku"), {})
            category = product.get("category") or "Other"
            if row.get("status") in confirmed_statuses:
                category_entry = category_totals.setdefault(category, {"checked": 0, "compliant": 0})
                category_entry["checked"] += 1
                category_entry["compliant"] += row["status"] == "Compliant"
            if row.get("status") != "Compliant":
                if row.get("status") in confirmed_statuses:
                    label = row.get("product") or f"SKU {row.get('sku')}"
                    issue_products[label] = issue_products.get(label, 0) + 1
                issue_type = "Missing" if row["status"] == "Missing Expected Product" else row["status"]
                if issue_type in issue_type_counts:
                    issue_type_counts[issue_type] += 1
                if issue_type in trend_counts:
                    trend_counts[issue_type] += 1
        for finding in report.get("planogram_audit", {}).get("detections", []):
            issue_type = finding.get("status", "Needs Review")
            if issue_type in issue_type_counts:
                issue_type_counts[issue_type] += 1
            if issue_type in trend_counts:
                trend_counts[issue_type] += 1
            if finding.get("sku"):
                issue_products[finding.get("product")] = issue_products.get(finding.get("product"), 0) + 1
        for price_row in report["price_results"]:
            if price_row["status"] == "Mismatch":
                issue_products[price_row["product"]] = issue_products.get(price_row["product"], 0) + 1
                price_issues_by_store[store_name] = price_issues_by_store.get(store_name, 0) + 1
                trend_counts["Price mismatch"] += 1
        if any(trend_counts.values()):
            issue_trends.append({"label": report.get("created_at", "Audit")[:6], **trend_counts})
    category_summaries = [
        {
            "category": category,
            "compliance_rate": round(values["compliant"] / values["checked"] * 100) if values["checked"] else 0,
        }
        for category, values in sorted(category_totals.items())
    ]
    top_issue_products = sorted(
        ({"product": product, "issues": count} for product, count in issue_products.items()),
        key=lambda item: (-item["issues"], item["product"]),
    )[:5]
    price_store_summaries = [
        {"store": store, "mismatches": count}
        for store, count in sorted(price_issues_by_store.items(), key=lambda item: (-item[1], item[0]))
    ]
    issue_trend_max = max(
        (
            trend["Missing"] + trend["Misplaced"] + trend["Unexpected Product"] + trend["Needs Review"] + trend["Price mismatch"]
            for trend in issue_trends
        ),
        default=1,
    )
    return render_template(
        "reports.html",
        reports=reports,
        trend_reports=trend_reports,
        audit_count=len(reports),
        average_compliance=round(compliant / total_placements * 100) if total_placements else None,
        issue_count=issue_count,
        mismatch_count=mismatch_count,
        store_summaries=store_summaries,
        status_counts=status_counts,
        issue_type_counts=issue_type_counts,
        issue_trends=list(reversed(issue_trends[:10])),
        issue_trend_max=issue_trend_max,
        category_summaries=category_summaries,
        top_issue_products=top_issue_products,
        price_store_summaries=price_store_summaries,
        page_title="Reports",
        active_page="reports",
    )


@app.get("/reports/download")
def download_reports():
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=REPORT_EXPORT_FIELDS)
    writer.writeheader()
    for report in _reports.values():
        writer.writerows(_report_export_rows(report))
    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=shelfsense-audit-history.csv"
    return response


@app.get("/settings")
def settings_page():
    return render_template(
        "settings.html",
        settings=_load_settings(),
        model_configured=bool(MODEL_PATH and Path(MODEL_PATH).is_file()),
        model=detector_status(),
        matcher=MATCHER_MODEL_ID,
        calibration_status=_calibration_summary(),
        results_dir=str(AUDIT_RESULTS_DIR),
        python_version=platform.python_version(),
        page_title="Settings",
        active_page="settings",
    )


@app.post("/settings/save")
def save_settings():
    settings = _load_settings()
    for field in ("default_store", "default_planogram", "default_catalogue", "theme", "language"):
        value = request.form.get(field)
        if value:
            settings[field] = value.strip()
    try:
        settings["detection_confidence"] = float(request.form.get("detection_confidence", settings["detection_confidence"]))
        settings["price_confidence"] = float(request.form.get("price_confidence", settings["price_confidence"]))
        settings["minimum_product_height"] = int(request.form.get("minimum_product_height", settings["minimum_product_height"]))
        settings["low_stock_threshold"] = int(request.form.get("low_stock_threshold", settings["low_stock_threshold"]))
        if not 0.1 <= settings["detection_confidence"] <= 0.9 or not 0.1 <= settings["price_confidence"] <= 0.95:
            raise ValueError("Confidence values are outside the allowed range.")
        if settings["minimum_product_height"] < 1 or settings["low_stock_threshold"] < 0:
            raise ValueError("Height and stock threshold must be zero or greater.")
    except ValueError as error:
        flash(f"Settings were not saved: {error}", "error")
        return redirect(url_for("settings_page"))
    settings["price_verification"] = "price_verification" in request.form
    settings["shelf_gap_detection"] = "shelf_gap_detection" in request.form
    settings["low_stock_alerts"] = "low_stock_alerts" in request.form
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    flash("Analysis settings saved.", "success")
    return redirect(url_for("settings_page"))


@app.get("/reports/<report_id>")
def report_detail(report_id: str):
    report = _reports.get(report_id)
    if report is None:
        abort(404)
    return render_template("report.html", report=report, report_id=report_id)


@app.get("/reports/<report_id>/download")
def download_report(report_id: str):
    report = _reports.get(report_id)
    if report is None:
        abort(404)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=REPORT_EXPORT_FIELDS)
    writer.writeheader()
    writer.writerows(_report_export_rows(report))
    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=shelfsense-report.csv"
    return response


def _calibration_summary() -> str:
    try:
        calibration = json.loads((DATA_DIR / "clip_calibration.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "Not calibrated (matches stay advisory)"
    if calibration.get("model_id") != MATCHER_MODEL_ID:
        return "Calibration file is for a different matcher"
    metrics = calibration.get("metrics", {})
    return (f"threshold {calibration.get('threshold')} / margin {calibration.get('minimum_margin')} · "
            f"validation precision {metrics.get('precision')}, coverage {metrics.get('known_coverage')}")


def _safe_preload() -> None:
    try:
        get_detector(MODEL_PATH).load()
        logger.info("Detector loaded: %s", detector_status().get("model_file"))
    except DetectorUnavailable as error:
        logger.warning("Detector preload failed: %s", error)


def _report_summary(report_id: str, report: dict) -> dict:
    return {
        "report_id": report_id,
        "created_at": report.get("created_at"),
        "store_name": report.get("store_name"),
        "image_name": report.get("image_name"),
        "detection_count": report.get("detection_count"),
        "inference_ms": report.get("inference_ms"),
        "compliance_rate": report.get("compliance_rate"),
        "report_url": url_for("report_detail", report_id=report_id),
        "annotated_image_url": url_for("report_annotated_image", report_id=report_id) if report.get("annotated_image") else None,
    }


@app.get("/api/health")
def api_health():
    model = detector_status()
    if request.args.get("load") == "1" and model.get("file_exists"):
        try:
            get_detector(MODEL_PATH).load()
        except DetectorUnavailable:
            pass
        model = detector_status()
    checks = {"model_file": bool(model.get("file_exists")), "model_load_error": model.get("load_error")}
    try:
        catalogue, planogram = _read_catalogue(), _read_planogram()
        checks.update(catalogue_products=len(catalogue), planogram_slots=len(planogram),
                      planogram_errors=validate_planogram(planogram, catalogue))
    except (OSError, csv.Error) as error:
        checks["data_error"] = str(error)
    results_target = AUDIT_RESULTS_DIR if AUDIT_RESULTS_DIR.exists() else AUDIT_RESULTS_DIR.parent
    checks["results_dir_writable"] = os.access(results_target, os.W_OK)
    healthy = (checks["model_file"] and not checks["model_load_error"]
               and not checks.get("data_error") and not checks.get("planogram_errors"))
    return jsonify(status="ok" if healthy else "degraded", model=model, checks=checks, saved_reports=len(_reports))


@app.get("/api/reports")
def api_reports():
    return jsonify(reports=[_report_summary(report_id, report) for report_id, report in reversed(list(_reports.items()))])


@app.get("/api/reports/<report_id>")
def api_report(report_id: str):
    report = _reports.get(report_id)
    if report is None:
        return jsonify(error="Report not found."), 404
    return jsonify({key: value for key, value in report.items() if key != "image_path"})


@app.post("/api/analyze")
def api_analyze():
    try:
        report_id, report = _run_audit(request.files, request.form)
    except AuditError as error:
        return jsonify(error=error.message), error.status
    return jsonify({**_report_summary(report_id, report), "persisted": report.get("persisted"),
                    "limitations": report.get("limitations", [])}), 201


@app.errorhandler(413)
def request_too_large(_error):
    if request.path.startswith("/api/"):
        return jsonify(error="That image is too large. The upload limit is 12 MB."), 413
    flash("That image is too large. The upload limit is 12 MB.", "error")
    return redirect(url_for("dashboard")), 303


@app.errorhandler(HTTPException)
def http_error(error: HTTPException):
    if request.path.startswith("/api/"):
        return jsonify(error=error.description), error.code
    return error


@app.errorhandler(Exception)
def unhandled_error(error: Exception):
    if isinstance(error, HTTPException):
        return http_error(error)
    logger.exception("Unhandled error on %s", request.path)
    if request.path.startswith("/api/"):
        return jsonify(error="Internal server error."), 500
    return "Internal server error. Check the server log for details.", 500


if __name__ == "__main__":
    # Load the detector in the background so the first analysis does not pay the model start-up cost.
    if MODEL_PATH and Path(MODEL_PATH).is_file():
        threading.Thread(target=_safe_preload, daemon=True).start()
    app.run(
        host=os.getenv("SHELFSENSE_HOST", "127.0.0.1"),
        port=int(os.getenv("SHELFSENSE_PORT", "5000")),
        debug=os.getenv("SHELFSENSE_DEBUG") == "1",
    )
