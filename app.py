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

from backend.matching.calibration import file_fingerprint, load_runtime_calibration
from backend.chatbot import ChatProviderError, audit_context as build_chat_audit_context, chat_configured, generate_reply as generate_chat_reply
from backend.matching.diagnostics import build_pipeline_diagnostics
from backend.matching.pipeline import ProductMatchingPipeline
from backend.matching.detections import suppress_duplicate_detections
from backend.matching.feedback import NOT_PRODUCT_SKU, feedback_accuracy, load_feedback, record_feedback
from backend.matching.shelf_mapper import CoordinateShelfMapper
from backend.matching.shelf_reconstruction import compare_expected_to_detected, reconstruct_detected_planogram
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
    PRICE_TAG_VOID_MODEL_PATH,
    PRICE_TAG_VOID_CONFIDENCE,
    ROOT_DIR,
    SAMPLE_IMAGE_PATH,
    SECRET_KEY,
    SKU_MATCH_THRESHOLD,
    TEST_SET_DIR,
    UPLOAD_DIR,
)
from config.compliance_scoring import SCORING_VERSION
from src.detector import (
    DetectorUnavailable, annotate_image, detect_price_tags_and_voids, detect_products, detector_status, get_detector,
)
from src.ocr import OcrUnavailable, read_price_crop, read_price_tags
from src.planogram import evaluate_planogram, validate_planogram
from src.price_checker import check_prices
from src.report import build_report
from src.scoring import calculate_compliance_score

logging.basicConfig(level=os.getenv("SHELFSENSE_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("shelfsense")

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_reports: dict[str, dict] = {}
_detect_results: dict[str, dict] = {}
_matching_pipelines: dict[tuple[str, str, str], ProductMatchingPipeline] = {}
_detector_fingerprints: dict[tuple[str, int, int], dict | None] = {}
REPORT_SCHEMA_VERSION = 3
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
    "compliance_rate", "score_status", "score_coverage", "compliant_checks", "violation_checks",
    "unresolved_checks", "applicable_check_count", "score_breakdown_json",
    "detection_count", "compliant_count", "product", "sku",
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
    score = report.get("compliance_score", {})
    metadata.update({
        "score_status": score.get("status", "unavailable"),
        "score_coverage": score.get("coverage", 0),
        "compliant_checks": score.get("compliant_checks", 0),
        "violation_checks": score.get("violation_checks", 0),
        "unresolved_checks": score.get("unresolved_checks", 0),
        "applicable_check_count": score.get("applicable_check_count", 0),
        "score_breakdown_json": json.dumps(score.get("breakdown", {}), ensure_ascii=False),
    })
    rows = [{**metadata, "record_type": "audit_summary", "status": score.get("status", "unavailable")}]
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


def _detector_fingerprint() -> dict | None:
    """Fingerprint of the active product-detector checkpoint, hashed once per file version."""
    path = Path(MODEL_PATH)
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _detector_fingerprints:
        _detector_fingerprints[key] = file_fingerprint(path)
    return _detector_fingerprints[key]


def _get_matching_pipeline() -> ProductMatchingPipeline | None:
    """Return the SKU-matching pipeline for the configured matcher, or None when disabled."""
    if MATCHER_MODEL_ID.casefold() == "none":
        return None
    detector = _detector_fingerprint()
    key = (str(DATA_DIR), MATCHER_MODEL_ID, (detector or {}).get("sha256", ""))
    if key not in _matching_pipelines:
        _matching_pipelines[key] = ProductMatchingPipeline(
            data_dir=DATA_DIR,
            model_id=MATCHER_MODEL_ID,
            top_k=int(os.getenv("SHELFSENSE_MATCH_TOP_K", "5")),
            detector_fingerprint=detector,
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
        "product_group_max_gap_ratio": 0.25,
        "product_group_min_vertical_overlap": 0.6,
        "price_verification": True,
        "shelf_gap_detection": True,
        "low_stock_alerts": True,
        "low_stock_threshold": 2,
        # Product-match similarity threshold chosen by the operator; None margin = calibrated margin.
        "match_threshold": DEFAULT_MATCH_THRESHOLD,
        "match_margin": None,
    }
    if SETTINGS_PATH.exists():
        try:
            defaults.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


DEFAULT_MATCH_THRESHOLD = SKU_MATCH_THRESHOLD
FALLBACK_MATCH_MARGIN = 0.05
MATCH_THRESHOLD_RANGE = (0.30, 0.95)
MATCH_MARGIN_RANGE = (0.0, 0.20)


def _calibrated_match_values() -> dict:
    """The validated calibration for the active matcher: threshold, margin, curve (None when unusable)."""
    calibration = load_runtime_calibration(DATA_DIR / "clip_calibration.json", MATCHER_MODEL_ID)
    payload = calibration["payload"]
    return {
        "threshold": calibration["threshold"], "margin": calibration["minimum_margin"], "state": calibration["state"],
        "curve": payload.get("curve", []) if calibration["threshold"] is not None else [],
        "precision": (payload.get("metrics") or {}).get("precision"),
    }


def _match_settings(settings: dict, form=None) -> tuple[float, float]:
    """Threshold and margin for one analysis: the form's values when given, else the saved settings."""
    form = form or {}
    try:
        threshold = float(form.get("match_threshold") or settings.get("match_threshold") or DEFAULT_MATCH_THRESHOLD)
        margin_value = form.get("match_margin") or settings.get("match_margin")
        margin = float(margin_value) if margin_value not in (None, "") else (
            _calibrated_match_values()["margin"] or FALLBACK_MATCH_MARGIN)
    except (TypeError, ValueError) as error:
        raise AuditError("The product match threshold must be a number.") from error
    if not MATCH_THRESHOLD_RANGE[0] <= threshold <= MATCH_THRESHOLD_RANGE[1]:
        raise AuditError(f"Product match threshold must be between {MATCH_THRESHOLD_RANGE[0]:.2f} and {MATCH_THRESHOLD_RANGE[1]:.2f}.")
    if not MATCH_MARGIN_RANGE[0] <= margin <= MATCH_MARGIN_RANGE[1]:
        raise AuditError(f"Product match margin must be between {MATCH_MARGIN_RANGE[0]:.2f} and {MATCH_MARGIN_RANGE[1]:.2f}.")
    return round(threshold, 3), round(margin, 3)


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
    for optional_field in ("barcode", "image_local_path"):
        if optional_field not in fields and any(product.get(optional_field) or product.get("_source", {}).get(optional_field) for product in products):
            fields.append(optional_field)
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
            for optional_field in ("barcode", "image_local_path"):
                if optional_field in fields and product.get(optional_field):
                    row[optional_field] = product[optional_field]
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
    temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as csv_file:
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
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _persist_planogram(rows: list[dict]) -> bool:
    try:
        _save_planogram(rows)
    except OSError:
        logger.exception("Could not persist planogram")
        flash("Planogram could not be saved. Check the data folder permissions and try again.", "error")
        return False
    return True


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


def _detected_planogram_urls(report_id: str, detected_planogram: dict) -> dict:
    """Attach stable report-image and crop references to structured geometry findings."""
    detected_planogram = dict(detected_planogram)
    detected_planogram["source_image_url"] = url_for("report_image", report_id=report_id)
    for row in detected_planogram.get("rows", []):
        for item in row.get("slots", []):
            if item.get("kind") == "product" and item.get("detection_index") is not None:
                item["crop_url"] = url_for(
                    "report_detection_crop", report_id=report_id,
                    detection_index=int(item["detection_index"]),
                )
    return detected_planogram


def _comparison_urls(report_id: str, comparison: dict) -> dict:
    comparison = dict(comparison)
    comparison["source_image_url"] = url_for("report_image", report_id=report_id)
    for finding in comparison.get("findings", []):
        if finding.get("detection_index") is not None:
            finding["crop_url"] = url_for(
                "report_detection_crop", report_id=report_id,
                detection_index=int(finding["detection_index"]),
            )
    return comparison


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
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError, ModuleNotFoundError) as error:
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
    reanalyze = form.get("reanalyze_image")
    if reanalyze:
        path = _uploaded_path(reanalyze)
        if path is None:
            raise AuditError("The original image of that report is no longer stored, so it cannot be re-analyzed.")
        return path, form.get("reanalyze_name") or path.name
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
    matches_by_detection = {}
    for index, detection in enumerate(detections):
        if index < len(matches):
            match = matches[index]
            source_index = int(match.get("detection_index", detection.get("_source_detection_index", index)))
            matches_by_detection[source_index] = match
    for index, detection in enumerate(detections):
        source_index = int(detection.get("_source_detection_index", index))
        match = matches_by_detection.get(source_index, {})
        mapping = slot_mapper.map_detection(detection, planogram)
        identity = mapping.get("slot_identity")
        identified.append({
            **detection,
            "detection_index": source_index,
            "sku": match.get("accepted_sku") or "",
            "match_status": match.get("status", "Unknown / Needs Review"),
            "match_reason_code": match.get("reason_code", "no_match_result"),
            "slot_identity": list(identity) if identity else None,
            "slot_mapping_status": mapping.get("status"),
            "slot_mapping_detail": mapping.get("detail", ""),
            "slot_mapping_confidence": mapping.get("confidence"),
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
            slot.update(status="Needs Review", reason_code="unidentified_objects_in_slot")
        elif unresolved.get(key):
            slot.update(status="Needs Review", reason_code="unidentified_objects_in_slot",
                        detail=f"{unresolved[key]} unidentified object(s) in an intentionally empty slot.")


def _attach_confirmed_sku_matches(slots: list[dict], identified: list[dict], layout_trusted: bool) -> None:
    """Attach exact SKU-in-slot evidence for scoring while leaving facing review intact."""
    if not layout_trusted:
        return
    confirmed = {
        tuple(str(value) for value in item["slot_identity"])
        for item in identified
        if item.get("sku") and item.get("slot_identity")
        and item.get("match_status") not in {"Unknown / Needs Review", "Needs Review"}
    }
    for slot in slots:
        identity = (
            str(slot.get("store_id", "")), str(slot.get("planogram_id", "")),
            str(slot.get("shelf", "")), str(slot.get("row", "")),
            str(slot.get("slot", slot.get("position", ""))),
        )
        if identity in confirmed and str(slot.get("sku", "")):
            slot["sku_match_confirmed"] = True


MIN_VOID_COVERAGE_FOR_ABSENCE = 0.5


def _void_slot_coverage(slot_voids: list[dict], slot_number: str, slot_count: int) -> float:
    """Share of an equal-grid slot's width covered by the union of its void boxes (0..1)."""
    try:
        index = int(slot_number) - 1
    except ValueError:
        return 0.0
    if not slot_voids or slot_count <= 0 or not 0 <= index < slot_count:
        return 0.0
    width = float(slot_voids[0].get("image_width") or 0)
    if width <= 0:
        return 0.0
    left, right = index * width / slot_count, (index + 1) * width / slot_count
    spans = sorted((max(left, float(item["box"][0])), min(right, float(item["box"][2]))) for item in slot_voids)
    covered, cursor = 0.0, left
    for start, end in spans:
        start = max(start, cursor)
        if end > start:
            covered += end - start
            cursor = end
    return covered / (right - left)


def _downgrade_unsupported_slot_statuses(slots: list[dict], voids: list[dict], layout_trusted: bool) -> None:
    """Require positive slot evidence before reporting a planogram result as confirmed.

    A void box is positive evidence of an empty area, so it can confirm an absence or a facing
    shortfall. It never overrides identified products: a slot whose expected facings were all
    identified is contradicted, not emptied, by a void box, and stays for review. Absence
    without a void box is not confirmed, because the detector may simply have missed products.
    """
    if not layout_trusted:
        return
    slots_per_row: dict[tuple[str, str], int] = {}
    for slot in slots:
        key = (str(slot.get("shelf", "")), str(slot.get("row", "")))
        slots_per_row[key] = slots_per_row.get(key, 0) + 1
    for slot in slots:
        identity = (str(slot.get("shelf", "")), str(slot.get("row", "")), str(slot.get("slot", slot.get("position", ""))))
        slot_voids = [
            item for item in voids
            if item.get("slot_identity") and len(item["slot_identity"]) >= 5
            and tuple(str(part) for part in item["slot_identity"][-3:]) == identity
        ]
        exact_void = bool(slot_voids)
        status = slot.get("status")
        if not slot.get("sku"):
            continue
        if status == "Missing Expected Product":
            observed = slot.get("observed") or 0
            coverage = _void_slot_coverage(slot_voids, identity[2], slots_per_row.get(identity[:2], 0))
            if exact_void and (observed or coverage >= MIN_VOID_COVERAGE_FOR_ABSENCE):
                slot.update(
                    reason_code="void_confirms_absence" if not observed else "void_confirms_facing_shortfall",
                    detail=(f"Void detections cover {coverage:.0%} of this assigned slot and no product was mapped here."
                            if not observed else
                            f"A void detection maps to this slot and only {observed} of {slot.get('expected')} "
                            "expected facings were identified."),
                )
            elif exact_void:
                slot.update(
                    status="Needs Review", reason_code="absence_not_confirmed",
                    detail=(f"No product was mapped here, but void detections cover only {coverage:.0%} of the slot; "
                            "a narrow void can sit beside a product the detector missed."),
                )
            else:
                slot.update(
                    status="Needs Review", reason_code="absence_not_confirmed",
                    detail=("Fewer products were mapped here than expected, but no void detection confirms the gap; "
                            "the detector may have missed products."),
                )
        elif exact_void and status == "Compliant":
            slot.update(
                status="Needs Review", reason_code="void_conflicts_with_products",
                detail=("All expected facings were identified here, but a void detection also maps to this slot; "
                        "the evidence is contradictory."),
            )
        elif exact_void:
            slot["detail"] = f"{slot.get('detail', '')} A void detection also maps to this slot.".strip()


def _read_shelf_prices(stored_path: Path, auxiliary_audit: dict, product_detections: list[dict],
                       min_confidence: float, report_id: str) -> dict:
    """OCR price-tag-model crops, then add whole-image OCR readings the tag model did not cover.

    The tag model misses tags (it finds none on many images), so whole-image price text that is
    not inside a detected tag and not printed on a detected product is kept as extra evidence
    instead of leaving those prices silently unverified.
    """
    result: dict = {"engine": "rapidocr-onnxruntime", "tags": []}
    model_tags = auxiliary_audit.get("price_tags", [])
    for detection in model_tags:
        tag = {**detection, "text": "", "raw_ocr_text": "", "price": None, "parsed_price": None,
               "ocr_confidence": None, "ocr_status": "unreadable", "sku": "", "slot": None,
               "association": "Needs Review", "tag_source": "price_tag_model"}
        try:
            ocr = read_price_crop(stored_path, detection["box"], min_confidence, detection.get("image_width"), detection.get("image_height"))
            tag.update(ocr)
            tag["text"] = ocr.get("text", "")
            tag["parsed_price"] = ocr["price"]
        except OcrUnavailable as error:
            result["error"] = str(error)
            tag["ocr_status"] = "unavailable"
            tag["ocr_diagnostic"] = str(error)
        except Exception as error:
            logger.exception("OCR failed for a detected price tag in report %s", report_id)
            result.setdefault("errors", []).append(str(error))
            tag["ocr_status"] = "error"
            tag["ocr_diagnostic"] = str(error)
        result["tags"].append(tag)

    def inside(reading: dict, boxes: list) -> bool:
        center_x = (reading["box"][0] + reading["box"][2]) / 2
        center_y = (reading["box"][1] + reading["box"][3]) / 2
        return any(box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3] for box in boxes)

    tag_boxes = [tag["box"] for tag in model_tags if len(tag.get("box") or []) == 4]
    product_boxes = [detection["box"] for detection in product_detections if len(detection.get("box") or []) == 4]
    supplemented = 0
    if not result.get("error"):
        try:
            for reading in read_price_tags(stored_path, min_confidence):
                if inside(reading, tag_boxes):
                    continue  # already read from the tag model's crop
                if inside(reading, product_boxes):
                    # Price-like text printed on a detected product's packaging is not a shelf tag.
                    result["readings_on_products"] = result.get("readings_on_products", 0) + 1
                    continue
                supplemented += 1
                result["tags"].append({
                    "label": "price_tag", "box": reading["box"], "confidence": reading["confidence"],
                    "text": reading["text"], "raw_ocr_text": reading.get("raw_ocr_text", reading["text"]), "price": reading["price"],
                    "parsed_price": reading["price"], "ocr_confidence": reading["confidence"],
                    "ocr_status": "read", "ocr_rule": reading["rule"], "tag_source": "full_image_ocr",
                    "sku": "", "slot": None, "association": "Needs Review",
                })
        except OcrUnavailable as error:
            result["error"] = str(error)
        except Exception as error:
            logger.exception("Whole-image price OCR failed in report %s", report_id)
            result.setdefault("errors", []).append(f"Whole-image price OCR failed: {error}")
    result["tag_source"] = (
        "price_tag_model+full_image_ocr" if model_tags and supplemented
        else "price_tag_model" if model_tags else "full_image_ocr"
    )
    if not model_tags:
        result["tag_source_reason"] = (
            f"Price-tag detector unavailable: {auxiliary_audit['error']}" if auxiliary_audit.get("error")
            else "The price-tag detector found no tags in this image."
        )
    if result.get("errors") and not result.get("error"):
        result["error"] = "OCR failed for one or more price readings."
    return result


def _assign_price_tags(tags: list[dict], planogram: list[dict], identified: list[dict], slot_mapper,
                       image_width: int, image_height: int) -> list[dict]:
    """Link each OCR price tag to one visible identified product using slot or geometry evidence."""
    slots_by_identity = {
        (row["store_id"], row["planogram_id"], row["physical_shelf"], str(row["row"]), str(row["slot"])): row
        for row in planogram
    }
    used_detections: set[int] = set()
    for tag in tags:
        tag.update(sku="", slot=None, association="unassigned")
        if slot_mapper.trusted:
            mapping = slot_mapper.map_detection(
                {"box": tag["box"], "image_width": image_width, "image_height": image_height}, planogram)
            row = slots_by_identity.get(tuple(mapping["slot_identity"])) if mapping.get("slot_identity") else None
            if row and row.get("sku"):
                identity = tuple(str(value) for value in mapping["slot_identity"])
                candidates = [
                    detection for detection in identified
                    if detection.get("sku") and detection.get("slot_identity")
                    and tuple(str(value) for value in detection["slot_identity"]) == identity
                    and int(detection.get("_source_detection_index", detection.get("detection_index", -1))) not in used_detections
                ]
                tag_center = (tag["box"][0] + tag["box"][2]) / 2
                candidates.sort(key=lambda detection: abs(
                    (detection["box"][0] + detection["box"][2]) / 2 - tag_center
                ))
                if candidates:
                    best = candidates[0]
                    if len(candidates) > 1 and candidates[1]["sku"] != best["sku"]:
                        # Equidistant facings of the same SKU do not make the tag's product ambiguous.
                        first_distance = abs((best["box"][0] + best["box"][2]) / 2 - tag_center)
                        second_distance = abs((candidates[1]["box"][0] + candidates[1]["box"][2]) / 2 - tag_center)
                        if second_distance - first_distance <= max(4, 0.02 * image_width):
                            tag.update(slot=f"R{row['row']}S{row['slot']}",
                                       association="Needs Review (ambiguous product in mapped slot)")
                            continue
                    if best["sku"] != row["sku"]:
                        # Shelf tags belong to the planned slot; under a misplaced product it is unclear
                        # whether the tag prices the planned SKU or the visible one.
                        tag.update(slot=f"R{row['row']}S{row['slot']}",
                                   association="Needs Review (identified product differs from the slot's planogram SKU)")
                        continue
                    detection_index = int(best.get("_source_detection_index", best.get("detection_index", -1)))
                    used_detections.add(detection_index)
                    tag.update(sku=best["sku"], slot=f"R{row['row']}S{row['slot']}",
                               associated_detection_index=detection_index,
                               association="calibrated slot and visible product")
                else:
                    tag.update(slot=f"R{row['row']}S{row['slot']}",
                               association="Needs Review (no identified product in mapped slot)")
            continue
        center_x, tag_top = (tag["box"][0] + tag["box"][2]) / 2, tag["box"][1]
        above = [
            detection for detection in identified
            if detection["sku"] and detection["box"][0] <= center_x <= detection["box"][2]
            and -0.02 * image_height <= tag_top - detection["box"][3] <= 0.25 * image_height
        ]
        if above:
            above.sort(key=lambda detection: tag_top - detection["box"][3])
            nearest = above[0]
            if len(above) > 1 and above[1]["sku"] != nearest["sku"] and abs(
                (tag_top - above[1]["box"][3]) - (tag_top - nearest["box"][3])
            ) <= 0.03 * image_height:
                tag.update(association="Needs Review (ambiguous product)")
            else:
                source_index = int(nearest.get("_source_detection_index", nearest.get("detection_index", -1)))
                if source_index in used_detections:
                    tag.update(association="Needs Review (product already linked to another tag)")
                else:
                    used_detections.add(source_index)
                    tag.update(sku=nearest["sku"], associated_detection_index=source_index,
                               association="nearest identified product above (heuristic)")
    # Two different prices read in one mapped slot contradict each other; neither is linked.
    prices_by_slot: dict[str, set[float]] = {}
    for tag in tags:
        if tag.get("slot") and tag.get("price") is not None:
            prices_by_slot.setdefault(tag["slot"], set()).add(round(float(tag["price"]), 2))
    for tag in tags:
        if tag.get("slot") and len(prices_by_slot.get(tag["slot"], ())) > 1:
            tag.pop("associated_detection_index", None)
            tag.update(sku="", association="Needs Review (conflicting price readings in slot)")
    return tags


def _price_results(planogram_products: list[dict], tags: list[dict]) -> list[dict]:
    observed: dict[str, list[float]] = {}
    for tag in tags:
        if tag.get("sku") and tag.get("price") is not None:
            observed.setdefault(tag["sku"], []).append(tag["price"])
    results = check_prices(planogram_products, {sku: max(set(values), key=values.count) for sku, values in observed.items()})
    for row in results:
        values = observed.get(row["sku"], [])
        row["tag_count"] = len(values)
        if len(set(values)) > 1:
            row["detail"] = f"Conflicting tag prices read: {sorted(set(values))}"
        methods = {tag["association"] for tag in tags if tag.get("sku") == row["sku"] and tag.get("price") is not None}
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


def _detect_shelf_gaps(detections: list[dict], image_width: int, image_height: int) -> list[dict]:
    """Estimate unusually wide horizontal openings between detected products, by image row."""
    gaps = []
    rows = CoordinateShelfMapper.cluster_detections_by_row(detections)
    for row_number, row in enumerate(rows, start=1):
        ordered = sorted(
            (item for item in row if item.get("box") and len(item["box"]) == 4),
            key=lambda item: (item["box"][0] + item["box"][2]) / 2,
        )
        widths = [max(0, item["box"][2] - item["box"][0]) for item in ordered]
        if not widths:
            continue
        typical_width = sorted(widths)[len(widths) // 2]
        minimum_gap = max(12, typical_width * 0.6)
        for left, right in zip(ordered, ordered[1:]):
            x1, x2 = left["box"][2], right["box"][0]
            if x2 - x1 < minimum_gap:
                continue
            y1 = max(0, min(left["box"][1], right["box"][1]))
            y2 = min(image_height, max(left["box"][3], right["box"][3]))
            gaps.append({"row": row_number, "box": [max(0, x1), y1, min(image_width, x2), y2],
                         "width": round(x2 - x1), "detail": "Unusually wide opening between detected product facings."})
    return gaps


def _low_stock_alerts(compliance: list[dict], threshold: int, enabled: bool, layout_trusted: bool) -> list[dict]:
    """Flag mapped slots that are below planogram facings and at or below the alert threshold."""
    if not enabled or not layout_trusted:
        return []
    return [
        {"product": row.get("product") or f"SKU {row.get('sku')}", "sku": row.get("sku", ""),
         "shelf": row.get("shelf", ""), "row": row.get("row", ""), "position": row.get("position", ""),
         "observed": row["observed"], "expected": row.get("expected", 0), "threshold": threshold,
         "detail": f"{row['observed']} visible facing(s); alert threshold is {threshold}."}
        for row in compliance
        if row.get("sku") and row.get("observed") is not None and row["observed"] <= threshold
        and row["observed"] < row.get("expected", 0)
        and row.get("status") != "Needs Review"
    ]


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
    elif calibration.get("acceptance_threshold") is None:
        issues = " ".join(issue["message"] for issue in calibration.get("issues", []))
        limitations.append(
            "SKU candidates are advisory: no valid validation-calibrated acceptance threshold is configured."
            + (f" {issues}" if issues else "")
        )
    else:
        accepted = sum(bool(match.get("accepted_sku")) for match in matching_audit.get("matches", []))
        source = "operator-set" if calibration.get("threshold_source") == "operator" else "calibrated"
        limitations.append(
            f"SKU matching ({calibration.get('model_id')}) accepted {accepted} of {len(matching_audit.get('matches', []))} "
            f"crops at the {source} similarity threshold {calibration.get('acceptance_threshold')}; "
            "the highest-scoring SKU is selected even when another candidate is close. "
            "Calibration used synthetic shelves, "
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
        tags = price_audit.get("tags", [])
        assigned = sum(bool(tag.get("sku")) for tag in tags)
        read = sum(tag.get("ocr_status") == "read" for tag in tags)
        limitations.append(
            f"OCR ({price_audit.get('engine')}) read a price from {read} of {len(tags)} price tag(s); {assigned} were linked "
            "to a SKU. Prices without a linked tag remain Unverified."
        )
    return limitations


def _run_audit(files, form) -> tuple[str, dict]:
    """Image upload -> detection -> SKU matching -> slot mapping -> planogram comparison -> saved report."""
    settings = _load_settings()
    source, display_name = _incoming_image(files, form)
    confidence = _parse_confidence(form)
    match_threshold, match_margin = _match_settings(settings, form)

    all_catalogue = _read_catalogue()
    catalogue = [product for product in all_catalogue if product.get("active", "true").casefold() == "true"]
    complete_planogram = _read_planogram()
    requested_planogram_id = str(form.get("planogram_id", "")).strip()
    saved_planogram_id = complete_planogram[0].get("planogram_id", "DEFAULT") if complete_planogram else "DEFAULT"
    if requested_planogram_id and requested_planogram_id != saved_planogram_id:
        raise AuditError("The selected planogram is no longer available. Reload the planogram and try again.", endpoint="planogram_page", status=409)
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

    for detection_index, detection in enumerate(detections):
        detection["detection_index"] = detection_index
    unique_detections, duplicate_map = suppress_duplicate_detections(detections)
    unique_detections = [
        {**detection, "_source_detection_index": detection["detection_index"]}
        for detection in unique_detections
    ]

    auxiliary_audit: dict = {"requested": True, "price_tags": [], "voids": []}
    try:
        auxiliary_detections = detect_price_tags_and_voids(stored_path, confidence=PRICE_TAG_VOID_CONFIDENCE)
        auxiliary_audit.update(auxiliary_detections)
    except DetectorUnavailable as error:
        auxiliary_audit["error"] = str(error)
    except Exception as error:
        logger.exception("Price-tag/void inference failed")
        auxiliary_audit["error"] = f"Auxiliary detector failed: {error}"

    report_id = uuid.uuid4().hex
    layout_requested = bool(form.get("calibrated_layout") or form.get("test_image"))
    expected_row_groups = {
        (str(slot.get("physical_shelf", slot.get("shelf", ""))), str(slot.get("row", "")))
        for slot in planogram
    }
    detected_row_count = len(CoordinateShelfMapper.cluster_detections_by_row(unique_detections, gap_ratio=0.65))
    one_shelf_group = len({shelf for shelf, _row in expected_row_groups}) == 1
    row_count_matches = detected_row_count == len({row for _shelf, row in expected_row_groups})
    layout_trusted = bool(layout_requested and one_shelf_group and row_count_matches)
    if not layout_requested:
        layout_validation = {"requested": False, "compatible": None, "detected_rows": detected_row_count,
                             "expected_rows": len({row for _shelf, row in expected_row_groups}),
                             "detail": "The operator did not enable fixed-camera slot confirmation."}
    elif layout_trusted:
        layout_validation = {"requested": True, "compatible": True, "detected_rows": detected_row_count,
                             "expected_rows": len({row for _shelf, row in expected_row_groups}),
                             "detail": "Detected row count agrees with the saved planogram row count."}
    else:
        detail = (
            "The saved planogram contains multiple physical shelf groups; the equal-grid mapper supports only one."
            if not one_shelf_group else
            f"Detected {detected_row_count} shelf rows, but the planogram defines "
            f"{len({row for _shelf, row in expected_row_groups})}; slot mapping remains Needs Review."
        )
        layout_validation = {"requested": True, "compatible": False, "detected_rows": detected_row_count,
                             "expected_rows": len({row for _shelf, row in expected_row_groups}), "detail": detail}
    slot_mapper = CoordinateShelfMapper(trusted=layout_trusted)
    for void in auxiliary_audit.get("voids", []):
        void["slot_identity"] = None
        void["slot_mapping_status"] = "needs_review"
        if layout_trusted:
            mapping = slot_mapper.map_detection(void, planogram)
            if mapping.get("slot_identity"):
                void["slot_identity"] = list(mapping["slot_identity"])
                void["slot_mapping_status"] = "calibrated location; review required"
    try:
        pipeline = _get_matching_pipeline()
        if pipeline is None:
            matching_audit = {"matches": [], "findings": [], "matching_error": "SKU matching is disabled (SHELFSENSE_MATCHER=none)."}
        else:
            matching_audit = pipeline.run(stored_path, unique_detections, catalogue, planogram, slot_mapper=slot_mapper,
                                          acceptance_threshold=match_threshold, minimum_margin=match_margin)
    except Exception as error:
        logger.exception("SKU matching pipeline failed for report %s", report_id)
        matching_audit = {"matches": [], "findings": [], "matching_error": f"Matching pipeline error: {error}"}
    identified = _identify_detections(unique_detections, matching_audit.get("matches", []), planogram, slot_mapper)
    names_by_sku = {product["sku"]: product.get("name") or f"SKU {product['sku']}" for product in all_catalogue}
    identified_by_source = {item["_source_detection_index"]: item for item in identified}
    kept_source_indices = set(identified_by_source)
    for detection in detections:
        source_index = detection["detection_index"]
        enriched = identified_by_source.get(source_index)
        if enriched:
            detection["matched_sku"] = enriched["sku"]
            detection["product_name"] = names_by_sku.get(enriched["sku"], "") if enriched["sku"] else ""
            detection["match_status"] = enriched["match_status"]
            detection["match_reason_code"] = enriched["match_reason_code"]
            detection["slot_identity"] = enriched["slot_identity"]
            detection["slot_mapping_status"] = enriched["slot_mapping_status"]
            detection["slot_mapping_detail"] = enriched["slot_mapping_detail"]
        elif source_index in duplicate_map:
            duplicate_of = duplicate_map[source_index]
            detection["duplicate_detection"] = True
            detection["duplicate_of"] = duplicate_of
            detection["matched_sku"] = ""
            detection["match_status"] = "Duplicate overlapping box excluded from counts."
            detection["slot_identity"] = None
    for source_index, duplicate_of in duplicate_map.items():
        detections[source_index]["product_name"] = detections[duplicate_of].get("product_name", "")
    matching_audit["raw_detection_count"] = len(detections)
    matching_audit["detection_count"] = len(unique_detections)
    matching_audit["duplicate_detection_map"] = {str(key): value for key, value in duplicate_map.items()}

    price_audit: dict = {"requested": bool(form.get("price_check")) and bool(settings.get("price_verification", True)), "tags": []}
    if price_audit["requested"]:
        price_audit.update(_read_shelf_prices(
            stored_path, auxiliary_audit, unique_detections, float(settings.get("price_confidence", 0.5)), report_id,
        ))
        # Assign all readable tags together so one product is linked to at most one tag and
        # conflicting readings in one slot are detected.
        _assign_price_tags([tag for tag in price_audit["tags"] if tag["ocr_status"] == "read"], planogram, identified,
                           slot_mapper, image_width, image_height)
    else:
        price_audit["tags"] = [
            {**detection, "text": "", "raw_ocr_text": "", "price": None, "parsed_price": None,
             "ocr_confidence": None, "ocr_status": "not_requested", "sku": "", "slot": None,
             "association": "OCR not requested"}
            for detection in auxiliary_audit.get("price_tags", [])
        ]

    annotated_name = None
    try:
        annotation_detections = detections + auxiliary_audit.get("price_tags", []) + auxiliary_audit.get("voids", [])
        annotated_name = annotate_image(stored_path, annotation_detections, UPLOAD_DIR / f"{report_id}_annotated.jpg").name
    except Exception:
        logger.exception("Could not draw annotated image for report %s", report_id)

    try:
        planogram_audit = evaluate_planogram(
            planogram, all_catalogue, identified,
            slot_mapper=(lambda detection, _rows: tuple(detection["slot_identity"]) if detection["slot_identity"] else None)
            if layout_trusted else None,
        )
        if layout_trusted:
            _downgrade_slots_with_unidentified(planogram_audit["slots"], identified)
            _downgrade_unsupported_slot_statuses(
                planogram_audit["slots"], auxiliary_audit.get("voids", []), layout_trusted,
            )
        compliance = planogram_audit["slots"]
        _attach_confirmed_sku_matches(compliance, identified, layout_trusted)
        planogram_skus = {slot["sku"] for slot in planogram if slot.get("sku")}
        price_results = (
            _price_results([product for product in catalogue if product["sku"] in planogram_skus], price_audit["tags"])
            if price_audit["requested"] else []
        )
        score = calculate_compliance_score(
            compliance, price_results, layout_trusted=layout_trusted,
            price_requested=price_audit["requested"],
        )
        report = build_report(
            compliance=compliance,
            price_results=price_results,
            detections=detections,
            image_name=display_name,
            store_name=form.get("store", settings["default_store"]).strip() or settings["default_store"],
            compliance_score=score,
        )
        # Keep the existing field/API contract as an alias of the versioned score.
        report["compliance_rate"] = score["score"]
        report["raw_detection_count"] = len(detections)
        report["detection_count"] = len(unique_detections)
        report["duplicate_detection_count"] = len(duplicate_map)
        report["duplicate_detection_map"] = {str(key): value for key, value in duplicate_map.items()}
        shelf_gaps = (
            _detect_shelf_gaps(unique_detections, image_width, image_height)
            if settings.get("shelf_gap_detection") and layout_trusted else []
        )
        low_stock_alerts = _low_stock_alerts(
            compliance, int(settings.get("low_stock_threshold", 2)),
            bool(settings.get("low_stock_alerts")), layout_trusted,
        )
        detected_planogram = reconstruct_detected_planogram(
            detections, auxiliary_audit.get("voids", []), price_audit.get("tags", []),
            all_catalogue, image_width, image_height,
            grouping_config={
                "max_gap_ratio": settings["product_group_max_gap_ratio"],
                "min_vertical_overlap": settings["product_group_min_vertical_overlap"],
            },
        )
        detected_planogram = _detected_planogram_urls(report_id, detected_planogram)
        planogram_comparison = compare_expected_to_detected(
            planogram, detected_planogram, planogram_audit, price_results, layout_trusted, all_catalogue,
            matching_audit,
        )
        planogram_comparison = _comparison_urls(report_id, planogram_comparison)
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
        planogram_id=saved_planogram_id,
        confidence_threshold=confidence,
        analysis_options={
            "calibrated_layout": layout_requested, "price_check": bool(form.get("price_check")),
            "match_threshold": match_threshold, "match_margin": match_margin,
        },
        planogram_audit=planogram_audit,
        detected_planogram=detected_planogram,
        planogram_comparison=planogram_comparison,
        matching_audit=matching_audit,
        image_file=stored_path.name,
        image_path=str(stored_path.resolve()),
        annotated_image=annotated_name,
        image_width=image_width,
        image_height=image_height,
        inference_ms=inference_ms,
        model=model_info,
        auxiliary_model=auxiliary_audit.get("model"),
        auxiliary_model_error=auxiliary_audit.get("error"),
        void_detections=auxiliary_audit.get("voids", []),
        price_tag_detections=auxiliary_audit.get("price_tags", []),
        facing_summary=_facing_summary(planogram, unique_detections),
        layout_trusted=layout_trusted,
        layout_validation=layout_validation,
        price_audit=price_audit,
        shelf_gap_detection={
            "enabled": bool(settings.get("shelf_gap_detection")),
            "available": layout_trusted,
            "gaps": shelf_gaps,
        },
        low_stock_alerts=low_stock_alerts,
        low_stock_threshold=int(settings.get("low_stock_threshold", 2)),
        identified_count=sum(bool(detection["sku"]) for detection in identified),
        limitations=_audit_limitations(model_info, matching_audit, layout_trusted, price_audit),
        raw_detection_count=len(detections),
        duplicate_detection_count=len(duplicate_map),
        duplicate_detection_map={str(key): value for key, value in duplicate_map.items()},
    )
    if layout_validation.get("compatible") is False:
        report["limitations"].append(layout_validation["detail"])
    try:
        diagnostics = build_pipeline_diagnostics(
            detections=detections, matching_audit=matching_audit, compliance=compliance,
            price_results=price_results, price_audit=price_audit, layout_trusted=layout_trusted,
            layout_validation=layout_validation, catalogue_size=len(catalogue), planogram=planogram,
        )
        report["pipeline_diagnostics"] = diagnostics
        report["limitations"].extend(
            issue["message"] for issue in diagnostics["scene"]["issues"] if issue["message"] not in report["limitations"]
        )
        logger.info("Audit %s diagnostics: scene=%s detection_reasons=%s slot_reasons=%s", report_id,
                    [issue["code"] for issue in diagnostics["scene"]["issues"]],
                    diagnostics["reason_counts"]["detections"], diagnostics["reason_counts"]["slots"])
    except Exception:
        logger.exception("Could not build pipeline diagnostics for report %s", report_id)
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
    report.setdefault("price_audit", {})
    for tag in report["price_audit"].get("tags", []):
        # Reports saved before tag-level OCR fields existed store whole-image readings as text/price.
        tag.setdefault("parsed_price", tag.get("price"))
        tag.setdefault("raw_ocr_text", tag.get("text", ""))
        tag.setdefault("ocr_status", "read" if tag.get("price") is not None else "unreadable")
        tag.setdefault("ocr_confidence", tag.get("confidence"))
        tag.setdefault("confidence", tag.get("ocr_confidence") or 0.0)
        tag.setdefault("box", [0, 0, 0, 0])
    report.setdefault("void_detections", [])
    report.setdefault("matching_audit", {})
    report.setdefault("model", {})
    report.setdefault("limitations", [])
    report.setdefault("facing_summary", {})
    report.setdefault("inference_ms", None)
    report.setdefault("confidence_threshold", None)
    report.setdefault("image_width", None)
    report.setdefault("image_height", None)
    report.setdefault("annotated_image", None)
    report.setdefault("detection_count", len(report["detections"]))
    report.setdefault("raw_detection_count", report["detection_count"])
    report.setdefault("duplicate_detection_count", 0)
    report.setdefault("layout_validation", {})
    report.setdefault("compliant_count", sum(row.get("status") == "Compliant" for row in report["compliance"]))
    report.setdefault("compliance_rate", None)
    report.setdefault("price_unverified", sum(row.get("status") == "Unverified" for row in report["price_results"]))
    report.setdefault("planogram_audit", {
        "slots": report["compliance"], "detections": [], "validation_errors": [],
        "slot_mapping_available": False,
        "empty_slot_count": sum(row.get("status") == "Empty Slot" for row in report["compliance"]),
    })
    if (not isinstance(report.get("compliance_score"), dict)
            or report["compliance_score"].get("version") != SCORING_VERSION):
        reviewed_matches = [
            {
                "sku": detection.get("matched_sku"),
                "slot_identity": detection.get("slot_identity"),
                "match_status": detection.get("match_status"),
            }
            for detection in report.get("detections", [])
            if detection.get("matched_sku") and detection.get("slot_identity")
        ]
        _attach_confirmed_sku_matches(
            report.get("compliance", []), reviewed_matches, bool(report.get("layout_trusted"))
        )
        report["compliance_score"] = calculate_compliance_score(
            report.get("compliance", []), report.get("price_results", []),
            layout_trusted=bool(report.get("layout_trusted")),
            price_requested=bool(report.get("price_audit", {}).get("requested")),
        )
    report["compliance_rate"] = report["compliance_score"].get("score")
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
    if latest_report:
        latest_report = _normalize_saved_report(latest_report_id, dict(latest_report))
    if latest_report and not latest_report.get("detected_planogram"):
        latest_report = dict(latest_report)
        latest_report["detected_planogram"] = reconstruct_detected_planogram(
            latest_report.get("detections", []), latest_report.get("void_detections", []),
            latest_report.get("price_audit", {}).get("tags", []), catalogue,
            int(latest_report.get("image_width") or 0), int(latest_report.get("image_height") or 0),
            grouping_config={
                "max_gap_ratio": _load_settings()["product_group_max_gap_ratio"],
                "min_vertical_overlap": _load_settings()["product_group_min_vertical_overlap"],
            },
        )
        latest_report["detected_planogram"] = _detected_planogram_urls(latest_report_id, latest_report["detected_planogram"])
        latest_report["planogram_comparison"] = compare_expected_to_detected(
            planogram, latest_report["detected_planogram"], latest_report.get("planogram_audit", {}),
            latest_report.get("price_results", []), bool(latest_report.get("layout_trusted")), catalogue,
            latest_report.get("matching_audit", {}),
        )
        latest_report["planogram_comparison"] = _comparison_urls(latest_report_id, latest_report["planogram_comparison"])
    if latest_report and latest_report.get("planogram_id") in {str(slot.get("planogram_id", "")) for slot in planogram}:
        # Rebuild comparison annotations for older saved reports too. This adds
        # SKU-presence evidence while leaving slot confirmation gated by the
        # report's original trusted-layout decision.
        latest_report = dict(latest_report)
        latest_report["planogram_comparison"] = compare_expected_to_detected(
            planogram, latest_report.get("detected_planogram", {}),
            latest_report.get("planogram_audit", {}), latest_report.get("price_results", []),
            bool(latest_report.get("layout_trusted")), catalogue, latest_report.get("matching_audit", {}),
        )
        latest_report["planogram_comparison"] = _comparison_urls(
            latest_report_id, latest_report["planogram_comparison"]
        )
    products_by_sku = {product["sku"]: product for product in catalogue}
    report_slots = {}
    detected_skus = set()
    if latest_report:
        detected_skus = {
            str(item.get("sku") or "")
            for item in latest_report.get("detected_planogram", {}).get("items", [])
            if item.get("identified") and item.get("sku")
        }
        for finding in latest_report.get("planogram_comparison", {}).get("findings", []):
            if finding.get("slot") is not None:
                identity = (str(finding.get("shelf", "")), str(finding.get("row", "")), str(finding["slot"]))
                report_slots[identity] = {
                    "status": finding["status"] if finding.get("confirmed") else "Needs Review"
                }
        if not report_slots:
            audit_slots = latest_report.get("planogram_audit", {}).get("slots", []) or latest_report.get("compliance", [])
            for audit_slot in audit_slots:
                report_slots[(str(audit_slot.get("shelf", "")), str(audit_slot.get("row", "")),
                              str(audit_slot.get("position", audit_slot.get("slot", ""))))] = audit_slot
    planogram_rows = []
    rows_by_identity = {}
    physical_shelves = {str(slot.get("physical_shelf", "")) for slot in planogram}
    ordered_slots = sorted(
        planogram,
        key=lambda item: (str(item.get("physical_shelf", "")).casefold(),
                          int(item.get("row") or 0), int(item.get("position") or 0)),
    )
    for slot in ordered_slots:
        row_identity = (str(slot.get("physical_shelf", "")), str(slot.get("row", "")))
        row = rows_by_identity.get(row_identity)
        if row is None:
            row_label = slot["shelf"] if len(physical_shelves) <= 1 else f"{slot['physical_shelf']} · {slot['shelf']}"
            row = {"shelf": row_label, "row": slot.get("row", ""), "slots": []}
            rows_by_identity[row_identity] = row
            planogram_rows.append(row)
        audit_slot = report_slots.get((str(slot.get("physical_shelf", "")), str(slot.get("row", "")), str(slot["position"])))
        sku_detected = str(slot.get("sku") or "") in detected_skus
        row["slots"].append({
            **slot,
            "product": products_by_sku.get(slot["sku"]),
            "expected_price": slot.get("min_price") or (products_by_sku.get(slot["sku"]) or {}).get("price"),
            "sku_detected": sku_detected,
            "audit_status": (
                audit_slot.get("status") if audit_slot and audit_slot.get("status") != "Needs Review"
                else "SKU detected · placement unverified" if sku_detected
                else audit_slot.get("status") if audit_slot else None
            ),
        })
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
    if latest_report:
        latest_issues.extend({**alert, "issue": "Low stock"} for alert in latest_report.get("low_stock_alerts", []))
        latest_issues.extend(
            {"product": "Shelf gap", "sku": "", "issue": "Shelf gap detected",
             "detail": gap["detail"], "shelf": f"Row {gap['row']}", "position": ""}
            for gap in latest_report.get("shelf_gap_detection", {}).get("gaps", [])
        )
    return render_template(
        "dashboard.html",
        catalogue=catalogue,
        planogram=planogram,
        latest_report=latest_report,
        latest_report_id=latest_report_id,
        latest_issues=latest_issues,
        planogram_rows=planogram_rows,
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


@app.get("/reports/<report_id>/detection/<int:detection_index>/crop")
def report_detection_crop(report_id: str, detection_index: int):
    report = _reports.get(report_id)
    if report is None:
        abort(404)
    crop = _detection_crop_image(report, detection_index)
    if crop is None:
        abort(404)
    output = io.BytesIO()
    crop.save(output, format="JPEG", quality=88)
    output.seek(0)
    return send_file(output, mimetype="image/jpeg", download_name=f"detection-{detection_index + 1}.jpg", max_age=0)


def _detection_crop_image(report: dict, detection_index: int) -> Image.Image | None:
    """The detection's box cropped from the report's stored image (None when unavailable)."""
    detections = report.get("detections", [])
    if not 0 <= detection_index < len(detections):
        return None
    box = detections[detection_index].get("box")
    image_path = _report_image_path(report)
    if not isinstance(box, (list, tuple)) or len(box) != 4 or image_path is None:
        return None
    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            left, top, right, bottom = (int(round(float(value))) for value in box)
            left, top = max(0, min(width, left)), max(0, min(height, top))
            right, bottom = max(0, min(width, right)), max(0, min(height, bottom))
            if right <= left or bottom <= top:
                return None
            return image.crop((left, top, right, bottom))
    except (OSError, TypeError, ValueError):
        return None


def _persist_report(report_id: str, report: dict) -> bool:
    try:
        save_audit_result(AUDIT_RESULTS_DIR, report_id, {key: value for key, value in report.items() if key != "image_path"})
        return True
    except (OSError, ValueError, TypeError):
        logger.exception("Could not persist report %s", report_id)
        return False


def _create_reviewed_product(fields: dict, crop: Image.Image | None) -> dict:
    """Add a catalogue product named during review; the reviewed crop becomes its reference image."""
    name = str(fields.get("name") or "").strip()
    if not name:
        raise ValueError("Enter a name for the new product.")
    price = str(fields.get("price") or "").strip()
    if price and not _valid_price(price):
        raise ValueError("Enter a valid numeric price, or leave it empty.")
    products = _read_catalogue()
    existing = next((product for product in products if (product.get("name") or "").strip().casefold() == name.casefold()), None)
    if existing:
        raise ValueError(f"{existing['name']} is already in the catalogue (SKU {existing['sku']}); pick it from the list instead.")
    sku = re.sub(r"[^A-Za-z0-9_-]", "", str(fields.get("sku") or "").strip()) or f"LOCAL-{uuid.uuid4().hex[:8].upper()}"
    if any(product["sku"].casefold() == sku.casefold() for product in products):
        raise ValueError(f"SKU {sku} already exists; choose it from the list instead.")
    product = {
        "sku": sku, "name": name, "category": str(fields.get("category") or "Uncategorised").strip(),
        "brand": str(fields.get("brand") or "").strip(), "pack_size": "", "price": price,
        "model_label": name.casefold(), "active": "true",
    }
    if crop is not None:
        (DATA_DIR / "product_images").mkdir(parents=True, exist_ok=True)
        crop.convert("RGB").save(DATA_DIR / "product_images" / f"{sku}.jpg", format="JPEG", quality=92)
        product["image_local_path"] = f"product_images/{sku}.jpg"
    products.append(product)
    _save_catalogue(products)
    return product


def _review_queue(report: dict) -> list[dict]:
    """One review item per non-duplicate detection: crop, identity, reason, top candidates, verdict."""
    names = {product["sku"]: product.get("name") or product["sku"] for product in _read_catalogue()}
    names[NOT_PRODUCT_SKU] = "Not a product (reviewed)"
    matches = {int(match.get("detection_index", -1)): match for match in report.get("matching_audit", {}).get("matches", [])}
    diagnostics = {item["detection_index"]: item for item in report.get("pipeline_diagnostics", {}).get("detections", [])}
    items = []
    for index, detection in enumerate(report.get("detections", [])):
        if detection.get("duplicate_detection"):
            continue
        match = matches.get(index, {})
        diagnostic = diagnostics.get(index, {})
        sku = detection.get("matched_sku") or ""
        review = (report.get("reviews") or {}).get(str(index))
        slot = detection.get("slot_identity")
        items.append({
            "index": index,
            "sku": sku,
            "name": names.get(sku, "") if sku else "",
            "identified": bool(sku),
            "confidence": detection.get("confidence"),
            "similarity": match.get("top_similarity"),
            "margin": match.get("top_margin"),
            "reason_code": diagnostic.get("reason_code") or match.get("reason_code") or "",
            "explanation": diagnostic.get("explanation") or match.get("detail") or "",
            "placement": diagnostic.get("placement_status", ""),
            "slot": f"R{slot[3]}S{slot[4]}" if slot and len(slot) >= 5 else "",
            # "Not a product" examples explain a rejection; they are not products a reviewer can pick.
            "candidates": [
                {**candidate, "product": candidate.get("product") or names.get(candidate.get("sku"), candidate.get("sku"))}
                for candidate in (match.get("candidates") or []) if candidate.get("sku") != NOT_PRODUCT_SKU
            ][:3],
            "review": review,
        })
    return items


@app.post("/reports/<report_id>/review")
def review_detection(report_id: str):
    """Record a reviewer's verdict on one detection's SKU match and keep its crop as evidence."""
    report = _reports.get(report_id)
    if report is None:
        return jsonify(error="Report not found."), 404
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        index = int(payload.get("detection_index"))
    except (TypeError, ValueError):
        return jsonify(error="detection_index must be a number."), 400
    detections = report.get("detections", [])
    if not 0 <= index < len(detections) or detections[index].get("duplicate_detection"):
        return jsonify(error="That detection does not exist in this report."), 404
    decision = str(payload.get("decision") or "")
    detection = detections[index]
    match = next((item for item in report.get("matching_audit", {}).get("matches", [])
                  if int(item.get("detection_index", -1)) == index), {})
    top = (match.get("candidates") or [{}])[0]
    suggestion = next((candidate for candidate in match.get("candidates") or [] if candidate.get("sku") != NOT_PRODUCT_SKU), {})
    crop = _detection_crop_image(report, index)
    catalogue = {product["sku"]: product for product in _read_catalogue()}
    new_product = payload.get("new_product") if isinstance(payload.get("new_product"), dict) else None
    if new_product is None and payload.get("new_name"):
        new_product = {"name": payload.get("new_name"), "price": payload.get("new_price"),
                       "category": payload.get("new_category"), "sku": payload.get("new_sku")}
    try:
        if decision == "correct":
            sku = str(payload.get("sku") or detection.get("matched_sku") or suggestion.get("sku") or "")
            if sku not in catalogue:
                raise ValueError("There is no suggested catalogue product to confirm; mark it Wrong and pick the product.")
        elif decision == "wrong":
            if new_product:
                product = _create_reviewed_product(new_product, crop)
                catalogue[product["sku"]] = product
                sku = product["sku"]
            else:
                sku = str(payload.get("sku") or "")
                if sku not in catalogue:
                    raise ValueError("Choose the correct product from the catalogue, or add it as a new product.")
        elif decision == "not_product":
            sku = ""
        else:
            raise ValueError("decision must be correct, wrong, or not_product.")
    except ValueError as error:
        return jsonify(error=str(error)), 400

    product_name = catalogue[sku].get("name", sku) if sku else "Not a product"
    record = record_feedback(DATA_DIR, {
        "report_id": report_id, "detection_index": index, "decision": decision,
        "true_sku": sku or None, "product_name": product_name,
        "model_top_sku": top.get("sku"), "model_top_similarity": match.get("top_similarity"),
        "model_top_margin": match.get("top_margin"), "model_accepted_sku": match.get("accepted_sku"),
        "threshold_used": match.get("threshold"), "image_name": report.get("image_name"),
        # A newly created product already uses this crop as its catalogue image.
        "use_as_reference": not (decision == "wrong" and new_product),
    }, crop)
    review = {"id": record["id"], "decision": decision, "sku": sku, "product_name": product_name,
              "created_at": record["created_at"]}
    report.setdefault("reviews", {})[str(index)] = review
    detection["review"] = review
    persisted = _persist_report(report_id, report)
    messages = {
        "correct": f"Confirmed as {product_name}. This photo is now a reference for it.",
        "wrong": f"Corrected to {product_name}. This photo is now a reference for it.",
        "not_product": "Marked as not a product. Similar crops will no longer be matched.",
    }
    return jsonify(ok=True, review=review, reviews=len(report["reviews"]), persisted=persisted,
                   message=messages[decision] + " Re-analyze to update the compliance results.")


@app.post("/reports/<report_id>/reanalyze")
def reanalyze_report(report_id: str):
    """Run the same stored photo again, with current reviews, catalogue, and threshold."""
    report = _reports.get(report_id)
    if report is None:
        abort(404)
    options = report.get("analysis_options") or {}
    form = {
        "reanalyze_image": report.get("image_file") or "", "reanalyze_name": report.get("image_name") or "",
        "store": report.get("store_name") or "", "planogram": report.get("planogram_name") or "",
        "match_threshold": request.form.get("match_threshold") or options.get("match_threshold")
        or (report.get("matching_audit", {}).get("calibration") or {}).get("acceptance_threshold") or "",
        "match_margin": options.get("match_margin") or "",
    }
    if report.get("confidence_threshold"):
        form["confidence_threshold"] = str(report["confidence_threshold"])
    if options.get("calibrated_layout", report.get("layout_validation", {}).get("requested")):
        form["calibrated_layout"] = "on"
    if request.form.get("confirm_layout") == "on":
        form["calibrated_layout"] = "on"
    if options.get("price_check", report.get("price_audit", {}).get("requested")):
        form["price_check"] = "on"
    try:
        new_id, new_report = _run_audit({}, form)
    except AuditError as error:
        flash(error.message, "error")
        return redirect(url_for("report_detail", report_id=report_id))
    reviews = len(report.get("reviews") or {})
    flash(f"Re-analyzed with {len(load_feedback(DATA_DIR))} saved review(s)"
          + (f" ({reviews} from this report)." if reviews else "."), "success")
    return redirect(url_for("report_detail", report_id=new_id))


@app.get("/favicon.ico")
def favicon():
    icon = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#1268f5"/>'
            '<path d="M8 11h16M8 16h16M8 21h16" stroke="#fff" stroke-width="2.5" stroke-linecap="round"/></svg>')
    return app.response_class(icon, mimetype="image/svg+xml", headers={"Cache-Control": "max-age=86400"})


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
    planogram_rows = _read_planogram()
    active_planogram_id = request.args.get("planogram_id") or (planogram_rows[0].get("planogram_id", "DEFAULT") if planogram_rows else "DEFAULT")
    return render_template(
        "upload.html",
        page_title="Upload & Analyze",
        active_page="upload",
        settings=_load_settings(),
        example=example,
        test_images=list(_test_images()),
        planogram_id=active_planogram_id,
        calibrated_threshold=_calibrated_match_values()["threshold"],
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
    planogram_id = planogram[0].get("planogram_id", "DEFAULT") if planogram else "DEFAULT"
    next_row_number = max((int(slot.get("_row_number", 0)) for slot in planogram), default=0) + 1
    return render_template(
        "planogram.html",
        shelf_rows=shelf_rows,
        products=[product for product in catalogue if product.get("active", "true").casefold() == "true"],
        total_slots=total_slots,
        assigned_slots=assigned_slots,
        empty_slots=total_slots - assigned_slots,
        planogram_errors=planogram_errors,
        planogram_id=planogram_id,
        next_row_number=next_row_number,
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
    if rows and not row_numbers:
        next_row_number = max((int(row.get("_row_number", 0)) for row in rows), default=0) + 1
    shelf = f"Row {next_row_number}"
    if any(str(row.get("_row_number")) == str(next_row_number) for row in rows):
        flash("Could not determine a unique next row number.", "error")
        return redirect(url_for("planogram_page"))
    source_shelf_id = rows[0].get("_shelf_id", "S1") if rows else "S1"
    rows.extend(
        {"shelf": shelf, "position": str(position), "slot": str(position), "sku": "", "expected_sku": "", "expected_facings": "0", "facing_count": "0", "min_price": "", "_shelf_id": source_shelf_id, "_row_number": next_row_number}
        for position in range(1, 6)
    )
    if not _persist_planogram(rows):
        return redirect(url_for("planogram_page"))
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
    if not shelf or position < 1 or position > 5 or facings < 1:
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
    replacing = bool(slot and slot.get("sku"))
    if slot is None:
        row_slots = [row for row in rows if row["shelf"].casefold() == shelf.casefold()]
        if not row_slots or request.form.get("replace") == "on":
            flash("That planogram slot no longer exists. Add a row before assigning a product.", "error")
            return redirect(url_for("planogram_page"))
        slot = {
            "shelf": shelf, "position": str(position), "sku": "", "expected_sku": "",
            "expected_facings": "0", "facing_count": "0", "min_price": "",
            "_shelf_id": row_slots[0].get("_shelf_id", "S1"), "_row_number": row_slots[0].get("_row_number", "1"),
            "planogram_id": row_slots[0].get("planogram_id", "DEFAULT"), "store_id": row_slots[0].get("store_id", "STORE001"),
        }
        rows.append(slot)
    elif slot.get("sku") and request.form.get("replace") != "on":
        flash("That slot is already assigned. Clear it before assigning another product.", "error")
        return redirect(url_for("planogram_page"))
    slot["sku"] = product["sku"]
    slot["expected_sku"] = product["sku"]
    slot["expected_facings"] = str(facings)
    slot["facing_count"] = str(facings)
    slot["min_price"] = product["price"]
    rows.sort(key=lambda row: (row["shelf"].casefold(), int(row["position"])))
    if not _persist_planogram(rows):
        return redirect(url_for("planogram_page"))
    action = "was changed in" if replacing else "was assigned to"
    flash(f"{product['name']} {action} {shelf}, slot {position}.", "success")
    return redirect(url_for("planogram_page"))


@app.post("/planogram/slot/remove")
def remove_planogram_slot():
    shelf = request.form.get("shelf", "").strip()
    position = request.form.get("position", "").strip()
    rows = _read_planogram()
    slot = next((row for row in rows if row["shelf"].casefold() == shelf.casefold() and row["position"] == position), None)
    if slot is None:
        flash("That planogram slot no longer exists.", "error")
        return redirect(url_for("planogram_page"))
    if not slot.get("sku"):
        flash("That slot is already empty.", "error")
        return redirect(url_for("planogram_page"))
    product = next((item for item in _read_catalogue() if item["sku"] == slot["sku"]), None)
    slot.update(sku="", expected_sku="", expected_facings="0", facing_count="0", min_price="")
    if not _persist_planogram(rows):
        return redirect(url_for("planogram_page"))
    flash(f"{product['name'] if product else 'Product'} was removed from {shelf}, slot {position}.", "success")
    return redirect(url_for("planogram_page"))


@app.post("/planogram/clear")
def clear_planogram():
    if not _persist_planogram([]):
        return redirect(url_for("planogram_page"))
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
    if not _persist_planogram(rows):
        return redirect(url_for("planogram_page"))
    flash(f"Planogram saved with {len(rows)} slots.", "success")
    return redirect(url_for("planogram_page"))


@app.post("/planogram/slot/move")
def move_planogram_slot():
    source_shelf, source_position = request.form.get("source_shelf", "").strip(), request.form.get("source_position", "").strip()
    target_shelf, target_position = request.form.get("target_shelf", "").strip(), request.form.get("target_position", "").strip()
    rows = _read_planogram()
    source = next((row for row in rows if row["shelf"] == source_shelf and row["position"] == source_position), None)
    target = next((row for row in rows if row["shelf"] == target_shelf and row["position"] == target_position), None)
    if source is None or target is None or not source.get("sku"):
        flash("The product or destination slot no longer exists.", "error")
    elif target.get("sku"):
        flash("Choose an empty destination slot to move this product.", "error")
    else:
        target.update(sku=source["sku"], expected_sku=source["sku"], expected_facings=source["expected_facings"], facing_count=source["facing_count"], min_price=source.get("min_price", ""))
        source.update(sku="", expected_sku="", expected_facings="0", facing_count="0", min_price="")
        if not _persist_planogram(rows):
            return redirect(url_for("planogram_page"))
        flash(f"Product moved to {target_shelf}, slot {target_position}.", "success")
    return redirect(url_for("planogram_page"))


@app.post("/planogram/slot/duplicate")
def duplicate_planogram_slot():
    source_shelf, source_position = request.form.get("source_shelf", "").strip(), request.form.get("source_position", "").strip()
    target_shelf, target_position = request.form.get("target_shelf", "").strip(), request.form.get("target_position", "").strip()
    rows = _read_planogram()
    source = next((row for row in rows if row["shelf"] == source_shelf and row["position"] == source_position), None)
    target = next((row for row in rows if row["shelf"] == target_shelf and row["position"] == target_position), None)
    if source is None or target is None or not source.get("sku"):
        flash("The product or destination slot no longer exists.", "error")
    elif target.get("sku"):
        flash("Choose an empty destination slot to duplicate this product.", "error")
    else:
        target.update(sku=source["sku"], expected_sku=source["sku"], expected_facings=source["expected_facings"], facing_count=source["facing_count"], min_price=source.get("min_price", ""))
        if not _persist_planogram(rows):
            return redirect(url_for("planogram_page"))
        flash(f"Product duplicated to {target_shelf}, slot {target_position}.", "success")
    return redirect(url_for("planogram_page"))


@app.post("/planogram/product/edit/<sku>")
def edit_planogram_product(sku: str):
    products = _read_catalogue()
    product = next((item for item in products if item["sku"].casefold() == sku.casefold()), None)
    if product is None:
        flash("That catalogue product no longer exists.", "error")
        return redirect(url_for("planogram_page"))
    name, category, price = (request.form.get(key, "").strip() for key in ("name", "category", "price"))
    if not name or not category or not _valid_price(price):
        flash("Enter a product name, category, and valid non-negative price.", "error")
        return redirect(url_for("planogram_page"))
    product.update(name=name, category=category, price=price, model_label=request.form.get("model_label", name).strip() or name)
    source = dict(product.get("_source", {}))
    source.update({"product_name": name, "category": category, "expected_price": price, "name": name, "price": price})
    image_upload = request.files.get("image")
    if image_upload and image_upload.filename:
        if not _allowed_file(image_upload.filename):
            flash("Choose a PNG, JPG, JPEG, or WEBP product image.", "error")
            return redirect(url_for("planogram_page"))
        try:
            with Image.open(image_upload.stream) as probe:
                probe.verify()
            image_upload.stream.seek(0)
            with Image.open(image_upload.stream) as opened:
                if opened.width * opened.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Product image exceeds the 40 megapixel limit.")
                image = ImageOps.exif_transpose(opened).convert("RGB")
            image_name = f"{secure_filename(sku)}.jpg"
            image_path = DATA_DIR / "product_images" / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path, format="JPEG", quality=92)
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError, ModuleNotFoundError) as error:
            flash(f"Could not read product image: {error}", "error")
            return redirect(url_for("planogram_page"))
        product.update(barcode=sku, image_local_path=f"product_images/{image_name}")
        source.update(barcode=sku, image_local_path=f"product_images/{image_name}")
    product["_source"] = source
    _save_catalogue(products)
    flash(f"{name} was updated in the catalogue.", "success")
    return redirect(url_for("planogram_page"))


@app.post("/planogram/product/add")
def create_and_assign_planogram_product():
    name, category, price = (request.form.get(key, "").strip() for key in ("name", "category", "price"))
    shelf, position = request.form.get("shelf", "").strip(), request.form.get("position", "").strip()
    image_upload = request.files.get("image")
    rows = _read_planogram()
    target = next((row for row in rows if row["shelf"] == shelf and row["position"] == position), None)
    if not name or not category or not _valid_price(price):
        flash("Enter a product name, category, and valid non-negative price.", "error")
        return redirect(url_for("planogram_page"))
    if not image_upload or not image_upload.filename or not _allowed_file(image_upload.filename):
        flash("Choose a PNG, JPG, JPEG, or WEBP product image.", "error")
        return redirect(url_for("planogram_page"))
    if target is None or target.get("sku"):
        flash("Choose an existing empty planogram slot.", "error")
        return redirect(url_for("planogram_page"))
    products = _read_catalogue()
    sku = "LOCAL" + uuid.uuid4().hex[:12].upper()
    image_dir = DATA_DIR / "product_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    try:
        image = Image.open(image_upload.stream)
        image.verify()
        image_upload.stream.seek(0)
        with Image.open(image_upload.stream) as opened:
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise ValueError("Product image exceeds the 40 megapixel limit.")
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image_name = f"{sku}.jpg"
        image.save(image_dir / image_name, format="JPEG", quality=92)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError, ModuleNotFoundError) as error:
        flash(f"Could not read product image: {error}", "error")
        return redirect(url_for("planogram_page"))
    fields = []
    if CATALOGUE_PATH.exists():
        with CATALOGUE_PATH.open(encoding="utf-8-sig", newline="") as csv_file:
            fields = next(csv.reader(csv_file), [])
    source = {field: "" for field in fields}
    source.update(sku=sku, name=name, product_name=name, category=category, price=price, expected_price=price,
                  brand="", model_label=name.casefold(), active="true", barcode=sku,
                  image_local_path=f"product_images/{image_name}")
    products.append({"sku": sku, "name": name, "category": category, "brand": "", "pack_size": "", "price": price,
                     "model_label": name.casefold(), "active": "true", "barcode": sku,
                     "image_local_path": f"product_images/{image_name}", "_source": source})
    _save_catalogue(products)
    target.update(sku=sku, expected_sku=sku, expected_facings="1", facing_count="1", min_price=price)
    if not _persist_planogram(rows):
        return redirect(url_for("planogram_page"))
    flash(f"{name} was added to the catalogue and assigned to {shelf}, slot {position}.", "success")
    return redirect(url_for("planogram_page"))


@app.post("/planogram/apply")
def apply_planogram_to_analysis():
    rows = _read_planogram()
    errors = validate_planogram(rows, _read_catalogue())
    if errors:
        flash("Planogram validation failed: " + " ".join(errors[:3]), "error")
        return redirect(url_for("planogram_page"))
    try:
        _save_planogram(rows)
    except OSError:
        logger.exception("Could not save planogram before analysis")
        flash("Planogram could not be saved. Check the data folder permissions and try again.", "error")
        return redirect(url_for("planogram_page"))
    planogram_id = rows[0].get("planogram_id", "DEFAULT") if rows else "DEFAULT"
    flash(f"Planogram {planogram_id} is saved and ready for analysis.", "success")
    return redirect(url_for("upload_page", planogram_id=planogram_id))


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
        (item for item in _read_catalogue() if item.get("barcode") == barcode or item.get("sku") == barcode),
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
    reports = [
        (report_id, _normalize_saved_report(report_id, dict(report)))
        for report_id, report in reversed(list(_reports.items()))
    ]
    trend_reports = [(report_id, report) for report_id, report in reports if report.get("compliance_score", {}).get("score") is not None]
    confirmed_statuses = {"Compliant", "Misplaced", "Missing Expected Product", "Unexpected Product"}
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
        entry = by_store.setdefault(store_name, {"store": store_name, "audits": 0, "score_total": 0.0, "scored_audits": 0, "placements": 0, "issues": 0})
        entry["audits"] += 1
        score = report.get("compliance_score", {}).get("score")
        if score is not None:
            entry["score_total"] += score
            entry["scored_audits"] += 1
        entry["placements"] += sum(row.get("status") in confirmed_statuses for row in report.get("compliance", []))
        entry["issues"] += sum(row.get("status") in {"Misplaced", "Missing Expected Product", "Unexpected Product"} for row in report.get("compliance", []))
    store_summaries = [
        {**store, "compliance_rate": round(store["score_total"] / store["scored_audits"], 2) if store["scored_audits"] else None}
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
        average_compliance=round(sum(report["compliance_score"]["score"] for _, report in trend_reports) / len(trend_reports), 2) if trend_reports else None,
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
    for report_id, report in _reports.items():
        writer.writerows(_report_export_rows(_normalize_saved_report(report_id, dict(report))))
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
        calibrated=_calibrated_match_values(),
        effective_margin=_match_settings(_load_settings())[1],
        feedback_stats=feedback_accuracy(load_feedback(DATA_DIR), *_match_settings(_load_settings())),
        match_threshold_range=MATCH_THRESHOLD_RANGE,
        feedback_points=[
            {"similarity": record["model_top_similarity"], "margin": record["model_top_margin"],
             "correct": record.get("decision") != "not_product" and record.get("true_sku") == record.get("model_top_sku")}
            for record in load_feedback(DATA_DIR)
            if record.get("model_top_similarity") is not None and record.get("model_top_margin") is not None
            and record.get("model_top_sku")
        ],
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
        settings["product_group_max_gap_ratio"] = float(request.form.get("product_group_max_gap_ratio", settings["product_group_max_gap_ratio"]))
        settings["product_group_min_vertical_overlap"] = float(request.form.get("product_group_min_vertical_overlap", settings["product_group_min_vertical_overlap"]))
        if request.form.get("match_reset"):
            settings["match_threshold"], settings["match_margin"] = DEFAULT_MATCH_THRESHOLD, None
        else:
            if request.form.get("match_threshold"):
                settings["match_threshold"] = float(request.form["match_threshold"])
            if request.form.get("match_margin"):
                settings["match_margin"] = float(request.form["match_margin"])
        try:
            _match_settings(settings)
        except AuditError as error:
            raise ValueError(error.message) from error
        if not 0.1 <= settings["detection_confidence"] <= 0.9 or not 0.1 <= settings["price_confidence"] <= 0.95:
            raise ValueError("Confidence values are outside the allowed range.")
        if settings["minimum_product_height"] < 1 or settings["low_stock_threshold"] < 0:
            raise ValueError("Height and stock threshold must be zero or greater.")
        if not 0 <= settings["product_group_max_gap_ratio"] <= 1 or not 0 <= settings["product_group_min_vertical_overlap"] <= 1:
            raise ValueError("Product grouping thresholds must be between 0 and 1.")
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
    report = _normalize_saved_report(report_id, dict(report))
    structured = _report_summary(report_id, report)
    report.setdefault("detected_planogram", structured.get("detected_planogram", {}))
    report.setdefault("planogram_comparison", structured.get("planogram_comparison", {}))
    catalogue = _read_catalogue()
    report_planogram = [
        slot for slot in _read_planogram()
        if str(slot.get("planogram_id", "")) == str(report.get("planogram_id", ""))
    ]
    layout_validation = report.get("layout_validation") or {}
    expected_rows = int(layout_validation.get("expected_rows") or 0)
    detected_rows = int(layout_validation.get("detected_rows") or 0)
    can_confirm_layout = bool(
        not report.get("layout_trusted") and expected_rows > 0 and detected_rows == expected_rows
        and len({str(slot.get("physical_shelf", slot.get("shelf", ""))) for slot in report_planogram}) == 1
    )
    return render_template(
        "report.html", report=report, report_id=report_id, review_items=_review_queue(report),
        catalogue_choices=sorted(({"sku": product["sku"], "name": product.get("name") or product["sku"]}
                                  for product in catalogue), key=lambda item: item["name"].casefold()),
        can_reanalyze=_report_image_path(report) is not None,
        can_confirm_layout=can_confirm_layout,
        page_title="Scan report", active_page="reports",
    )


@app.get("/reports/<report_id>/download")
def download_report(report_id: str):
    report = _reports.get(report_id)
    if report is None:
        abort(404)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=REPORT_EXPORT_FIELDS)
    writer.writeheader()
    writer.writerows(_report_export_rows(_normalize_saved_report(report_id, dict(report))))
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
    active_threshold = float(_load_settings().get("match_threshold", DEFAULT_MATCH_THRESHOLD))
    return (f"active threshold {active_threshold:.2f} · validation reference {calibration.get('threshold')} / "
            f"margin {calibration.get('minimum_margin')} · validation precision {metrics.get('precision')}, "
            f"coverage {metrics.get('known_coverage')}")


def _safe_preload() -> None:
    try:
        get_detector(MODEL_PATH).load()
        logger.info("Detector loaded: %s", detector_status().get("model_file"))
    except DetectorUnavailable as error:
        logger.warning("Detector preload failed: %s", error)


def _report_summary(report_id: str, report: dict) -> dict:
    report = _normalize_saved_report(report_id, dict(report))
    detected_planogram = report.get("detected_planogram")
    comparison = report.get("planogram_comparison")
    if not detected_planogram:
        catalogue = _read_catalogue()
        detected_planogram = _detected_planogram_urls(report_id, reconstruct_detected_planogram(
            report.get("detections", []), report.get("void_detections", []),
            report.get("price_audit", {}).get("tags", []), catalogue,
            int(report.get("image_width") or 0), int(report.get("image_height") or 0),
            grouping_config={
                "max_gap_ratio": _load_settings()["product_group_max_gap_ratio"],
                "min_vertical_overlap": _load_settings()["product_group_min_vertical_overlap"],
            },
        ))
        comparison = compare_expected_to_detected(
            _read_planogram(), detected_planogram, report.get("planogram_audit", {}),
            report.get("price_results", []), bool(report.get("layout_trusted")), catalogue,
            report.get("matching_audit", {}),
        )
        comparison = _comparison_urls(report_id, comparison)
    return {
        "report_id": report_id,
        "created_at": report.get("created_at"),
        "store_name": report.get("store_name"),
        "image_name": report.get("image_name"),
        "detection_count": report.get("detection_count"),
        "raw_detection_count": report.get("raw_detection_count", report.get("detection_count")),
        "duplicate_detection_count": report.get("duplicate_detection_count", 0),
        "layout_validation": report.get("layout_validation", {}),
        "inference_ms": report.get("inference_ms"),
        "compliance_rate": report.get("compliance_rate"),
        "compliance_score": report.get("compliance_score", {}),
        "report_url": url_for("report_detail", report_id=report_id),
        "annotated_image_url": url_for("report_annotated_image", report_id=report_id) if report.get("annotated_image") else None,
        "price_audit": report.get("price_audit", {}),
        "price_tag_detections": report.get("price_tag_detections", []),
        "void_detections": report.get("void_detections", []),
        "detected_planogram": detected_planogram or {},
        "planogram_comparison": comparison or {},
        "auxiliary_model": report.get("auxiliary_model", {}),
        "auxiliary_model_error": report.get("auxiliary_model_error"),
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
    auxiliary_detector = get_detector(PRICE_TAG_VOID_MODEL_PATH) if PRICE_TAG_VOID_MODEL_PATH else None
    auxiliary_model = auxiliary_detector.info() if auxiliary_detector else {
        "file_exists": False, "loaded": False, "load_error": "SHELFSENSE_PRICE_TAG_VOID_MODEL is not set.",
        "class_names": [],
    }
    if request.args.get("load") == "1" and auxiliary_model.get("file_exists"):
        try:
            auxiliary_detector.load()
        except DetectorUnavailable:
            pass
        auxiliary_model = auxiliary_detector.info()
    checks = {"model_file": bool(model.get("file_exists")), "model_load_error": model.get("load_error"),
              "auxiliary_model_file": bool(auxiliary_model.get("file_exists")),
              "auxiliary_model_load_error": auxiliary_model.get("load_error")}
    try:
        catalogue, planogram = _read_catalogue(), _read_planogram()
        checks.update(catalogue_products=len(catalogue), planogram_slots=len(planogram),
                      planogram_errors=validate_planogram(planogram, catalogue))
    except (OSError, csv.Error) as error:
        checks["data_error"] = str(error)
    results_target = AUDIT_RESULTS_DIR if AUDIT_RESULTS_DIR.exists() else AUDIT_RESULTS_DIR.parent
    checks["results_dir_writable"] = os.access(results_target, os.W_OK)
    healthy = (checks["model_file"] and not checks["model_load_error"] and checks["auxiliary_model_file"]
               and not checks["auxiliary_model_load_error"]
               and not checks.get("data_error") and not checks.get("planogram_errors"))
    return jsonify(status="ok" if healthy else "degraded", model=model, auxiliary_model=auxiliary_model,
                   checks=checks, saved_reports=len(_reports))


@app.get("/api/reports")
def api_reports():
    return jsonify(reports=[_report_summary(report_id, report) for report_id, report in reversed(list(_reports.items()))])


@app.route("/api/chat", methods=["GET", "POST"])
def api_chat():
    latest_report_id = next(reversed(_reports), None)
    if request.method == "GET":
        configured = chat_configured()
        return jsonify(
            configured=configured,
            report_id=latest_report_id,
            message=None if configured else (
                "AI chat is not configured. Set SHELFSENSE_CHAT_API_KEY on the server and restart ShelfSense AI."
            ),
        )

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Send a JSON object with a message."), 400
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify(error="Enter a message to continue."), 400
    if len(message) > 1500:
        return jsonify(error="Messages must be 1,500 characters or fewer."), 400

    requested_report_id = body.get("report_id")
    report_id = str(requested_report_id).strip() if requested_report_id else latest_report_id
    report = _reports.get(report_id) if report_id else None
    if report_id and report is None:
        return jsonify(error="That audit report was not found."), 404
    report = _normalize_saved_report(report_id, dict(report)) if report is not None else None
    context = build_chat_audit_context(report_id, report)
    try:
        answer = generate_chat_reply(message, context, body.get("history"))
    except ChatProviderError as error:
        status = 503 if not chat_configured() else 502
        return jsonify(error=str(error), configured=chat_configured()), status
    return jsonify(reply=answer, report_id=report_id)


@app.get("/api/reports/<report_id>")
def api_report(report_id: str):
    report = _reports.get(report_id)
    if report is None:
        return jsonify(error="Report not found."), 404
    report = _normalize_saved_report(report_id, dict(report))
    payload = {key: value for key, value in report.items() if key != "image_path"}
    summary = _report_summary(report_id, report)
    payload["detected_planogram"] = summary["detected_planogram"]
    payload["planogram_comparison"] = summary["planogram_comparison"]
    return jsonify(payload)


@app.post("/api/analyze")
def api_analyze():
    try:
        report_id, report = _run_audit(request.files, request.form)
    except AuditError as error:
        unavailable_score = calculate_compliance_score(
            [], [], layout_trusted=False, price_requested=False,
        )
        unavailable_score["reason"] = f"Audit failed before compliance checks could be evaluated: {error.message}"
        return jsonify(error=error.message, compliance_score=unavailable_score), error.status
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
