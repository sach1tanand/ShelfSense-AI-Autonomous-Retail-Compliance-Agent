"""ShelfSense AI web application."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import platform
import re
import uuid
from pathlib import Path

from flask import Flask, abort, flash, make_response, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from backend.matching.pipeline import ProductMatchingPipeline
from backend.matching.storage import save_audit_result
from config.settings import (
    ALLOWED_EXTENSIONS,
    DATA_DIR,
    DETECTION_CONFIDENCE,
    MAX_CONTENT_LENGTH,
    MODEL_PATH,
    ROOT_DIR,
    UPLOAD_DIR,
)
from src.compliance import evaluate_compliance
from src.detector import DetectorUnavailable, detect_products
from src.planogram import evaluate_planogram, validate_planogram
from src.price_checker import check_prices
from src.report import build_report

app = Flask(__name__)
app.config["SECRET_KEY"] = "shelfsense-local-dev-key"
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_reports: dict[str, dict] = {}
_matching_pipeline: ProductMatchingPipeline | None = None
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


def _get_matching_pipeline() -> ProductMatchingPipeline:
    global _matching_pipeline
    if _matching_pipeline is None:
        _matching_pipeline = ProductMatchingPipeline(
            data_dir=DATA_DIR,
            model_id=os.getenv("SHELFSENSE_CLIP_MODEL", "openai/clip-vit-base-patch32"),
            top_k=int(os.getenv("SHELFSENSE_MATCH_TOP_K", "5")),
        )
    return _matching_pipeline


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
        latest_issues.extend(
            {**finding, "issue": finding.get("status", "Needs Review"), "shelf": "", "position": ""}
            for finding in latest_report.get("planogram_audit", {}).get("detections", [])
        )
        latest_issues.extend(
            {**row, "issue": "Price mismatch", "detail": f"Detected ${row['observed_price']:.2f}; expected ${row['expected_price']:.2f}."}
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
        active_page="dashboard",
    )


@app.get("/reports/<report_id>/image")
def report_image(report_id: str):
    report = _reports.get(report_id)
    if report is None and report_id == DEMO_REPORT_ID:
        example_path = UPLOAD_DIR / "demo-shelf.svg"
        if example_path.is_file():
            return send_file(example_path, mimetype="image/svg+xml", max_age=3600)
    image_path = Path(report["image_path"]) if report and report.get("image_path") else None
    if image_path is None or not image_path.is_file() or image_path.parent.resolve() != UPLOAD_DIR.resolve():
        abort(404)
    suffix = image_path.suffix.lower()
    mime_type = "image/svg+xml" if suffix == ".svg" else "image/jpeg"
    return send_file(io.BytesIO(image_path.read_bytes()), mimetype=mime_type, download_name=image_path.name, max_age=0)


@app.get("/upload")
def upload_page():
    return render_template(
        "upload.html",
        page_title="Upload & Analyze",
        active_page="upload",
        settings=_load_settings(),
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
    image = request.files.get("shelf_image")
    if image is None or not image.filename:
        flash("Choose a shelf image to analyze.", "error")
        return redirect(url_for("upload_page"))
    if not _allowed_file(image.filename):
        flash("Upload a PNG, JPG, JPEG, or WEBP image.", "error")
        return redirect(url_for("upload_page"))

    safe_name = secure_filename(image.filename)
    stored_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    image.save(stored_path)

    settings = _load_settings()
    try:
        confidence = float(request.form.get("confidence_threshold", settings["detection_confidence"]))
    except (TypeError, ValueError):
        confidence = float(settings["detection_confidence"])
    if not 0.1 <= confidence <= 0.95:
        flash("Detection confidence must be between 0.10 and 0.95.", "error")
        stored_path.unlink(missing_ok=True)
        return redirect(url_for("upload_page"))

    all_catalogue = _read_catalogue()
    catalogue = [product for product in all_catalogue if product.get("active", "true").casefold() == "true"]
    complete_planogram = _read_planogram()
    planogram_errors = validate_planogram(complete_planogram, all_catalogue)
    if planogram_errors:
        stored_path.unlink(missing_ok=True)
        flash("Planogram validation failed: " + " ".join(planogram_errors[:3]), "error")
        return redirect(url_for("planogram_page"))
    active_skus = {product["sku"] for product in catalogue}
    planogram = [slot for slot in complete_planogram if not slot.get("sku") or slot["sku"] in active_skus]
    try:
        detections = detect_products(stored_path, confidence=confidence)
    except DetectorUnavailable as error:
        stored_path.unlink(missing_ok=True)
        flash(str(error), "error")
        return redirect(url_for("upload_page"))
    except Exception:
        stored_path.unlink(missing_ok=True)
        app.logger.exception("Shelf image analysis failed")
        flash("Image analysis failed unexpectedly. Check the model and image, then try again.", "error")
        return redirect(url_for("upload_page"))

    try:
        planogram_audit = evaluate_planogram(planogram, all_catalogue, detections)
        compliance = planogram_audit["slots"]
        price_results = check_prices(catalogue, {}) if request.form.get("price_check") else []
        report = build_report(
            compliance=compliance,
            price_results=price_results,
            detections=detections,
            image_name=safe_name,
            store_name=request.form.get("store", settings["default_store"]).strip() or settings["default_store"],
        )
    except Exception:
        stored_path.unlink(missing_ok=True)
        app.logger.exception("Shelf audit report generation failed")
        flash("The audit could not be completed. Check the catalogue and planogram, then try again.", "error")
        return redirect(url_for("upload_page"))
    report["planogram_name"] = request.form.get("planogram", settings["default_planogram"]).strip()
    report["confidence_threshold"] = confidence
    report["planogram_audit"] = planogram_audit
    report_id = uuid.uuid4().hex
    report["image_path"] = str(stored_path.resolve())
    _reports[report_id] = report
    return redirect(url_for("report_detail", report_id=report_id))


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
                issue_products[row.get("product") or "Unidentified product"] = issue_products.get(row.get("product") or "Unidentified product", 0) + 1
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
            issue_products[finding.get("product") or "Unidentified product"] = issue_products.get(finding.get("product") or "Unidentified product", 0) + 1
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


@app.errorhandler(413)
def request_too_large(_error):
    flash("That image is too large. The upload limit is 12 MB.", "error")
    return redirect(url_for("dashboard")), 303


if __name__ == "__main__":
    app.run(debug=True)