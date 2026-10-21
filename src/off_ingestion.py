"""Download Indian packaged-food product records and front images from Open Food Facts."""

from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError

from config.settings import DATA_DIR

API_URL = "https://world.openfoodfacts.org/api/v2/search"
USER_AGENT = "ShelfSenseAI/1.0 (Open Food Facts catalogue ingestion; local development project)"
PAGE_SIZE = 100
MAX_PRODUCTS = 200
MAX_RETRIES = 5
REQUEST_TIMEOUT = (10, 30)
CSV_FIELDS = [
    "sku",
    "product_name",
    "brand",
    "category",
    "quantity",
    "expected_price",
    "barcode",
    "front_image_url",
    "product_url",
    "image_local_path",
]
LOGGER = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    valid_products: int
    downloaded_images: int
    failed_downloads: int
    output_csv: Path
    image_directory: Path


def is_valid_gtin(value: Any) -> bool:
    """Return whether a value is a checksum-valid 8–14 digit GTIN barcode."""
    barcode = str(value or "").strip()
    if not re.fullmatch(r"(?:\d{8}|\d{12}|\d{13}|\d{14})", barcode):
        return False
    digits = [int(digit) for digit in barcode]
    check_digit = digits.pop()
    weighted_sum = sum(digit * (3 if index % 2 == 0 else 1) for index, digit in enumerate(reversed(digits)))
    return (10 - weighted_sum % 10) % 10 == check_digit


def _retry_delay(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, min(float(retry_after), 120.0))
            except ValueError:
                try:
                    retry_time = parsedate_to_datetime(retry_after).timestamp()
                    return max(0.0, min(retry_time - time.time(), 120.0))
                except (TypeError, ValueError, OverflowError):
                    pass
    return min(2.0**attempt, 30.0)


def _request_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    stream: bool = False,
    max_retries: int = MAX_RETRIES,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        response = None
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT, stream=stream)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 == max_retries:
                    response.raise_for_status()
                delay = _retry_delay(response, attempt)
                response.close()
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if response is not None:
                response.close()
            if attempt + 1 == max_retries:
                raise
            time.sleep(_retry_delay(response, attempt))
    if last_error:
        raise last_error
    raise requests.RequestException(f"Request failed after {max_retries} attempts: {url}")


def _text(value: Any) -> str:
    if isinstance(value, list):
        value = ", ".join(str(part).strip() for part in value if str(part).strip())
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _image_url(product: dict[str, Any]) -> str:
    candidates = (
        product.get("image_front_url"),
        product.get("image_front_small_url"),
        product.get("image_front_thumb_url"),
    )
    for value in candidates:
        url = _text(value)
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return url
    return ""


def _product_page_url(product: dict[str, Any]) -> str:
    value = _text(product.get("url"))
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else ""


def _verified_image_format(content: bytes) -> str:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            image_format = (image.format or "").upper()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError("Response is not a readable image") from error
    extension = Image.registered_extensions()
    extensions = [ext.lower() for ext, fmt in extension.items() if fmt.upper() == image_format]
    if not extensions:
        raise ValueError(f"Unsupported image format: {image_format or 'unknown'}")
    return ".jpg" if image_format == "JPEG" else sorted(extensions, key=len)[0]


def _download_front_image(
    session: requests.Session,
    url: str,
    barcode: str,
    image_directory: Path,
) -> Path:
    response = _request_with_retries(session, url, stream=True)
    try:
        content_type = response.headers.get("Content-Type", "").lower()
        if content_type and not content_type.startswith("image/"):
            raise ValueError(f"Image URL returned content type {content_type!r}")
        chunks = []
        size = 0
        max_image_bytes = 20 * 1024 * 1024
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            size += len(chunk)
            if size > max_image_bytes:
                raise ValueError("Image exceeds the 20 MB safety limit")
            chunks.append(chunk)
        content = b"".join(chunks)
    finally:
        response.close()
    if not content:
        raise ValueError("Image response was empty")
    suffix = _verified_image_format(content)
    image_directory.mkdir(parents=True, exist_ok=True)
    destination = image_directory / f"{barcode}{suffix}"
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(content)
    try:
        _verified_image_format(temporary.read_bytes())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _existing_verified_image(image_directory: Path, barcode: str) -> Path | None:
    for candidate in image_directory.glob(f"{barcode}.*"):
        if candidate.suffix == ".part" or not candidate.is_file():
            continue
        try:
            _verified_image_format(candidate.read_bytes())
        except (OSError, ValueError):
            candidate.unlink(missing_ok=True)
            continue
        return candidate
    return None


def _write_catalogue(output_csv: Path, rows: list[dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output_csv.with_suffix(".csv.part")
    with temporary_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(output_csv)


def _fetch_page(session: requests.Session, page: int, page_size: int) -> list[dict[str, Any]]:
    params = {
        "countries_tags_en": "india",
        "page": page,
        "page_size": page_size,
        "fields": "code,product_name,brands,categories,quantity,image_front_url,image_front_small_url,image_front_thumb_url,url",
    }
    response = _request_with_retries(session, API_URL, params=params)
    try:
        payload = response.json()
    finally:
        response.close()
    products = payload.get("products", []) if isinstance(payload, dict) else []
    if not isinstance(products, list):
        raise ValueError(f"Unexpected Open Food Facts response on page {page}")
    return [product for product in products if isinstance(product, dict)]


def _make_row(product: dict[str, Any], barcode: str, image_path: Path, data_dir: Path) -> dict[str, str]:
    try:
        relative_image_path = image_path.relative_to(data_dir).as_posix()
    except ValueError as error:
        raise ValueError("Downloaded product image is outside the data directory") from error
    return {
        "sku": barcode,
        "product_name": _text(product.get("product_name")),
        "brand": _text(product.get("brands")),
        "category": _text(product.get("categories")),
        "quantity": _text(product.get("quantity")),
        "expected_price": "",
        "barcode": barcode,
        "front_image_url": _image_url(product),
        "product_url": _product_page_url(product),
        "image_local_path": relative_image_path,
    }


def ingest_products(
    *,
    max_products: int = MAX_PRODUCTS,
    data_dir: Path = DATA_DIR,
    session: requests.Session | None = None,
    page_size: int = PAGE_SIZE,
    max_pages: int | None = None,
    request_pause: float = 1.0,
) -> IngestionResult:
    """Fetch up to max_products valid products and export verified local images and CSV rows."""
    if not 1 <= max_products <= MAX_PRODUCTS:
        raise ValueError(f"max_products must be between 1 and {MAX_PRODUCTS}")
    if page_size < 1:
        raise ValueError("page_size must be at least 1")
    data_dir = Path(data_dir).resolve()
    image_directory = data_dir / "product_images"
    output_csv = data_dir / "catalogue.csv"
    owns_session = session is None
    session = session or requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json,image/*"})

    rows: list[dict[str, str]] = []
    seen_barcodes: set[str] = set()
    failed_downloads = 0
    page = 1
    consecutive_page_errors = 0
    try:
        while len(rows) < max_products and (max_pages is None or page <= max_pages):
            try:
                products = _fetch_page(session, page, page_size)
                consecutive_page_errors = 0
            except (requests.RequestException, ValueError) as error:
                consecutive_page_errors += 1
                LOGGER.warning("Open Food Facts page %d failed (%d consecutive): %s", page, consecutive_page_errors, error)
                if consecutive_page_errors >= 3:
                    LOGGER.error("Stopping after three consecutive API page failures; retaining collected products")
                    break
                page += 1
                if request_pause > 0:
                    time.sleep(request_pause)
                continue
            if not products:
                break
            for product in products:
                barcode = str(product.get("code", "")).strip()
                if barcode in seen_barcodes or not is_valid_gtin(barcode):
                    continue
                image_url = _image_url(product)
                if not image_url:
                    continue
                image_path = _existing_verified_image(image_directory, barcode)
                if image_path is None:
                    try:
                        image_path = _download_front_image(session, image_url, barcode, image_directory)
                    except (requests.RequestException, ValueError, OSError) as error:
                        failed_downloads += 1
                        LOGGER.warning("Could not download front image for barcode %s: %s", barcode, error)
                        continue
                seen_barcodes.add(barcode)
                rows.append(_make_row(product, barcode, image_path, data_dir))
                if len(rows) >= max_products:
                    break
            _write_catalogue(output_csv, rows)
            page += 1
            if len(rows) < max_products and products and request_pause > 0:
                time.sleep(request_pause)
    finally:
        if owns_session:
            session.close()

    _write_catalogue(output_csv, rows)

    return IngestionResult(
        valid_products=len(rows),
        downloaded_images=len(rows),
        failed_downloads=failed_downloads,
        output_csv=output_csv,
        image_directory=image_directory,
    )


def verify_catalogue(data_dir: Path = DATA_DIR) -> dict[str, int]:
    """Verify every exported barcode is unique and every local image exists and decodes."""
    data_dir = Path(data_dir).resolve()
    catalogue_path = data_dir / "catalogue.csv"
    with catalogue_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    barcodes: set[str] = set()
    errors = []
    verified_images = 0
    for row_number, row in enumerate(rows, start=2):
        barcode = row.get("barcode", "")
        if not is_valid_gtin(barcode):
            errors.append(f"row {row_number}: invalid barcode {barcode!r}")
        if barcode in barcodes:
            errors.append(f"row {row_number}: duplicate barcode {barcode!r}")
        barcodes.add(barcode)
        relative_path = row.get("image_local_path", "")
        image_path = (data_dir / relative_path).resolve()
        if image_path.parent != (data_dir / "product_images").resolve():
            errors.append(f"row {row_number}: image path is not in product_images")
            continue
        if not image_path.is_file():
            errors.append(f"row {row_number}: missing image {relative_path!r}")
            continue
        try:
            _verified_image_format(image_path.read_bytes())
        except (OSError, ValueError) as error:
            errors.append(f"row {row_number}: unreadable image {relative_path!r}: {error}")
            continue
        verified_images += 1
    if errors:
        raise ValueError("Catalogue verification failed: " + "; ".join(errors[:20]))
    image_files = sum(path.is_file() for path in (data_dir / "product_images").glob("*") if path.suffix != ".part")
    return {
        "valid_products": len(rows),
        "unique_barcodes": len(barcodes),
        "verified_images": verified_images,
        "image_files": image_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=MAX_PRODUCTS, help="maximum valid products to collect (1–200)")
    parser.add_argument("--max-pages", type=int, default=None, help="optional page limit for bounded runs")
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE, help="products requested per API page")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between pages to respect API rate limits")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = ingest_products(
        max_products=args.limit,
        max_pages=args.max_pages,
        page_size=args.page_size,
        request_pause=args.pause,
    )
    verification = verify_catalogue()
    LOGGER.info("Valid products exported: %d", result.valid_products)
    LOGGER.info("Verified images downloaded: %d", result.downloaded_images)
    LOGGER.info("Failed image downloads: %d", result.failed_downloads)
    LOGGER.info("Catalogue verification: %d products, %d unique barcodes, %d readable images, %d image files", verification["valid_products"], verification["unique_barcodes"], verification["verified_images"], verification["image_files"])
    LOGGER.info("Catalogue CSV: %s", result.output_csv)
    LOGGER.info("Image directory: %s", result.image_directory)


if __name__ == "__main__":
    main()
