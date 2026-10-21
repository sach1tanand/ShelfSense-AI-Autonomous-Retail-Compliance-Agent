"""Shelf price-tag OCR using RapidOCR (PP-OCR models on ONNX Runtime, Apache-2.0).

Only text that looks like a price is kept: a currency marker (₹, Rs, MRP) followed by a
number, or a number with exactly two decimals (e.g. "45.00"). Plain numbers such as pack
sizes ("500 g") are ignored to avoid reading weights as prices.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

PRICE_PATTERN = re.compile(
    r"(?:(?:₹|rs\.?|inr|mrp)\s*:?\s*(?P<marked>\d{1,5}(?:[.,]\d{1,2})?))|(?P<decimal>\d{1,5}[.,]\d{2})(?!\d)",
    re.IGNORECASE,
)


# Shelf tags often print the rupee sign as a symbol OCR drops ("'85"); such bare numbers are
# accepted only when the text sits on a yellow label (see _on_yellow_label).
BARE_NUMBER = re.compile(r"[₹'`’]?\s*(\d{2,5})")


def _on_yellow_label(region) -> bool:
    """True when the median colour of an RGB numpy region looks like a yellow price sticker."""
    import numpy as np

    if region.size == 0:
        return False
    red, green, blue = (float(value) for value in np.median(region.reshape(-1, 3), axis=0))
    return red > 140 and green > 110 and blue < 0.7 * min(red, green)


class OcrUnavailable(RuntimeError):
    """Raised when the OCR runtime cannot be loaded."""


def parse_price(text: str) -> float | None:
    """Return the first price-like value in text, or None."""
    match = PRICE_PATTERN.search(text.replace(" ", " "))
    if not match:
        return None
    raw = (match.group("marked") or match.group("decimal")).replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if 0 < value < 100000 else None


class PriceTagReader:
    engine = "rapidocr-onnxruntime"

    def __init__(self):
        self._ocr = None
        self._lock = threading.Lock()

    def _load(self):
        if self._ocr is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except Exception as error:
                raise OcrUnavailable(
                    f"Price OCR requires rapidocr_onnxruntime (see requirements.txt): {error}"
                ) from error
            self._ocr = RapidOCR()
        return self._ocr

    def read_prices(self, image_path: str | Path, min_confidence: float = 0.5) -> list[dict]:
        """Return price tags as {text, price, confidence, box: [x1, y1, x2, y2]} in image pixels."""
        import numpy as np
        from PIL import Image

        with Image.open(image_path) as opened:
            image = np.array(opened.convert("RGB"))
        with self._lock:
            results, _ = self._load()(image)
        tags = []
        for polygon, text, confidence in results or []:
            confidence = float(confidence)
            if confidence < min_confidence:
                continue
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            box = [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]
            price, rule = parse_price(str(text)), "currency marker or two decimals"
            if price is None:
                bare = BARE_NUMBER.fullmatch(str(text).strip())
                region = image[int(box[1]):int(box[3]) + 1, int(box[0]):int(box[2]) + 1]
                if bare and int(bare.group(1)) > 0 and _on_yellow_label(region):
                    price, rule = float(bare.group(1)), "bare number on yellow label (inferred)"
            if price is None:
                continue
            tags.append({"text": str(text), "price": price, "confidence": round(confidence, 4), "box": box, "rule": rule})
        return tags


_reader = PriceTagReader()


def read_price_tags(image_path: str | Path, min_confidence: float = 0.5) -> list[dict]:
    return _reader.read_prices(image_path, min_confidence)
