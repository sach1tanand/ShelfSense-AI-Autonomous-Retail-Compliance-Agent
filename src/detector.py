"""Replaceable shelf-product detector interface.

The application depends only on ``detect_products``, ``detector_status`` and
``annotate_image``. The concrete backend is chosen from ``SHELFSENSE_MODEL``;
any checkpoint Ultralytics can load (YOLOv5u/v8/v11 ``.pt``, ``.onnx``, ...) works.

Each detection is ``{"label", "class_id", "confidence", "box", "image_width",
"image_height"}``. ``label`` is the model's own class name. It is never a SKU:
SKU identity is decided later by catalogue matching.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config.settings import (
    DETECTION_CONFIDENCE, MAX_DETECTIONS, MODEL_DEVICE, MODEL_PATH, NMS_IOU,
    PRICE_TAG_VOID_CONFIDENCE,
    PRICE_TAG_VOID_MODEL_PATH,
)


class DetectorUnavailable(RuntimeError):
    """Raised when image inference has not been configured or cannot load."""


class YoloDetector:
    """Ultralytics YOLO backend, loaded once and reused across requests."""

    backend = "ultralytics-yolo"

    def __init__(self, model_path: str | Path, device: str | None = None, max_detections: int = MAX_DETECTIONS):
        self.model_path = Path(model_path)
        self.device = device
        self.max_detections = max_detections
        self._model = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self._model is not None:
                return self._model
            if not self.model_path.is_file():
                raise DetectorUnavailable(
                    f"Detector weights were not found at {self.model_path}. Set SHELFSENSE_MODEL to a YOLO weights file."
                )
            try:
                from ultralytics import YOLO
            except Exception as error:  # ImportError, or a blocked torch DLL (OSError)
                self._load_error = f"Ultralytics/PyTorch could not be imported: {error}"
                raise DetectorUnavailable(
                    "The detector is configured, but Ultralytics/PyTorch could not be loaded. "
                    "Install the packages in requirements.txt."
                ) from error
            try:
                model = YOLO(str(self.model_path))
            except Exception as error:
                self._load_error = f"Could not load {self.model_path.name}: {error}"
                raise DetectorUnavailable(self._load_error) from error
            if getattr(model, "task", "detect") != "detect":
                self._load_error = f"{self.model_path.name} is a '{model.task}' model; an object-detection model is required."
                raise DetectorUnavailable(self._load_error)
            self._model = model
            self._load_error = None
            return model

    def detect(self, image_path: str | Path, confidence: float) -> list[dict]:
        model = self.load()
        with self._lock:
            results = model.predict(
                source=str(image_path),
                conf=confidence,
                iou=NMS_IOU,
                max_det=self.max_detections,
                device=self.device,
                verbose=False,
            )
        detections = []
        for result in results:
            image_height, image_width = result.orig_shape
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append(
                    {
                        "label": str(result.names[class_id]).strip().lower(),
                        "class_id": class_id,
                        "confidence": round(float(box.conf.item()), 4),
                        "box": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                        "image_width": int(image_width),
                        "image_height": int(image_height),
                    }
                )
        return detections

    def info(self) -> dict:
        names = getattr(self._model, "names", None) or {}
        return {
            "backend": self.backend,
            "model_path": str(self.model_path),
            "model_file": self.model_path.name,
            "file_exists": self.model_path.is_file(),
            "loaded": self._model is not None,
            "load_error": self._load_error,
            "class_names": [str(names[key]) for key in sorted(names)] if names else [],
            "class_agnostic": len(names) == 1,
            "device": self.device or "auto",
            "max_detections": self.max_detections,
            "nms_iou": NMS_IOU,
        }


_detectors: dict[str, YoloDetector] = {}
_registry_lock = threading.Lock()


def get_detector(model_path: str | Path | None = None) -> YoloDetector:
    """Return the cached detector for the configured (or given) checkpoint."""
    path = str(model_path or MODEL_PATH or "")
    if not path:
        raise DetectorUnavailable(
            "Image analysis is not configured yet. Set SHELFSENSE_MODEL to a trained YOLO weights file."
        )
    with _registry_lock:
        if path not in _detectors:
            _detectors[path] = YoloDetector(path, device=MODEL_DEVICE)
        return _detectors[path]


def detect_products(
    image_path: str | Path, confidence: float = DETECTION_CONFIDENCE
) -> list[dict]:
    """Return detected objects with model label, confidence, and xyxy bounding boxes."""
    if not MODEL_PATH or not Path(MODEL_PATH).is_file():
        raise DetectorUnavailable(
            "Image analysis is not configured yet. Set SHELFSENSE_MODEL to a trained YOLO weights file."
        )
    return get_detector(MODEL_PATH).detect(image_path, confidence)


def detect_price_tags_and_voids(image_path: str | Path, confidence: float = PRICE_TAG_VOID_CONFIDENCE) -> dict:
    """Run the separate price-tag/void model, mapping detections by checkpoint metadata."""
    if not PRICE_TAG_VOID_MODEL_PATH or not Path(PRICE_TAG_VOID_MODEL_PATH).is_file():
        raise DetectorUnavailable(
            f"Price-tag/void detector weights were not found at {PRICE_TAG_VOID_MODEL_PATH}. "
            "Set SHELFSENSE_PRICE_TAG_VOID_MODEL to its detection checkpoint."
        )
    detector = get_detector(PRICE_TAG_VOID_MODEL_PATH)
    model = detector.load()
    class_names = getattr(model, "names", {}) or {}
    items = class_names.items() if isinstance(class_names, dict) else enumerate(class_names)
    by_label = {
        str(name).strip().casefold().replace("-", "_").replace(" ", "_"): int(class_id)
        for class_id, name in items
    }
    price_tag_id = by_label.get("price_tag")
    void_id = by_label.get("void")
    if price_tag_id is None or void_id is None:
        raise DetectorUnavailable(
            f"{Path(PRICE_TAG_VOID_MODEL_PATH).name} must define Price_Tag and void classes; "
            f"checkpoint classes are {list(by_label)}."
        )
    detections = detector.detect(image_path, confidence)
    price_tags = [{**item, "kind": "price_tag"} for item in detections if item["class_id"] == price_tag_id]
    voids = [{**item, "kind": "void"} for item in detections if item["class_id"] == void_id]
    return {"price_tags": price_tags, "voids": voids, "model": detector.info()}


def detector_status() -> dict:
    """Describe the configured detector without forcing it to load."""
    if not MODEL_PATH:
        return {"backend": None, "model_path": "", "model_file": "", "file_exists": False, "loaded": False,
                "load_error": "SHELFSENSE_MODEL is not set.", "class_names": [], "class_agnostic": False}
    return get_detector(MODEL_PATH).info()


def annotate_image(image_path: str | Path, detections: list[dict], output_path: str | Path) -> Path:
    """Draw detection boxes with label and confidence onto a copy of the image."""
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) / 300))
    try:
        font = ImageFont.load_default(size=max(12, round(min(image.size) / 60)))
    except TypeError:
        font = ImageFont.load_default()
    for index, detection in enumerate(detections, start=1):
        box = detection.get("box")
        if not box or len(box) != 4:
            continue
        scale_x = image.width / float(detection.get("image_width") or image.width)
        scale_y = image.height / float(detection.get("image_height") or image.height)
        x1, y1, x2, y2 = box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y
        kind = detection.get("kind")
        # Distinct colors keep auxiliary detections visible without changing product annotations.
        colour = (236, 72, 153) if kind == "price_tag" else (139, 92, 246) if kind == "void" else (
            (245, 158, 11) if "matched_sku" in detection and not detection["matched_sku"] else (16, 185, 129)
        )
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=line_width)
        name = detection.get("product_name")
        caption = (
            f"{detection.get('label', kind)} {float(detection.get('confidence', 0)):.2f}"
            if kind else f"#{index} {name[:22] + ' ' if name else ''}{float(detection.get('confidence', 0)):.2f}"
        )
        text_box = draw.textbbox((x1, y1), caption, font=font)
        text_height = text_box[3] - text_box[1]
        top = max(0, y1 - text_height - 4)
        draw.rectangle((x1, top, x1 + (text_box[2] - text_box[0]) + 4, top + text_height + 4), fill=colour)
        draw.text((x1 + 2, top + 1), caption, fill=(255, 255, 255), font=font)
    output = Path(output_path)
    image.save(output, format="JPEG", quality=90)
    return output
