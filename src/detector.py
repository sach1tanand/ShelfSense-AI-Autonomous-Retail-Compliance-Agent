"""Optional YOLO-based product detection adapter."""

from __future__ import annotations

from pathlib import Path

from config.settings import DETECTION_CONFIDENCE, MODEL_PATH


class DetectorUnavailable(RuntimeError):
    """Raised when image inference has not been configured."""


def detect_products(
    image_path: str | Path, confidence: float = DETECTION_CONFIDENCE
) -> list[dict]:
    """Return detected product labels, confidence, and bounding boxes."""
    if not MODEL_PATH or not Path(MODEL_PATH).is_file():
        raise DetectorUnavailable(
            "Image analysis is not configured yet. Set SHELFSENSE_MODEL to a trained YOLO weights file."
        )
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise DetectorUnavailable(
            "The model is configured, but Ultralytics is not installed. Install it to enable image analysis."
        ) from error

    model = YOLO(MODEL_PATH)
    results = model.predict(source=str(image_path), conf=confidence, verbose=False)
    detections = []
    for result in results:
        image_height, image_width = result.orig_shape
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append(
                {
                    "label": str(result.names[class_id]).strip().lower(),
                    "confidence": round(float(box.conf.item()), 4),
                    "box": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                    "image_width": image_width,
                    "image_height": image_height,
                }
            )
    return detections