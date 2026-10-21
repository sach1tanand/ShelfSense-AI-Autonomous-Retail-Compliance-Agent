"""Application paths and upload settings."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = Path(os.getenv("SHELFSENSE_UPLOAD_DIR", ROOT_DIR / "uploads"))
MODEL_PATH = os.getenv("SHELFSENSE_MODEL", str(ROOT_DIR / "models" / "external" / "sku110k-yolov8n.pt"))
MAX_CONTENT_LENGTH = 12 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
DETECTION_CONFIDENCE = float(os.getenv("SHELFSENSE_CONFIDENCE", "0.35"))