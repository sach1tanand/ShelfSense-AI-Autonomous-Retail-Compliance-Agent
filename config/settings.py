"""Application paths, upload settings, and detector configuration.

Every value can be overridden with an environment variable so the detector
checkpoint can be swapped without code changes (see README "Replacing the model").
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = Path(os.getenv("SHELFSENSE_UPLOAD_DIR", ROOT_DIR / "uploads"))
AUDIT_RESULTS_DIR = Path(os.getenv("SHELFSENSE_RESULTS_DIR", DATA_DIR / "audit_results"))
SAMPLE_IMAGE_PATH = DATA_DIR / "samples" / "real_supermarket_shelf.jpg"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "external" / "shelfsense-product-detector.pt"
MODEL_PATH = os.getenv("SHELFSENSE_MODEL") or str(DEFAULT_MODEL_PATH)
DEFAULT_PRICE_TAG_VOID_MODEL_PATH = ROOT_DIR / "models" / "external" / "price-tag-void.pt"
PRICE_TAG_VOID_MODEL_PATH = os.getenv("SHELFSENSE_PRICE_TAG_VOID_MODEL") or str(DEFAULT_PRICE_TAG_VOID_MODEL_PATH)
PRICE_TAG_VOID_CONFIDENCE = float(os.getenv("SHELFSENSE_PRICE_TAG_VOID_CONFIDENCE", "0.25"))
MODEL_DEVICE = os.getenv("SHELFSENSE_DEVICE") or None
MAX_DETECTIONS = int(os.getenv("SHELFSENSE_MAX_DETECTIONS", "1000"))
# NMS IoU: 0.5 removes duplicate boxes on one product; adjacent facings rarely overlap this much.
NMS_IOU = float(os.getenv("SHELFSENSE_NMS_IOU", "0.5"))
# SKU matcher: torchvision-mobilenet_v3_large-imagenet1k_v2 (default), a CLIP model id, or "none".
MATCHER_MODEL_ID = os.getenv("SHELFSENSE_MATCHER", "torchvision-mobilenet_v3_large-imagenet1k_v2")
SKU_MATCH_THRESHOLD = float(os.getenv("SHELFSENSE_MATCH_THRESHOLD", "0.40"))
TEST_SET_DIR = DATA_DIR / "test_set"
MAX_CONTENT_LENGTH = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
DETECTION_CONFIDENCE = float(os.getenv("SHELFSENSE_CONFIDENCE", "0.35"))
SECRET_KEY = os.getenv("SHELFSENSE_SECRET_KEY", "shelfsense-local-dev-key")
