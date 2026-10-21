# ShelfSense AI

See [docs/DEVELOPER_REPORT.md](docs/DEVELOPER_REPORT.md) for architecture, measured accuracy, limitations, and how to replace the model.

A lightweight Flask dashboard for reviewing shelf images against a store catalogue and planogram. It includes product-count compliance checks, price-checking hooks, and downloadable CSV reports.

## Run locally (Windows PowerShell)

```powershell
cd D:\ShelfSense-AI\ShelfSense-AI-Autonomous-Retail-Compliance-Agent
py -3.11 -m venv .venv                      # skip if .venv already exists
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000. The detector preloads in the background at start-up (about 10 s on first load); `GET /api/health?load=1` reports whether it is ready.

Workflow: **Upload & Analyze** (or **Use Example**) → real detector inference → annotated image with boxes and confidences → optional CLIP SKU candidates → equal-grid slot proposals → planogram comparison → report saved to `data/audit_results/<id>.json` → listed under **Reports** after restarts. **Run Product Detection** on the dashboard is a detection-only check that does not save a report.

## Detection model (temporary)

The default detector is `models/external/sku110k-yolov8n.pt`, a YOLOv8n checkpoint trained on the SKU-110K retail-shelf dataset. It has a single class, `object`: it finds product facings but does not identify SKUs itself. SKU identity comes from catalogue image matching (below), and every report lists its method and limitations. Detection labels are model classes, kept separate from catalogue SKUs.

Configuration (all optional environment variables):

| Variable | Default | Purpose |
|---|---|---|
| `SHELFSENSE_MODEL` | `models/external/sku110k-yolov8n.pt` | Detector weights (any Ultralytics-loadable detection model: `.pt`, `.onnx`, ...) |
| `SHELFSENSE_CONFIDENCE` | `0.35` | Default confidence threshold (the upload form can override it per run) |
| `SHELFSENSE_MAX_DETECTIONS` | `1000` | Max boxes per image (dense shelves exceed YOLO's default of 300) |
| `SHELFSENSE_NMS_IOU` | `0.5` | Duplicate-box suppression overlap |
| `SHELFSENSE_MATCHER` | MobileNetV3 | SKU matcher model id, or `none` |
| `SHELFSENSE_DEVICE` | auto | e.g. `cpu`, `0` for the first CUDA GPU |
| `SHELFSENSE_RESULTS_DIR` | `data/audit_results` | Where audit reports are saved |
| `SHELFSENSE_UPLOAD_DIR` | `uploads` | Stored shelf images and annotated copies |
| `SHELFSENSE_PORT` / `SHELFSENSE_HOST` | `5000` / `127.0.0.1` | Server address |

### Replacing the model with your trained weights

The app depends only on the interface in `src/detector.py` (`detect_products`, `detector_status`, `annotate_image`), never on a checkpoint path. To switch models:

1. Train an Ultralytics YOLO **detection** model (YOLOv8/YOLO11) and copy `best.pt` to, for example, `models\shelfsense-best.pt`.
2. Point the app at it and restart:
   ```powershell
   $env:SHELFSENSE_MODEL = "D:\ShelfSense-AI\ShelfSense-AI-Autonomous-Retail-Compliance-Agent\models\shelfsense-best.pt"
   .\.venv\Scripts\python.exe app.py
   ```
3. Check `http://127.0.0.1:5000/api/health?load=1`: `model.class_names` should list your classes and `load_error` should be `null`.
4. If your classes are product identities, set each catalogue row's `model_label` to the matching class name (case-insensitive). A detection whose label resolves to exactly one catalogue SKU is then treated as that SKU in the planogram audit; anything else stays `Needs Review`.

To use a different framework entirely, add a class with the same `detect(image_path, confidence) -> list[dict]` and `info()` methods as `YoloDetector` and return it from `get_detector()`. Each detection dict must contain `label`, `confidence`, `box` (`[x1, y1, x2, y2]` in pixels), `image_width`, and `image_height`.

### SKU matching

Each detected box is cropped and compared with the 200 catalogue images in `data/product_images/` using ImageNet-pretrained MobileNetV3-Large features (`SHELFSENSE_MATCHER`, default `torchvision-mobilenet_v3_large-imagenet1k_v2`; weights 21 MB, BSD-3, downloaded once by torchvision). A crop is accepted as a SKU only above the threshold and top-2 margin in `data/clip_calibration.json`; otherwise it stays `Needs Review`. `SHELFSENSE_MATCHER=none` disables matching; a CLIP model id works after `pip install -r requirements-matching.txt` (recalibrate after switching).

### Slot mapping

Slot findings (Compliant, Missing Expected Product, Misplaced, Unexpected Product) are only confirmed when the photo is declared to follow the **calibrated layout**: a fixed-camera front view where the planogram rows and slots are equal bands of the image ("Calibrated shelf layout" checkbox, or `calibrated_layout=on` in the API). Otherwise slots stay `Needs Review`. A slot holding an object the matcher could not identify is never confirmed.

### Price OCR

With "Price label verification" ticked, RapidOCR (PP-OCR on ONNX Runtime, Apache-2.0) reads the whole image. Only price-like text is kept: a ₹/Rs/MRP marker or a number with two decimals. Each tag is linked to the slot's planogram SKU (calibrated layout) or to the identified product directly above it (heuristic), then compared with the catalogue `expected_price`. Products without a linked tag stay `Unverified`. The 13 planogram SKUs carry demo prices in `data/catalogue.csv`: replace them with your store's prices.

### Labelled test set

`data/test_set/` holds synthetic shelf scenes built from the catalogue images, with exact ground truth:

- `test/`: 6 scenes on the planogram grid (compliant, misplaced, missing, unexpected, price mismatch, mixed), using the 13 planogram SKUs plus 3 distractors
- `validation/`: 8 scenes using 80 *other* products (60 known, 20 removed from the gallery as "unknown"). These are used only to calibrate the match threshold.

```powershell
.\.venv\Scripts\python.exe -m tools.build_test_set          # regenerate scenes + ground_truth.json
.\.venv\Scripts\python.exe -m tools.calibrate_matching      # threshold for >=98% precision on validation crops
.\.venv\Scripts\python.exe -m tools.evaluate_test_set       # score the full pipeline on the test scenes
.\.venv\Scripts\python.exe -m tools.evaluate_test_set --save  # also add the audits to the Reports history
```

The test scenes also appear as buttons on **Upload & Analyze**. They are synthetic: accuracy on real store photos will be lower, and real photos need the calibrated-layout declaration to be true before slot findings mean anything.

### Windows note

Torch 2.5.1 (CPU) is pinned because Windows Smart App Control blocked the DLLs of newer torch builds on the development machine. CUDA builds that support recent GPUs (for example the RTX 50 series) need torch 2.7 or newer. Install one only if Smart App Control allows it, then set `SHELFSENSE_DEVICE=0`.

## API

- `GET /api/health`: model, catalogue, planogram, and storage checks (`?load=1` loads the model)
- `POST /api/analyze`: multipart `shelf_image` (plus optional `confidence_threshold`, `store`, `planogram`, `price_check`); returns the report summary (201) or `{"error": ...}`
- `GET /api/reports`, `GET /api/reports/<id>`: saved audits as JSON

## Import Open Food Facts products

Run `python -m src.off_ingestion` to fetch up to 200 Indian packaged-food products with valid barcodes and readable front images. Product images are stored under `data/product_images/`, and `data/catalogue.csv` is replaced with the ingested product data and local image paths. Use `--limit`, `--page-size`, and `--pause` to adjust the collection size or request pacing. The command retries transient API errors, checkpoints collected pages, and validates exported image paths and image files.

## Project layout

- `app.py`: Flask routes, image upload, and report downloads.
- `config/settings.py`: paths and upload configuration.
- `src/detector.py`: replaceable detector interface (Ultralytics YOLO backend), box annotation.
- `src/compliance.py`: compares detected products with expected facings.
- `src/price_checker.py`: compares observed prices when supplied by an OCR integration.
- `src/report.py`: report summary generation.
- `src/off_ingestion.py`: Open Food Facts product and front-image ingestion.
- `src/planogram.py`: planogram validation and conservative slot audit findings.
- `data/`: editable example catalogue and planogram.
- `tests/`: automated compliance tests.

Audit reports are saved as JSON in `data/audit_results/` and reloaded at start-up. Uploaded images are re-encoded (upright, EXIF stripped) and kept with their annotated copy in `uploads/` so saved reports can show them; delete files there to free space. Uploads are checked to be readable PNG/JPG/WEBP images of at most 12 MB and 40 megapixels. 

The planogram uses `store_id,planogram_id,shelf,row,slot,expected_sku,facing_count`. The audit validates slot uniqueness, catalogue SKUs, and facing counts. Product detector boxes alone do not prove exact physical shelf-slot mapping. Until a trusted slot-mapping implementation is explicitly provided to the planogram audit, expected slots are labelled `Needs Review`; intentionally empty slots are separately identified. A future mapper must return the exact store/planogram/shelf/row/slot identity and return no match when uncertain.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

`tests/test_end_to_end.py` includes a test that runs the real detector on `data/samples/real_supermarket_shelf.jpg`.