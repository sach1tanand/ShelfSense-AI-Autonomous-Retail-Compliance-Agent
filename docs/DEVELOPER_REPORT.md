# ShelfSense AI: Developer Report

Status as of 22 Oct 2026. This report describes what the system does today, how each stage works, what was measured, what is still missing, and what must change before production. In particular it covers how to replace the temporary detection model.

---

## 1. Summary

ShelfSense AI is a local Flask application that audits a shelf photo against a product catalogue (200 Indian packaged-food SKUs from Open Food Facts) and a 15-slot planogram. The full workflow runs end to end with real models:

```
upload → validate image → detect products (YOLO) → match each box to a catalogue SKU (image embeddings)
      → map boxes to planogram slots → read price tags (OCR) → compare with planogram and catalogue prices
      → annotated image + saved JSON report → dashboard / report history / CSV / JSON API
```

| Area | State |
|---|---|
| Web app (dashboard, upload, planogram, catalogue, reports, settings) | Working |
| Product detection | Working, with a **temporary** pretrained model that detects "object" only |
| SKU identification | Working via catalogue image matching, calibrated threshold |
| Slot mapping | Working only for the *calibrated layout* (fixed-camera, planogram-grid photos) |
| Price OCR | Working (RapidOCR); tag-to-product linking is reliable only in the calibrated layout |
| Report persistence and history | Working (JSON files, survive restarts) |
| Labelled test set, calibration, evaluation tooling | Working (synthetic scenes) |
| Production readiness (auth, deployment, real-photo accuracy) | **Not done** |

---

## 2. Running, testing, evaluating

```powershell
cd D:\ShelfSense-AI\ShelfSense-AI-Autonomous-Retail-Compliance-Agent
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # once
.\.venv\Scripts\python.exe app.py                                # http://127.0.0.1:5000
.\.venv\Scripts\python.exe -m pytest -q                          # 65 tests

.\.venv\Scripts\python.exe -m tools.build_test_set              # regenerate labelled scenes
.\.venv\Scripts\python.exe -m tools.calibrate_matching          # recompute SKU-match threshold
.\.venv\Scripts\python.exe -m tools.evaluate_test_set [--save]  # score the full pipeline
```

Health check: `GET /api/health?load=1` (loads the model and reports class names, load errors, data validity).

**Windows constraint:** Smart App Control on the development machine blocked the DLLs of torch 2.14. torch 2.5.1 (CPU) is pinned for that reason. Do not upgrade torch without re-checking that it imports.

---

## 3. Architecture

### 3.1 Code map

| File | Responsibility |
|---|---|
| `app.py` | Flask routes; `_run_audit()` orchestrates the pipeline; report persistence; reports analytics; JSON API |
| `config/settings.py` | All paths and environment-variable configuration |
| `src/detector.py` | **Detector interface** (`detect_products`, `detector_status`, `annotate_image`) and the `YoloDetector` backend |
| `backend/matching/encoder.py` | Image encoders: `TorchvisionEncoder` (default), `TransformersClipEncoder` (optional), `PerceptualFeatureEncoder` (tests); `create_encoder()` |
| `backend/matching/recognizer.py` | Crops boxes, embeds them, ranks catalogue candidates, applies threshold and margin |
| `backend/matching/catalog.py` | Loads catalogue reference images and caches their embeddings (`data/cache/clip_catalogue_embeddings.json`) |
| `backend/matching/shelf_mapper.py` | `CoordinateShelfMapper`: equal-grid box→slot mapping; `trusted` flag; row clustering |
| `backend/matching/planogram_compare.py` | Match-level comparison (stored in `matching_audit`) |
| `backend/matching/pipeline.py` | Runs recognizer + mapper + comparison; reads calibration |
| `src/planogram.py` | Planogram validation and the slot-status rules used for the report's compliance table |
| `src/ocr.py` | RapidOCR wrapper and price parsing rules |
| `src/price_checker.py` | Catalogue vs observed price comparison |
| `src/report.py` | Report summary and compliance-rate formula |
| `tools/` | Test-set builder, matcher calibration, end-to-end evaluation |
| `templates/`, `static/` | Server-rendered UI |

### 3.2 Pipeline stages (inside `_run_audit`)

1. **Input**: an upload, the example image, or a test-set scene. Must be a readable PNG/JPG/WEBP image, ≤ 12 MB, ≤ 40 MP, ≥ 32 px. It is re-encoded as an upright JPEG (EXIF stripped) in `uploads/`.
2. **Planogram validation** runs before any inference. Invalid planograms abort the audit.
3. **Detection**: `detect_products(path, confidence)`. Output per box: `label` (model class), `class_id`, `confidence`, `box` [x1, y1, x2, y2], `image_width`, `image_height`. NMS IoU 0.5, max 1000 boxes.
4. **SKU matching**: every box is cropped and embedded with MobileNetV3-Large (ImageNet), and compared by cosine similarity with the catalogue images. A SKU is **accepted** only if top similarity ≥ threshold **and** top-1 minus top-2 ≥ margin (from `data/clip_calibration.json`). Otherwise the box stays unidentified.
5. **Slot mapping**: box centre → (row, slot) using equal bands of the image. It is trusted only when the request declares `calibrated_layout` (or uses a test-set scene). Boxes near band edges are not mapped.
6. **Price OCR**: RapidOCR on the full image. Only price-like text is kept (see 5.3). Tags are linked to a SKU by slot (calibrated layout) or by the identified product directly above (heuristic).
7. **Planogram comparison** (`evaluate_planogram`):
   - calibrated layout: Compliant / Missing Expected Product / Misplaced / Unexpected Product / Empty Slot per slot. Any slot that contains an unidentified object is downgraded to Needs Review.
   - otherwise: every assigned slot is Needs Review.
8. **Prices**: planogram SKUs only. Match / Mismatch / Unverified (no linked tag).
9. **Report**: annotated image (catalogue names on boxes; green = identified, amber = unidentified), facing-count summary, method & limitations, JSON saved atomically to `data/audit_results/<id>.json`.

The model class (`label`) and the catalogue identity (`matched_sku`, `product_name`) are stored as separate fields on purpose. Never overwrite `label` with a SKU.

### 3.3 Status semantics

- **Compliance rate** = Compliant ÷ (Compliant + Misplaced + Missing Expected Product + Unexpected Product). Needs Review and Empty Slot are excluded; if nothing is confirmed, the rate is `None`.
- **Needs Review** means "the system cannot confirm either way". It is never counted as an issue in Reports.

### 3.4 Interfaces

HTML routes: `/`, `/upload`, `/analyze` (POST), `/detect` (GET/POST, detection only, not saved), `/planogram/*`, `/catalogue/*`, `/reports`, `/reports/<id>`, `/reports/<id>/image|annotated|download`, `/reports/download`, `/settings`.

JSON API:

| Route | Purpose |
|---|---|
| `GET /api/health[?load=1]` | model/data/storage checks |
| `POST /api/analyze` | multipart `shelf_image` or `test_image`, optional `confidence_threshold`, `store`, `planogram`, `price_check=on`, `calibrated_layout=on`; returns 201 summary or `{"error"}` with 400/413/422/500/503 |
| `GET /api/reports`, `GET /api/reports/<id>` | saved audits |

---

## 4. Data and artefacts

| Path | Content | Notes |
|---|---|---|
| `data/catalogue.csv` | 200 products (Open Food Facts) | Only the 13 planogram SKUs have `expected_price`. These are **demo prices**; replace them with real store prices. SKU 7622202243516 has no product name. |
| `data/product_images/` | 200 front images | Reference gallery for SKU matching |
| `data/planogram.csv` | 15 slots, 3 rows × 5, 2 intentionally empty | One store / planogram / shelf group |
| `data/test_set/` | 6 test + 8 validation synthetic scenes, `ground_truth.json` | Rebuild with `tools.build_test_set` |
| `data/clip_calibration.json` | Accepted-match threshold and margin + validation metrics | Name is historical; applies to any matcher whose `model_id` matches |
| `data/cache/` | Cached catalogue embeddings | Safe to delete; recomputed on demand |
| `data/audit_results/` | Saved reports (JSON) | Includes 6 "Test Set Store" audits and older reports from manual testing |
| `uploads/` | Stored photos and annotated copies | Not cleaned automatically |
| `models/external/sku110k-yolov8n.pt` | Temporary detector (24 MB) | Keep until replaced |

**Test-set split** (seeded, disjoint): test = 13 planogram SKUs + 3 distractors; validation = 60 other products; holdout "unknown" = 20 products removed from the gallery during calibration.

---

## 5. Models

### 5.1 Detector (temporary): `sku110k-yolov8n.pt`
- YOLOv8n trained on SKU-110K (dense retail shelves). One class: `object`.
- Finds product facings; does **not** identify products, price tags, shelf edges or gaps.
- CPU latency: ~70–90 ms per image warm, about 1 s on the first call. Loaded once and reused.
- Measured recall on the synthetic test scenes: **86.6%** (IoU ≥ 0.5). It consistently misses very flat, wide packs (e.g. the chocolate-bar SKU in R1S5).

### 5.2 SKU matcher: MobileNetV3-Large (ImageNet1K_V2, torchvision, 21 MB, BSD-3)
- Generic image features, not trained on retail products. It works best when the shelf product looks like its catalogue photo.
- Calibration (validation crops from the real detector): threshold **0.66**, margin **0.045** → precision **98.4%**, known-product coverage **83%**, unknown rejection **97%**.

### 5.3 OCR: RapidOCR (PP-OCR via ONNX Runtime, Apache-2.0, models bundled)
- A price is accepted when the text has a ₹/Rs/MRP marker, or a number with exactly two decimals (`45.00`).
- A bare 2–5-digit number is accepted only when its text box sits on a **yellow** label (median colour test), because real tags print ₹ as a glyph OCR drops (`'85`). These prices are marked "inferred".
- Pack sizes such as "500 g" are deliberately ignored.

---

## 6. Measured results

**Labelled synthetic test set** (`tools.evaluate_test_set`, 6 scenes, 179 facings, 90 slots, 78 tags):

| Metric | Value |
|---|---|
| Detection recall | 86.6% |
| Accepted SKU precision | 100% |
| Facings correctly identified | 80.5% |
| Slot status accuracy | 82.2% |
| Price tags read correctly | 100% |
| Price Match/Mismatch accuracy | 100% |

Slot errors come from detector misses (a missed facing reads as Missing Expected Product) and from accepted-but-unconfirmed facings (slot downgraded to Needs Review).

**Real photo (India Gate rice shelf, phone camera):**
- 55 boxes detected.
- 1 crop accepted as a SKU, and it was **wrong**: a Good Life rice bag was matched to Haldiram Aloo Bhujia (similarity 0.688). The products aren't in the catalogue, so the open-set rejection calibrated on synthetic data did not transfer.
- OCR read 4 yellow-tag prices (₹165, 85, 499, 73). None could be linked to a SKU.
- Slots: correctly left as Needs Review (layout not calibrated).

**Treat all synthetic numbers as upper bounds.** Real-photo accuracy has not been measured because no labelled real photos exist yet.

---

## 7. Replacing the temporary detection model

The application depends only on the detector interface in `src/detector.py`. Changing the model needs **no application code changes** for any Ultralytics-compatible detection checkpoint.

### 7.1 Drop-in replacement (Ultralytics YOLOv8/YOLO11 detection model)
1. Train a **detection** model (not segmentation/classification; `YoloDetector.load()` rejects other tasks).
2. Copy weights, e.g. `models\shelfsense-best.pt` (`.onnx` also works through Ultralytics).
3. `$env:SHELFSENSE_MODEL = "...\models\shelfsense-best.pt"` and restart.
4. Verify `GET /api/health?load=1`: `class_names` lists your classes, `load_error` is `null`.
5. Run `python -m tools.evaluate_test_set` and compare with section 6.
6. Retune `SHELFSENSE_CONFIDENCE` and `SHELFSENSE_NMS_IOU` on a validation set if box quality differs.

### 7.2 Which kind of model to train

| Option | What the classes are | Effect on the pipeline | Recommended when |
|---|---|---|---|
| **A. Better class-agnostic product detector** | `product` (optionally `price_tag`, `shelf_edge`, `gap`) | Keeps the matcher as SKU identifier; adding `price_tag` lets OCR run on tag crops only; `shelf_edge` enables real slot mapping | Catalogue changes often (new SKUs need only a reference image) |
| **B. SKU-level detector** | one class per SKU | Set each catalogue row's `model_label` to the class name; `evaluate_planogram` resolves labels to SKUs directly. The matcher becomes optional (it can confirm low-confidence boxes) | Fixed, small assortment with enough labelled images per SKU (dozens of facings each) |

For option B, detections already resolve through `model_label` (`src/planogram.py::_catalogue_label_index`). The remaining work is to prefer a detector SKU over the matcher result when the detector confidence is high, and to record which source decided the SKU.

### 7.3 Training data needed
- Real shelf photos from the target stores (lighting, angle, occlusion, price-tag styles), annotated in YOLO format.
- Minimum practical: a few hundred images for option A; for option B, 30–100+ labelled facings per SKU.
- Hold out a labelled **real** test split and add it to `tools/evaluate_test_set.py` (the ground-truth format is in `data/test_set/ground_truth.json`).

### 7.4 After changing any model
- **Detector changed** → rerun `tools.calibrate_matching` (crop boxes change, so similarity distributions change).
- **Matcher changed** (`SHELFSENSE_MATCHER`) → recalibrate. The calibration file is ignored when its `model_id` differs, and all matches stay advisory until recalibrated.
- **Catalogue images changed** → the embedding cache refreshes automatically; recalibrate if many products change.

### 7.5 Using a non-Ultralytics framework
Implement a class with `detect(image_path, confidence) -> list[dict]` and `info() -> dict` (same fields as `YoloDetector`) and return it from `get_detector()`. Each detection must contain `label`, `confidence`, `box` (pixels, xyxy), `image_width`, `image_height`.

---

## 8. Limitations and challenges

### Detection
- Temporary model: single class; misses thin/flat packs; occasional duplicate boxes (reduced with NMS IoU 0.5).
- No price-tag, shelf-edge or gap detection.
- CPU only. The RTX 5050 GPU needs torch ≥ 2.7 (CUDA 12.8), which Smart App Control currently blocks.

### SKU identification
- Generic ImageNet features confuse visually similar packs (colour/layout) and are weak for products not in the catalogue (open set). This was observed on a real photo.
- Calibration uses synthetic scenes made from the same catalogue photos; real-photo precision is unknown and likely lower.
- One reference image per SKU. No variants (pack redesigns, sizes, angles).

### Slot mapping
- Equal-grid mapping only. It assumes a fixed, front-facing camera where planogram rows and slots are equal image bands.
- The "calibrated layout" flag is an **operator declaration**, not verified by the system. Ticking it on an arbitrary photo produces wrong confirmed findings.
- One shelf group per planogram is supported. No perspective correction, no shelf-edge detection.

### Counting and compliance
- Counts are visible facings in one image, not stock. Occlusion and depth are invisible.
- A missed detection in a calibrated layout becomes "Missing Expected Product" (false negative finding).
- Non-planogram SKUs are reported as separate "Unexpected Product" findings, not against a slot.

### OCR / prices
- Tag-to-SKU linking is reliable only in the calibrated layout; otherwise it needs an identified product directly above the tag.
- The bare-number rule needs yellow tags; white or other coloured tags without ₹/decimals are ignored.
- Promotions, multi-buy offers and per-unit prices are not interpreted.
- Expected prices are demo values for 13 SKUs only.

### Application / operations
- Flask development server; no authentication, users or roles; single local instance.
- No upload retention policy (`uploads/` grows). Detection-only (`/detect`) results are in memory only.
- The Settings toggles **shelf-gap detection** and **low-stock alerts** are saved but have no effect (not implemented).
- Reports are JSON files without a database, so concurrent multi-process writes and querying at scale are not supported.

---

## 9. Recommended work, in priority order

1. **Collect and label real shelf photos** (with SKU boxes, slot positions and tag prices) from target stores. This is the prerequisite for everything below.
2. **Train the replacement detector** (option A with `product` + `price_tag` + `shelf_edge` classes recommended). Swap it in via `SHELFSENSE_MODEL`.
3. **Real-photo evaluation**: extend `tools/evaluate_test_set.py` to read a real labelled split; report metrics per store.
4. **Better SKU matcher**: a retail-trained or fine-tuned embedding (metric learning on shelf crops), multiple reference images per SKU, and recalibration on real crops.
5. **Real slot mapping**: shelf-edge detection + per-row segmentation (or per-store camera calibration) to replace the operator-declared equal grid, and to verify the layout automatically.
6. **OCR on tag crops** (from a `price_tag` class) instead of the full image; support white tags and promo formats; link tags by shelf-edge position.
7. **Catalogue data**: real expected prices for all SKUs; fill missing names (e.g. 7622202243516).
8. **Remove or implement** the shelf-gap and low-stock settings.
9. **Production hardening**: WSGI server (waitress on Windows), authentication, database for reports, upload retention, structured logging/monitoring, deployment packaging.
10. **GPU**: once Smart App Control policy allows newer torch, install a CUDA build and set `SHELFSENSE_DEVICE=0`.

---

## 10. Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `SHELFSENSE_MODEL` | `models/external/sku110k-yolov8n.pt` | Detector weights |
| `SHELFSENSE_CONFIDENCE` | 0.35 | Default detection confidence |
| `SHELFSENSE_NMS_IOU` | 0.5 | Duplicate-box suppression |
| `SHELFSENSE_MAX_DETECTIONS` | 1000 | Max boxes per image |
| `SHELFSENSE_DEVICE` | auto | `cpu` or CUDA index |
| `SHELFSENSE_MATCHER` | `torchvision-mobilenet_v3_large-imagenet1k_v2` | SKU matcher id, a CLIP id, or `none` |
| `SHELFSENSE_MATCH_TOP_K` | 5 | Candidates kept per crop |
| `SHELFSENSE_RESULTS_DIR` | `data/audit_results` | Report storage |
| `SHELFSENSE_UPLOAD_DIR` | `uploads` | Stored images |
| `SHELFSENSE_HOST` / `SHELFSENSE_PORT` | 127.0.0.1 / 5000 | Server address |
| `SHELFSENSE_SECRET_KEY` | dev key | Flask session key (set in any shared deployment) |
| `SHELFSENSE_LOG_LEVEL` | INFO | Logging |

---

## 11. Dependencies and licences

| Package | Version | Licence | Note |
|---|---|---|---|
| Flask | 3.1 | BSD-3 | |
| Ultralytics | 8.4 | **AGPL-3.0** | Commercial/closed deployment needs an Ultralytics Enterprise licence or a different detector framework |
| torch / torchvision | 2.5.1 / 0.20.1 CPU | BSD-3 | Pinned (Smart App Control) |
| MobileNetV3 weights | IMAGENET1K_V2 | BSD-3 | Downloaded to `~/.cache/torch` on first use |
| rapidocr_onnxruntime / onnxruntime | 1.4 / 1.31 | Apache-2.0 / MIT | PP-OCR models bundled |
| SKU-110K YOLOv8n checkpoint | n/a | Check the origin of the checkpoint and the SKU-110K dataset terms (research use) before commercial use | |
| Catalogue data/images | n/a | Open Food Facts (ODbL data, CC BY-SA images) | Attribution required |
