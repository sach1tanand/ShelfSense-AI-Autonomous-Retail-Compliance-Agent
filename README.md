# ShelfSense AI

A lightweight Flask dashboard for reviewing shelf images against a store catalogue and planogram. It includes product-count compliance checks, price-checking hooks, and downloadable CSV reports.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the website with `python app.py` and open http://127.0.0.1:5000.

The dashboard, catalogue, and planogram work without an inference model. To analyze images, install the optional `ultralytics` package and provide a compatible YOLO weights file using the `SHELFSENSE_MODEL` environment variable. The model's class names should correspond to the `model_label` values in `data/catalogue.csv`. No model weights are bundled. Until a model is configured, the site explains that image analysis is unavailable instead of presenting invented detections.

## Import Open Food Facts products

Run `python -m src.off_ingestion` to fetch up to 200 Indian packaged-food products with valid barcodes and readable front images. Product images are stored under `data/product_images/`, and `data/catalogue.csv` is replaced with the ingested product data and local image paths. Use `--limit`, `--page-size`, and `--pause` to adjust the collection size or request pacing. The command retries transient API errors, checkpoints collected pages, and validates exported image paths and image files.

## Project layout

- `app.py`: Flask routes, image upload, and report downloads.
- `config/settings.py`: paths and upload configuration.
- `src/detector.py`: optional YOLO inference adapter.
- `src/compliance.py`: compares detected products with expected facings.
- `src/price_checker.py`: compares observed prices when supplied by an OCR integration.
- `src/report.py`: report summary generation.
- `src/off_ingestion.py`: Open Food Facts product and front-image ingestion.
- `src/planogram.py`: planogram validation and conservative slot audit findings.
- `data/`: editable example catalogue and planogram.
- `tests/`: automated compliance tests.

Reports are kept in application memory and are cleared when the server restarts. Uploaded images are removed after analysis. This starter does not include OCR; price values are marked as unverified until an OCR source supplies observed prices.

The planogram uses `store_id,planogram_id,shelf,row,slot,expected_sku,facing_count`. The audit validates slot uniqueness, catalogue SKUs, and facing counts. Product detector boxes alone do not prove exact physical shelf-slot mapping. Until a trusted slot-mapping implementation is explicitly provided to the planogram audit, expected slots are labelled `Needs Review`; intentionally empty slots are separately identified. A future mapper must return the exact store/planogram/shelf/row/slot identity and return no match when uncertain.

## Tests

Run `python -m unittest discover -s tests`.