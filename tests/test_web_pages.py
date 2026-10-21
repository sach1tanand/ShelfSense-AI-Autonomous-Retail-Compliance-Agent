import csv
import io
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import app as web_app
from PIL import Image


def jpeg_bytes(size=(64, 64), color=(200, 120, 40)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


class WebPageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        temp_root = Path(self.temp_dir.name)
        self.catalogue_path = temp_root / "catalogue.csv"
        self.settings_path = temp_root / "user_settings.json"
        self.catalogue_path.write_text(
            "sku,name,category,brand,pack_size,price,model_label,active\n"
            "A1,Oat Milk,Plant-based,Northstar,1 L,4.00,oat milk,true\n",
            encoding="utf-8",
        )
        (temp_root / "planogram.csv").write_text(
            "shelf,position,sku,expected_facings,min_price\nTop,1,A1,1,4.00\n",
            encoding="utf-8",
        )
        self.catalogue_patch = patch.object(web_app, "CATALOGUE_PATH", self.catalogue_path)
        self.data_patch = patch.object(web_app, "DATA_DIR", temp_root)
        self.settings_patch = patch.object(web_app, "SETTINGS_PATH", self.settings_path)
        self.catalogue_patch.start()
        self.data_patch.start()
        self.settings_patch.start()
        self.addCleanup(self.catalogue_patch.stop)
        self.addCleanup(self.data_patch.stop)
        self.addCleanup(self.settings_patch.stop)
        self.results_dir = temp_root / "audit_results"
        self.upload_dir = temp_root / "default_uploads"
        self.upload_dir.mkdir()
        for name, value in (("AUDIT_RESULTS_DIR", self.results_dir), ("UPLOAD_DIR", self.upload_dir)):
            extra_patch = patch.object(web_app, name, value)
            extra_patch.start()
            self.addCleanup(extra_patch.stop)
        self.old_reports = web_app._reports.copy()
        web_app._reports.clear()
        self.addCleanup(lambda: (web_app._reports.clear(), web_app._reports.update(self.old_reports)))
        web_app.app.config["TESTING"] = True
        self.client = web_app.app.test_client()

    def test_all_navigation_pages_render(self):
        for route in ("/", "/upload", "/planogram", "/catalogue", "/reports", "/settings"):
            with self.subTest(route=route):
                response = self.client.get(route)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"Product Catalogue" if route == "/catalogue" else b"ShelfSense AI", response.data)

    def test_add_product_updates_catalogue(self):
        response = self.client.post(
            "/catalogue/add",
            data={
                "sku": "B2",
                "name": "Granola",
                "category": "Pantry",
                "brand": "Meadow",
                "pack_size": "400 g",
                "price": "6.25",
                "model_label": "granola",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Granola was added", response.data)
        with self.catalogue_path.open(encoding="utf-8", newline="") as csv_file:
            products = list(csv.DictReader(csv_file))
        self.assertEqual([product["sku"] for product in products], ["A1", "B2"])

    def test_catalogue_import_merges_products_by_sku(self):
        csv_upload = io.BytesIO(
            b"sku,name,category,brand,pack_size,price,model_label\n"
            b"B2,Granola,Pantry,Meadow,400 g,6.25,granola\n"
        )
        response = self.client.post(
            "/catalogue/import",
            data={"catalogue_file": (csv_upload, "products.csv")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Imported 1 catalogue products", response.data)
        self.assertIn("B2", self.catalogue_path.read_text(encoding="utf-8"))

    def test_catalogue_page_displays_local_product_image(self):
        image_dir = Path(self.temp_dir.name) / "product_images"
        image_dir.mkdir()
        image_path = image_dir / "8901234567890.png"
        image = Image.new("RGB", (3, 3), "blue")
        image.save(image_path, format="PNG")
        self.catalogue_path.write_text(
            "sku,product_name,brand,category,quantity,expected_price,barcode,front_image_url,product_url,image_local_path\n"
            "8901234567890,Test Product,Brand,Snacks,50 g,,8901234567890,,,product_images/8901234567890.png\n",
            encoding="utf-8",
        )

        page = self.client.get("/catalogue")
        image_response = self.client.get("/catalogue/image/8901234567890")

        self.assertEqual(page.status_code, 200)
        self.assertIn(b"catalogue-product-image", page.data)
        self.assertIn(b"/catalogue/image/8901234567890", page.data)
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(image_response.mimetype, "image/png")
        self.assertEqual(image_response.data, image_path.read_bytes())

    def test_catalogue_includes_ten_item_pagination(self):
        original_contents = self.catalogue_path.read_text(encoding="utf-8")
        with self.catalogue_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["sku", "name", "category", "brand", "pack_size", "price", "model_label", "active"])
            for product_number in range(1, 24):
                writer.writerow([f"SKU{product_number}", f"Product {product_number}", "Snacks", "Brand", "50 g", "2.50", f"product {product_number}", "true"])
        self.addCleanup(self.catalogue_path.write_text, original_contents, encoding="utf-8")

        response = self.client.get("/catalogue")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"catalogue-pagination", response.data)
        self.assertIn(b"pageSize=10", response.data)
        self.assertIn(b"Showing 10 of 23 products", response.data)

    def test_catalogue_image_route_rejects_unknown_barcode(self):
        response = self.client.get("/catalogue/image/unknown")

        self.assertEqual(response.status_code, 404)

    def test_catalogue_edit_rejects_missing_required_fields(self):
        response = self.client.post(
            "/catalogue/A1/edit",
            data={"name": "", "category": "Plant-based", "price": "4.00", "model_label": "oat milk", "active": "on"},
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Product name, category, price, and model label are required", response.data)
        self.assertIn("Oat Milk", self.catalogue_path.read_text(encoding="utf-8"))

    def test_planogram_row_and_slot_changes_persist_to_csv(self):
        add_row = self.client.post("/planogram/row/add", data={"shelf": "Eye Level"})
        assign_slot = self.client.post(
            "/planogram/slot/add",
            data={"shelf": "Top", "position": "2", "sku": "A1", "expected_facings": "3"},
            follow_redirects=True,
        )

        self.assertEqual(add_row.status_code, 302)
        self.assertEqual(assign_slot.status_code, 200)
        self.assertIn(b"Oat Milk was assigned", assign_slot.data)
        with (Path(self.temp_dir.name) / "planogram.csv").open(encoding="utf-8", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        self.assertEqual(len(rows), 7)
        assigned = next(row for row in rows if row["shelf"] == "Top" and row["position"] == "2")
        self.assertEqual(assigned["sku"], "A1")
        self.assertEqual(assigned["expected_facings"], "3")

    def test_planogram_rejects_duplicate_shelf_row(self):
        response = self.client.post(
            "/planogram/row/add",
            data={"shelf": "top"},
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"That shelf row already exists", response.data)

    def test_current_planogram_schema_has_fifteen_valid_slots(self):
        catalogue_rows = [
            {"sku": f"890000000000{i}", "name": f"Product {i}", "category": "Food", "price": "", "model_label": f"product {i}", "active": "true"}
            for i in range(1, 14)
        ]
        planogram_rows = []
        for slot in range(1, 16):
            product_index = slot if slot <= 13 else None
            planogram_rows.append({
                "store_id": "STORE001",
                "planogram_id": "SNACKS_V1",
                "shelf": "S1",
                "row": str((slot - 1) // 5 + 1),
                "slot": str((slot - 1) % 5 + 1),
                "expected_sku": f"890000000000{product_index}" if product_index else "",
                "facing_count": "2" if product_index else "0",
            })
        with self.catalogue_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(catalogue_rows[0]))
            writer.writeheader()
            writer.writerows(catalogue_rows)
        with (Path(self.temp_dir.name) / "planogram.csv").open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["store_id", "planogram_id", "shelf", "row", "slot", "expected_sku", "facing_count"])
            writer.writeheader()
            writer.writerows(planogram_rows)

        response = self.client.get("/planogram")
        normalized_planogram = web_app._read_planogram()
        validation_errors = web_app.validate_planogram(normalized_planogram, web_app._read_catalogue())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(normalized_planogram), 15)
        self.assertEqual(sum(not row["sku"] for row in normalized_planogram), 2)
        self.assertEqual(validation_errors, [])

    def test_planogram_grid_renders_local_product_package_image(self):
        barcode = "8901234567890"
        image_dir = Path(self.temp_dir.name) / "product_images"
        image_dir.mkdir()
        image_path = image_dir / f"{barcode}.png"
        Image.new("RGB", (4, 6), "orange").save(image_path, format="PNG")
        self.catalogue_path.write_text(
            "sku,product_name,brand,category,quantity,expected_price,barcode,front_image_url,product_url,image_local_path\n"
            f"{barcode},Test Snack,Test Brand,Snacks,50 g,,{barcode},,,product_images/{barcode}.png\n",
            encoding="utf-8",
        )
        (Path(self.temp_dir.name) / "planogram.csv").write_text(
            "store_id,planogram_id,shelf,row,slot,expected_sku,facing_count\n"
            f"STORE001,PLAN1,S1,1,1,{barcode},2\n",
            encoding="utf-8",
        )

        page = self.client.get("/planogram")
        image_response = self.client.get(f"/catalogue/image/{barcode}")

        self.assertEqual(page.status_code, 200)
        self.assertIn(f"/catalogue/image/{barcode}".encode(), page.data)
        self.assertIn(b"slot-product-art", page.data)
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(image_response.data, image_path.read_bytes())

    def test_current_live_planogram_csv_uses_only_catalogue_skus(self):
        with patch.object(web_app, "DATA_DIR", web_app.ROOT_DIR / "data"), patch.object(web_app, "CATALOGUE_PATH", web_app.ROOT_DIR / "data" / "catalogue.csv"):
            live_catalogue = web_app._read_catalogue()
            live_planogram = web_app._read_planogram()
        self.assertEqual(len(live_planogram), 15)
        self.assertEqual(
            web_app.validate_planogram(live_planogram, live_catalogue),
            [],
        )
        self.assertEqual(sum(not row["sku"] for row in live_planogram), 2)

    def test_aggregate_report_export_has_header_when_empty(self):
        response = self.client.get("/reports/download")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"record_type,created_at,store_name", response.data)

    def test_settings_are_saved_and_reloaded(self):
        response = self.client.post(
            "/settings/save",
            data={
                "default_store": "Demo Store",
                "default_planogram": "Snacks",
                "default_catalogue": "Retail",
                "theme": "Light",
                "language": "English",
                "detection_confidence": "0.60",
                "price_confidence": "0.55",
                "minimum_product_height": "24",
                "low_stock_threshold": "3",
                "price_verification": "on",
                "low_stock_alerts": "on",
            },
            follow_redirects=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Analysis settings saved", response.data)
        self.assertIn(b"Demo Store", response.data)
        self.assertFalse(web_app._load_settings()["shelf_gap_detection"])

    def test_reports_page_and_download_include_saved_audits(self):
        web_app._reports["test-report"] = {
            "created_at": "Oct 20, 2026",
            "image_name": "shelf.jpg",
            "compliance_rate": 50,
            "detection_count": 2,
            "compliant_count": 1,
            "price_unverified": 1,
            "price_results": [],
            "compliance": [
                {"product": "Oat Milk", "sku": "A1", "shelf": "Top", "position": "1", "expected": 2, "observed": 2, "status": "Compliant"},
                {"product": "Granola", "sku": "B2", "shelf": "Top", "position": "2", "expected": 2, "observed": 0, "status": "Understocked"},
            ],
        }

        page = self.client.get("/reports")
        download = self.client.get("/reports/download")

        self.assertEqual(page.status_code, 200)
        self.assertIn(b"shelf.jpg", page.data)
        self.assertIn("shelfsense-audit-history.csv", download.headers["Content-Disposition"])
        self.assertIn(b"Understocked", download.data)

    def test_empty_dashboard_and_reports_do_not_create_fabricated_audit(self):
        self.assertEqual(web_app._reports, {})

        dashboard = self.client.get("/")
        reports = self.client.get("/reports")

        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(b"Awaiting Analysis", dashboard.data)
        self.assertNotIn(b"sample_shelf.jpg", dashboard.data)
        self.assertEqual(reports.status_code, 200)
        self.assertIn(b"No saved audits yet", reports.data)
        self.assertEqual(web_app._reports, {})

    def test_dashboard_explains_missing_model_instead_of_offering_unavailable_scan(self):
        with patch.object(web_app, "MODEL_PATH", None):
            response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Detection model is not configured", response.data)
        self.assertIn(b"View model setup", response.data)
        self.assertNotIn(b"Your shelf scan will appear here", response.data)

    def test_analyze_uses_confidence_and_excludes_inactive_products(self):
        with self.catalogue_path.open("a", encoding="utf-8") as catalogue_file:
            catalogue_file.write("B2,Granola,Pantry,Meadow,400 g,6.00,granola,false\n")
        with (Path(self.temp_dir.name) / "planogram.csv").open("a", encoding="utf-8") as planogram_file:
            planogram_file.write("Top,2,B2,1,6.00\n")
        detection = {"label": "oat milk", "confidence": 0.91, "box": [1, 2, 3, 4]}

        with patch.object(web_app, "detect_products", return_value=[detection]) as detector:
            response = self.client.post(
                "/analyze",
                data={
                    "shelf_image": (io.BytesIO(jpeg_bytes()), "shelf.jpg"),
                    "store": "Demo Store",
                    "planogram": "Weekend Planogram",
                    "confidence_threshold": "0.72",
                    "price_check": "on",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 302)
        detector.assert_called_once()
        self.assertEqual(detector.call_args.kwargs["confidence"], 0.72)
        report_id = response.location.rsplit("/", 1)[-1]
        report = web_app._reports[report_id]
        detail = self.client.get(response.location)
        self.assertEqual(report["store_name"], "Demo Store")
        self.assertEqual(report["planogram_name"], "Weekend Planogram")
        self.assertEqual([row["sku"] for row in report["compliance"]], ["A1"])
        self.assertEqual(report["compliance"][0]["status"], "Needs Review")
        self.assertIsNone(report["compliance"][0]["observed"])
        self.assertIsNone(report["compliance_rate"])
        self.assertEqual(len(report["price_results"]), 1)
        self.assertEqual(detail.status_code, 200)
        self.assertIn(b"Needs Review", detail.data)
        self.assertIn(b"bounding boxes", detail.data)

    def test_analyze_rejects_planogram_sku_not_in_catalogue(self):
        planogram_path = Path(self.temp_dir.name) / "planogram.csv"
        planogram_path.write_text(
            "store_id,planogram_id,shelf,row,slot,expected_sku,facing_count\n"
            "STORE001,PLAN1,S1,1,1,NOT-A-CATALOGUE-SKU,1\n",
            encoding="utf-8",
        )
        upload_dir = Path(self.temp_dir.name) / "uploads"
        upload_dir.mkdir()
        upload_patch = patch.object(web_app, "UPLOAD_DIR", upload_dir)
        upload_patch.start()
        self.addCleanup(upload_patch.stop)
        with patch.object(web_app, "detect_products") as detector:
            response = self.client.post(
                "/analyze",
                data={"shelf_image": (io.BytesIO(b"image bytes"), "shelf.jpg")},
                content_type="multipart/form-data",
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"unknown expected_sku", response.data)
        detector.assert_not_called()
        self.assertFalse(any(upload_dir.glob("*_shelf.jpg")))

    def test_analyze_handles_unexpected_detector_failure(self):
        upload_dir = Path(self.temp_dir.name) / "uploads"
        upload_dir.mkdir()
        upload_patch = patch.object(web_app, "UPLOAD_DIR", upload_dir)
        upload_patch.start()
        self.addCleanup(upload_patch.stop)
        with patch.object(web_app, "detect_products", side_effect=RuntimeError("model crashed")):
            response = self.client.post(
                "/analyze",
                data={"shelf_image": (io.BytesIO(jpeg_bytes()), "shelf.jpg")},
                content_type="multipart/form-data",
                follow_redirects=True,
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Image analysis failed unexpectedly", response.data)
        self.assertFalse(any(upload_dir.glob("*_shelf.jpg")))

    def test_report_exports_include_price_and_detection_rows(self):
        web_app._reports["export-test"] = {
            "created_at": "Oct 20, 2026",
            "image_name": "shelf.jpg",
            "store_name": "Demo Store",
            "compliance_rate": 75,
            "detection_count": 1,
            "compliant_count": 1,
            "compliance": [{"product": "Oat Milk", "sku": "A1", "shelf": "Top", "position": "1", "expected": 1, "observed": 1, "status": "Compliant"}],
            "price_results": [{"product": "Oat Milk", "sku": "A1", "expected_price": 4.0, "observed_price": 5.0, "status": "Mismatch"}],
            "detections": [{"label": "oat milk", "confidence": 0.9, "box": [1, 2, 3, 4]}],
        }

        response = self.client.get("/reports/export-test/download")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"price_check", response.data)
        self.assertIn(b"detection", response.data)
        self.assertIn(b"Mismatch", response.data)

    def test_dashboard_displays_completed_scan_and_image(self):
        upload_dir = Path(self.temp_dir.name) / "uploads"
        upload_dir.mkdir()
        image_path = upload_dir / "shelf.jpg"
        image_path.write_bytes(b"test image content")
        upload_patch = patch.object(web_app, "UPLOAD_DIR", upload_dir)
        upload_patch.start()
        self.addCleanup(upload_patch.stop)
        web_app._reports["dashboard-report"] = {
            "created_at": "Oct 20, 2026",
            "image_name": "shelf.jpg",
            "image_path": str(image_path.resolve()),
            "store_name": "Demo Store",
            "compliance_rate": 100,
            "detection_count": 1,
            "compliant_count": 1,
            "price_unverified": 1,
            "price_results": [],
            "detections": [
                {"label": "oat milk", "confidence": 0.96, "box": [10, 10, 50, 50], "image_width": 100, "image_height": 100}
            ],
            "compliance": [
                {"product": "Oat Milk", "sku": "A1", "shelf": "Top", "position": "1", "expected": 1, "observed": 1, "difference": 0, "status": "Compliant"}
            ],
        }

        response = self.client.get("/")
        image_response = self.client.get("/reports/dashboard-report/image")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Shelf Audit Result", response.data)
        self.assertIn(b"Demo Store", response.data)
        self.assertIn(b"oat milk", response.data)
        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(image_response.data, b"test image content")


if __name__ == "__main__":
    unittest.main()
