"""OCR price parsing/reading, price-tag linking, calibrated-layout audits, and the labelled test set."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

import app as web_app
from backend.matching.shelf_mapper import CoordinateShelfMapper
from config.settings import DEFAULT_MODEL_PATH, DEFAULT_PRICE_TAG_VOID_MODEL_PATH, TEST_SET_DIR
from src import detector as detector_module
from src.ocr import OcrUnavailable, PriceTagReader, parse_price, read_price_crop, read_price_tags
from src.planogram import evaluate_planogram
from tools.build_test_set import expected_status, load_font


class PriceParsingTests(unittest.TestCase):
    def test_prices_require_currency_marker_or_two_decimals(self):
        cases = {"₹ 45.00": 45.0, "45.00": 45.0, "Rs. 120": 120.0, "MRP:99": 99.0, "₹85": 85.0,
                 "500 g": None, "2024": None, "12.5": None, "": None}
        cases["\u20b9 45"] = 45.0
        for text, expected in cases.items():
            self.assertEqual(parse_price(text), expected, text)

    def test_rapidocr_reads_a_rendered_price_tag(self):
        image = Image.new("RGB", (420, 120), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 410, 110), fill=(255, 214, 0))
        draw.text((40, 32), "₹ 129.00", font=load_font(48), fill="black")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tag.png"
            image.save(path)
            try:
                tags = read_price_tags(path)
            except OcrUnavailable as error:
                self.skipTest(str(error))
        self.assertEqual([tag["price"] for tag in tags], [129.0])
        x1, y1, x2, y2 = tags[0]["box"]
        self.assertTrue(10 <= x1 < x2 <= 410 and 10 <= y1 < y2 <= 110)

    def test_price_formats_and_ocr_spacing_are_normalized(self):
        for text, expected in (("₹20", 20.0), ("Rs. 20", 20.0), ("₹120", 120.0),
                               ("20.00", 20.0), ("50 .00", 50.0)):
            self.assertEqual(parse_price(text), expected, text)

    def test_invalid_crop_boundary_is_reported_unreadable(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shelf.png"
            Image.new("RGB", (40, 30), "white").save(path)
            result = read_price_crop(path, [50, 50, 20, 20])
        self.assertEqual(result["ocr_status"], "unreadable")
        self.assertIsNone(result["price"])

    def test_blank_price_tag_crop_is_not_given_a_fabricated_price(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "blank.png"
            Image.new("RGB", (220, 100), "white").save(path)
            try:
                result = read_price_crop(path, [0, 0, 220, 100])
            except OcrUnavailable as error:
                self.skipTest(str(error))
        self.assertEqual(result["ocr_status"], "unreadable")
        self.assertIsNone(result["price"])

    def test_crop_coordinates_scale_from_detector_dimensions_to_stored_image(self):
        import numpy as np

        class FakeOcr:
            def __call__(self, image):
                self.shape = image.shape
                return [([], "45.00", 0.99)], None

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shelf.png"
            Image.new("RGB", (200, 100), "white").save(path)
            reader = PriceTagReader()
            fake = FakeOcr()
            reader._ocr = fake
            result = reader.read_crop(path, [20, 10, 40, 30], image_width=100, image_height=50)
        self.assertEqual(result["price"], 45.0)
        self.assertEqual(result["text"], "45.00")
        self.assertEqual(fake.shape[:2], (80, 80))  # 40x40 image crop, enlarged 2x

    def test_ocr_value_contains_only_the_parsed_numeric_amount(self):
        class FakeOcr:
            def __call__(self, _image):
                return [([], "MRP Rs. 129.00 only", 0.99)], None

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tag.png"
            Image.new("RGB", (220, 100), "white").save(path)
            reader = PriceTagReader()
            reader._ocr = FakeOcr()
            result = reader.read_crop(path, [0, 0, 220, 100])

        self.assertEqual(result["ocr_status"], "read")
        self.assertEqual(result["text"], "129.00")
        self.assertEqual(result["raw_ocr_text"], "MRP Rs. 129.00 only")

    def test_multiple_price_readings_are_ambiguous(self):
        class FakeOcr:
            def __call__(self, _image):
                return [([], "₹20", 0.99), ([], "30.00", 0.98)], None

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tag.png"
            Image.new("RGB", (220, 100), "white").save(path)
            reader = PriceTagReader()
            reader._ocr = FakeOcr()
            result = reader.read_crop(path, [0, 0, 220, 100])
        self.assertEqual(result["ocr_status"], "ambiguous")
        self.assertIsNone(result["price"])
        self.assertIn("₹20", result["raw_ocr_text"])


class AuxiliaryDetectorTests(unittest.TestCase):
    def test_missing_auxiliary_checkpoint_has_a_useful_error(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(
            detector_module, "PRICE_TAG_VOID_MODEL_PATH", str(Path(temp) / "missing.pt")
        ):
            with self.assertRaisesRegex(detector_module.DetectorUnavailable, "missing.pt"):
                detector_module.detect_price_tags_and_voids("shelf.jpg")

    def test_empty_auxiliary_inference_returns_empty_class_lists(self):
        class FakeModel:
            names = {0: "Price_Tag", 1: "void"}

        class FakeDetector:
            def load(self):
                return FakeModel()

            def detect(self, *_args, **_kwargs):
                return []

            def info(self):
                return {"class_names": ["Price_Tag", "void"]}

        with patch.object(detector_module, "get_detector", return_value=FakeDetector()):
            result = detector_module.detect_price_tags_and_voids("shelf.jpg")
        self.assertEqual(result["price_tags"], [])
        self.assertEqual(result["voids"], [])

    @unittest.skipUnless(DEFAULT_PRICE_TAG_VOID_MODEL_PATH.is_file(), "price-tag/void checkpoint missing")
    def test_real_auxiliary_model_maps_classes_and_reads_price_tag_crop(self):
        image = TEST_SET_DIR / "test" / "05_price_mismatch.jpg"
        self.assertTrue(image.is_file(), "labelled shelf test image missing")
        result = detector_module.detect_price_tags_and_voids(image)
        self.assertEqual(result["model"]["class_names"], ["Price_Tag", "void"])
        self.assertGreaterEqual(len(result["price_tags"]), 1)
        self.assertGreaterEqual(len(result["voids"]), 1)
        tag = result["price_tags"][0]
        self.assertEqual(tag["label"], "price_tag")
        self.assertEqual(tag["class_id"], 0)
        self.assertTrue(all(void["class_id"] == 1 and void["label"] == "void" for void in result["voids"]))
        self.assertTrue(0 <= tag["confidence"] <= 1)
        with Image.open(image) as opened:
            width, height = opened.size
        x1, y1, x2, y2 = tag["box"]
        self.assertTrue(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)
        ocr = read_price_crop(image, tag["box"], min_confidence=0.5)
        self.assertEqual(ocr["ocr_status"], "read")
        self.assertEqual(ocr["raw_ocr_text"], "50 .00")
        self.assertEqual(ocr["text"], "50.00")
        self.assertEqual(ocr["price"], 50.0)


class PriceLinkingTests(unittest.TestCase):
    planogram = [
        {"store_id": "S", "planogram_id": "P", "physical_shelf": "S1", "row": "1", "slot": "1", "sku": "A", "expected_facings": "1"},
        {"store_id": "S", "planogram_id": "P", "physical_shelf": "S1", "row": "1", "slot": "2", "sku": "B", "expected_facings": "1"},
    ]

    def test_calibrated_layout_links_tag_to_slot_sku(self):
        tags = [{"text": "45.00", "price": 45.0, "confidence": 0.99, "box": [650, 330, 750, 370]}]
        unsupported = web_app._assign_price_tags(tags, self.planogram, [], CoordinateShelfMapper(trusted=True), 1000, 400)
        self.assertEqual(unsupported[0]["sku"], "")
        self.assertIn("no identified product", unsupported[0]["association"])
        tags = [{"text": "45.00", "price": 45.0, "confidence": 0.99, "box": [650, 330, 750, 370]}]
        identified = [{"box": [600, 50, 800, 300], "sku": "B", "slot_identity": ("S", "P", "S1", "1", "2"), "detection_index": 7}]
        linked = web_app._assign_price_tags(tags, self.planogram, identified, CoordinateShelfMapper(trusted=True), 1000, 400)
        self.assertEqual((linked[0]["sku"], linked[0]["slot"]), ("B", "R1S2"))

    def test_calibrated_price_tag_between_different_products_in_slot_needs_review(self):
        tags = [{"text": "45.00", "price": 45.0, "box": [650, 330, 750, 370]}]
        identity = ("S", "P", "S1", "1", "2")
        identified = [
            {"box": [590, 50, 690, 300], "sku": "B", "slot_identity": identity, "detection_index": 7},
            {"box": [710, 50, 810, 300], "sku": "A", "slot_identity": identity, "detection_index": 8},
        ]
        linked = web_app._assign_price_tags(tags, self.planogram, identified, CoordinateShelfMapper(trusted=True), 1000, 400)
        self.assertEqual(linked[0]["sku"], "")
        self.assertIn("ambiguous", linked[0]["association"])

    def test_calibrated_price_tag_between_facings_of_one_sku_is_linked(self):
        # Two equidistant facings of the same SKU leave no doubt about which product the tag prices.
        tags = [{"text": "45.00", "price": 45.0, "box": [650, 330, 750, 370]}]
        identity = ("S", "P", "S1", "1", "2")
        identified = [
            {"box": [590, 50, 690, 300], "sku": "B", "slot_identity": identity, "detection_index": 7},
            {"box": [710, 50, 810, 300], "sku": "B", "slot_identity": identity, "detection_index": 8},
        ]
        linked = web_app._assign_price_tags(tags, self.planogram, identified, CoordinateShelfMapper(trusted=True), 1000, 400)
        self.assertEqual((linked[0]["sku"], linked[0]["slot"]), ("B", "R1S2"))

    def test_calibrated_price_tag_under_misplaced_product_is_not_linked(self):
        # The shelf tag belongs to the slot's planned SKU (B); the visible product (A) is misplaced,
        # so the tag must not produce a price verdict for either product.
        tags = [{"text": "45.00", "price": 45.0, "box": [650, 330, 750, 370]}]
        identified = [{"box": [600, 50, 800, 300], "sku": "A", "slot_identity": ("S", "P", "S1", "1", "2"), "detection_index": 7}]
        linked = web_app._assign_price_tags(tags, self.planogram, identified, CoordinateShelfMapper(trusted=True), 1000, 400)
        self.assertEqual(linked[0]["sku"], "")
        self.assertIn("differs from the slot's planogram SKU", linked[0]["association"])

    def test_conflicting_price_readings_in_one_slot_are_not_linked(self):
        tags = [{"text": "45.00", "price": 45.0, "box": [650, 330, 750, 370]},
                {"text": "1.00", "price": 1.0, "box": [620, 150, 660, 170]}]
        identified = [
            {"box": [590, 50, 690, 300], "sku": "B", "slot_identity": ("S", "P", "S1", "1", "2"), "detection_index": 7},
            {"box": [710, 50, 810, 300], "sku": "B", "slot_identity": ("S", "P", "S1", "1", "2"), "detection_index": 8},
        ]
        linked = web_app._assign_price_tags(tags, self.planogram, identified, CoordinateShelfMapper(trusted=True), 1000, 400)
        self.assertEqual([tag["sku"] for tag in linked], ["", ""])
        self.assertTrue(all("conflicting price readings" in tag["association"] for tag in linked))

    def test_uncalibrated_layout_links_tag_to_identified_product_above(self):
        identified = [{"box": [100, 50, 220, 300], "sku": "A"}, {"box": [600, 50, 700, 300], "sku": ""}]
        tags = [
            {"text": "₹20.00", "price": 20.0, "confidence": 0.9, "box": [120, 320, 200, 350]},
            {"text": "₹30.00", "price": 30.0, "confidence": 0.9, "box": [610, 320, 690, 350]},
        ]
        linked = web_app._assign_price_tags(tags, self.planogram, identified, CoordinateShelfMapper(), 1000, 400)
        self.assertEqual(linked[0]["sku"], "A")
        self.assertEqual(linked[1]["sku"], "")  # product above was not identified
        results = web_app._price_results(
            [{"sku": "A", "name": "Alpha", "price": "25.00"}, {"sku": "B", "name": "Beta", "price": "30.00"}], linked)
        self.assertEqual({row["sku"]: row["status"] for row in results}, {"A": "Mismatch", "B": "Unverified"})


class CalibratedLayoutAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "catalogue.csv").write_text(
            "sku,name,category,brand,pack_size,price,model_label,active\n"
            "A,Alpha,Snacks,X,1,10.00,alpha,true\nB,Beta,Drinks,X,1,20.00,beta,true\n",
            encoding="utf-8",
        )
        (root / "planogram.csv").write_text(
            "store_id,planogram_id,shelf,row,slot,expected_sku,facing_count\n"
            "S,P,S1,1,1,A,1\nS,P,S1,1,2,B,1\nS,P,S1,1,3,,0\n",
            encoding="utf-8",
        )
        (root / "uploads").mkdir()
        for name, value in (("DATA_DIR", root), ("CATALOGUE_PATH", root / "catalogue.csv"),
                            ("SETTINGS_PATH", root / "settings.json"), ("AUDIT_RESULTS_DIR", root / "results"),
                            ("UPLOAD_DIR", root / "uploads")):
            patcher = patch.object(web_app, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        old_reports = web_app._reports.copy()
        web_app._reports.clear()
        self.addCleanup(lambda: (web_app._reports.clear(), web_app._reports.update(old_reports)))
        self.client = web_app.app.test_client()
        auxiliary = patch.object(web_app, "detect_price_tags_and_voids", return_value={
            "price_tags": [], "voids": [], "model": {"class_names": ["Price_Tag", "void"]},
        })
        auxiliary.start()
        self.addCleanup(auxiliary.stop)

    def _run(self, accepted: list, calibrated: bool = True, detections: list | None = None):
        detections = detections or [
            {"label": "object", "confidence": 0.9, "box": [40, 40, 160, 160], "image_width": 600, "image_height": 200},
            {"label": "object", "confidence": 0.9, "box": [240, 40, 360, 160], "image_width": 600, "image_height": 200},
        ]
        matches = [{"accepted_sku": sku, "status": "Matched" if sku else "Unknown / Needs Review", "candidates": []} for sku in accepted]

        class Pipeline:
            def run(self, *_args, slot_mapper=None, **_kwargs):
                return {"matches": matches, "findings": [], "calibration": {"calibration_status": "calibrated_on_validation_set"}}

        buffer = io.BytesIO()
        Image.new("RGB", (600, 200), "gray").save(buffer, format="JPEG")
        form = {"shelf_image": (io.BytesIO(buffer.getvalue()), "shelf.jpg"), "store": "Calibrated Store"}
        if calibrated:
            form["calibrated_layout"] = "on"
        with patch.object(web_app, "detect_products", return_value=detections), \
                patch.object(web_app, "_get_matching_pipeline", return_value=Pipeline()):
            response = self.client.post("/api/analyze", data=form, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 201, response.json)
        return web_app._reports[response.json["report_id"]]

    def test_identified_products_in_calibrated_layout_give_confirmed_findings(self):
        report = self._run(["B", "A"])  # swapped
        statuses = {row["slot"]: row["status"] for row in report["compliance"]}
        self.assertEqual(statuses, {"1": "Misplaced", "2": "Misplaced", "3": "Empty Slot"})
        self.assertEqual(report["compliance_rate"], 33.33)  # trusted empty slot passes; both assigned slots fail
        self.assertTrue(report["layout_trusted"])
        reports_page = self.client.get("/reports")
        self.assertIn(b"Calibrated Store", reports_page.data)

        compliant = self._run(["A", "B"])
        self.assertEqual(compliant["compliance_rate"], 100)
        page = self.client.get("/reports").data
        self.assertIn(b"Snacks", page)  # category-wise compliance now has data

    def test_score_is_consistent_across_report_api_dashboard_and_csv(self):
        report = self._run(["A", "B"])
        score = report["compliance_score"]
        self.assertEqual(score["status"], "complete")
        self.assertEqual(score["score"], 100.0)
        api_report = self.client.get(f"/api/reports/{report['report_id']}").json
        self.assertEqual(api_report["compliance_score"], score)
        dashboard = self.client.get("/").data
        self.assertIn(b"Compliance score", dashboard)
        detail = self.client.get(f"/reports/{report['report_id']}").data
        self.assertIn(b"Compliance score breakdown", detail)
        csv_export = self.client.get(f"/reports/{report['report_id']}/download").data
        self.assertIn(b"score_status", csv_export)
        self.assertIn(b"complete", csv_export)
        self.assertIn(b"100.0", csv_export)

    def test_unidentified_object_keeps_slot_in_review(self):
        report = self._run(["A", None])
        statuses = {row["slot"]: row["status"] for row in report["compliance"]}
        self.assertEqual(statuses["1"], "Compliant")
        self.assertEqual(statuses["2"], "Needs Review")
        self.assertEqual([d["product_name"] for d in report["detections"]], ["Alpha", ""])
        self.assertEqual([d["label"] for d in report["detections"]], ["object", "object"])  # model class kept separately
        dashboard = self.client.get("/").data.decode("utf-8")
        self.assertIn("Alpha · 90%", dashboard)
        self.assertIn("unidentified · 90%", dashboard)
        self.assertNotIn("object · 90%", dashboard)

    def test_uncalibrated_layout_never_confirms_slots(self):
        report = self._run(["A", "B"], calibrated=False)
        self.assertTrue(all(row["status"] in {"Needs Review", "Empty Slot"} for row in report["compliance"]))
        self.assertIsNone(report["compliance_rate"])

    def test_detected_row_count_mismatch_disables_confirmed_slot_comparison(self):
        detections = [
            {"label": "object", "confidence": 0.9, "box": [40, 30, 160, 150], "image_width": 600, "image_height": 400},
            {"label": "object", "confidence": 0.9, "box": [240, 30, 360, 150], "image_width": 600, "image_height": 400},
            {"label": "object", "confidence": 0.9, "box": [40, 250, 160, 370], "image_width": 600, "image_height": 400},
        ]
        report = self._run(["A", "B", "A"], detections=detections)
        self.assertFalse(report["layout_trusted"])
        self.assertFalse(report["layout_validation"]["compatible"])
        self.assertEqual(report["layout_validation"]["detected_rows"], 2)
        self.assertTrue(all(slot["status"] in {"Needs Review", "Empty Slot"} for slot in report["compliance"]))

    def test_overlapping_duplicate_box_is_not_counted_or_matched_twice(self):
        detections = [
            {"label": "object", "confidence": 0.9, "box": [40, 40, 160, 160], "image_width": 600, "image_height": 200},
            {"label": "object", "confidence": 0.8, "box": [41, 41, 161, 161], "image_width": 600, "image_height": 200},
            {"label": "object", "confidence": 0.9, "box": [240, 40, 360, 160], "image_width": 600, "image_height": 200},
        ]
        report = self._run(["A", "B"], detections=detections)
        self.assertEqual(report["raw_detection_count"], 3)
        self.assertEqual(report["detection_count"], 2)
        self.assertEqual(report["duplicate_detection_count"], 1)
        self.assertEqual(sum(slot["observed"] or 0 for slot in report["compliance"]), 2)
        self.assertTrue(report["detections"][1]["duplicate_detection"])

    def test_auxiliary_inference_failure_does_not_hide_product_audit(self):
        image = Image.new("RGB", (80, 60), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        with patch.object(web_app, "detect_products", return_value=[]), \
                patch.object(web_app, "detect_price_tags_and_voids", side_effect=RuntimeError("aux model failed")), \
                patch.object(web_app, "_get_matching_pipeline", return_value=None):
            response = self.client.post(
                "/api/analyze", data={"shelf_image": (io.BytesIO(buffer.getvalue()), "shelf.jpg")},
                content_type="multipart/form-data")
        self.assertEqual(response.status_code, 201, response.json)
        report = web_app._reports[response.json["report_id"]]
        self.assertIn("Auxiliary detector failed", report["auxiliary_model_error"])
        self.assertEqual(report["detections"], [])
        self.assertIn(b"aux model failed", self.client.get("/").data)


class ExpectedStatusTests(unittest.TestCase):
    def test_expected_status_rules(self):
        self.assertEqual(expected_status("A", 2, ["A", "A"]), "Compliant")
        self.assertEqual(expected_status("A", 2, ["A"]), "Missing Expected Product")
        self.assertEqual(expected_status("A", 2, []), "Missing Expected Product")
        self.assertEqual(expected_status("A", 2, ["B", "B"]), "Misplaced")
        self.assertEqual(expected_status("A", 1, ["A", "A"]), "Unexpected Product")
        self.assertEqual(expected_status("", 0, ["B"]), "Unexpected Product")
        self.assertEqual(expected_status("", 0, []), "Empty Slot")

    def test_expected_status_matches_runtime_rules_for_non_planogram_products(self):
        # A non-planogram catalogue product ("Z") that is identified and mapped is evidence in the
        # runtime audit, so the ground truth must expect the same status the audit can produce.
        catalogue = [{"sku": sku, "name": sku} for sku in ("A", "Z")]
        planogram = [
            {"store_id": "S", "planogram_id": "P", "shelf": "1", "row": "1", "slot": "1", "expected_sku": "A", "facing_count": "2"},
            {"store_id": "S", "planogram_id": "P", "shelf": "1", "row": "1", "slot": "2", "expected_sku": "", "facing_count": "0"},
        ]
        for slot_number, expected_sku, facings in (("1", "A", 2), ("2", "", 0)):
            detections = [{"sku": "Z", "slot": slot_number}, {"sku": "Z", "slot": slot_number}]
            audit = evaluate_planogram(planogram, catalogue, detections,
                                       slot_mapper=lambda detection, _rows: ("S", "P", "1", "1", detection["slot"]))
            runtime = next(row["status"] for row in audit["slots"] if row["slot"] == slot_number)
            self.assertEqual(expected_status(expected_sku, facings, ["Z", "Z"]), runtime)


@unittest.skipUnless((TEST_SET_DIR / "ground_truth.json").is_file() and DEFAULT_MODEL_PATH.is_file(), "test set or model missing")
class LabelledTestSetTests(unittest.TestCase):
    """Runs the real detector, matcher, and OCR on labelled scenes."""

    def test_price_tag_void_ocr_results_reach_api_and_dashboard(self):
        image_name = "05_price_mismatch.jpg"
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(web_app, "AUDIT_RESULTS_DIR", Path(temp)), patch.object(web_app, "UPLOAD_DIR", Path(temp)):
            response = web_app.app.test_client().post(
                "/api/analyze", data={"test_image": image_name, "price_check": "on"},
                content_type="multipart/form-data")
            self.assertEqual(response.status_code, 201, response.json)
            report_id = response.json["report_id"]
            self.addCleanup(lambda: web_app._reports.pop(report_id, None))
            report = web_app._reports[report_id]
            self.assertEqual(report["auxiliary_model"]["class_names"], ["Price_Tag", "void"])
            self.assertGreaterEqual(len(report["void_detections"]), 1)
            # Synthetic tags sit near the tag model's confidence threshold; when it finds none,
            # whole-image OCR must still supply readings (and say so) instead of skipping prices.
            if report["price_tag_detections"]:
                self.assertEqual(report["price_audit"]["tag_source"], "price_tag_model")
            else:
                self.assertEqual(report["price_audit"]["tag_source"], "full_image_ocr")
            tags = report["price_audit"]["tags"]
            self.assertTrue(any(tag["parsed_price"] is not None for tag in tags), report["price_audit"])
            self.assertTrue(any(tag["parsed_price"] is not None for tag in response.json["price_audit"]["tags"]))
            self.assertTrue(all(tag["ocr_status"] in {"read", "unreadable", "needs_review", "ambiguous", "error", "unavailable"}
                                for tag in tags))
            api_report = web_app.app.test_client().get(f"/api/reports/{report_id}").json
            self.assertEqual(api_report["void_detections"], report["void_detections"])
            client = web_app.app.test_client()
            annotated = client.get(f"/reports/{report_id}/annotated")
            self.assertEqual(annotated.status_code, 200)
            with Image.open(io.BytesIO(annotated.data)) as marked_up:
                self.assertGreater(marked_up.width, 0)
                self.assertGreater(marked_up.height, 0)
            report_page = client.get(f"/reports/{report_id}")
            self.assertEqual(report_page.status_code, 200)
            self.assertIn(b"Potential shelf voids", report_page.data)
            dashboard = client.get("/").data
            self.assertIn(b"Price Tags &amp; Voids", dashboard)

    def test_ocr_failure_is_kept_as_review_and_does_not_fail_analysis(self):
        detection = {"label": "Price_Tag", "class_id": 0, "confidence": 0.9,
                     "box": [10, 10, 50, 40], "image_width": 80, "image_height": 60, "kind": "price_tag"}
        auxiliary = {"price_tags": [detection], "voids": [], "model": {"class_names": ["Price_Tag", "void"]}}
        image = Image.new("RGB", (80, 60), "white")
        payload = io.BytesIO()
        image.save(payload, format="JPEG")
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(web_app, "AUDIT_RESULTS_DIR", Path(temp)), patch.object(web_app, "UPLOAD_DIR", Path(temp)), \
                patch.object(web_app, "detect_products", return_value=[]), \
                patch.object(web_app, "detect_price_tags_and_voids", return_value=auxiliary), \
                patch.object(web_app, "read_price_crop", side_effect=RuntimeError("OCR engine crashed")), \
                patch.object(web_app, "_get_matching_pipeline", return_value=None):
            response = web_app.app.test_client().post(
                "/api/analyze", data={"shelf_image": (io.BytesIO(payload.getvalue()), "blank.jpg"), "price_check": "on"},
                content_type="multipart/form-data")
            self.assertEqual(response.status_code, 201, response.json)
            report_id = response.json["report_id"]
            self.addCleanup(lambda: web_app._reports.pop(report_id, None))
            tag = web_app._reports[report_id]["price_audit"]["tags"][0]
            self.assertEqual(tag["ocr_status"], "error")
            self.assertIsNone(tag["parsed_price"])
            self.assertIn("OCR failed", web_app._reports[report_id]["price_audit"]["error"])

    def test_price_mismatch_scene_end_to_end(self):
        from tools.evaluate_test_set import score_scene

        truth = json.loads((TEST_SET_DIR / "ground_truth.json").read_text(encoding="utf-8"))
        split = truth["split"]
        self.assertFalse(set(split["validation_skus"]) & set(split["test_planogram_skus"] + split["test_distractor_skus"]))
        scene = next(item for item in truth["test"] if "price_mismatch" in item["image"])
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(web_app, "AUDIT_RESULTS_DIR", Path(temp)), patch.object(web_app, "UPLOAD_DIR", Path(temp)):
            response = web_app.app.test_client().post(
                "/api/analyze", data={"test_image": Path(scene["image"]).name, "price_check": "on"},
                content_type="multipart/form-data")
            self.assertEqual(response.status_code, 201, response.json)
            report = web_app._reports.pop(response.json["report_id"])
        if report["matching_audit"].get("matching_error"):
            self.skipTest(report["matching_audit"]["matching_error"])
        score = score_scene(scene, report)
        self.assertGreaterEqual(score["recalled"] / score["facings"], 0.75)
        self.assertEqual(score["accepted"], score["accepted_correct"])  # no wrong SKU accepted
        self.assertGreaterEqual(score["identified_correct"] / score["facings"], 0.6)
        self.assertGreaterEqual(score["price_tags_read"], 12)
        mismatched = {row["sku"] for row in report["price_results"] if row["status"] == "Mismatch"}
        expected = {sku for sku, status in scene["expected_price_status"].items() if status == "Mismatch"}
        self.assertEqual(mismatched, expected)
        self.assertIsNotNone(report["compliance_rate"])


if __name__ == "__main__":
    unittest.main()
