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
from config.settings import DEFAULT_MODEL_PATH, TEST_SET_DIR
from src.ocr import OcrUnavailable, parse_price, read_price_tags
from tools.build_test_set import expected_status, load_font


class PriceParsingTests(unittest.TestCase):
    def test_prices_require_currency_marker_or_two_decimals(self):
        cases = {"₹ 45.00": 45.0, "45.00": 45.0, "Rs. 120": 120.0, "MRP:99": 99.0, "₹85": 85.0,
                 "500 g": None, "2024": None, "12.5": None, "": None}
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


class PriceLinkingTests(unittest.TestCase):
    planogram = [
        {"store_id": "S", "planogram_id": "P", "physical_shelf": "S1", "row": "1", "slot": "1", "sku": "A", "expected_facings": "1"},
        {"store_id": "S", "planogram_id": "P", "physical_shelf": "S1", "row": "1", "slot": "2", "sku": "B", "expected_facings": "1"},
    ]

    def test_calibrated_layout_links_tag_to_slot_sku(self):
        tags = [{"text": "45.00", "price": 45.0, "confidence": 0.99, "box": [650, 330, 750, 370]}]
        linked = web_app._assign_price_tags(tags, self.planogram, [], CoordinateShelfMapper(trusted=True), 1000, 400)
        self.assertEqual((linked[0]["sku"], linked[0]["slot"], linked[0]["association"]), ("B", "R1S2", "calibrated slot"))

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

    def _run(self, accepted: list, calibrated: bool = True):
        detections = [
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
        self.assertEqual(report["compliance_rate"], 0)
        self.assertTrue(report["layout_trusted"])
        reports_page = self.client.get("/reports")
        self.assertIn(b"Calibrated Store", reports_page.data)

        compliant = self._run(["A", "B"])
        self.assertEqual(compliant["compliance_rate"], 100)
        page = self.client.get("/reports").data
        self.assertIn(b"Snacks", page)  # category-wise compliance now has data

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


class ExpectedStatusTests(unittest.TestCase):
    def test_expected_status_rules(self):
        planogram_skus = {"A", "B"}
        self.assertEqual(expected_status("A", 2, ["A", "A"], planogram_skus), "Compliant")
        self.assertEqual(expected_status("A", 2, ["A"], planogram_skus), "Missing Expected Product")
        self.assertEqual(expected_status("A", 2, [], planogram_skus), "Missing Expected Product")
        self.assertEqual(expected_status("A", 2, ["B", "B"], planogram_skus), "Misplaced")
        self.assertEqual(expected_status("A", 1, ["A", "A"], planogram_skus), "Unexpected Product")
        self.assertEqual(expected_status("", 0, ["B"], planogram_skus), "Unexpected Product")
        self.assertEqual(expected_status("", 0, ["Z"], planogram_skus), "Empty Slot")


@unittest.skipUnless((TEST_SET_DIR / "ground_truth.json").is_file() and DEFAULT_MODEL_PATH.is_file(), "test set or model missing")
class LabelledTestSetTests(unittest.TestCase):
    """Runs the real detector, matcher, and OCR on labelled scenes."""

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
