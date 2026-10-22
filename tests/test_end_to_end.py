"""Upload validation, persistence, API, and real-detector workflow tests."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app as web_app
from config.settings import DEFAULT_MODEL_PATH, SAMPLE_IMAGE_PATH
from src import detector as detector_module


def jpeg_bytes(size=(320, 240), color=(30, 140, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


DETECTIONS = [
    {"label": "object", "class_id": 0, "confidence": 0.88, "box": [10, 20, 60, 110], "image_width": 320, "image_height": 240},
    {"label": "object", "class_id": 0, "confidence": 0.74, "box": [80, 20, 130, 110], "image_width": 320, "image_height": 240},
]


class AuditWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "catalogue.csv").write_text(
            "sku,name,category,brand,pack_size,price,model_label,active\n"
            "A1,Oat Milk,Plant-based,Northstar,1 L,4.00,oat milk,true\n",
            encoding="utf-8",
        )
        (self.root / "planogram.csv").write_text(
            "store_id,planogram_id,shelf,row,slot,expected_sku,facing_count\n"
            "STORE001,P1,S1,1,1,A1,2\n",
            encoding="utf-8",
        )
        self.results_dir = self.root / "audit_results"
        self.upload_dir = self.root / "uploads"
        self.upload_dir.mkdir()
        for name, value in (
            ("DATA_DIR", self.root),
            ("CATALOGUE_PATH", self.root / "catalogue.csv"),
            ("SETTINGS_PATH", self.root / "user_settings.json"),
            ("AUDIT_RESULTS_DIR", self.results_dir),
            ("UPLOAD_DIR", self.upload_dir),
        ):
            patcher = patch.object(web_app, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        old_reports = web_app._reports.copy()
        web_app._reports.clear()
        self.addCleanup(lambda: (web_app._reports.clear(), web_app._reports.update(old_reports)))
        web_app.app.config["TESTING"] = True
        self.client = web_app.app.test_client()
        auxiliary = patch.object(web_app, "detect_price_tags_and_voids", return_value={
            "price_tags": [], "voids": [], "model": {"class_names": ["Price_Tag", "void"]},
        })
        auxiliary.start()
        self.addCleanup(auxiliary.stop)

    def _analyze(self, detections=DETECTIONS, **extra):
        data = {"shelf_image": (io.BytesIO(jpeg_bytes()), "shelf.jpg"), "store": "Demo Store", **extra}
        with patch.object(web_app, "detect_products", return_value=detections):
            return self.client.post("/analyze", data=data, content_type="multipart/form-data")

    def test_invalid_image_is_rejected_before_detection(self):
        with patch.object(web_app, "detect_products") as detector:
            response = self.client.post(
                "/analyze",
                data={"shelf_image": (io.BytesIO(b"not an image"), "shelf.jpg")},
                content_type="multipart/form-data",
                follow_redirects=True,
            )
        self.assertIn(b"not a readable PNG, JPG, or WEBP image", response.data)
        detector.assert_not_called()
        self.assertEqual(list(self.upload_dir.iterdir()), [])

    def test_report_is_persisted_reloaded_and_rendered(self):
        response = self._analyze()
        self.assertEqual(response.status_code, 302)
        report_id = response.location.rsplit("/", 1)[-1]
        saved = self.results_dir / f"{report_id}.json"
        self.assertTrue(saved.is_file())
        payload = json.loads(saved.read_text(encoding="utf-8"))
        self.assertEqual(payload["detection_count"], 2)
        self.assertIn("compliance_score", payload)
        self.assertEqual(payload["compliance_score"]["status"], "unavailable")
        self.assertEqual(payload["compliance_rate"], payload["compliance_score"]["score"])
        self.assertNotIn("image_path", payload)
        self.assertEqual(payload["facing_summary"]["expected_facings"], 2)
        self.assertEqual(payload["facing_summary"]["detected_objects"], 2)
        self.assertIsNotNone(payload["inference_ms"])
        self.assertTrue(payload["annotated_image"])

        # Simulate a server restart.
        web_app._reports.clear()
        web_app._reports.update(web_app._load_saved_reports(self.results_dir))
        self.assertIn(report_id, web_app._reports)
        detail = self.client.get(f"/reports/{report_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertIn(b"Detected products", detail.data)
        self.assertIn(b"Audit method &amp; limitations", detail.data)
        annotated = self.client.get(f"/reports/{report_id}/annotated")
        self.assertEqual(annotated.status_code, 200)
        self.assertEqual(annotated.mimetype, "image/jpeg")
        Image.open(io.BytesIO(annotated.data)).verify()
        self.assertEqual(self.client.get(f"/reports/{report_id}/image").status_code, 200)
        self.assertIn(b"Demo Store", self.client.get("/reports").data)
        dashboard = self.client.get("/")
        self.assertIn(b"Open annotated image", dashboard.data)
        self.assertIn(b"2 detected objects", dashboard.data)

    def test_legacy_saved_report_without_new_fields_renders(self):
        self.results_dir.mkdir()
        (self.results_dir / "abc123.json").write_text(json.dumps({
            "created_at": "Oct 21, 2026", "image_name": "old.jpg", "compliance": [],
            "detections": [{"label": "object", "confidence": 0.8, "box": [1, 2, 3, 4]}],
        }), encoding="utf-8")
        (self.results_dir / "broken.json").write_text("{not json", encoding="utf-8")
        web_app._reports.update(web_app._load_saved_reports(self.results_dir))
        self.assertEqual(list(web_app._reports), ["abc123"])
        self.assertEqual(web_app._reports["abc123"]["compliance_score"]["status"], "unavailable")
        for route in ("/", "/reports", "/reports/abc123", "/reports/abc123/download"):
            self.assertEqual(self.client.get(route).status_code, 200, route)
        self.assertEqual(self.client.get("/reports/abc123/image").status_code, 404)

    def test_legacy_whole_image_price_tags_render_on_dashboard(self):
        self.results_dir.mkdir()
        (self.results_dir / "def456.json").write_text(json.dumps({
            "created_at": "Oct 21, 2026", "image_name": "old.jpg", "compliance": [], "detections": [],
            "price_audit": {"requested": True, "tags": [
                {"text": "₹ 20.00", "price": 20.0, "confidence": 0.9, "box": [1, 2, 30, 40], "sku": "", "association": "unassigned"},
            ]},
        }), encoding="utf-8")
        web_app._reports.update(web_app._load_saved_reports(self.results_dir))
        for route in ("/", "/reports/def456"):
            self.assertEqual(self.client.get(route).status_code, 200, route)
        self.assertIn("₹20.00".encode(), self.client.get("/").data)

    def test_example_image_can_be_analyzed_without_upload(self):
        if not SAMPLE_IMAGE_PATH.is_file():
            self.skipTest("sample image missing")
        with patch.object(web_app, "detect_products", return_value=[]) as detector:
            response = self.client.post("/analyze", data={"use_example": "1"}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 302)
        detector.assert_called_once()
        report = web_app._reports[response.location.rsplit("/", 1)[-1]]
        self.assertEqual(report["image_name"], SAMPLE_IMAGE_PATH.name)

    def test_api_endpoints_return_json(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertIn(health.json["status"], {"ok", "degraded"})
        self.assertEqual(health.json["checks"]["catalogue_products"], 1)

        missing = self.client.post("/api/analyze", data={}, content_type="multipart/form-data")
        self.assertEqual(missing.status_code, 400)
        self.assertIn("Choose a shelf image", missing.json["error"])
        self.assertEqual(missing.json["compliance_score"]["status"], "unavailable")
        self.assertEqual(self.client.get("/api/reports/nope").status_code, 404)
        self.assertEqual(self.client.get("/api/does-not-exist").json["error"] is not None, True)

        with patch.object(web_app, "detect_products", return_value=DETECTIONS):
            created = self.client.post(
                "/api/analyze",
                data={"shelf_image": (io.BytesIO(jpeg_bytes()), "shelf.png")},
                content_type="multipart/form-data",
            )
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json["detection_count"], 2)
        self.assertIn("compliance_score", created.json)
        self.assertIn("applicable_check_count", created.json["compliance_score"])
        self.assertTrue(created.json["persisted"])
        listed = self.client.get("/api/reports").json["reports"]
        self.assertEqual(listed[0]["report_id"], created.json["report_id"])

    def test_detector_unavailable_is_reported_honestly(self):
        with patch.object(web_app, "detect_products", side_effect=web_app.DetectorUnavailable("Model missing")):
            response = self.client.post(
                "/api/analyze",
                data={"shelf_image": (io.BytesIO(jpeg_bytes()), "shelf.jpg")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["error"], "Model missing")
        self.assertEqual(web_app._reports, {})
        self.assertEqual(list(self.upload_dir.iterdir()), [])

    def test_detect_only_page_and_dashboard_link(self):
        self.assertEqual(self.client.get("/detect").status_code, 200)
        self.assertIn(b"/detect", self.client.get("/").data)
        with patch.object(web_app, "detect_products", return_value=DETECTIONS):
            response = self.client.post(
                "/detect",
                data={"shelf_image": (io.BytesIO(jpeg_bytes()), "shelf.jpg"), "confidence_threshold": "0.4"},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"2 detections", response.data)
        result_id = next(iter(web_app._detect_results))
        self.assertEqual(self.client.get(f"/detect/{result_id}/image").status_code, 200)


class DetectorUnitTests(unittest.TestCase):
    def test_missing_checkpoint_reports_the_configured_path(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing.pt"
            detector = detector_module.YoloDetector(missing)
            with self.assertRaisesRegex(detector_module.DetectorUnavailable, "missing.pt"):
                detector.load()

    def test_inference_preserves_checkpoint_class_mapping_and_scores(self):
        class Scalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class Coordinates:
            def tolist(self):
                return [1.234, 2.345, 90.456, 110.567]

        class Box:
            cls = Scalar(1)
            conf = Scalar(0.87654)
            xyxy = [Coordinates()]

        class Result:
            orig_shape = (120, 100)
            names = {0: "product", 1: "beverage"}
            boxes = [Box()]

        class FakeModel:
            task = "detect"
            names = Result.names

            def predict(self, **kwargs):
                return [Result()]

        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model.pt"
            checkpoint.touch()
            detector = detector_module.YoloDetector(checkpoint)
            with patch("ultralytics.YOLO", return_value=FakeModel()):
                detections = detector.detect(Path(temp) / "shelf.jpg", confidence=0.4)

        self.assertEqual(detections, [{
            "label": "beverage", "class_id": 1, "confidence": 0.8765,
            "box": [1.23, 2.35, 90.46, 110.57], "image_width": 100, "image_height": 120,
        }])


@unittest.skipUnless(DEFAULT_MODEL_PATH.is_file() and SAMPLE_IMAGE_PATH.is_file(), "model or sample image missing")
class RealDetectorTests(unittest.TestCase):
    """Runs the actual configured checkpoint; skipped only when weights are absent."""

    def test_real_model_detects_products_on_sample_shelf(self):
        try:
            detector = detector_module.get_detector(DEFAULT_MODEL_PATH)
            detector.load()
        except detector_module.DetectorUnavailable as error:
            self.skipTest(f"detector runtime unavailable: {error}")
        detections = detector.detect(SAMPLE_IMAGE_PATH, confidence=0.35)
        self.assertGreater(len(detections), 10)
        with Image.open(SAMPLE_IMAGE_PATH) as sample:
            width, height = sample.size
        for detection in detections:
            self.assertEqual((detection["image_width"], detection["image_height"]), (width, height))
            self.assertGreaterEqual(detection["confidence"], 0.35)
            x1, y1, x2, y2 = detection["box"]
            self.assertTrue(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)
        info = detector.info()
        self.assertTrue(info["loaded"])
        self.assertEqual(info["class_names"], ["object"])
        with tempfile.TemporaryDirectory() as temp:
            output = detector_module.annotate_image(SAMPLE_IMAGE_PATH, detections, Path(temp) / "annotated.jpg")
            with Image.open(output) as annotated:
                self.assertEqual(annotated.size, (width, height))


if __name__ == "__main__":
    unittest.main()
