"""Regression tests for why products end up Needs Review, end to end.

Uses a deterministic colour-mean encoder (no model download) so similarities are exact:
pure red/blue crops match their references with cosine 1.0, a grey crop is ~0.58 from every
reference (below threshold) and a purple crop is equally close to red and blue (no margin).
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

import app as web_app
from backend.matching.calibration import file_fingerprint, load_runtime_calibration
from backend.matching.diagnostics import build_pipeline_diagnostics
from backend.matching.feedback import feedback_accuracy, load_feedback
from backend.matching.pipeline import ProductMatchingPipeline
from backend.matching.recognizer import ProductRecognizer
from backend.matching.shelf_mapper import CoordinateShelfMapper

RED, BLUE, GREY, PURPLE = (230, 20, 20), (20, 20, 230), (128, 128, 128), (128, 0, 128)


class ColourEncoder:
    model_id = "colour-mean-test"

    def encode_images(self, images):
        vectors = []
        for image in images:
            red, green, blue = image.convert("RGB").resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
            norm = (red * red + green * green + blue * blue) ** 0.5 or 1.0
            vectors.append([red / norm, green / norm, blue / norm])
        return vectors


def shelf_image(products: list[tuple[tuple[int, int, int], list[int]]], size=(400, 200)) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for colour, box in products:
        draw.rectangle(box, fill=colour)
    return image


def detection(box, width=400, height=200):
    return {"label": "object", "confidence": 0.9, "box": box, "image_width": width, "image_height": height}


SLOT1_BOX, SLOT2_BOX = [60, 40, 140, 160], [260, 40, 340, 160]


class DataDirMixin:
    def make_data_dir(self) -> Path:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "product_images").mkdir()
        for sku, colour in (("RED", RED), ("BLUE", BLUE)):
            Image.new("RGB", (32, 32), colour).save(root / "product_images" / f"{sku}.png")
        self.catalogue = [
            {"sku": "RED", "name": "Red Sauce", "image_local_path": "product_images/RED.png", "price": "10.00"},
            {"sku": "BLUE", "name": "Blue Soda", "image_local_path": "product_images/BLUE.png", "price": "20.00"},
        ]
        self.planogram = [
            {"store_id": "S", "planogram_id": "P", "shelf": "S1", "row": "1", "slot": "1", "expected_sku": "RED", "facing_count": "1"},
            {"store_id": "S", "planogram_id": "P", "shelf": "S1", "row": "1", "slot": "2", "expected_sku": "BLUE", "facing_count": "1"},
        ]
        return root

    def write_calibration(self, root: Path, **overrides) -> Path:
        payload = {"status": "calibrated_on_validation_set", "model_id": ColourEncoder.model_id,
                   "threshold": 0.9, "minimum_margin": 0.05, **overrides}
        path = root / "clip_calibration.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path


class RecognitionAndComparisonTests(DataDirMixin, unittest.TestCase):
    def setUp(self):
        self.root = self.make_data_dir()
        self.write_calibration(self.root)

    def pipeline(self, **kwargs) -> ProductMatchingPipeline:
        return ProductMatchingPipeline(data_dir=self.root, model_id=ColourEncoder.model_id, encoder=ColourEncoder(), **kwargs)

    def run_pipeline(self, products, boxes, trusted=True, **kwargs):
        image_path = self.root / "shelf.png"
        shelf_image(products).save(image_path)
        detections = [detection(box) for box in boxes]
        return self.pipeline(**kwargs).run(image_path, detections, self.catalogue, self.planogram,
                                           slot_mapper=CoordinateShelfMapper(trusted=trusted))

    @staticmethod
    def slot_statuses(result) -> dict[str, list[str]]:
        statuses: dict[str, list[str]] = {}
        for finding in result["slot_findings"]:
            statuses.setdefault(finding["slot"], []).append(finding["status"])
        return statuses

    # 1
    def test_confident_match_with_valid_calibration_and_trusted_slot_is_correct(self):
        result = self.run_pipeline([(RED, SLOT1_BOX), (BLUE, SLOT2_BOX)], [SLOT1_BOX, SLOT2_BOX])
        self.assertEqual(result["calibration"]["state"], "valid")
        self.assertEqual(result["calibration"]["acceptance_threshold"], 0.4)
        self.assertEqual(result["calibration"]["calibrated_threshold"], 0.9)
        self.assertEqual(result["calibration"]["threshold_source"], "configured_default")
        self.assertEqual([match["reason_code"] for match in result["matches"]], ["accepted", "accepted"])
        self.assertEqual([match["accepted_sku"] for match in result["matches"]], ["RED", "BLUE"])
        self.assertEqual(self.slot_statuses(result), {"1": ["Correct"], "2": ["Correct"]})

    def test_sku_score_above_point_five_identifies_independently_of_detector_confidence(self):
        image_path = self.root / "shelf.png"
        dark_red = (150, 60, 60)
        shelf_image([(dark_red, SLOT1_BOX)]).save(image_path)
        result = self.pipeline().run(
            image_path, [detection(SLOT1_BOX) | {"confidence": 0.10}],
            self.catalogue, self.planogram, slot_mapper=CoordinateShelfMapper(trusted=True),
            acceptance_threshold=0.50, minimum_margin=0.05,
        )
        match = result["matches"][0]
        self.assertGreater(match["top_similarity"], 0.50)
        self.assertEqual(match["accepted_sku"], "RED")
        self.assertEqual(match["reason_code"], "accepted")

    def test_high_detector_confidence_does_not_bypass_low_sku_similarity(self):
        result = self.run_pipeline([(GREY, SLOT1_BOX)], [SLOT1_BOX], acceptance_threshold=0.6)
        self.assertEqual(result["matches"][0]["detector_confidence"], 0.9)
        self.assertIsNone(result["matches"][0]["accepted_sku"])
        self.assertEqual(result["matches"][0]["reason_code"], "below_threshold")

    # 2
    def test_insufficient_similarity_stays_needs_review(self):
        result = self.run_pipeline([(GREY, SLOT1_BOX), (BLUE, SLOT2_BOX)], [SLOT1_BOX, SLOT2_BOX], acceptance_threshold=0.6)
        grey = result["matches"][0]
        self.assertIsNone(grey["accepted_sku"])
        self.assertEqual(grey["reason_code"], "below_threshold")
        self.assertLess(grey["top_similarity"], grey["threshold"])
        self.assertEqual(grey["candidates"][0]["sku"], "RED")  # candidates stay visible for review
        self.assertEqual(self.slot_statuses(result)["1"], ["Needs Review"])

    # 3
    def test_best_sku_is_identified_when_score_passes_even_with_small_margin(self):
        self.write_calibration(self.root, threshold=0.6, minimum_margin=0.05)
        result = self.run_pipeline([(PURPLE, SLOT1_BOX)], [SLOT1_BOX])
        purple = result["matches"][0]
        self.assertGreaterEqual(purple["top_similarity"], 0.6)
        self.assertLess(purple["top_margin"], 0.05)
        self.assertEqual(purple["reason_code"], "accepted")
        self.assertEqual(purple["accepted_sku"], "RED")  # deterministic top-ranked candidate
        self.assertEqual(self.slot_statuses(result)["1"], ["Correct"])

    # 4
    def test_missing_or_invalid_calibration_fails_closed_with_actionable_issue(self):
        cases = {
            "missing": (None, "calibration_missing"),
            "malformed": ("{not json", "calibration_unreadable"),
            "values": (json.dumps({"status": "calibrated_on_validation_set", "model_id": ColourEncoder.model_id,
                                   "threshold": "abc", "minimum_margin": 0.05}), "calibration_values_invalid"),
            "range": (json.dumps({"status": "calibrated_on_validation_set", "model_id": ColourEncoder.model_id,
                                  "threshold": 2.0, "minimum_margin": 0.05}), "calibration_values_out_of_range"),
            "uncalibrated": (json.dumps({"status": "uncalibrated", "threshold": None}), "calibration_status"),
        }
        for name, (content, code) in cases.items():
            with self.subTest(name):
                path = self.root / "clip_calibration.json"
                path.unlink(missing_ok=True)
                if content is not None:
                    path.write_text(content, encoding="utf-8")
                result = self.run_pipeline([(RED, SLOT1_BOX), (BLUE, SLOT2_BOX)], [SLOT1_BOX, SLOT2_BOX])
                calibration = result["calibration"]
                self.assertEqual(calibration["acceptance_threshold"], 0.4)
                self.assertEqual([issue["code"] for issue in calibration["issues"]], [code])
                self.assertIn("python -m tools.calibrate_matching", calibration["issues"][0]["message"])
                self.assertEqual([match["accepted_sku"] for match in result["matches"]], ["RED", "BLUE"])

    # 5
    def test_recognised_sku_with_untrusted_mapping_is_not_correct(self):
        result = self.run_pipeline([(RED, SLOT1_BOX), (BLUE, SLOT2_BOX)], [SLOT1_BOX, SLOT2_BOX], trusted=False)
        self.assertEqual([match["accepted_sku"] for match in result["matches"]], ["RED", "BLUE"])
        statuses = [finding["status"] for finding in result["findings"]]
        self.assertNotIn("Correct", statuses)
        proposals = [finding for finding in result["findings"] if finding.get("slot_mapping", {}).get("slot_identity")]
        self.assertEqual(len(proposals), 2)  # the proposed slot is kept for the reviewer

    # 6
    def test_reliably_identified_sku_in_wrong_slot_is_misplaced(self):
        result = self.run_pipeline([(RED, SLOT2_BOX)], [SLOT2_BOX])
        self.assertIn("Misplaced", self.slot_statuses(result)["2"])
        misplaced = next(finding for finding in result["slot_findings"] if finding["status"] == "Misplaced")
        self.assertEqual((misplaced["sku"], misplaced["expected_sku"]), ("RED", "BLUE"))

    # 7 (pipeline level; the app-level void rule is in AuditFlowTests)
    def test_no_detection_is_not_reported_missing(self):
        result = self.run_pipeline([(RED, SLOT1_BOX)], [SLOT1_BOX])
        self.assertEqual(self.slot_statuses(result)["2"], ["Needs Review"])

    # 9
    def test_invalid_box_and_empty_catalogue_are_handled(self):
        image = shelf_image([(RED, SLOT1_BOX)])
        recognizer = ProductRecognizer(data_dir=self.root, encoder=ColourEncoder(), model_id=ColourEncoder.model_id,
                                       acceptance_threshold=0.9, minimum_margin=0.05)
        invalid = [detection([150, 150, 100, 100]), {"label": "object", "box": None}, detection(SLOT1_BOX)]
        matches = recognizer.match_detections(image, invalid, self.catalogue)
        self.assertEqual([match["reason_code"] for match in matches], ["invalid_crop", "invalid_crop", "accepted"])
        self.assertEqual([match["detection_index"] for match in matches], [0, 1, 2])

        empty = recognizer.match_detections(image, [detection(SLOT1_BOX)], [])
        self.assertEqual(empty[0]["reason_code"], "no_catalogue_references")
        self.assertIsNone(empty[0]["accepted_sku"])

        image_path = self.root / "shelf.png"
        image.save(image_path)
        result = self.pipeline().run(image_path, invalid, [], self.planogram, slot_mapper=CoordinateShelfMapper(trusted=True))
        self.assertTrue(all(match["accepted_sku"] is None for match in result["matches"]))
        diagnostics = build_pipeline_diagnostics(
            detections=[{**item, "detection_index": index} for index, item in enumerate(invalid)],
            matching_audit=result, compliance=[], price_results=[], price_audit={}, layout_trusted=True,
            layout_validation={}, catalogue_size=0, planogram=self.planogram,
        )
        self.assertIn("empty_catalogue", [issue["code"] for issue in diagnostics["scene"]["issues"]])
        # With no catalogue references at all, that is the reason reported for every detection.
        self.assertEqual({item["reason_code"] for item in diagnostics["detections"]}, {"no_catalogue_references"})

    def test_feedback_accuracy_measures_the_matcher_on_reviewed_crops(self):
        records = [
            {"decision": "correct", "true_sku": "A", "model_top_sku": "A", "model_top_similarity": 0.80, "model_top_margin": 0.10},
            {"decision": "wrong", "true_sku": "B", "model_top_sku": "A", "model_top_similarity": 0.55, "model_top_margin": 0.08},
            {"decision": "not_product", "true_sku": None, "model_top_sku": "C", "model_top_similarity": 0.52, "model_top_margin": 0.20},
            {"decision": "correct", "true_sku": "D", "model_top_sku": "D", "model_top_similarity": 0.70, "model_top_margin": 0.01},
            {"decision": "correct", "true_sku": "E"},  # recorded without model scores: not counted
        ]
        strict = feedback_accuracy(records, 0.60, 0.05)
        self.assertEqual((strict["reviewed"], strict["accepted"], strict["correct"], strict["precision"]), (4, 2, 2, 1.0))
        loose = feedback_accuracy(records, 0.50, 0.05)
        self.assertEqual((loose["accepted"], loose["correct"], loose["precision"]), (4, 2, 0.5))
        self.assertEqual(loose["coverage"], 0.6667)  # 2 of 3 in-catalogue crops found

    # 11
    def test_matcher_and_detector_mismatch_are_detected_and_reported(self):
        self.write_calibration(self.root, model_id="torchvision-mobilenet_v3_large-imagenet1k_v2")
        result = self.run_pipeline([(RED, SLOT1_BOX)], [SLOT1_BOX])
        self.assertEqual(result["calibration"]["state"], "model_mismatch")
        self.assertEqual(result["calibration"]["issues"][0]["code"], "calibration_model_mismatch")
        self.assertEqual(result["matches"][0]["accepted_sku"], "RED")

        detector_file = self.root / "detector.pt"
        detector_file.write_bytes(b"current checkpoint")
        current = file_fingerprint(detector_file)
        self.write_calibration(self.root, detector_model={"file": "old.pt", "size": 1, "sha256": "0" * 64})
        stale = load_runtime_calibration(self.root / "clip_calibration.json", ColourEncoder.model_id,
                                         detector_fingerprint=current)
        self.assertEqual(stale["state"], "stale")
        self.assertEqual(stale["issues"][0]["code"], "calibration_detector_changed")
        self.assertEqual(stale["threshold"], 0.9)  # well-formed thresholds still apply, with a recalibration warning

        self.write_calibration(self.root, detector_model=current)
        fresh = load_runtime_calibration(self.root / "clip_calibration.json", ColourEncoder.model_id,
                                         detector_fingerprint=current)
        self.assertEqual((fresh["state"], fresh["issues"]), ("valid", []))


class AuditFlowTests(DataDirMixin, unittest.TestCase):
    """Runs POST /api/analyze with a real matching pipeline; detectors and OCR are stubbed."""

    def setUp(self):
        root = self.make_data_dir()
        (root / "catalogue.csv").write_text(
            "sku,name,category,brand,pack_size,price,model_label,active,image_local_path\n"
            "RED,Red Sauce,Sauces,X,1,10.00,red sauce,true,product_images/RED.png\n"
            "BLUE,Blue Soda,Drinks,X,1,20.00,blue soda,true,product_images/BLUE.png\n",
            encoding="utf-8",
        )
        (root / "planogram.csv").write_text(
            "store_id,planogram_id,shelf,row,slot,expected_sku,facing_count\nS,P,S1,1,1,RED,1\nS,P,S1,1,2,BLUE,1\n",
            encoding="utf-8",
        )
        (root / "uploads").mkdir()
        self.root = root
        for name, value in (("DATA_DIR", root), ("CATALOGUE_PATH", root / "catalogue.csv"),
                            ("SETTINGS_PATH", root / "settings.json"), ("AUDIT_RESULTS_DIR", root / "results"),
                            ("UPLOAD_DIR", root / "uploads")):
            patcher = patch.object(web_app, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # These scenarios use a strict operator threshold so the colour similarities stay unambiguous.
        (root / "settings.json").write_text(json.dumps({"match_threshold": 0.9, "match_margin": 0.05}), encoding="utf-8")
        old_reports = web_app._reports.copy()
        self.addCleanup(lambda: (web_app._reports.clear(), web_app._reports.update(old_reports)))
        self.pipeline = ProductMatchingPipeline(data_dir=root, model_id=ColourEncoder.model_id, encoder=ColourEncoder(),
                                                acceptance_threshold=0.9, minimum_margin=0.05)
        self.client = web_app.app.test_client()

    def stubs(self, boxes, voids=(), price_tags=(), crop_ocr=None, full_ocr=()):
        """Patch the detectors, OCR and matcher factory (the matching pipeline itself is real)."""
        from contextlib import ExitStack
        auxiliary = {"price_tags": [dict(tag) for tag in price_tags],
                     "voids": [{"label": "void", "confidence": 0.8, "image_width": 400, "image_height": 200, **void}
                               for void in voids],
                     "model": {"class_names": ["Price_Tag", "void"]}}
        crop_result = crop_ocr or {"raw_ocr_text": "", "price": None, "ocr_confidence": None, "ocr_status": "unreadable"}
        stack = ExitStack()
        for name, value in (("detect_products", [detection(box) for box in boxes]),
                            ("detect_price_tags_and_voids", auxiliary), ("read_price_crop", crop_result),
                            ("read_price_tags", list(full_ocr)), ("_get_matching_pipeline", self.pipeline)):
            stack.enter_context(patch.object(web_app, name, return_value=value))
        return stack

    def audit(self, products, boxes, voids=(), price_tags=(), crop_ocr=None, full_ocr=(), calibrated=True, **form_fields):
        buffer = io.BytesIO()
        shelf_image(products).save(buffer, format="PNG")
        form = {"shelf_image": (io.BytesIO(buffer.getvalue()), "shelf.png"), "price_check": "on", **form_fields}
        if calibrated:
            form["calibrated_layout"] = "on"
        with self.stubs(boxes, voids, price_tags, crop_ocr, full_ocr):
            response = self.client.post("/api/analyze", data=form, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 201, response.get_json())
        return web_app._reports[response.get_json()["report_id"]]

    def review(self, report, index, **body):
        response = self.client.post(f"/reports/{report['report_id']}/review", json={"detection_index": index, **body})
        return response.status_code, response.get_json()

    def reanalyze(self, report, boxes):
        with self.stubs(boxes):
            response = self.client.post(f"/reports/{report['report_id']}/reanalyze")
        self.assertEqual(response.status_code, 302)
        new_id = response.location.rstrip("/").rsplit("/", 1)[-1]
        self.assertNotEqual(new_id, report["report_id"])
        return web_app._reports[new_id]

    @staticmethod
    def statuses(report):
        return {row["slot"]: row["status"] for row in report["compliance"]}

    @staticmethod
    def displayed(report):
        return {finding["slot"]: finding["status"] for finding in report["planogram_comparison"]["findings"]
                if finding.get("slot") is not None}

    def test_correct_products_are_compliant_through_every_stage(self):
        report = self.audit([(RED, SLOT1_BOX), (BLUE, SLOT2_BOX)], [SLOT1_BOX, SLOT2_BOX])
        self.assertTrue(report["layout_trusted"])
        self.assertEqual(self.statuses(report), {"1": "Compliant", "2": "Compliant"})
        self.assertEqual(self.displayed(report), {"1": "MATCH", "2": "MATCH"})
        self.assertEqual(report["compliance_score"]["breakdown"]["placement"]["compliant"], 2)
        diagnostics = report["pipeline_diagnostics"]
        self.assertEqual([item["reason_code"] for item in diagnostics["detections"]], ["compliant", "compliant"])
        self.assertEqual(diagnostics["detections"][0]["mapping"]["slot"], "R1S1")
        self.assertEqual(diagnostics["detections"][0]["planogram"]["expected_sku"], "RED")

    # 7
    def test_absent_product_is_missing_only_with_a_covering_void(self):
        report = self.audit([(RED, SLOT1_BOX)], [SLOT1_BOX])
        self.assertEqual(self.statuses(report)["2"], "Needs Review")
        self.assertEqual(report["compliance"][1]["reason_code"], "absence_not_confirmed")
        self.assertEqual(self.displayed(report)["2"], "Needs Review")

        report = self.audit([(RED, SLOT1_BOX)], [SLOT1_BOX], voids=[{"box": [210, 20, 395, 180]}])
        self.assertEqual(self.statuses(report)["2"], "Missing Expected Product")
        self.assertEqual(self.displayed(report)["2"], "MISSING PRODUCT")

    # 8
    def test_unreadable_or_missing_price_tag_does_not_affect_identity_or_placement(self):
        tag = {"label": "price_tag", "confidence": 0.8, "box": [70, 170, 130, 195]}
        report = self.audit([(RED, SLOT1_BOX), (BLUE, SLOT2_BOX)], [SLOT1_BOX, SLOT2_BOX], price_tags=[tag])
        self.assertEqual(self.statuses(report), {"1": "Compliant", "2": "Compliant"})
        self.assertEqual({row["sku"]: row["status"] for row in report["price_results"]}, {"RED": "Unverified", "BLUE": "Unverified"})
        self.assertEqual(report["price_audit"]["tags"][0]["ocr_status"], "unreadable")
        self.assertEqual([item["identity_status"] for item in report["pipeline_diagnostics"]["detections"]],
                         ["Identified", "Identified"])
        score = report["compliance_score"]
        self.assertEqual((score["breakdown"]["placement"]["compliant"], score["breakdown"]["price"]["unresolved"]), (2, 2))

        readable = [{"text": "₹ 10.00", "price": 10.0, "confidence": 0.9, "box": [80, 172, 120, 192],
                     "rule": "currency marker or two decimals"}]
        report = self.audit([(RED, SLOT1_BOX), (BLUE, SLOT2_BOX)], [SLOT1_BOX, SLOT2_BOX], full_ocr=readable)
        self.assertEqual(report["price_audit"]["tag_source"], "full_image_ocr")
        self.assertEqual({row["sku"]: row["status"] for row in report["price_results"]}, {"RED": "Match", "BLUE": "Unverified"})

    # 10
    def test_later_stages_do_not_overwrite_or_contradict_slot_status(self):
        report = self.audit([(RED, SLOT1_BOX), (BLUE, SLOT2_BOX)], [SLOT1_BOX, SLOT2_BOX],
                            voids=[{"box": [205, 20, 255, 180]}])
        self.assertEqual(self.statuses(report), {"1": "Compliant", "2": "Needs Review"})
        self.assertEqual(report["compliance"][1]["reason_code"], "void_conflicts_with_products")
        # The displayed comparison and the score agree with the authoritative slot audit.
        self.assertEqual(self.displayed(report), {"1": "MATCH", "2": "Needs Review"})
        outcomes = {check["slot"]: check["outcome"] for check in report["compliance_score"]["checks"]
                    if check["category"] == "placement"}
        self.assertEqual(outcomes, {"1": "pass", "2": "unresolved"})

    def test_products_outside_the_catalogue_are_explained(self):
        report = self.audit([(GREY, SLOT1_BOX), (GREY, SLOT2_BOX)], [SLOT1_BOX, SLOT2_BOX])
        self.assertEqual(self.statuses(report), {"1": "Needs Review", "2": "Needs Review"})
        issues = {issue["code"]: issue["message"] for issue in report["pipeline_diagnostics"]["scene"]["issues"]}
        self.assertIn("no_catalogue_matches", issues)
        self.assertIn("not in the 2-product catalogue", issues["no_catalogue_matches"])
        self.assertIn(issues["no_catalogue_matches"], report["limitations"])
        self.assertEqual(report["pipeline_diagnostics"]["reason_counts"]["detections"], {"below_threshold": 2})

    def test_operator_threshold_from_settings_and_form_controls_acceptance(self):
        dark_red = (150, 60, 60)  # cosine ~0.92 to the red reference, ~0.45 to blue
        strict = self.audit([(dark_red, SLOT1_BOX)], [SLOT1_BOX], match_threshold="0.95")
        self.assertEqual(strict["detections"][0]["matched_sku"], "")
        self.assertEqual(strict["detections"][0]["match_reason_code"], "below_threshold")
        relaxed = self.audit([(dark_red, SLOT1_BOX)], [SLOT1_BOX], match_threshold="0.90")
        self.assertEqual(relaxed["detections"][0]["matched_sku"], "RED")
        calibration = relaxed["matching_audit"]["calibration"]
        self.assertEqual((calibration["acceptance_threshold"], calibration["threshold_source"]), (0.9, "operator"))
        self.assertEqual(relaxed["analysis_options"]["match_threshold"], 0.9)
        self.assertIn("operator-set similarity threshold 0.9", " ".join(relaxed["limitations"]))

        (self.root / "settings.json").write_text(json.dumps({"match_threshold": 0.95}), encoding="utf-8")
        from_settings = self.audit([(dark_red, SLOT1_BOX)], [SLOT1_BOX])
        self.assertEqual(from_settings["matching_audit"]["calibration"]["acceptance_threshold"], 0.95)

        buffer = io.BytesIO()
        shelf_image([(RED, SLOT1_BOX)]).save(buffer, format="PNG")
        with self.stubs([SLOT1_BOX]):
            response = self.client.post("/api/analyze", data={"shelf_image": (buffer, "shelf.png"), "match_threshold": "0.1"},
                                        content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertIn("between 0.30 and 0.95", response.get_json()["error"])

    def test_lower_threshold_than_calibrated_is_flagged_on_the_report(self):
        self.write_calibration(self.root, threshold=0.95)
        self.pipeline = ProductMatchingPipeline(data_dir=self.root, model_id=ColourEncoder.model_id, encoder=ColourEncoder())
        report = self.audit([(RED, SLOT1_BOX)], [SLOT1_BOX], match_threshold="0.60")
        codes = [issue["code"] for issue in report["pipeline_diagnostics"]["scene"]["issues"]]
        self.assertIn("operator_threshold_below_calibrated", codes)
        page = self.client.get(f"/reports/{report['report_id']}").data
        self.assertIn(b"calibrated 0.95", page)

    def test_settings_save_validates_and_resets_match_threshold(self):
        response = self.client.post("/settings/save", data={"match_threshold": "0.45", "match_margin": "0.03"})
        self.assertEqual(response.status_code, 302)
        saved = json.loads((self.root / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["match_threshold"], saved["match_margin"]), (0.45, 0.03))
        self.client.post("/settings/save", data={"match_threshold": "0.99"})
        self.assertEqual(json.loads((self.root / "settings.json").read_text(encoding="utf-8"))["match_threshold"], 0.45)
        self.client.post("/settings/save", data={"match_reset": "1"})
        saved = json.loads((self.root / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["match_threshold"], saved["match_margin"]), (0.40, None))
        page = self.client.get("/settings").data
        self.assertIn(b"Product Matching", page)
        self.assertIn(b'name="match_threshold"', page)
        self.assertIn(b'name="match_threshold"', self.client.get("/upload").data)

    def test_confirming_a_match_stores_feedback_and_a_reference(self):
        report = self.audit([(RED, SLOT1_BOX)], [SLOT1_BOX])
        status, body = self.review(report, 0, decision="correct")
        self.assertEqual(status, 200, body)
        self.assertEqual((body["review"]["decision"], body["review"]["sku"]), ("correct", "RED"))
        records = load_feedback(self.root)
        self.assertEqual(len(records), 1)
        self.assertEqual((records[0]["true_sku"], records[0]["model_top_sku"]), ("RED", "RED"))
        self.assertTrue((self.root / "feedback" / "references" / records[0]["reference_image"]).is_file())
        saved = json.loads((self.root / "results" / f"{report['report_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["reviews"]["0"]["decision"], "correct")
        # A second verdict on the same detection replaces the first instead of piling up.
        self.review(report, 0, decision="correct")
        self.assertEqual(len(load_feedback(self.root)), 1)
        page = self.client.get(f"/reports/{report['report_id']}").data
        self.assertIn("✓ Confirmed".encode(), page)

    def test_correcting_an_unknown_product_teaches_the_matcher(self):
        green = (20, 200, 20)  # no green product in the catalogue
        report = self.audit([(green, SLOT1_BOX)], [SLOT1_BOX])
        self.assertEqual(report["detections"][0]["matched_sku"], "")
        status, body = self.review(report, 0, decision="wrong", sku="BLUE")
        self.assertEqual(status, 200, body)
        again = self.reanalyze(report, [SLOT1_BOX])
        self.assertEqual(again["detections"][0]["matched_sku"], "BLUE")
        match = again["matching_audit"]["matches"][0]
        self.assertEqual(match["candidates"][0]["reference"], "review")
        self.assertIn("reviewed reference photo", match["detail"])
        self.assertEqual(again["analysis_options"]["match_threshold"], 0.9)

    def test_new_product_from_review_is_added_and_recognised(self):
        green = (20, 200, 20)
        report = self.audit([(green, SLOT1_BOX)], [SLOT1_BOX])
        status, body = self.review(report, 0, decision="wrong",
                                   new_product={"name": "Green Tea", "price": "45.00", "category": "Drinks"})
        self.assertEqual(status, 200, body)
        sku = body["review"]["sku"]
        self.assertTrue(sku.startswith("LOCAL-"))
        catalogue = {row["sku"]: row for row in web_app._read_catalogue()}
        self.assertEqual((catalogue[sku]["name"], catalogue[sku]["price"]), ("Green Tea", "45.00"))
        self.assertTrue((self.root / catalogue[sku]["image_local_path"]).is_file())
        self.assertEqual(self.reanalyze(report, [SLOT1_BOX])["detections"][0]["matched_sku"], sku)

    def test_not_a_product_verdict_suppresses_similar_crops(self):
        report = self.audit([(RED, SLOT1_BOX)], [SLOT1_BOX])
        self.assertEqual(report["detections"][0]["matched_sku"], "RED")
        status, _body = self.review(report, 0, decision="not_product")
        self.assertEqual(status, 200)
        again = self.reanalyze(report, [SLOT1_BOX])
        self.assertEqual(again["detections"][0]["matched_sku"], "")
        self.assertEqual(again["detections"][0]["match_reason_code"], "known_non_product")

    def test_new_product_with_an_existing_name_is_refused(self):
        report = self.audit([(GREY, SLOT1_BOX)], [SLOT1_BOX])
        status, body = self.review(report, 0, decision="wrong", new_product={"name": "red sauce"})
        self.assertEqual(status, 400)
        self.assertIn("already in the catalogue (SKU RED)", body["error"])
        self.assertEqual(len(web_app._read_catalogue()), 2)

    def test_not_a_product_examples_are_not_offered_as_products(self):
        report = self.audit([(RED, SLOT1_BOX)], [SLOT1_BOX])
        self.review(report, 0, decision="not_product")
        again = self.reanalyze(report, [SLOT1_BOX])
        self.assertEqual(again["matching_audit"]["matches"][0]["candidates"][0]["sku"], "__not_product__")
        item = web_app._review_queue(again)[0]
        self.assertNotIn("__not_product__", [candidate["sku"] for candidate in item["candidates"]])
        self.assertEqual(item["candidates"][0]["sku"], "RED")
        # "Correct" confirms the best real product, never the not-a-product example.
        status, body = self.review(again, 0, decision="correct")
        self.assertEqual((status, body["review"]["sku"]), (200, "RED"))

    def test_invalid_review_requests_are_rejected(self):
        report = self.audit([(GREY, SLOT1_BOX)], [SLOT1_BOX])
        self.assertEqual(self.review(report, 5, decision="correct")[0], 404)
        self.assertEqual(self.review(report, 0, decision="maybe")[0], 400)
        self.assertEqual(self.review(report, 0, decision="wrong")[0], 400)
        self.assertEqual(self.review(report, 0, decision="wrong", sku="NOPE")[0], 400)
        self.assertEqual(self.review(report, 0, decision="wrong", new_product={"name": ""})[0], 400)
        self.assertEqual(self.review(report, 0, decision="wrong", new_product={"name": "X", "price": "abc"})[0], 400)
        self.assertEqual(self.client.post("/reports/missing/review", json={}).status_code, 404)
        self.assertEqual(load_feedback(self.root), [])

    def test_untrusted_layout_never_confirms_identified_products(self):
        report = self.audit([(RED, SLOT1_BOX), (BLUE, SLOT2_BOX)], [SLOT1_BOX, SLOT2_BOX], calibrated=False)
        self.assertFalse(report["layout_trusted"])
        self.assertEqual(self.statuses(report), {"1": "Needs Review", "2": "Needs Review"})
        self.assertEqual({item["reason_code"] for item in report["pipeline_diagnostics"]["detections"]}, {"layout_untrusted"})


if __name__ == "__main__":
    unittest.main()
