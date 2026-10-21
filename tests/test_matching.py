"""Comprehensive tests for the ShelfSense AI product matching pipeline.

Covers:
1. Catalogue image loading and error handling.
2. Embedding cache generation, persistence, and cache hit verification.
3. Crop extraction and cosine similarity ranking.
4. Fail-closed rejection of uncertain/uncalibrated matches.
5. Shelf row clustering and multi-row slot mapping.
6. Planogram comparison findings (Correct, Misplaced, Missing, Unexpected, Needs Review).
7. End-to-end pipeline execution and atomic JSON storage.
8. Web application audit workflow and CSV export integration.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app as web_app
from backend.matching.catalog import CatalogueEmbeddingCache, load_catalogue_references
from backend.matching.encoder import ImageEncoder, PerceptualFeatureEncoder
from backend.matching.pipeline import ProductMatchingPipeline
from backend.matching.planogram_compare import compare_planogram
from backend.matching.recognizer import ProductRecognizer, cosine_similarity, crop_detection
from backend.matching.shelf_mapper import CoordinateShelfMapper
from backend.matching.storage import save_audit_result


class DummyEncoder:
    """Deterministic encoder for testing."""

    model_id = "dummy-test-v1"

    def encode_images(self, images: list[Image.Image]) -> list[list[float]]:
        results = []
        for img in images:
            # Deterministic vector based on average color
            raw = list(img.convert("RGB").resize((2, 2)).tobytes())
            avg_r = sum(raw[0::3]) / (4 * 255.0)
            avg_g = sum(raw[1::3]) / (4 * 255.0)
            avg_b = sum(raw[2::3]) / (4 * 255.0)
            vec = [avg_r, avg_g, avg_b, 1.0]
            norm = sum(v * v for v in vec) ** 0.5
            results.append([v / norm for v in vec])
        return results


class ProductMatchingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.image_dir = self.root / "product_images"
        self.image_dir.mkdir(parents=True)

        # Create two distinct test product images (Red and Blue)
        self.red_path = self.image_dir / "SKU_RED.jpg"
        red_img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        red_img.save(self.red_path, format="JPEG")

        self.blue_path = self.image_dir / "SKU_BLUE.jpg"
        blue_img = Image.new("RGB", (64, 64), color=(0, 0, 255))
        blue_img.save(self.blue_path, format="JPEG")

        self.catalogue = [
            {
                "sku": "SKU_RED",
                "name": "Red Tomato Sauce",
                "image_local_path": "product_images/SKU_RED.jpg",
                "active": "true",
            },
            {
                "sku": "SKU_BLUE",
                "name": "Blue Energy Drink",
                "image_local_path": "product_images/SKU_BLUE.jpg",
                "active": "true",
            },
            {
                "sku": "SKU_MISSING_IMG",
                "name": "Missing Photo Item",
                "image_local_path": "product_images/nonexistent.jpg",
                "active": "true",
            },
        ]

        self.planogram = [
            {"store_id": "S1", "planogram_id": "P1", "shelf": "Shelf 1", "row": "1", "slot": "1", "expected_sku": "SKU_RED", "facing_count": "1"},
            {"store_id": "S1", "planogram_id": "P1", "shelf": "Shelf 1", "row": "1", "slot": "2", "expected_sku": "SKU_BLUE", "facing_count": "1"},
            {"store_id": "S1", "planogram_id": "P1", "shelf": "Shelf 1", "row": "2", "slot": "1", "expected_sku": "SKU_RED", "facing_count": "2"},
            {"store_id": "S1", "planogram_id": "P1", "shelf": "Shelf 1", "row": "2", "slot": "2", "expected_sku": "", "facing_count": "0"},
        ]

    # --- Criterion 1: Catalogue Image Loading ---
    def test_catalogue_images_can_be_loaded(self):
        refs = load_catalogue_references(self.catalogue, self.root)
        skus = [r["sku"] for r in refs]
        self.assertIn("SKU_RED", skus)
        self.assertIn("SKU_BLUE", skus)
        self.assertNotIn("SKU_MISSING_IMG", skus)
        self.assertEqual(len(refs), 2)

    def test_corrupt_catalogue_images_are_skipped(self):
        corrupt_path = self.image_dir / "CORRUPT.jpg"
        corrupt_path.write_bytes(b"not a valid jpeg file")
        catalogue = [
            {"sku": "SKU_CORRUPT", "name": "Broken", "image_local_path": "product_images/CORRUPT.jpg"}
        ]
        refs = load_catalogue_references(catalogue, self.root)
        self.assertEqual(refs, [])

    # --- Criterion 2: Embedding Cache Generation & Persistence ---
    def test_reference_embeddings_created_and_cached(self):
        cache_path = self.root / "cache" / "embeddings.json"
        cache = CatalogueEmbeddingCache(cache_path)
        encoder = DummyEncoder()
        refs = load_catalogue_references(self.catalogue, self.root)

        # First run: should compute all
        embeddings1, stats1 = cache.get_embeddings(refs, encoder)
        self.assertEqual(stats1["computed"], 2)
        self.assertEqual(stats1["cached"], 0)
        self.assertIn("SKU_RED", embeddings1)
        self.assertIn("SKU_BLUE", embeddings1)
        self.assertTrue(cache_path.is_file())

        # Second run: should load all from cache
        embeddings2, stats2 = cache.get_embeddings(refs, encoder)
        self.assertEqual(stats2["computed"], 0)
        self.assertEqual(stats2["cached"], 2)
        self.assertEqual(embeddings1["SKU_RED"], embeddings2["SKU_RED"])

    # --- Criterion 3: Crop Extraction & Known SKU Ranking ---
    def test_known_product_crop_returns_plausible_ranked_candidates(self):
        # Create shelf image with a red product on the left and a blue product on the right
        shelf = Image.new("RGB", (200, 100), color=(255, 255, 255))
        shelf.paste(Image.new("RGB", (40, 40), (255, 0, 0)), (10, 10))
        shelf.paste(Image.new("RGB", (40, 40), (0, 0, 255)), (110, 10))

        recognizer = ProductRecognizer(
            data_dir=self.root,
            encoder=DummyEncoder(),
            top_k=2,
        )

        detections = [
            {"label": "red product", "confidence": 0.95, "box": [10, 10, 50, 50], "image_width": 200, "image_height": 100},
            {"label": "blue product", "confidence": 0.92, "box": [110, 10, 150, 50], "image_width": 200, "image_height": 100},
        ]

        matches = recognizer.match_detections(shelf, detections, self.catalogue)
        self.assertEqual(len(matches), 2)

        # Red crop should rank SKU_RED first
        red_candidates = matches[0]["candidates"]
        self.assertEqual(red_candidates[0]["sku"], "SKU_RED")
        self.assertGreater(red_candidates[0]["similarity"], red_candidates[1]["similarity"])

        # Blue crop should rank SKU_BLUE first
        blue_candidates = matches[1]["candidates"]
        self.assertEqual(blue_candidates[0]["sku"], "SKU_BLUE")
        self.assertGreater(blue_candidates[0]["similarity"], blue_candidates[1]["similarity"])

    # --- Criterion 4: Fail-closed Rejection of Uncertain Matches ---
    def test_uncertain_matches_rejected_rather_than_silently_accepted(self):
        shelf = Image.new("RGB", (100, 100), color=(128, 128, 128))
        detection = {"label": "grey crop", "box": [10, 10, 50, 50], "image_width": 100, "image_height": 100}

        # Case A: No threshold configured -> advisory only, status is Needs Review
        recognizer_uncalibrated = ProductRecognizer(data_dir=self.root, encoder=DummyEncoder(), acceptance_threshold=None)
        match_uncalibrated = recognizer_uncalibrated.match_detections(shelf, [detection], self.catalogue)[0]
        self.assertIsNone(match_uncalibrated["accepted_sku"])
        self.assertEqual(match_uncalibrated["status"], "Unknown / Needs Review")

        # Case B: Score is below configured threshold -> rejected
        recognizer_strict = ProductRecognizer(data_dir=self.root, encoder=DummyEncoder(), acceptance_threshold=0.999)
        match_strict = recognizer_strict.match_detections(shelf, [detection], self.catalogue)[0]
        self.assertIsNone(match_strict["accepted_sku"])
        self.assertEqual(match_strict["status"], "Unknown / Needs Review")

        # Case C: Invalid crop box -> returns Needs Review
        invalid_detection = {"label": "bad box", "box": [-10, 50, 0, 50], "image_width": 100, "image_height": 100}
        match_invalid = recognizer_uncalibrated.match_detections(shelf, [invalid_detection], self.catalogue)[0]
        self.assertIsNone(match_invalid["accepted_sku"])
        self.assertEqual(match_invalid["status"], "Unknown / Needs Review")

    # --- Criterion 5: Shelf Row Clustering & Multi-Row Slot Mapping ---
    def test_multiple_product_detections_assigned_to_separate_shelf_rows(self):
        # 4 detections: 2 on row 1 (y ≈ 50), 2 on row 2 (y ≈ 250)
        # Deliberately scrambled x coordinates across rows
        detections = [
            {"id": "r2_left", "box": [20, 240, 60, 280], "image_width": 300, "image_height": 400},
            {"id": "r1_right", "box": [200, 40, 240, 80], "image_width": 300, "image_height": 400},
            {"id": "r1_left", "box": [30, 40, 70, 80], "image_width": 300, "image_height": 400},
            {"id": "r2_right", "box": [210, 240, 250, 280], "image_width": 300, "image_height": 400},
        ]

        clustered = CoordinateShelfMapper.cluster_detections_by_row(detections)
        self.assertEqual(len(clustered), 2)
        row1_ids = [d["id"] for d in clustered[0]]
        row2_ids = [d["id"] for d in clustered[1]]
        self.assertEqual(row1_ids, ["r1_left", "r1_right"])
        self.assertEqual(row2_ids, ["r2_left", "r2_right"])

    def test_coordinate_shelf_mapper_maps_rows_and_boundary_rejection(self):
        mapper = CoordinateShelfMapper(boundary_margin=0.05, trusted=True)
        # Row 1 (y=50 in h=300 -> 16%), Slot 1 (x=50 in w=300 -> 16%)
        det_r1_s1 = {"box": [30, 30, 70, 70], "image_width": 300, "image_height": 300}
        res1 = mapper.map_detection(det_r1_s1, self.planogram)
        self.assertEqual(res1["slot_identity"][-2:], ("1", "1"))

        # Close to horizontal boundary between row 1 and row 2 (y ≈ 150 in h=300) -> Needs Review
        det_boundary = {"box": [30, 145, 70, 155], "image_width": 300, "image_height": 300}
        res_boundary = mapper.map_detection(det_boundary, self.planogram)
        self.assertIsNone(res_boundary["slot_identity"])
        self.assertEqual(res_boundary["status"], "Needs Review")

    # --- Criterion 6: Planogram Comparison Known Test Cases ---
    def test_planogram_comparison_identifies_known_test_cases(self):
        mapper = CoordinateShelfMapper(trusted=True)

        # Case A: Correct match on Shelf 1, Row 1, Slot 1
        matches_correct = [
            {
                "detection_index": 0,
                "box": [30, 30, 70, 70], "image_width": 300, "image_height": 300,
                "accepted_sku": "SKU_RED",
                "candidates": [{"sku": "SKU_RED", "product": "Red Tomato Sauce", "similarity": 0.98}],
            }
        ]
        res = compare_planogram(self.planogram, matches_correct, mapper)
        slot_r1_s1 = next(f for f in res["findings"] if f.get("row") == "1" and f.get("slot") == "1")
        self.assertEqual(slot_r1_s1["status"], "Correct")

        # Case B: Misplaced (SKU_BLUE placed at slot expecting SKU_RED)
        matches_misplaced = [
            {
                "detection_index": 0,
                "box": [30, 30, 70, 70], "image_width": 300, "image_height": 300,
                "accepted_sku": "SKU_BLUE",
                "candidates": [{"sku": "SKU_BLUE", "product": "Blue Energy Drink", "similarity": 0.95}],
            }
        ]
        res_misplaced = compare_planogram(self.planogram, matches_misplaced, mapper)
        misplaced_finding = next(f for f in res_misplaced["findings"] if f.get("status") == "Misplaced")
        self.assertEqual(misplaced_finding["sku"], "SKU_BLUE")
        self.assertEqual(misplaced_finding["row"], "1")
        self.assertEqual(misplaced_finding["slot"], "1")

        # Case C: Unexpected product at intentionally empty slot (Row 2, Slot 2)
        matches_unexpected = [
            {
                "detection_index": 0,
                "box": [220, 220, 260, 260], "image_width": 300, "image_height": 300,
                "accepted_sku": "SKU_RED",
                "candidates": [{"sku": "SKU_RED", "product": "Red Tomato Sauce", "similarity": 0.92}],
            }
        ]
        res_unexpected = compare_planogram(self.planogram, matches_unexpected, mapper)
        unexpected_finding = next(f for f in res_unexpected["findings"] if f.get("status") == "Unexpected")
        self.assertEqual(unexpected_finding["row"], "2")
        self.assertEqual(unexpected_finding["slot"], "2")

    # --- Criterion 7: Pipeline Execution & Atomic JSON Storage ---
    def test_pipeline_execution_and_storage(self):
        pipeline = ProductMatchingPipeline(
            data_dir=self.root,
            encoder=DummyEncoder(),
            acceptance_threshold=0.5,
        )

        shelf_path = self.root / "shelf_test.jpg"
        shelf = Image.new("RGB", (200, 200), (255, 0, 0))
        shelf.save(shelf_path, format="JPEG")

        detections = [
            {"label": "red", "confidence": 0.9, "box": [10, 10, 50, 50], "image_width": 200, "image_height": 200}
        ]

        result = pipeline.run(shelf_path, detections, self.catalogue, self.planogram)
        self.assertIn("matches", result)
        self.assertIn("findings", result)
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["matches"][0]["accepted_sku"], "SKU_RED")

        # Test atomic storage
        audit_id = "abcd-1234-ef56"
        results_dir = self.root / "audit_results"
        saved_file = save_audit_result(results_dir, audit_id, result)
        self.assertTrue(saved_file.is_file())
        reloaded = json.loads(saved_file.read_text(encoding="utf-8"))
        self.assertEqual(len(reloaded["matches"]), 1)

    # --- Criterion 8: Web App Integration & CSV Export ---
    def test_web_app_analyze_generates_matching_audit_and_exports(self):
        shelf_path = self.root / "web_shelf.jpg"
        shelf = Image.new("RGB", (100, 100), (255, 0, 0))
        shelf.save(shelf_path, format="JPEG")

        client = web_app.app.test_client()
        web_app.app.config["TESTING"] = True

        pipeline = ProductMatchingPipeline(
            data_dir=self.root,
            encoder=DummyEncoder(),
            acceptance_threshold=0.5,
        )

        with patch.object(web_app, "DATA_DIR", self.root), \
             patch.object(web_app, "CATALOGUE_PATH", self.root / "catalogue.csv"), \
             patch.object(web_app, "_get_matching_pipeline", return_value=pipeline), \
             patch.object(web_app, "detect_products", return_value=[{"label": "red product", "confidence": 0.95, "box": [5, 5, 45, 45], "image_width": 100, "image_height": 100}]):

            # Write temp catalogue & planogram
            with (self.root / "catalogue.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["sku", "name", "category", "brand", "pack_size", "price", "model_label", "active", "image_local_path"])
                writer.writeheader()
                for item in self.catalogue:
                    writer.writerow({**item, "category": "General", "brand": "Brand", "pack_size": "100g", "price": "1.00", "model_label": item["name"].lower()})

            with (self.root / "planogram.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["store_id", "planogram_id", "shelf", "row", "slot", "expected_sku", "facing_count"])
                writer.writeheader()
                writer.writerows(self.planogram)

            with open(shelf_path, "rb") as img_file:
                res = client.post(
                    "/analyze",
                    data={"shelf_image": (img_file, "shelf.jpg"), "store": "Store 1", "planogram": "Plan 1"},
                    content_type="multipart/form-data",
                )

            self.assertEqual(res.status_code, 302)
            report_id = res.location.rsplit("/", 1)[-1]
            report = web_app._reports[report_id]
            self.assertIn("matching_audit", report)
            self.assertEqual(len(report["matching_audit"]["matches"]), 1)
            self.assertEqual(report["matching_audit"]["matches"][0]["accepted_sku"], "SKU_RED")

            # Check CSV export contains sku_match rows
            csv_res = client.get(f"/reports/{report_id}/download")
            self.assertEqual(csv_res.status_code, 200)
            self.assertIn(b"sku_match", csv_res.data)
            self.assertIn(b"SKU_RED", csv_res.data)

            # Check report page renders SKU Recognition table
            detail_res = client.get(res.location)
            self.assertEqual(detail_res.status_code, 200)
            self.assertIn(b"SKU RECOGNITION", detail_res.data)
            self.assertIn(b"Crop-to-SKU Candidates", detail_res.data)


if __name__ == "__main__":
    unittest.main()
