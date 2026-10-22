import unittest

from backend.matching.shelf_reconstruction import (
    compare_expected_to_detected,
    reconstruct_detected_planogram,
)


class ShelfReconstructionTests(unittest.TestCase):
    def test_three_adjacent_identical_detections_form_one_cell_with_three_facings(self):
        detections = [
            {"box": [10, 10, 50, 90], "confidence": .9, "matched_sku": "A1"},
            {"box": [52, 11, 92, 91], "confidence": .9, "matched_sku": "A1"},
            {"box": [94, 9, 134, 89], "confidence": .9, "matched_sku": "A1"},
        ]
        result = reconstruct_detected_planogram(detections, [], [], [{"sku": "A1", "name": "Oats"}], 180, 120)
        cells = result["rows"][0]["slots"]
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["facing_count"], 3)
        self.assertEqual(cells[0]["detection_indices"], [0, 1, 2])
        self.assertEqual(cells[0]["box"], [10, 9, 134, 91])
        self.assertEqual([item["box"] for item in result["items"]], [item["box"] for item in detections])

    def test_identical_adjacent_products_mapped_to_distinct_expected_slots_stay_separate(self):
        detections = [
            {"box": [10, 10, 50, 90], "matched_sku": "A1", "slot_identity": ["S", "P", "A", "1", "1"]},
            {"box": [52, 10, 92, 90], "matched_sku": "A1", "slot_identity": ["S", "P", "A", "1", "2"]},
        ]
        result = reconstruct_detected_planogram(detections, [], [], [], 120, 100)
        cells = result["rows"][0]["slots"]
        self.assertEqual(len(cells), 2)
        self.assertEqual([cell["facing_count"] for cell in cells], [1, 1])

    def test_adjacent_different_skus_remain_separate_cells(self):
        detections = [
            {"box": [10, 10, 50, 90], "matched_sku": "A1"},
            {"box": [52, 10, 92, 90], "matched_sku": "B2"},
        ]
        result = reconstruct_detected_planogram(detections, [], [], [], 120, 100)
        cells = result["rows"][0]["slots"]
        self.assertEqual(len(cells), 2)
        self.assertEqual([cell["sku"] for cell in cells], ["A1", "B2"])
        self.assertEqual([cell["facing_count"] for cell in cells], [1, 1])

    def test_identical_skus_on_different_rows_are_not_grouped(self):
        detections = [
            {"box": [10, 10, 50, 90], "matched_sku": "A1"},
            {"box": [52, 210, 92, 290], "matched_sku": "A1"},
        ]
        result = reconstruct_detected_planogram(detections, [], [], [], 120, 320)
        self.assertEqual(result["row_count"], 2)
        self.assertEqual([row["slots"][0]["facing_count"] for row in result["rows"]], [1, 1])

    def test_uncertain_adjacent_skus_stay_separate_and_are_flagged(self):
        detections = [
            {"box": [10, 10, 50, 90], "matched_sku": "", "match_status": "Unknown / Needs Review"},
            {"box": [52, 10, 92, 90], "matched_sku": "", "match_status": "Unknown / Needs Review"},
        ]
        result = reconstruct_detected_planogram(detections, [], [], [], 120, 100)
        cells = result["rows"][0]["slots"]
        self.assertEqual(len(cells), 2)
        self.assertEqual([cell["facing_count"] for cell in cells], [1, 1])
        self.assertTrue(all("adjacent identity is uncertain" in cell["group_status"] for cell in cells))

    def test_grouped_facings_do_not_compress_later_products_into_earlier_expected_slots(self):
        detections = [
            {"box": [25, 10, 45, 90], "matched_sku": "A1"},
            {"box": [47, 10, 67, 90], "matched_sku": "A1"},
            {"box": [69, 10, 89, 90], "matched_sku": "A1"},
            {"box": [140, 10, 180, 90], "matched_sku": "B2"},
        ]
        detected = reconstruct_detected_planogram(detections, [], [], [], 200, 100)
        planogram = [
            {"physical_shelf": "A", "shelf": "Row 1", "row": "1", "position": "1", "sku": "A1"},
            {"physical_shelf": "A", "shelf": "Row 1", "row": "1", "position": "2", "sku": "B2"},
        ]
        result = compare_expected_to_detected(planogram, detected, {}, [], False)
        self.assertEqual(len(detected["rows"][0]["slots"]), 2)
        self.assertEqual([item["detected_sku"] for item in result["findings"]], ["A1", "B2"])
        self.assertEqual(result["findings"][0]["facing_count"], 3)

    def test_reconstructs_rows_columns_voids_and_skips_duplicate_overlap(self):
        detections = [
            {"box": [10, 10, 60, 110], "confidence": .92, "matched_sku": "A1", "product_name": "Oats"},
            {"box": [11, 11, 61, 111], "confidence": .80, "matched_sku": "A1", "product_name": "Oats"},
            {"box": [100, 12, 150, 112], "confidence": .78, "matched_sku": "", "product_name": ""},
            {"box": [12, 220, 62, 320], "confidence": .91, "matched_sku": "B2", "product_name": "Tea"},
        ]
        result = reconstruct_detected_planogram(
            detections,
            [{"box": [200, 10, 250, 110], "confidence": .76}],
            [],
            [{"sku": "A1", "name": "Oats"}, {"sku": "B2", "name": "Tea"}],
            300, 360,
        )

        self.assertEqual(result["row_count"], 2)
        self.assertEqual([row["slot_count"] for row in result["rows"]], [3, 1])
        self.assertEqual(result["rows"][0]["slots"][0]["product"], "Oats")
        self.assertEqual(result["rows"][0]["slots"][1]["product"], "Unidentified")
        self.assertEqual(result["rows"][0]["slots"][2]["kind"], "void")
        self.assertEqual(result["duplicate_detection_indices"], [1])
        self.assertEqual(result["geometry_status"], "Needs Review")

    def test_price_tag_association_is_one_to_one_and_ambiguous_tags_stay_unassigned(self):
        products = [
            {"box": [10, 10, 70, 100], "confidence": .9, "matched_sku": "A1", "product_name": "Oats"},
            {"box": [11, 10, 71, 100], "confidence": .9, "matched_sku": "B2", "product_name": "Tea"},
        ]
        tags = [{"box": [20, 105, 60, 125], "price": None, "raw_ocr_text": "", "ocr_status": "unreadable"}]
        result = reconstruct_detected_planogram(products, [], tags, [], 300, 400)

        self.assertIsNone(result["price_tags"][0].get("associated_detection_index"))
        self.assertEqual(result["price_tags"][0]["association_status"], "Needs Review")
        self.assertTrue(all(not item["price_tags"] for item in result["items"]))
        planogram = [{"physical_shelf": "A", "shelf": "Row 1", "row": "1", "position": "1", "sku": "A1"}]
        comparison = compare_expected_to_detected(planogram, result, {}, [], False, [{"sku": "A1", "name": "Oats"}])
        self.assertEqual(comparison["summary"]["price_unreadable"], 1)
        self.assertEqual(comparison["summary"]["price_mismatches"], 0)

    def test_comparison_only_counts_confirmed_existing_slot_results(self):
        planogram = [
            {"store_id": "S", "planogram_id": "P", "physical_shelf": "A", "shelf": "Row 1", "row": "1", "position": "1", "sku": "A1", "expected_facings": "1"},
            {"store_id": "S", "planogram_id": "P", "physical_shelf": "A", "shelf": "Row 1", "row": "1", "position": "2", "sku": "B2", "expected_facings": "1"},
        ]
        detected = {"rows": [{"row": 1, "slots": [], "slot_count": 0}]}
        audit = {"slots": [
            {"shelf": "A", "row": "1", "position": "1", "status": "Compliant", "detail": "matched"},
            {"shelf": "A", "row": "1", "position": "2", "status": "Needs Review", "detail": "uncertain"},
        ]}
        matching_audit = {"findings": [
            {"shelf": "A", "row": "1", "slot": "1", "status": "Correct"},
            {"shelf": "A", "row": "1", "slot": "2", "status": "Needs Review"},
        ]}
        result = compare_expected_to_detected(planogram, detected, audit, [], True, [
            {"sku": "A1", "name": "Oats", "price": "20.00"},
            {"sku": "B2", "name": "Tea", "price": "30.00"},
        ], matching_audit)

        self.assertEqual(result["summary"]["matches"], 1)
        self.assertEqual(result["summary"]["needs_review"], 1)
        self.assertEqual(result["findings"][0]["expected_product"], "Oats")
        self.assertEqual(result["findings"][0]["expected_price"], "20.00")
        self.assertFalse(result["findings"][1]["confirmed"])

    def test_price_mismatch_requires_trusted_slot_identity_and_readable_unique_tag(self):
        identity = ["S", "P", "A", "1", "1"]
        planogram = [{
            "store_id": "S", "planogram_id": "P", "physical_shelf": "A", "shelf": "Row 1",
            "row": "1", "position": "1", "sku": "A1", "expected_facings": "1",
        }]
        detections = [{
            "box": [20, 10, 80, 80], "confidence": .95, "matched_sku": "A1",
            "product_name": "Oats", "slot_identity": identity,
        }]
        tags = [{
            "box": [30, 82, 60, 96], "confidence": .91, "price": 25.0,
            "raw_ocr_text": "Rs 25", "ocr_status": "read", "ocr_confidence": .9,
        }]
        detected = reconstruct_detected_planogram(detections, [], tags, [{"sku": "A1", "name": "Oats"}], 100, 120)
        audit = {"slots": [{"shelf": "A", "row": "1", "position": "1", "status": "Compliant", "detail": "mapped"}]}
        matching = {"findings": [{"shelf": "A", "row": "1", "slot": "1", "status": "Correct"}]}
        prices = [{"sku": "A1", "expected_price": 20.0}]

        trusted = compare_expected_to_detected(planogram, detected, audit, prices, True,
                                              [{"sku": "A1", "name": "Oats", "price": "20"}], matching)
        untrusted = compare_expected_to_detected(planogram, detected, audit, prices, False,
                                                [{"sku": "A1", "name": "Oats", "price": "20"}], matching)

        self.assertEqual(trusted["findings"][0]["price_status"], "PRICE MISMATCH")
        self.assertEqual(trusted["summary"]["price_mismatches"], 1)
        self.assertEqual(untrusted["findings"][0]["price_status"], "Needs Review")


if __name__ == "__main__":
    unittest.main()
