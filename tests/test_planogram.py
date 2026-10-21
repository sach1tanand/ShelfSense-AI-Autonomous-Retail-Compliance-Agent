import unittest

from src.planogram import evaluate_planogram, validate_planogram


class PlanogramTests(unittest.TestCase):
    def setUp(self):
        self.catalogue = [
            {"sku": "8900000000001", "name": "Oat Drink", "model_label": "oat drink", "active": "true"},
            {"sku": "8900000000002", "name": "Granola", "model_label": "granola", "active": "true"},
            {"sku": "8900000000003", "name": "Unplanned Soda", "model_label": "soda", "active": "true"},
        ]
        self.planogram = [
            {"store_id": "STORE1", "planogram_id": "PLAN1", "shelf": "S1", "row": "1", "slot": "1", "expected_sku": "8900000000001", "facing_count": "1"},
            {"store_id": "STORE1", "planogram_id": "PLAN1", "shelf": "S1", "row": "1", "slot": "2", "expected_sku": "8900000000002", "facing_count": "2"},
            {"store_id": "STORE1", "planogram_id": "PLAN1", "shelf": "S1", "row": "1", "slot": "3", "expected_sku": "", "facing_count": "0"},
        ]

    def test_intentionally_empty_slot_is_valid(self):
        self.assertEqual(validate_planogram(self.planogram, self.catalogue), [])

    def test_validation_catches_unknown_sku_duplicate_slot_and_invalid_facings(self):
        invalid = [
            *self.planogram,
            {**self.planogram[0], "expected_sku": "NO-SUCH-SKU"},
            {**self.planogram[1], "facing_count": "-2"},
            {**self.planogram[2], "slot": "4", "facing_count": "1"},
        ]
        errors = validate_planogram(invalid, self.catalogue)
        self.assertTrue(any("duplicate slot" in error for error in errors))
        self.assertTrue(any("unknown expected_sku" in error for error in errors))
        self.assertTrue(any("non-negative whole number" in error for error in errors))
        self.assertTrue(any("empty slots must have facing_count 0" in error for error in errors))

    def test_without_slot_mapper_expected_slots_are_needs_review_not_missing(self):
        audit = evaluate_planogram(
            self.planogram,
            self.catalogue,
            [{"label": "oat drink", "box": [0, 0, 20, 20]}],
        )
        self.assertEqual(audit["slots"][0]["status"], "Needs Review")
        self.assertIn("bounding boxes do not establish", audit["slots"][0]["detail"])
        self.assertIsNone(audit["slots"][0]["observed"])
        self.assertEqual(audit["slots"][2]["status"], "Empty Slot")
        self.assertFalse(audit["slot_mapping_available"])

    def test_exact_injected_mapper_can_confirm_slot_findings(self):
        target = ("STORE1", "PLAN1", "S1", "1", "1")
        mapper = lambda detection, slots: target
        audit = evaluate_planogram(
            self.planogram,
            self.catalogue,
            [{"label": "oat drink"}],
            slot_mapper=mapper,
        )
        self.assertEqual(audit["slots"][0]["status"], "Compliant")
        self.assertEqual(audit["slots"][0]["observed"], 1)
        self.assertEqual(audit["slots"][1]["status"], "Missing Expected Product")

    def test_wrong_product_mapped_to_slot_is_misplaced_and_unknown_is_review(self):
        target = ("STORE1", "PLAN1", "S1", "1", "1")
        mapper = lambda detection, slots: target
        audit = evaluate_planogram(
            self.planogram,
            self.catalogue,
            [{"label": "granola"}, {"label": "unknown detector class"}],
            slot_mapper=mapper,
        )
        self.assertEqual(audit["slots"][0]["status"], "Misplaced")
        self.assertTrue(any(item["status"] == "Needs Review" for item in audit["detections"]))

    def test_unique_catalogue_product_absent_from_planogram_is_unexpected(self):
        audit = evaluate_planogram(
            self.planogram,
            self.catalogue,
            [{"label": "soda"}],
        )
        self.assertEqual(audit["detections"][0]["status"], "Unexpected Product")
        self.assertEqual(audit["detections"][0]["sku"], "8900000000003")

    def test_ambiguous_identity_needs_review(self):
        catalogue = [
            *self.catalogue,
            {"sku": "8900000000004", "name": "Oat Drink", "model_label": "oat drink"},
        ]
        audit = evaluate_planogram(self.planogram, catalogue, [{"label": "oat drink"}])
        self.assertEqual(audit["detections"][0]["status"], "Needs Review")

    def test_explicit_catalogue_barcode_can_resolve_identity(self):
        audit = evaluate_planogram(
            self.planogram,
            self.catalogue,
            [{"label": "model-class-7", "barcode": "8900000000003"}],
        )

        self.assertEqual(audit["detections"][0]["status"], "Unexpected Product")
        self.assertEqual(audit["detections"][0]["sku"], "8900000000003")


if __name__ == "__main__":
    unittest.main()
