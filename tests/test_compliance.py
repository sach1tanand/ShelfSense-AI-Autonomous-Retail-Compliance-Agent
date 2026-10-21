import unittest

from src.compliance import evaluate_compliance
from src.price_checker import check_prices


class ComplianceTests(unittest.TestCase):
    def setUp(self):
        self.catalogue = [
            {"sku": "A1", "name": "Oat Milk", "model_label": "oat milk", "price": "4.00"},
            {"sku": "B2", "name": "Granola", "model_label": "granola", "price": "6.00"},
        ]
        self.planogram = [
            {"shelf": "Top", "position": "1", "sku": "A1", "expected_facings": "2"},
            {"shelf": "Top", "position": "2", "sku": "B2", "expected_facings": "1"},
        ]

    def test_counts_detected_products_and_reports_status(self):
        detections = [{"label": "Oat Milk"}, {"label": "oat milk"}, {"label": "granola"}]

        result = evaluate_compliance(self.planogram, self.catalogue, detections)

        self.assertEqual([row["observed"] for row in result], [2, 1])
        self.assertEqual([row["status"] for row in result], ["Compliant", "Compliant"])

    def test_marks_missing_facings_as_understocked(self):
        result = evaluate_compliance(self.planogram, self.catalogue, [])

        self.assertEqual([row["status"] for row in result], ["Understocked", "Understocked"])

    def test_unknown_detection_does_not_change_catalogue_counts(self):
        result = evaluate_compliance(self.planogram, self.catalogue, [{"label": "unknown item"}])

        self.assertEqual([row["observed"] for row in result], [0, 0])

    def test_price_checker_marks_unavailable_ocr_unverified(self):
        result = check_prices(self.catalogue, {"A1": 4.00})

        self.assertEqual([row["status"] for row in result], ["Match", "Unverified"])

    def test_price_checker_keeps_missing_catalogue_prices_unverified(self):
        result = check_prices(
            [{"sku": "C3", "name": "Unknown price", "price": ""}],
            {},
        )

        self.assertEqual(result[0]["status"], "Unverified")
        self.assertIsNone(result[0]["expected_price"])


if __name__ == "__main__":
    unittest.main()