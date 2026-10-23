import unittest

import app as web_app
from src.report import build_report
from src.scoring import calculate_compliance_score


class ComplianceScoringTests(unittest.TestCase):
    def test_complete_placement_and_price_audit_uses_weighted_denominator(self):
        score = calculate_compliance_score(
            [
                {"status": "Compliant", "shelf": "S1", "row": "1", "position": "1"},
                {"status": "Missing Expected Product", "shelf": "S1", "row": "1", "position": "2"},
            ],
            [{"sku": "A", "expected_price": 4.0, "status": "Match"},
             {"sku": "B", "expected_price": 6.0, "status": "Mismatch"}],
            layout_trusted=True, price_requested=True,
        )
        self.assertEqual(score["score"], 50.0)
        self.assertEqual(score["status"], "complete")
        self.assertEqual(score["coverage"], 100.0)
        self.assertEqual((score["compliant_checks"], score["violation_checks"], score["unresolved_checks"]), (2, 2, 0))
        self.assertEqual(score["applicable_check_count"], 4)
        self.assertEqual(score["version"], "1.1.0")

    def test_unresolved_checks_reduce_score_and_coverage(self):
        score = calculate_compliance_score(
            [{"status": "Compliant"}, {"status": "Needs Review"}], [],
            layout_trusted=True, price_requested=False,
        )
        self.assertEqual(score["score"], 50.0)
        self.assertEqual(score["coverage"], 50.0)
        self.assertEqual(score["status"], "partial")
        self.assertEqual(score["unresolved_checks"], 1)

    def test_trusted_expected_sku_match_earns_partial_credit_when_facings_need_review(self):
        score = calculate_compliance_score(
            [{
                "status": "Needs Review", "sku": "A1", "sku_match_confirmed": True,
                "expected": 2, "observed": 1, "detail": "Detected 1 of 2 expected facings.",
            }], [], layout_trusted=True, price_requested=False,
        )

        self.assertEqual(score["score"], 50.0)
        self.assertEqual(score["coverage"], 50.0)
        self.assertEqual(score["compliant_checks"], 1)
        self.assertEqual(score["unresolved_checks"], 1)
        self.assertEqual(score["breakdown"]["placement"]["applicable"], 2)

    def test_unidentified_objects_do_not_get_partial_sku_match_credit(self):
        score = calculate_compliance_score(
            [{
                "status": "Needs Review", "sku": "A1", "sku_match_confirmed": True,
                "reason_code": "unidentified_objects_in_slot",
            }], [], layout_trusted=True, price_requested=False,
        )

        self.assertIsNone(score["score"])
        self.assertEqual(score["unresolved_checks"], 1)

    def test_saved_v1_report_recalculates_partial_credit_from_exact_mapped_sku(self):
        report = web_app._normalize_saved_report("legacy", {
            "layout_trusted": True,
            "compliance": [{
                "store_id": "S", "planogram_id": "P", "shelf": "A", "row": "1", "slot": "1",
                "sku": "A1", "expected": 2, "observed": 1, "status": "Needs Review",
                "reason_code": "absence_not_confirmed",
            }],
            "detections": [{
                "matched_sku": "A1", "match_status": "Matched",
                "slot_identity": ["S", "P", "A", "1", "1"],
            }],
            "compliance_score": {"version": "1.0.0", "score": 0.0},
        })

        self.assertEqual(report["compliance_score"]["score"], 50.0)
        self.assertEqual(report["compliance_score"]["version"], "1.1.0")

    def test_untrusted_layout_has_no_numeric_score_when_all_checks_are_unresolved(self):
        score = calculate_compliance_score(
            [{"status": "Compliant"}, {"status": "Empty Slot", "sku": ""}], [],
            layout_trusted=False, price_requested=False,
        )
        self.assertIsNone(score["score"])
        self.assertEqual(score["status"], "unavailable")
        self.assertEqual(score["coverage"], 0.0)
        self.assertEqual(score["unresolved_checks"], 2)

    def test_intentionally_empty_slot_is_a_pass_only_with_trusted_mapping(self):
        score = calculate_compliance_score(
            [{"status": "Empty Slot", "sku": ""}], [],
            layout_trusted=True, price_requested=False,
        )
        self.assertEqual(score["score"], 100.0)
        self.assertEqual(score["compliant_checks"], 1)

    def test_price_checks_are_not_applicable_when_not_requested(self):
        score = calculate_compliance_score(
            [], [{"sku": "A", "status": "Mismatch"}],
            layout_trusted=False, price_requested=False,
        )
        self.assertIsNone(score["score"])
        self.assertEqual(score["applicable_check_count"], 0)
        self.assertEqual(score["status"], "unavailable")

    def test_price_without_catalogue_reference_is_not_applicable(self):
        score = calculate_compliance_score(
            [], [{"sku": "A", "expected_price": None, "status": "Unverified"}],
            layout_trusted=False, price_requested=True,
        )
        self.assertEqual(score["applicable_check_count"], 0)
        self.assertEqual(score["breakdown"]["price"]["applicable"], 0)

    def test_score_is_bounded_and_zero_violations_are_resolved(self):
        score = calculate_compliance_score(
            [{"status": "Unexpected Product"}], [],
            layout_trusted=True, price_requested=False,
        )
        self.assertEqual(score["score"], 0.0)
        self.assertEqual(score["status"], "complete")
        self.assertGreaterEqual(score["score"], 0)
        self.assertLessEqual(score["score"], 100)

    def test_report_builder_keeps_structured_score_and_compatibility_alias_in_sync(self):
        score = calculate_compliance_score(
            [{"status": "Compliant"}], [], layout_trusted=True, price_requested=False,
        )
        report = build_report(
            compliance=[{"status": "Compliant"}], price_results=[], detections=[],
            image_name="shelf.jpg", compliance_score=score,
        )
        self.assertEqual(report["compliance_score"], score)
        self.assertEqual(report["compliance_rate"], score["score"])


if __name__ == "__main__":
    unittest.main()
