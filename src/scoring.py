"""Evidence-aware, deterministic audit compliance scoring."""

from __future__ import annotations

from collections import Counter
from typing import Any

from config.compliance_scoring import SCORING_VERSION, SCORING_WEIGHTS


def calculate_compliance_score(
    compliance: list[dict[str, Any]],
    price_results: list[dict[str, Any]],
    *,
    layout_trusted: bool,
    price_requested: bool,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score confirmed slot and price checks; retain unknowns in the denominator.

    The score is weighted earned points / weighted applicable checks. Confirmed
    passes earn 1, confirmed violations earn 0, and unresolved checks earn 0.
    Coverage is the resolved share of applicable weighted checks. No numeric
    score is returned when evidence is absent or no checks apply.
    """
    configured = dict(weights or SCORING_WEIGHTS)
    placement_weight = float(configured.get("placement", 0))
    price_weight = float(configured.get("price", 0))
    if any(value < 0 for value in (placement_weight, price_weight)):
        raise ValueError("Compliance scoring weights must be non-negative.")
    if placement_weight == 0 and price_weight == 0:
        raise ValueError("At least one compliance scoring weight must be positive.")

    checks: list[dict[str, Any]] = []
    if compliance:
        per_slot = placement_weight / len(compliance)
        confirmed_pass = {"Compliant"}
        confirmed_fail = {"Misplaced", "Missing Expected Product", "Unexpected Product"}
        for row in compliance:
            status = str(row.get("status", "Needs Review"))
            if not layout_trusted:
                outcome = "unresolved"
                reason = "Slot mapping is not trusted for this audit."
            elif status in confirmed_pass:
                outcome, reason = "pass", "Confirmed compliant slot."
            elif status in confirmed_fail:
                outcome, reason = "violation", status
            elif status == "Empty Slot" and not str(row.get("sku", "")).strip():
                outcome, reason = "pass", "Confirmed intentionally empty planogram slot."
            else:
                outcome, reason = "unresolved", row.get("detail") or "Slot result needs review."
            checks.append({
                "category": "placement", "outcome": outcome, "weight": per_slot,
                "shelf": row.get("shelf", ""), "row": row.get("row", ""),
                "slot": row.get("slot", row.get("position", "")),
                "sku": row.get("sku", ""), "status": status, "reason": reason,
            })

    if price_requested and price_results:
        price_checks = [row for row in price_results if row.get("expected_price") is not None]
        per_price = price_weight / len(price_checks) if price_checks else 0
        for row in price_checks:
            status = str(row.get("status", "Unverified"))
            if status == "Match":
                outcome, reason = "pass", "OCR price matches the expected price."
            elif status in {"Mismatch", "Price Mismatch"}:
                outcome, reason = "violation", "Confirmed shelf price differs from expected price."
            else:
                outcome, reason = "unresolved", row.get("detail") or "Price was not verified."
            checks.append({
                "category": "price", "outcome": outcome, "weight": per_price,
                "sku": row.get("sku", ""), "status": status, "reason": reason,
            })

    applicable_weight = sum(check["weight"] for check in checks)
    resolved = [check for check in checks if check["outcome"] != "unresolved"]
    resolved_weight = sum(check["weight"] for check in resolved)
    passed_weight = sum(check["weight"] for check in checks if check["outcome"] == "pass")
    denominator = applicable_weight
    score = round(100 * passed_weight / denominator, 2) if denominator > 0 and resolved_weight > 0 else None
    coverage = round(100 * resolved_weight / applicable_weight, 2) if applicable_weight > 0 else 0.0
    outcomes = Counter(check["outcome"] for check in checks)
    if not checks:
        status, reason = "unavailable", "No applicable placement or price checks were available."
    elif not resolved:
        status, reason = "unavailable", "No checks had enough evidence for a confirmed result."
    elif outcomes["unresolved"]:
        status, reason = "partial", "Some checks remain unresolved; they reduce the score and coverage."
    else:
        status, reason = "complete", "All applicable checks have confirmed results."

    breakdown = {}
    for category in ("placement", "price"):
        category_checks = [check for check in checks if check["category"] == category]
        breakdown[category] = {
            "weight": configured.get(category, 0),
            "applicable": len(category_checks),
            "compliant": sum(check["outcome"] == "pass" for check in category_checks),
            "violations": sum(check["outcome"] == "violation" for check in category_checks),
            "unresolved": sum(check["outcome"] == "unresolved" for check in category_checks),
        }

    return {
        "version": SCORING_VERSION,
        "score": score,
        "status": status,
        "reason": reason,
        "coverage": coverage,
        "earned_weight": round(passed_weight, 8),
        "resolved_weight": round(resolved_weight, 8),
        "applicable_weight": round(applicable_weight, 8),
        "compliant_checks": outcomes["pass"],
        "violation_checks": outcomes["violation"],
        "unresolved_checks": outcomes["unresolved"],
        "applicable_check_count": len(checks),
        "breakdown": breakdown,
        "weights": configured,
        "formula": "100 * earned weighted checks / all applicable weighted checks; unresolved checks earn zero and reduce coverage.",
        "checks": checks,
    }
