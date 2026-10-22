"""Build concise report summaries for the web interface."""

from __future__ import annotations

from datetime import datetime, timezone


def build_report(
    *, compliance: list[dict], price_results: list[dict], detections: list[dict], image_name: str,
    store_name: str = "Local Store", compliance_score: dict | None = None,
) -> dict:
    compliant_count = sum(row.get("status") == "Compliant" for row in compliance)
    score = compliance_score or {
        "version": "1.0.0", "score": None, "status": "unavailable",
        "reason": "A scored audit result was not supplied.", "coverage": 0.0,
    }
    return {
        "image_name": image_name,
        "store_name": store_name,
        "created_at": datetime.now(timezone.utc).strftime("%b %d, %Y · %H:%M UTC"),
        "compliance": compliance,
        "price_results": price_results,
        "detections": detections,
        "detection_count": len(detections),
        "compliant_count": compliant_count,
        "compliance_score": score,
        "compliance_rate": score.get("score"),
        "price_unverified": sum(row["status"] == "Unverified" for row in price_results),
    }
