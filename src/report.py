"""Build concise report summaries for the web interface."""

from __future__ import annotations

from datetime import datetime, timezone


def build_report(
    *, compliance: list[dict], price_results: list[dict], detections: list[dict], image_name: str,
    store_name: str = "Local Store",
) -> dict:
    compliant_count = sum(row.get("status") == "Compliant" for row in compliance)
    confirmed_statuses = {"Compliant", "Misplaced", "Missing Expected Product", "Unexpected Product"}
    total = sum(row.get("status") in confirmed_statuses for row in compliance)
    compliance_rate = round(compliant_count / total * 100) if total else None
    return {
        "image_name": image_name,
        "store_name": store_name,
        "created_at": datetime.now(timezone.utc).strftime("%b %d, %Y · %H:%M UTC"),
        "compliance": compliance,
        "price_results": price_results,
        "detections": detections,
        "detection_count": len(detections),
        "compliant_count": compliant_count,
        "compliance_rate": compliance_rate,
        "price_unverified": sum(row["status"] == "Unverified" for row in price_results),
    }