"""Compare catalogue prices with prices supplied by an OCR integration."""

from __future__ import annotations


def check_prices(catalogue: list[dict], observed_prices: dict[str, float]) -> list[dict]:
    """Return price validation rows; absent OCR values remain unverified."""
    results = []
    for product in catalogue:
        raw_expected = product.get("price")
        try:
            expected = float(raw_expected) if raw_expected not in (None, "") else None
        except (TypeError, ValueError):
            expected = None
        observed = observed_prices.get(product["sku"])
        if expected is None or observed is None:
            status = "Unverified"
        elif abs(float(observed) - expected) < 0.01:
            status = "Match"
        else:
            status = "Mismatch"
        results.append(
            {
                "sku": product["sku"],
                "product": product["name"],
                "expected_price": expected,
                "observed_price": observed,
                "status": status,
            }
        )
    return results