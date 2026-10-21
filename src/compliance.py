"""Planogram compliance calculations."""

from __future__ import annotations


def _normalize_label(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").split())


def evaluate_compliance(
    planogram: list[dict], catalogue: list[dict], detections: list[dict]
) -> list[dict]:
    """Compare detected facings with planogram expectations, per SKU."""
    products_by_sku = {product["sku"]: product for product in catalogue}
    labels_by_sku = {
        sku: _normalize_label(product.get("model_label", product["name"]))
        for sku, product in products_by_sku.items()
    }
    counts = {sku: 0 for sku in products_by_sku}
    for detection in detections:
        label = _normalize_label(str(detection.get("label", "")))
        for sku, model_label in labels_by_sku.items():
            if label == model_label:
                counts[sku] += 1
                break

    results = []
    for placement in planogram:
        sku = placement["sku"]
        product = products_by_sku.get(sku)
        if product is None:
            continue
        expected = int(placement["expected_facings"])
        observed = counts[sku]
        difference = observed - expected
        status = "Compliant" if difference == 0 else ("Understocked" if difference < 0 else "Overstocked")
        results.append(
            {
                "shelf": placement["shelf"],
                "position": placement["position"],
                "sku": sku,
                "product": product["name"],
                "expected": expected,
                "observed": observed,
                "difference": difference,
                "status": status,
            }
        )
    return results