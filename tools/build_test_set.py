"""Build a labelled synthetic shelf test set from the catalogue's product images.

Products are split so calibration and testing never share products:

* test: the planogram SKUs plus a few distractor SKUs that are not in the planogram
* validation: other catalogue products, used only to calibrate the SKU-match threshold
* holdout: products removed from the gallery during calibration to measure unknown rejection

Each scene is a fixed-camera, front-facing shelf whose rows and slots follow the planogram
grid exactly (the "calibrated layout"), with ₹ price tags on the shelf edge. Product facings
are augmented (scale, rotation, brightness, blur) so they never equal the gallery images.
Ground truth (facing boxes, slot contents, tag prices, expected statuses) is written to
data/test_set/ground_truth.json, with SHA-256 fingerprints of the planogram and catalogue it was
built from so calibration and evaluation can refuse a stale test set.

Planogram SKUs without an expected_price get no rendered tag and an "Unverified" price truth;
no price is invented for them.

The scenes are synthetic (catalogue photos pasted on a flat shelf). Metrics measured on them do
not predict accuracy on real store photos.

Usage:  python -m tools.build_test_set
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "test_set"
SEED = 20261022
WIDTH, ROW_HEIGHT, SLOT_WIDTH = 1500, 400, 300
PRODUCT_TOP, SHELF_TOP, SHELF_BOTTOM = 50, 300, 318
TAG_BOX = (325, 375)
N_DISTRACTORS, N_VALIDATION, N_HOLDOUT, N_VALIDATION_SCENES = 3, 60, 20, 8


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_status(expected_sku: str, expected_facings: int, placed: list[str]) -> str:
    """Status the planogram audit should report for one slot under the calibrated layout.

    Mirrors src.planogram.evaluate_planogram with a slot mapper: every identified catalogue
    product mapped to a slot counts, including products that are not in the planogram.
    """
    mapped = list(placed)
    if not expected_sku:
        return "Unexpected Product" if mapped else "Empty Slot"
    if not mapped:
        return "Missing Expected Product"
    if any(sku != expected_sku for sku in mapped):
        return "Misplaced"
    if len(mapped) < expected_facings:
        return "Missing Expected Product"
    if len(mapped) > expected_facings:
        return "Unexpected Product"
    return "Compliant"


class ShelfRenderer:
    def __init__(self, catalogue: dict[str, dict], rng: random.Random, rows: int, slots: int):
        self.catalogue = catalogue
        self.rng = rng
        self.rows, self.slots = rows, slots
        self.font = load_font(30)
        self._images: dict[str, Image.Image] = {}

    def product_image(self, sku: str) -> Image.Image:
        if sku not in self._images:
            with Image.open(DATA_DIR / self.catalogue[sku]["image_local_path"]) as opened:
                self._images[sku] = opened.convert("RGB")
        return self._images[sku]

    def augmented(self, sku: str, max_width: int, max_height: int) -> Image.Image:
        image = self.product_image(sku)
        image = image.rotate(self.rng.uniform(-2.5, 2.5), expand=True, fillcolor=(245, 245, 245))
        image = ImageEnhance.Brightness(image).enhance(self.rng.uniform(0.85, 1.15))
        image = ImageEnhance.Contrast(image).enhance(self.rng.uniform(0.9, 1.1))
        scale = min(max_width / image.width, max_height / image.height) * self.rng.uniform(0.88, 1.0)
        image = image.resize((max(8, int(image.width * scale)), max(8, int(image.height * scale))), Image.Resampling.LANCZOS)
        return image.filter(ImageFilter.GaussianBlur(self.rng.uniform(0, 0.8)))

    def render(self, slot_contents: dict[tuple[int, int], list[str]], tag_prices: dict[tuple[int, int], float]):
        height = ROW_HEIGHT * self.rows
        canvas = Image.new("RGB", (WIDTH, height), (58, 64, 74))
        draw = ImageDraw.Draw(canvas)
        for x in range(0, WIDTH, 6):  # faint back-panel texture
            shade = 58 + self.rng.randint(-3, 3)
            draw.line((x, 0, x, height), fill=(shade, shade + 6, shade + 16))
        facings, tags = [], []
        for row in range(1, self.rows + 1):
            top = (row - 1) * ROW_HEIGHT
            draw.rectangle((0, top + SHELF_TOP, WIDTH, top + SHELF_BOTTOM), fill=(205, 208, 212))
            draw.rectangle((0, top + SHELF_BOTTOM, WIDTH, top + ROW_HEIGHT - 10), fill=(232, 234, 236))
            for slot in range(1, self.slots + 1):
                left = (slot - 1) * SLOT_WIDTH
                skus = slot_contents.get((row, slot), [])
                if skus:
                    count = len(skus)
                    gap = 8
                    facing_width = min(120, (SLOT_WIDTH - 40 - gap * (count - 1)) // count)
                    total = facing_width * count + gap * (count - 1)
                    x = left + (SLOT_WIDTH - total) // 2
                    for sku in skus:
                        product = self.augmented(sku, facing_width, SHELF_TOP - PRODUCT_TOP)
                        px = x + (facing_width - product.width) // 2
                        py = top + SHELF_TOP - product.height
                        canvas.paste(product, (px, py))
                        facings.append({"sku": sku, "row": row, "slot": slot,
                                        "box": [px, py, px + product.width, py + product.height]})
                        x += facing_width + gap
                price = tag_prices.get((row, slot))
                if price is not None:
                    center = left + SLOT_WIDTH // 2
                    box = [center - 75, top + TAG_BOX[0], center + 75, top + TAG_BOX[1]]
                    draw.rectangle(box, fill=(255, 214, 0), outline=(40, 40, 40), width=2)
                    text = f"₹ {price:.2f}"
                    bounds = draw.textbbox((0, 0), text, font=self.font)
                    draw.text((center - (bounds[2] - bounds[0]) / 2, box[1] + 8), text, font=self.font, fill=(15, 15, 15))
                    tags.append({"row": row, "slot": slot, "price": price, "box": box})
        return canvas, facings, tags


def build(output_dir: Path = OUTPUT_DIR, seed: int = SEED) -> dict:
    rng = random.Random(seed)
    catalogue_rows = read_csv(DATA_DIR / "catalogue.csv")
    catalogue = {row["sku"]: row for row in catalogue_rows if (DATA_DIR / row.get("image_local_path", "")).is_file()}
    planogram = read_csv(DATA_DIR / "planogram.csv")
    planogram_skus = {row["expected_sku"] for row in planogram if row["expected_sku"]}
    rows = max(int(row["row"]) for row in planogram)
    slots = max(int(row["slot"]) for row in planogram)
    expected = {(int(row["row"]), int(row["slot"])): (row["expected_sku"], int(row["facing_count"] or 0)) for row in planogram}

    missing_images = sorted(planogram_skus - set(catalogue))
    if missing_images:
        raise SystemExit(f"Planogram SKUs without a catalogue reference image: {missing_images}")
    prices = {}
    for sku in planogram_skus:
        raw = catalogue[sku].get("expected_price") or catalogue[sku].get("price")
        if raw:
            prices[sku] = float(raw)
    unpriced = sorted(planogram_skus - set(prices))

    others = sorted(sku for sku in catalogue if sku not in planogram_skus)
    rng.shuffle(others)
    distractors = others[:N_DISTRACTORS]
    holdout = others[N_DISTRACTORS:N_DISTRACTORS + N_HOLDOUT]
    validation = others[N_DISTRACTORS + N_HOLDOUT:N_DISTRACTORS + N_HOLDOUT + N_VALIDATION]
    groups = {"planogram": planogram_skus, "distractor": set(distractors), "holdout": set(holdout), "validation": set(validation)}
    names = list(groups)
    for position, left in enumerate(names):
        for right in names[position + 1:]:
            if groups[left] & groups[right]:
                raise SystemExit(f"SKU split overlap between {left} and {right}: {sorted(groups[left] & groups[right])}")

    renderer = ShelfRenderer(catalogue, rng, rows, slots)
    for sub in ("test", "validation"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    def planogram_contents() -> dict[tuple[int, int], list[str]]:
        return {key: [sku] * facings for key, (sku, facings) in expected.items() if sku}

    def correct_tags() -> dict[tuple[int, int], float]:
        return {key: prices[sku] for key, (sku, _) in expected.items() if sku in prices}

    def wrong_price(tags: dict, key: tuple[int, int], delta: float) -> None:
        if sku_at.get(key) in prices:  # an unpriced SKU has no tag to falsify
            tags[key] = prices[sku_at[key]] + delta

    sku_at = {key: sku for key, (sku, _) in expected.items()}
    scenarios = []

    contents, tags = planogram_contents(), correct_tags()
    scenarios.append(("01_compliant", "Every slot matches the planogram; all tag prices correct.", contents, tags))

    contents, tags = planogram_contents(), correct_tags()
    contents[(1, 1)], contents[(1, 2)] = [sku_at[(1, 2)]] * 3, [sku_at[(1, 1)]] * 3
    contents[(2, 3)] = [distractors[0]] * 2
    scenarios.append(("02_misplaced", "Row 1 slots 1 and 2 swapped; row 2 slot 3 holds a non-planogram product.", contents, tags))

    contents, tags = planogram_contents(), correct_tags()
    contents[(2, 4)] = []
    contents[(1, 5)] = [sku_at[(1, 5)]]
    contents[(3, 2)] = []
    scenarios.append(("03_missing", "Row 2 slot 4 and row 3 slot 2 empty; row 1 slot 5 has 1 of 3 facings.", contents, tags))

    contents, tags = planogram_contents(), correct_tags()
    contents[(3, 4)] = [sku_at[(2, 2)]] * 2
    contents[(3, 5)] = [distractors[1]] * 2
    contents[(2, 1)] = [sku_at[(2, 1)]] * 3
    scenarios.append(("04_unexpected", "R3S4 holds the R2S2 product, non-planogram product in R3S5, R2S1 overstocked.", contents, tags))

    contents, tags = planogram_contents(), correct_tags()
    for key, delta in (((1, 3), 20.0), ((2, 2), -5.0), ((3, 1), 10.0)):
        wrong_price(tags, key, delta)
    scenarios.append(("05_price_mismatch", "Placement compliant; tags for R1S3, R2S2, R3S1 show wrong prices.", contents, tags))

    contents, tags = planogram_contents(), correct_tags()
    contents[(2, 5)], contents[(3, 3)] = [sku_at[(3, 3)]] * 2, [sku_at[(2, 5)]] * 2
    contents[(1, 4)] = [sku_at[(1, 4)]]
    contents[(3, 5)] = [distractors[2]]
    wrong_price(tags, (2, 1), 50.0)
    scenarios.append(("06_mixed", "R2S5/R3S3 swapped, R1S4 short one facing, distractor in R3S5, R2S1 price wrong.", contents, tags))

    ground_truth = {
        "seed": seed,
        "synthetic": True,
        "planogram_sha256": file_sha256(DATA_DIR / "planogram.csv"),
        "catalogue_sha256": file_sha256(DATA_DIR / "catalogue.csv"),
        "unpriced_planogram_skus": unpriced,
        "layout": {"type": "equal-grid", "rows": rows, "slots": slots, "width": WIDTH, "height": ROW_HEIGHT * rows,
                   "description": "Fixed-camera front view; rows and slots are equal bands matching the planogram."},
        "split": {"test_planogram_skus": sorted(planogram_skus), "test_distractor_skus": distractors,
                  "validation_skus": validation, "holdout_unknown_skus": holdout},
        "catalogue_prices": prices,
        "test": [],
        "validation": [],
    }
    for name, description, contents, tags in scenarios:
        image, facings, rendered_tags = renderer.render(contents, tags)
        path = output_dir / "test" / f"{name}.jpg"
        image.save(path, quality=92)
        slots_truth = []
        for (row, slot), (sku, facings_expected) in sorted(expected.items()):
            placed = contents.get((row, slot), [])
            slots_truth.append({
                "row": row, "slot": slot, "expected_sku": sku, "expected_facings": facings_expected,
                "placed_skus": placed, "tag_price": tags.get((row, slot)),
                "expected_status": expected_status(sku, facings_expected, placed),
            })
        price_truth = {sku: "Unverified" for sku in unpriced}
        for (row, slot), price in tags.items():
            sku = sku_at[(row, slot)]
            price_truth[sku] = "Match" if abs(price - prices[sku]) < 0.01 else "Mismatch"
        ground_truth["test"].append({
            "image": f"test/{name}.jpg", "scenario": description, "slots": slots_truth,
            "facings": facings, "price_tags": rendered_tags, "expected_price_status": price_truth,
            "unplaced_unexpected_skus": sorted({sku for skus in contents.values() for sku in skus if sku not in planogram_skus}),
        })

    pool = validation + holdout
    for index in range(1, N_VALIDATION_SCENES + 1):
        contents = {}
        for key in expected:
            if rng.random() < 0.9:
                contents[key] = [rng.choice(pool)] * rng.randint(1, 3)
        image, facings, _ = renderer.render(contents, {})
        path = output_dir / "validation" / f"val_{index:02d}.jpg"
        image.save(path, quality=92)
        ground_truth["validation"].append({"image": f"validation/val_{index:02d}.jpg", "facings": facings})

    (output_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False), encoding="utf-8")
    return ground_truth


if __name__ == "__main__":
    truth = build()
    print(f"Wrote {len(truth['test'])} test and {len(truth['validation'])} validation scenes to {OUTPUT_DIR}")
