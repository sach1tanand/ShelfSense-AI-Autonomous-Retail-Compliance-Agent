import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.off_ingestion import CSV_FIELDS, ingest_products, is_valid_gtin, verify_catalogue


def make_gtin(prefix):
    digits = [int(value) for value in prefix]
    weighted = sum(value * (3 if index % 2 == 0 else 1) for index, value in enumerate(reversed(digits)))
    return prefix + str((10 - weighted % 10) % 10)


def png_bytes():
    image = Image.new("RGB", (4, 4), color="orange")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, *, payload=None, body=b"", status_code=200, content_type="image/png"):
        self.payload = payload
        self.body = body
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        yield self.body

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, pages, images):
        self.pages = pages
        self.images = images
        self.headers = {}
        self.closed = False
        self.api_calls = 0

    def get(self, url, *, params=None, timeout=None, stream=False):
        if "api/v2/search" in url:
            self.api_calls += 1
            payload = self.pages.get(params["page"], {"products": []})
            return FakeResponse(payload=payload, content_type="application/json")
        return self.images[url]

    def close(self):
        self.closed = True


class OpenFoodFactsIngestionTests(unittest.TestCase):
    def test_gtin_checksum_validation(self):
        barcode = make_gtin("89012345678")
        self.assertTrue(is_valid_gtin(barcode))
        self.assertFalse(is_valid_gtin(barcode[:-1] + str((int(barcode[-1]) + 1) % 10)))
        self.assertFalse(is_valid_gtin("123abc"))

    def test_ingestion_deduplicates_downloads_and_verifies_csv_paths(self):
        first_barcode = make_gtin("89012345678")
        broken_barcode = make_gtin("89054321098")
        second_barcode = make_gtin("89011111222")
        products = [
            {"code": "invalid", "product_name": "Not valid", "image_front_url": "https://img.test/invalid"},
            {"code": first_barcode, "product_name": "Indian snack", "brands": "Example", "categories": "Snacks", "quantity": "50 g", "image_front_url": "https://img.test/good", "url": "https://world.openfoodfacts.org/product/" + first_barcode},
            {"code": first_barcode, "product_name": "Duplicate", "image_front_url": "https://img.test/good"},
            {"code": broken_barcode, "product_name": "Broken image", "image_front_url": "https://img.test/broken"},
            {"code": second_barcode, "product_name": "Missing image"},
        ]
        session = FakeSession(
            {1: {"products": products}, 2: {"products": [{"code": second_barcode, "product_name": "Second snack", "image_front_url": "https://img.test/second"}]}, 3: {"products": []}},
            {
                "https://img.test/good": FakeResponse(body=png_bytes()),
                "https://img.test/broken": FakeResponse(body=b"not an image"),
                "https://img.test/second": FakeResponse(body=png_bytes()),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            result = ingest_products(max_products=10, data_dir=Path(directory), session=session, page_size=5, request_pause=0)
            with result.output_csv.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(result.valid_products, 2)
            self.assertEqual(result.downloaded_images, 2)
            self.assertEqual(result.failed_downloads, 1)
            self.assertEqual(session.api_calls, 3)
            self.assertEqual(list(rows[0]), CSV_FIELDS)
            self.assertEqual(len({row["barcode"] for row in rows}), 2)
            self.assertEqual(rows[0]["expected_price"], "")
            self.assertEqual(rows[0]["brand"], "Example")
            self.assertEqual(rows[0]["image_local_path"], f"product_images/{first_barcode}.png")
            for row in rows:
                local_path = Path(directory) / row["image_local_path"]
                self.assertTrue(local_path.is_file())
                with Image.open(local_path) as image:
                    image.verify()
            self.assertFalse((result.image_directory / f"{broken_barcode}.png").exists())
            verification = verify_catalogue(Path(directory))
            self.assertEqual(verification["valid_products"], 2)
            self.assertEqual(verification["verified_images"], 2)

    def test_retries_rate_limited_requests(self):
        import requests
        from src.off_ingestion import _request_with_retries

        session = FakeSession({}, {})
        limited = FakeResponse(status_code=429, content_type="application/json")
        success = FakeResponse(payload={"products": []}, content_type="application/json")
        session.get = unittest.mock.Mock(side_effect=[limited, success])
        with patch("src.off_ingestion.time.sleep") as sleep:
            response = _request_with_retries(session, "https://example.test", max_retries=2)

        self.assertIs(response, success)
        sleep.assert_called_once()
        failing_session = FakeSession({}, {})
        failing_session.get = unittest.mock.Mock(return_value=FakeResponse(status_code=503))
        with self.assertRaises(requests.RequestException):
            _request_with_retries(failing_session, "https://example.test", max_retries=1)


if __name__ == "__main__":
    unittest.main()
