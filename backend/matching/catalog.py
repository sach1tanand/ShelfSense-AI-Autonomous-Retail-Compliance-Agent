"""Catalogue reference-image loading and persistent CLIP embedding cache."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .encoder import ImageEncoder


def load_catalogue_references(catalogue: list[dict[str, Any]], data_dir: Path) -> list[dict[str, Any]]:
    """Return products with safe, existing local reference images only."""
    root = Path(data_dir).resolve()
    image_root = (root / "product_images").resolve()
    references = []
    for product in catalogue:
        sku = str(product.get("sku", "")).strip()
        relative = str(product.get("image_local_path", "")).strip()
        if not sku or not relative:
            continue
        image_path = (root / relative).resolve()
        if image_path.parent != image_root or not image_path.is_file():
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError, ValueError):
            continue
        references.append({"sku": sku, "name": str(product.get("name", "")), "image_path": image_path})
    return references


class CatalogueEmbeddingCache:
    """Caches normalized catalogue vectors by SKU, file signature, and CLIP model id."""

    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)

    @staticmethod
    def _signature(reference: dict[str, Any], model_id: str) -> str:
        image_path = Path(reference["image_path"])
        stat = image_path.stat()
        return f"{model_id}|{image_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"

    def get_embeddings(
        self,
        references: list[dict[str, Any]],
        encoder: ImageEncoder,
    ) -> tuple[dict[str, list[float]], dict[str, Any]]:
        cache: dict[str, Any] = {"version": 1, "embeddings": {}}
        try:
            loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if loaded.get("version") == 1 and isinstance(loaded.get("embeddings"), dict):
                cache = loaded
        except (OSError, json.JSONDecodeError):
            pass

        embeddings: dict[str, list[float]] = {}
        pending: list[dict[str, Any]] = []
        cache_entries = cache["embeddings"]
        for reference in references:
            sku = str(reference["sku"])
            signature = self._signature(reference, encoder.model_id)
            entry = cache_entries.get(sku)
            if entry and entry.get("signature") == signature and isinstance(entry.get("vector"), list):
                embeddings[sku] = [float(value) for value in entry["vector"]]
            else:
                pending.append({**reference, "signature": signature})

        if pending:
            images = []
            for reference in pending:
                with Image.open(reference["image_path"]) as image:
                    images.append(image.convert("RGB"))
            vectors = encoder.encode_images(images)
            if len(vectors) != len(pending):
                raise ValueError("Image encoder returned a different number of vectors than input images")
            for reference, vector in zip(pending, vectors):
                normalized = _normalize_vector(vector)
                sku = str(reference["sku"])
                embeddings[sku] = normalized
                cache_entries[sku] = {"signature": reference["signature"], "vector": normalized}
            self._write_cache(cache)
        return embeddings, {"cached": len(references) - len(pending), "computed": len(pending)}

    def _write_cache(self, cache: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="", dir=self.cache_path.parent,
                delete=False, suffix=".tmp",
            ) as temporary:
                temporary_name = temporary.name
                json.dump(cache, temporary, separators=(",", ":"))
            os.replace(temporary_name, self.cache_path)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = sum(float(value) * float(value) for value in vector) ** 0.5
    if norm <= 0:
        raise ValueError("Image encoder returned a zero-length embedding")
    return [float(value) / norm for value in vector]
