"""Modular crop-to-SKU matching and planogram comparison pipeline."""

from .catalog import load_catalogue_references
from .recognizer import ProductRecognizer, MatchingUnavailable
from .shelf_mapper import CoordinateShelfMapper
from .planogram_compare import compare_planogram
from .pipeline import ProductMatchingPipeline

__all__ = [
    "CoordinateShelfMapper",
    "MatchingUnavailable",
    "ProductMatchingPipeline",
    "ProductRecognizer",
    "compare_planogram",
    "load_catalogue_references",
]
