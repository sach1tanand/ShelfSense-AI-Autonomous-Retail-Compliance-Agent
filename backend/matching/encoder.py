"""CLIP image encoder adapter, imported lazily to keep basic UI use lightweight."""

from __future__ import annotations

from typing import Protocol

from PIL import Image


class ImageEncoder(Protocol):
    """Minimal interface required by the matching and test pipelines."""

    model_id: str

    def encode_images(self, images: list[Image.Image]) -> list[list[float]]: ...


class ClipEncoderUnavailable(RuntimeError):
    """Raised when optional CLIP runtime packages or model weights are unavailable."""


class TransformersClipEncoder:
    """Pretrained CLIP image encoder using Hugging Face Transformers + PyTorch."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str | None = None):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise ClipEncoderUnavailable(
                "CLIP matching requires the optional packages in requirements-matching.txt."
            ) from error

        self.model_id = model_id
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        except Exception as error:
            raise ClipEncoderUnavailable(
                f"Could not load pretrained CLIP model {model_id!r}: {error}"
            ) from error

    def encode_images(self, images: list[Image.Image]) -> list[list[float]]:
        if not images:
            return []
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        return features.detach().cpu().tolist()
