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


class PerceptualFeatureEncoder:
    """Dependency-free 192-d colour-layout encoder (8x8 RGB grid).

    Deterministic and offline, useful for tests and smoke checks. It captures
    coarse colour layout only and is not a semantic product recogniser.
    """

    model_id = "perceptual-grid-v1"

    def __init__(self, grid_size: int = 8):
        self.grid_size = grid_size

    def encode_images(self, images: list[Image.Image]) -> list[list[float]]:
        vectors = []
        for image in images:
            small = image.convert("RGB").resize((self.grid_size, self.grid_size), Image.Resampling.BILINEAR)
            values = [channel / 255.0 for pixel in small.getdata() for channel in pixel]
            norm = sum(value * value for value in values) ** 0.5 or 1.0
            vectors.append([value / norm for value in values])
        return vectors


class TorchvisionEncoder:
    """ImageNet-pretrained MobileNetV3-Large features (960-d) for image-to-image retrieval.

    Weights (21 MB, BSD-3) are downloaded once by torchvision from download.pytorch.org.
    """

    model_id = "torchvision-mobilenet_v3_large-imagenet1k_v2"

    def __init__(self, device: str | None = None):
        try:
            import torch
            from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
        except Exception as error:  # ImportError, or a blocked torch DLL (OSError)
            raise ClipEncoderUnavailable(f"torchvision could not be imported: {error}") from error
        self._torch = torch
        self.device = device or "cpu"
        try:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
            self.model = mobilenet_v3_large(weights=weights).to(self.device).eval()
        except Exception as error:
            raise ClipEncoderUnavailable(f"Could not load MobileNetV3 ImageNet weights: {error}") from error
        self.transform = weights.transforms()

    def encode_images(self, images: list[Image.Image]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(images), 32):
            batch = self._torch.stack([self.transform(image.convert("RGB")) for image in images[start:start + 32]])
            with self._torch.inference_mode():
                features = self.model.avgpool(self.model.features(batch.to(self.device))).flatten(1)
                features = self._torch.nn.functional.normalize(features, dim=1)
            vectors.extend(features.cpu().tolist())
        return vectors


def create_encoder(model_id: str) -> ImageEncoder:
    """Build the encoder named by model_id (torchvision-*, perceptual-grid-v1, or a CLIP model id)."""
    if model_id == TorchvisionEncoder.model_id:
        return TorchvisionEncoder()
    if model_id == PerceptualFeatureEncoder.model_id:
        return PerceptualFeatureEncoder()
    return TransformersClipEncoder(model_id)


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
