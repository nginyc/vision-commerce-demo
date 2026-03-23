"""
Fine-tuned DINOv2 ViT-B/14 + SEA-LION hybrid scoring model.

DINOv2 provides overall_score (1–10).
SEA-LION provides background_cleanliness, text_watermark_score,
product_prominence, commercial_appeal, bg_class, and reason.

Preprocessing matches the val_transform used during training:
    ScaleImage(332) → PadToSquare(fill=255) → Resize(224) → ToTensor → ImageNet normalise
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms  # type: ignore

from .score import OpenAIScoringModel, ScoringModel
from .types import ImageScoring


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class _ScaleImage:
    """Downscale so the longest side ≤ max_px. Never upscales."""

    def __init__(self, max_px: int = 332) -> None:
        self.max_px = max_px

    def __call__(self, img: Image.Image) -> Image.Image:
        img.thumbnail((self.max_px, self.max_px), resample=Image.Resampling.LANCZOS)
        return img


class _PadToSquare:
    """Pad to square with a solid fill colour. Preserves aspect ratio."""

    def __init__(self, fill: int = 255) -> None:
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        pl, pt = (s - w) // 2, (s - h) // 2
        return ImageOps.expand(img, border=(pl, pt, s - w - pl, s - h - pt), fill=self.fill)


_INFERENCE_TRANSFORM = transforms.Compose([
    _ScaleImage(332),
    _PadToSquare(fill=255),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


class _DINOv2Net(nn.Module):
    """Matches the DINOv2Scorer architecture used during training."""

    def __init__(self, pretrained: str, dropout: float) -> None:
        super().__init__()
        from transformers import AutoModel
        self.backbone: Any = AutoModel.from_pretrained(pretrained)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        cls = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        return self.head(cls).squeeze(-1)


class DINOv2WithSealionScoringModel(ScoringModel):
    """
    Hybrid scoring model: DINOv2 for overall_score, SEA-LION for sub-scores.

    Args:
        sealion_model:   An already-constructed OpenAIScoringModel pointed at SEA-LION.
        checkpoint_path: Path to the DINOv2 .pt checkpoint saved by train_dinov2.py.
        device:          Torch device string, e.g. "cpu", "cuda", "mps".
        pretrained:      HuggingFace backbone ID — must match the one used at training time.
        dropout:         Dropout rate — must match the one used at training time.
    """

    def __init__(
        self,
        sealion_model: OpenAIScoringModel,
        checkpoint_path: str,
        device: str = "cpu",
        pretrained: str = "facebook/dinov2-base",
        dropout: float = 0.1,
    ) -> None:
        self._device = torch.device(device)
        self._net = _DINOv2Net(pretrained=pretrained, dropout=dropout).to(self._device)

        ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        self._net.load_state_dict(state_dict)
        self._net.eval()

        self._sealion = sealion_model

    def score_image(self, image: Image.Image) -> ImageScoring:
        try:
            # DINOv2: overall score
            tensor = torch.as_tensor(_INFERENCE_TRANSFORM(image.convert("RGB"))).unsqueeze(0).to(self._device)
            with torch.no_grad():
                raw = self._net(tensor).item()
            overall = round(float(max(1.0, min(10.0, raw))), 2)

            # SEA-LION: sub-dimension scores
            sealion = self._sealion.score_image(image)

            return {
                "overall_score": overall,
                "background_cleanliness": sealion["background_cleanliness"],
                "text_watermark_score": sealion["text_watermark_score"],
                "product_prominence": sealion["product_prominence"],
                "commercial_appeal": sealion["commercial_appeal"],
                "bg_class": sealion["bg_class"],
                "reason": sealion["reason"],
            }
        except Exception as exc:
            return {
                "overall_score": -1,
                "background_cleanliness": -1,
                "text_watermark_score": -1,
                "product_prominence": -1,
                "commercial_appeal": -1,
                "bg_class": "",
                "reason": f"error: {exc}",
            }
