"""
DINOv2 ViT-B/14 backbone with a regression head for aesthetic scoring.

Backbone: facebook/dinov2-base (HuggingFace) — same weights as torch.hub dinov2_vitb14.
Uses the 768-dim CLS token (last_hidden_state[:, 0, :]) for regression.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv2Scorer(nn.Module):
    def __init__(
        self,
        pretrained: str = "facebook/dinov2-base",
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        unfreeze_last_n: int = 4,
    ):
        """
        Args:
            pretrained:       HuggingFace model ID. Default: facebook/dinov2-base (ViT-B/14, 86M).
            dropout:          Dropout before the regression head.
            freeze_backbone:  If True, freeze the backbone. Use with unfreeze_last_n
                              for partial fine-tuning (recommended for small datasets).
            unfreeze_last_n:  When freeze_backbone=True, unfreeze the last N transformer
                              blocks + final layernorm. Recommended: 4 for 4k samples.
        """
        super().__init__()

        self.backbone = AutoModel.from_pretrained(pretrained)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if unfreeze_last_n > 0:
                # HuggingFace DINOv2: backbone.encoder.layer (ModuleList of 12 blocks)
                for block in self.backbone.encoder.layer[-unfreeze_last_n:]:
                    for p in block.parameters():
                        p.requires_grad = True
                for p in self.backbone.layernorm.parameters():
                    p.requires_grad = True

        # Regression head: 768-dim CLS token -> scalar score (1-10 range)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        cls = outputs.last_hidden_state[:, 0, :]  # [B, 768] CLS token
        return self.head(cls).squeeze(-1)          # [B]
