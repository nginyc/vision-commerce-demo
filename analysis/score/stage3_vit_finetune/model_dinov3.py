"""
DINOv3 ViT-B/16 backbone with a regression head for aesthetic scoring.

Backbone: facebook/dinov3-vitb16-pretrain-lvd1689m (HuggingFace transformers).
Pretrained on LVD-1689M (1.69B images) -- ~12x the scale of DINOv2's LVD-142M.
Uses the 768-dim CLS token (last_hidden_state[:, 0, :]) for regression.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv2Scorer(nn.Module):
    def __init__(
        self,
        pretrained: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        unfreeze_last_n: int = 4,
    ):
        """
        Args:
            pretrained:       HuggingFace model ID. Default: facebook/dinov3-vitb16-pretrain-lvd1689m.
                              Larger options: facebook/dinov3-vitl16-pretrain-lvd1689m.
            dropout:          Dropout before the regression head.
            freeze_backbone:  If True, freeze the backbone. Use with unfreeze_last_n
                              for partial fine-tuning (recommended for small datasets).
            unfreeze_last_n:  When freeze_backbone=True, unfreeze the last N transformer
                              blocks + final layernorm. Recommended: 4 for 4k samples.
        """
        super().__init__()

        # Load DINOv3 ViT-B/16 from HuggingFace (transformers).
        # Requires: huggingface_hub login + accepted facebook/dinov3 licence.
        self.backbone = AutoModel.from_pretrained(pretrained, trust_remote_code=True)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if unfreeze_last_n > 0:
                # backbone.layer = ModuleList of DINOv3ViTLayer blocks
                for block in self.backbone.layer[-unfreeze_last_n:]:
                    for p in block.parameters():
                        p.requires_grad = True
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True

        # Regression head: 768-dim CLS token -> scalar score (1-10 range)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        cls = outputs.last_hidden_state[:, 0, :]  # [B, 768] post-layernorm CLS token
        return self.head(cls).squeeze(-1)          # [B]
