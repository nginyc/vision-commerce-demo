from typing import Any, cast

import numpy as np
import torch
from PIL import Image

from .segment import SegmentationModel
from .types import InstanceMasks, InstanceScores, MergedMask, SegmentConfig, SegmentInstances


class Sam3SegmentationModel(SegmentationModel):
    """SAM3 segmentation backend with instance and merged-mask APIs."""

    SCORE_THRESHOLD_DEFAULT = 0.5
    MASK_THRESHOLD_DEFAULT = 0.5

    def __init__(self, model_id: str, device: str):
        """Load SAM3 processor/model onto the target device."""
        from transformers import Sam3Model, Sam3Processor

        # Use fp16 only on CUDA for better compatibility on CPU/MPS.
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        self._device = device
        self._processor: Any = Sam3Processor.from_pretrained(model_id)  # type: ignore[reportUnknownMemberType]
        self._model: Any = Sam3Model.from_pretrained(model_id, torch_dtype=torch_dtype)  # type: ignore[reportUnknownMemberType]
        self._model = self._model.to(device).eval()  # type: ignore[reportUnknownMemberType]

    @staticmethod
    def get_config_defaults() -> SegmentConfig:
        """Return default thresholds used during SAM3 post-processing."""
        return {
            "score_threshold": Sam3SegmentationModel.SCORE_THRESHOLD_DEFAULT,
            "mask_threshold": Sam3SegmentationModel.MASK_THRESHOLD_DEFAULT,
        }

    @staticmethod
    def _combine_masks(masks: InstanceMasks, scores: InstanceScores, strategy: str = "union") -> MergedMask:
        """Merge multiple instance masks into one binary mask."""
        if not masks:
            return None

        if strategy == "best":
            if not scores:
                return None
            best_idx = int(np.argmax(scores))
            return masks[best_idx].astype(np.uint8) * 255

        combined = np.zeros_like(masks[0], dtype=bool)
        for mask in masks:
            combined |= mask
        return combined.astype(np.uint8) * 255

    def segment_instances(
        self,
        image: Image.Image,
        prompt: str,
        config: SegmentConfig,
    ) -> SegmentInstances:
        """Run SAM3 instance segmentation and return masks with confidence scores."""
        inputs = self._processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        image_h, image_w = image.size[1], image.size[0]
        score_threshold = float(config.get("score_threshold", self.SCORE_THRESHOLD_DEFAULT))
        mask_threshold = float(config.get("mask_threshold", self.MASK_THRESHOLD_DEFAULT))
        post_processed = cast(list[dict[str, Any]], self._processor.post_process_instance_segmentation(  # type: ignore[reportUnknownMemberType]
            outputs,
            threshold=score_threshold,
            mask_threshold=mask_threshold,
            target_sizes=[(image_h, image_w)],
        ))
        result = post_processed[0] if post_processed else {}

        raw_masks = cast(list[torch.Tensor], result.get("masks", []))
        raw_scores = cast(list[Any], result.get("scores", []))

        masks = [mask.detach().cpu().numpy().astype(bool) for mask in raw_masks]
        scores = [float(score) for score in raw_scores]
        return masks, scores

    def generate_mask(
        self,
        image: Image.Image,
        prompt: str,
        config: SegmentConfig,
    ) -> MergedMask:
        """Generate a single merged binary mask from SAM3 instance predictions."""
        masks, scores = self.segment_instances(image, prompt, config)
        return self._combine_masks(masks, scores, strategy="union")
