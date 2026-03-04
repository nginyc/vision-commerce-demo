from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from .config import CATEGORY_OVERRIDES, MODELS


class SegmentationModel(ABC):
    """Base interface for segmentation backends."""

    @abstractmethod
    def segment_instances(
        self,
        image: Image.Image,
        prompt: str,
        config: dict[str, float | int],
    ) -> tuple[list[np.ndarray], list[float]]:
        """Return instance-level masks and scores for a text prompt."""
        pass

    @abstractmethod
    def generate_mask(
        self,
        image: Image.Image,
        prompt: str,
        config: dict[str, float | int],
    ) -> np.ndarray | None:
        """Return a single merged binary mask for the given prompt."""
        pass

    @staticmethod
    def get_config_defaults() -> dict[str, float | int]:
        """Return backend-specific default configuration values."""
        return {}


def get_model_config_defaults(model_id: str) -> dict[str, float | int]:
    if model_id == MODELS["sam3"]:
        from .sam3 import Sam3SegmentationModel

        return Sam3SegmentationModel.get_config_defaults()
    raise ValueError(f"Unsupported segmentation model id: {model_id}")


def build_model(model_id: str, device: str) -> SegmentationModel:
    if model_id == MODELS["sam3"]:
        from .sam3 import Sam3SegmentationModel

        return Sam3SegmentationModel(model_id, device)
    raise ValueError(f"Unsupported segmentation model id: {model_id}")


def normalize_category(category: str) -> str:
    """Normalize dataset category text using configured overrides."""
    return CATEGORY_OVERRIDES.get(category.strip().lower(), category.strip())
