from abc import ABC, abstractmethod
from PIL import Image
from .config import MODELS
from .preprocessing import make_fill_mask, resize_to_multiple

class InpaintingModel(ABC):
    @abstractmethod
    def inpaint(
        self, image: Image.Image, mask: Image.Image, prompt: str, seed: int, 
        config: dict[str, float | int],
    ) -> Image.Image:
        pass

    @staticmethod
    def get_config_defaults() -> dict[str, float | int]:
        return {}

def get_model_config_defaults(model_id: str) -> dict[str, float | int]:
    if model_id == MODELS['FLUX.1-Fill-dev']:
        from .flux1 import Flux1InpaintingModel
        return Flux1InpaintingModel.get_config_defaults()
    else:
        from .huggingface import HuggingFaceInpaintingModel
        return HuggingFaceInpaintingModel.get_config_defaults()


def build_model(model_id: str, device: str) -> InpaintingModel:
    if model_id == MODELS['FLUX.1-Fill-dev']:
        from .flux1 import Flux1InpaintingModel
        return Flux1InpaintingModel()
    else:
        from .huggingface import HuggingFaceInpaintingModel
        return HuggingFaceInpaintingModel(model_id, device)

def inpaint_background(
    model: InpaintingModel,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    config: dict[str, float | int],
    seed: int = 42,
) -> Image.Image:
    '''
    Inpaint the background of the given image using the specified model and parameters.
    Returns the generated image.
    '''
    image_rgb = resize_to_multiple(image.convert("RGB"))
    mask_for_inpaint = make_fill_mask(mask)
    mask_for_inpaint = mask_for_inpaint.resize(image_rgb.size, resample=Image.Resampling.NEAREST)

    output_image = model.inpaint(
        image=image_rgb,
        mask=mask_for_inpaint,
        prompt=prompt,
        seed=seed,
        config=config,
    )
    return output_image
