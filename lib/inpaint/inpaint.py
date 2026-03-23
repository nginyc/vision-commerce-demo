from abc import ABC, abstractmethod
from PIL import Image
from .preprocessing import make_fill_mask, resize_to_multiple
from .types import InpaintConfig

class InpaintingModel(ABC):
    @abstractmethod
    def inpaint(
        self, image: Image.Image, mask: Image.Image, prompt: str, seed: int, 
        config: InpaintConfig,
    ) -> Image.Image:
        pass

    @staticmethod
    def get_config_defaults() -> InpaintConfig:
        return {}

_FLUX_MODEL_IDS = {"flux", "flux-finetuned", "flux-finetuned-fixed-prompt"}


def get_model_config_defaults(model_id: str) -> InpaintConfig:
    if model_id in _FLUX_MODEL_IDS:
        from .flux1 import Flux1InpaintingModel
        return Flux1InpaintingModel.get_config_defaults()
    else:
        from .huggingface import HuggingFaceInpaintingModel
        return HuggingFaceInpaintingModel.get_config_defaults()


def build_model(model_id: str, device: str, lora_path: str | None = None) -> InpaintingModel:
    if model_id == "flux":
        from .flux1 import Flux1InpaintingModel
        return Flux1InpaintingModel()
    elif model_id == "flux-finetuned-fixed-prompt":
        from .flux1 import Flux1FinetunedFixedPromptInpaintingModel
        if lora_path is None:
            raise ValueError("lora_path is required for flux-finetuned-fixed-prompt")
        return Flux1FinetunedFixedPromptInpaintingModel(lora_path=lora_path)
    elif model_id == "flux-finetuned":
        from .flux1 import Flux1FinetunedInpaintingModel
        if lora_path is None:
            raise ValueError("lora_path is required for flux-finetuned")
        return Flux1FinetunedInpaintingModel(lora_path=lora_path)
    else:
        from .huggingface import HuggingFaceInpaintingModel
        return HuggingFaceInpaintingModel(model_id, device)

def inpaint_background(
    model: InpaintingModel,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    config: InpaintConfig,
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
