from typing import cast
from PIL import Image
import torch
from .inpaint import InpaintingModel
from .types import InpaintConfig

class HuggingFaceInpaintingModel(InpaintingModel):
    def __init__(self, model_id: str, device: str):
        from diffusers import DiffusionPipeline # type: ignore

        dtype = torch.float16 if device == "cuda" else torch.float32

        kwargs: dict[str, object] = {}
        if dtype == torch.float16:
            kwargs["variant"] = "fp16"

        pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, **kwargs) # type: ignore
        if device == "cuda":
            pipeline.enable_model_cpu_offload() 
        else:
            pipeline = pipeline.to(device) # type: ignore

        self._model_id = model_id
        self._device = device
        self._pipeline = pipeline

    @staticmethod
    def get_config_defaults() -> InpaintConfig:
        return {
            "strength": 1.0,
            "guidance_scale": 7.5,
            "num_inference_steps": 50
        }

    def inpaint(
        self, image: Image.Image, mask: Image.Image, prompt: str, seed: int, 
        config: InpaintConfig,
    ) -> Image.Image:
        width, height = image.size
        
        generator = torch.Generator(device="cpu").manual_seed(seed)
        call_kwargs: dict[str, object] = {
            "prompt": prompt,
            "image": image,
            "mask_image": mask,
            "width": width,
            "height": height,
            "generator": generator,
            **config,
        }

        output_images = cast(list[Image.Image], self._pipeline(**call_kwargs).images)  # type: ignore
        if not output_images:
            raise RuntimeError("Inpainting pipeline returned no images")
        return output_images[0]
    