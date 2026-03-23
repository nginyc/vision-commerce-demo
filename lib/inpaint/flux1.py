from PIL import Image
from pathlib import Path
import os
import tempfile
from .inpaint import InpaintingModel
from .types import InpaintConfig


_ACTIVATION_TOKEN = "A JD background"


class Flux1InpaintingModel(InpaintingModel):
    def __init__(self, quantize: int = 8, lora_paths: list[str] | None = None, lora_scales: list[float] | None = None):
        from mflux.models.flux.variants.fill.flux_fill import Flux1Fill  # type: ignore
        self._pipeline = Flux1Fill(quantize=quantize, lora_paths=lora_paths, lora_scales=lora_scales)

    @staticmethod
    def get_config_defaults() -> InpaintConfig:
        return {
            "guidance": 30.0,
            "num_inference_steps": 28
        }

    def inpaint(
        self, image: Image.Image, mask: Image.Image, prompt: str, seed: int, 
        config: InpaintConfig,
    ) -> Image.Image:
        from mflux.models.flux.variants.fill.flux_fill import Config  # type: ignore

        width, height = image.size
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = os.path.join(tmp_dir, "input.png")
            mask_path = os.path.join(tmp_dir, "fill_mask.png")
            output_path = os.path.join(tmp_dir, "output.png")

            image.save(image_path)
            mask.save(mask_path)

            output = self._pipeline.generate_image(
                seed=seed,
                prompt=prompt,
                width=width,
                height=height,
                guidance=config['guidance'],
                image_path=Path(image_path),
                num_inference_steps=int(config['num_inference_steps']),
                masked_image_path=Path(mask_path),
            )
            output.save(path=output_path)
            return Image.open(output_path).convert("RGB").copy()


class Flux1FinetunedFixedPromptInpaintingModel(Flux1InpaintingModel):
    def __init__(self, lora_path: str):
        super().__init__(lora_paths=[lora_path], lora_scales=[1.0])

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str, seed: int, config: InpaintConfig) -> Image.Image:
        return super().inpaint(image, mask, _ACTIVATION_TOKEN, seed, config)


class Flux1FinetunedInpaintingModel(Flux1InpaintingModel):
    def __init__(self, lora_path: str):
        super().__init__(lora_paths=[lora_path], lora_scales=[1.0])

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str, seed: int, config: InpaintConfig) -> Image.Image:
        return super().inpaint(image, mask, f"{_ACTIVATION_TOKEN} {prompt}", seed, config)
