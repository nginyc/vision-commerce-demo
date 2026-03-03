from PIL import Image
from pathlib import Path
import os
import tempfile
from .inpaint import InpaintingModel


class Flux1InpaintingModel(InpaintingModel):
    def __init__(self, quantize: int = 8):
        from mflux.models.flux.variants.fill.flux_fill import Flux1Fill  # type: ignore
        self._pipeline = Flux1Fill(quantize=quantize)

    @staticmethod
    def get_config_defaults() -> dict[str, float | int]:
        return {
            "guidance": 30.0,
            "num_inference_steps": 28
        }

    def inpaint(
        self, image: Image.Image, mask: Image.Image, prompt: str, seed: int, 
        config: dict[str, float | int],
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
                config=Config(
                    width=width,
                    height=height,
                    guidance=config['guidance'],
                    image_path=Path(image_path),
                    num_inference_steps=int(config['num_inference_steps']),
                    masked_image_path=Path(mask_path),
                )
            )
            output.save(path=output_path)
            return Image.open(output_path).convert("RGB").copy()
        