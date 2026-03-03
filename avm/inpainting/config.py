
from typing import Literal

ModelKey = Literal[
    "FLUX.1-Fill-dev",
    "stable-diffusion-xl-1.0-inpainting-0.1",
    "stable-diffusion-inpainting",
    "kandinsky-2-2-decoder-inpaint",
]

MODELS: dict[ModelKey, str] = {
    "FLUX.1-Fill-dev": "FLUX.1-Fill-dev",
    "stable-diffusion-xl-1.0-inpainting-0.1": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "stable-diffusion-inpainting": "stable-diffusion-v1-5/stable-diffusion-inpainting"
}
DEFAULT_MODEL_ID = MODELS['FLUX.1-Fill-dev']