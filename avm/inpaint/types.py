from typing import Literal, TypeAlias

InpaintModelKey = Literal[
    "FLUX.1-Fill-dev",
    "stable-diffusion-xl-1.0-inpainting-0.1",
    "stable-diffusion-inpainting",
]

InpaintConfig: TypeAlias = dict[str, float | int]
