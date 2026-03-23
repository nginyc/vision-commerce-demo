from typing import Literal, TypeAlias

InpaintModelKey = Literal[
    "flux",
    "flux-finetuned",
    "flux-finetuned-fixed-prompt",
    "stable-diffusion-xl",
    "stable-diffusion",
]

InpaintConfig: TypeAlias = dict[str, float | int]
