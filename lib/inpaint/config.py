from .types import InpaintModelKey

MODELS: dict[InpaintModelKey, str] = {
    "flux-finetuned-fixed-prompt": "flux-finetuned-fixed-prompt",
    "flux-finetuned": "flux-finetuned",
    "flux": "flux",
    "stable-diffusion-xl": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "stable-diffusion": "stable-diffusion-v1-5/stable-diffusion-inpainting",
}
DEFAULT_MODEL_ID = MODELS["flux-finetuned"]