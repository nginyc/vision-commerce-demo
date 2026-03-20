from typing import Literal

ModelKey = Literal[
    "sealionv4",
    "dinov2-finetuned-with-sealionv4",
]

MODELS: dict[ModelKey, str] = {
    "sealionv4": "aisingapore/Gemma-SEA-LION-v4-27B-IT",
    "dinov2-finetuned-with-sealionv4": "dinov2-finetuned-with-sealionv4",
}

DEFAULT_MODEL_ID = MODELS["dinov2-finetuned-with-sealionv4"]
DEFAULT_MAX_TOKENS = 400
DEFAULT_TEMPERATURE = 0.0
DEFAULT_JPEG_QUALITY = 92
