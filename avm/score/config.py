from typing import Literal

ModelKey = Literal[
    "Gemma-SEA-LION-v4-27B-IT",
]

MODELS: dict[ModelKey, str] = {
    "Gemma-SEA-LION-v4-27B-IT": "aisingapore/Gemma-SEA-LION-v4-27B-IT",
}

DEFAULT_MODEL_ID = MODELS["Gemma-SEA-LION-v4-27B-IT"]
DEFAULT_MAX_TOKENS = 400
DEFAULT_TEMPERATURE = 0.0
DEFAULT_JPEG_QUALITY = 92
