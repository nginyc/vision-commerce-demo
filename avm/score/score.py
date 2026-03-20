from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

from PIL import Image

from .config import DEFAULT_JPEG_QUALITY, DEFAULT_MAX_TOKENS, DEFAULT_MODEL_ID, DEFAULT_TEMPERATURE
from .prompts import SCORING_PROMPT
from .types import ImageScoring
from ..utils import encode_image_to_base64


class ScoringModel(ABC):
    @abstractmethod
    def score_image(self, image: Image.Image) -> ImageScoring:
        pass


class OpenAIScoringModel(ScoringModel):
    def __init__(
        self,
        client: Any,
        model_id: str = DEFAULT_MODEL_ID,
        prompt: str = SCORING_PROMPT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ):
        self._client = client
        self._model_id = model_id
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._jpeg_quality = jpeg_quality

    def score_image(self, image: Image.Image) -> ImageScoring:
        payload = encode_image_to_base64(image, format="JPEG", quality=self._jpeg_quality)

        try:
            response = self._client.chat.completions.create(
                model=self._model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{payload}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )

            raw = _extract_text_from_message_content(response.choices[0].message.content)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                return _error_scoring(f"no_json: {raw[:120]}")

            import json

            parsed_any = json.loads(raw[start:end])
            if not isinstance(parsed_any, dict):
                return _error_scoring("invalid_json_payload")

            parsed = cast(dict[str, Any], parsed_any)
            overall_raw = parsed.get("overall_score")
            overall = _to_score(overall_raw)
            if overall == -1:
                return _error_scoring("invalid_overall_score")

            if not (1 <= overall <= 10):
                return _error_scoring(f"out_of_range: {overall}")

            return {
                "overall_score": overall,
                "background_cleanliness": _to_score(parsed.get("background_cleanliness")),
                "text_watermark_score": _to_score(parsed.get("text_watermark_score")),
                "product_prominence": _to_score(parsed.get("product_prominence")),
                "commercial_appeal": _to_score(parsed.get("commercial_appeal")),
                "bg_class": str(parsed.get("background_class", "")),
                "reason": str(parsed.get("primary_failure_reason") or parsed.get("background_description", "")),
            }
        except Exception as exc:
            return _error_scoring(f"error: {exc}")

def _error_scoring(reason: str) -> ImageScoring:
    return {
        "overall_score": -1,
        "background_cleanliness": -1,
        "text_watermark_score": -1,
        "product_prominence": -1,
        "commercial_appeal": -1,
        "bg_class": "",
        "reason": reason,
    }


def _to_score(value: Any) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return -1



def _extract_text_from_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        items = cast(list[Any], content)
        for item in items:
            if not isinstance(item, dict):
                continue
            item_dict = cast(dict[str, Any], item)
            if item_dict.get("type") == "text":
                parts.append(str(item_dict.get("text", "")))
        return "\n".join(parts)
    return ""