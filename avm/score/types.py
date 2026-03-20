from typing import TypedDict


class ImageScoring(TypedDict):
    overall_score: float
    background_cleanliness: float
    text_watermark_score: float
    product_prominence: float
    commercial_appeal: float
    bg_class: str
    needs_bg_replacement: bool | None
    reason: str
