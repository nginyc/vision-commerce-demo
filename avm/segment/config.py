from .types import SegmentModelKey

MODELS: dict[SegmentModelKey, str] = {
    "sam3": "facebook/sam3",
}
DEFAULT_MODEL_ID = MODELS["sam3"]

CATEGORY_OVERRIDES: dict[str, str] = {
    "second hand watches": "watch",
    "national watch": "watch",
    "fabric sofa": "sofa",
    "decorative ornaments": "ornament",
    "women casual shoes": "shoe",
    "men casual shoes": "shoe",
    "sports and casual shoes": "shoe",
    "running shoes": "shoe",
    "bracelet or anklet": "bracelet",
    "women shoulder or crossbody bag": "bag",
    "instant food": "food",
}
