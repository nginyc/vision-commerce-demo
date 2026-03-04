import numpy as np
from PIL import Image

def create_mask_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    if mask.dtype == np.bool_:
        alpha = mask.astype(np.uint8) * 255
    else:
        alpha = mask.astype(np.uint8)

    out_mask = np.array(image.convert("RGB"), dtype=np.uint8)
    out_mask[alpha == 0] = 0
    out_mask = np.dstack([out_mask, alpha])
    return Image.fromarray(out_mask, mode="RGBA")
