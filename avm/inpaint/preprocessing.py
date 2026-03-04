import numpy as np
from PIL import Image


def make_fill_mask(mask: Image.Image) -> Image.Image:
    '''
    Convert a RGBA mask to a single-channel fill mask.
    '''
    mask_rgba = mask.convert("RGBA")
    alpha = np.array(mask_rgba)[:, :, 3]
    fill = np.where(alpha > 128, 0, 255).astype(np.uint8)
    return Image.fromarray(fill, mode="L")


def resize_to_multiple(
    image: Image.Image,
    multiple: int = 64,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    '''
    Resize the image so that its dimensions are multiples of the specified value.
    '''
    w, h = image.size
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple

    if new_w == 0 or new_h == 0:
        return image

    if (new_w, new_h) != (w, h):
        return image.resize((new_w, new_h), resample=resample)
    return image
