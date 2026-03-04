from pathlib import Path
from PIL import Image
from typing import Generator, TypedDict

import torch

def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class InfoRecord(TypedDict):
    image_id: str
    image_name: str
    mask_image_name: str
    category: str

def load_info_records(info_path: str) -> Generator[InfoRecord, None, None]:
    '''
    Load from the BG1K info records TXT. Each line should have at least two tab-separated values: image_name and category.
    '''
    path = Path(info_path)
    if not path.exists():
        raise FileNotFoundError(f'Info file not found: {path}')
    
    with path.open('r', encoding='utf-8') as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                print(f'Skipping malformed line {line_no}: {line!r}')
                continue

            image_name = parts[0].strip()
            category = parts[1].strip()
            mask_image_name = f"{Path(image_name).stem}_mask.png"
            image_id = Path(image_name).stem
            yield InfoRecord(
                image_id=image_id, image_name=image_name, category=category,
                mask_image_name=mask_image_name,
            )

def make_image_grid(
    images: list[Image.Image],
    rows: int,
    cols: int,
    gap: int = 8,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    assert len(images) == rows * cols, "rows * cols must match number of images"
    w, h = images[0].size
    grid_w = cols * w + (cols - 1) * gap
    grid_h = rows * h + (rows - 1) * gap
    grid = Image.new("RGB", (grid_w, grid_h), color=bg_color)

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x = c * (w + gap)
        y = r * (h + gap)
        grid.paste(img.convert("RGB"), (x, y))

    return grid