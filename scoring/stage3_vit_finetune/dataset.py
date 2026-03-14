"""
Dataset for product image aesthetic scoring.

Handles two image sources:
  - BG60k:        10 zip shards at bg60k_zip_dir/bg60k_imgs_{N%10}.zip
  - Ecommerce118k: files on disk at ecom_img_dir/{category_id:02d}/{filename}
"""

import atexit
import io
import zipfile
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Process-local zip cache: populated lazily, safe under DataLoader multi-process
# (each worker has its own copy after fork).
_ZIP_CACHE: dict[str, zipfile.ZipFile] = {}


def _open_zip(path: Path) -> zipfile.ZipFile:
    key = str(path)
    if key not in _ZIP_CACHE:
        _ZIP_CACHE[key] = zipfile.ZipFile(path, "r")
    return _ZIP_CACHE[key]


@atexit.register
def _close_zips():
    for zf in _ZIP_CACHE.values():
        try:
            zf.close()
        except Exception:
            pass


class ProductImageDataset(Dataset):
    def __init__(self, csv_path, bg60k_zip_dir, ecom_img_dir, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.bg60k_zip_dir = Path(bg60k_zip_dir)
        self.ecom_img_dir = Path(ecom_img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _load_bg60k(self, filename: str) -> Image.Image:
        img_id = int(Path(filename).stem)
        shard = img_id % 10
        zip_path = self.bg60k_zip_dir / f"bg60k_imgs_{shard}.zip"
        inner = f"bg60k_imgs_{shard}/{img_id}.png"
        zf = _open_zip(zip_path)
        with zf.open(inner) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")

    def _load_ecom(self, filename: str, category_id) -> Image.Image:
        path = self.ecom_img_dir / str(int(category_id)).zfill(2) / filename
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if row["dataset"] == "bg60k":
            image = self._load_bg60k(row["filename"])
        else:
            image = self._load_ecom(row["filename"], row["category_id"])

        if self.transform:
            image = self.transform(image)

        score = torch.tensor(float(row["overall_score"]), dtype=torch.float32)
        return image, score
