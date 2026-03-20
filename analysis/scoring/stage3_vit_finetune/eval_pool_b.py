#!/usr/bin/env python3
"""
Evaluate the best DINOv3 checkpoint against Pool B human annotations.

Usage (from repo root):
    python scoring/stage3_vit_finetune/eval_pool_b.py
    python scoring/stage3_vit_finetune/eval_pool_b.py --checkpoint scoring/stage3_vit_finetune/checkpoints/dinov3_vitb16_best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from dataset import ProductImageDataset
# model class selected at runtime based on checkpoint pretrained arg

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent.parent.parent
DATA_DIR    = ROOT / "data"
SCORING_DIR = ROOT / "scoring"

BG60K_ZIP_DIR = DATA_DIR / "ICASSP2025-BG60k"
ECOM_IMG_DIR  = DATA_DIR / "Ecommerce_118K" / "train" / "train"
POOL_B_XLSX   = SCORING_DIR / "stage1_annotation" / "pool_b_300_images.xlsx"
DEFAULT_CKPT  = Path(__file__).parent / "checkpoints" / "dinov3_vitb16_best.pt"

# ── Transform (same as val_transform in train_dinov3.py / train_dinov2.py) ───

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class ScaleImage:
    def __init__(self, max_px=332):
        self.max_px = max_px
    def __call__(self, img):
        img.thumbnail((self.max_px, self.max_px), resample=Image.LANCZOS)
        return img


class PadToSquare:
    def __init__(self, fill=255):
        self.fill = fill
    def __call__(self, img):
        w, h = img.size
        s = max(w, h)
        pl, pt = (s - w) // 2, (s - h) // 2
        return ImageOps.expand(img, border=(pl, pt, s - w - pl, s - h - pt), fill=self.fill)


eval_transform = transforms.Compose([
    ScaleImage(332),
    PadToSquare(fill=255),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── Dataset wrapper for Pool B ────────────────────────────────────────────────


class PoolBDataset(Dataset):
    """Loads Pool B images; returns (image_tensor, human_score, pool_b_id)."""

    def __init__(self, xlsx_path, bg60k_zip_dir, ecom_img_dir, transform=None):
        df = pd.read_excel(xlsx_path)
        # Normalise column names to match ProductImageDataset conventions
        self.df = df.rename(columns={
            "Dataset":              "dataset",
            "Filename":             "filename",
            "Category ID":          "category_id",
            "[HUMAN] Overall Score":"human_score",
            "Pool B ID":            "pool_b_id",
        }).reset_index(drop=True)
        self.bg60k_zip_dir = Path(bg60k_zip_dir)
        self.ecom_img_dir  = Path(ecom_img_dir)
        self.transform     = transform

        # Re-use the zip-loading logic from ProductImageDataset
        self._inner = ProductImageDataset.__new__(ProductImageDataset)
        self._inner.bg60k_zip_dir = self.bg60k_zip_dir
        self._inner.ecom_img_dir  = self.ecom_img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dataset = row["dataset"].lower()
        if dataset == "bg60k":
            image = self._inner._load_bg60k(row["filename"])
        else:
            image = self._inner._load_ecom(row["filename"], row["category_id"])

        if self.transform:
            image = self.transform(image)

        score = torch.tensor(float(row["human_score"]), dtype=torch.float32)
        return image, score, row["pool_b_id"]


# ── Inference ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_targets, all_ids = [], [], []
    for images, scores, ids in tqdm(loader, desc="Pool B inference"):
        images = images.to(device)
        preds  = model(images).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(scores.numpy().tolist())
        all_ids.extend(ids)
    return np.array(all_preds), np.array(all_targets), all_ids


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args):
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})
    print(f"Checkpoint epoch : {ckpt.get('epoch', '?')}")
    print(f"Checkpoint val SRCC: {ckpt['metrics']['srcc']:.4f}  (on SEA-LION labels)")

    # Select model class based on which backbone was used during training
    pretrained_id = saved_args.get("pretrained", "")
    if "dinov2" in pretrained_id or pretrained_id == "":
        from model_dinov2 import DINOv2Scorer
    else:
        from model_dinov3 import DINOv2Scorer

    # Rebuild model with the same args used during training
    model = DINOv2Scorer(
        pretrained      = saved_args.get("pretrained",       "facebook/dinov2-base"),
        dropout         = saved_args.get("dropout",          0.1),
        freeze_backbone = saved_args.get("freeze_backbone",  False),
        unfreeze_last_n = saved_args.get("unfreeze_last_n",  4),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print("Model loaded.\n")

    # Dataset
    ds = PoolBDataset(POOL_B_XLSX, BG60K_ZIP_DIR, ECOM_IMG_DIR, eval_transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    print(f"Pool B: {len(ds)} images\n")

    # Inference
    preds, targets, ids = run_inference(model, loader, device)

    # Metrics
    srcc, _ = spearmanr(preds, targets)
    plcc, _ = pearsonr(preds, targets)
    mae     = float(np.mean(np.abs(preds - targets)))

    print("=" * 45)
    print("Pool B evaluation (vs human annotations)")
    print("=" * 45)
    print(f"  SRCC : {srcc:.4f}")
    print(f"  PLCC : {plcc:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  N    : {len(preds)}")
    print("=" * 45)

    # Save per-image results
    results_df = pd.DataFrame({
        "pool_b_id":    ids,
        "human_score":  targets,
        "model_pred":   preds,
        "error":        preds - targets,
    }).sort_values("pool_b_id")

    out_csv = Path(__file__).parent.parent / "stage4_evaluation" / f"{Path(args.checkpoint).stem}_pool_b_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nPer-image results saved → {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pool B evaluation for DINOv3 scorer")
    parser.add_argument("--checkpoint",   default=str(DEFAULT_CKPT),
                        help="Path to best.pt checkpoint")
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--num-workers",  type=int, default=4)
    main(parser.parse_args())
