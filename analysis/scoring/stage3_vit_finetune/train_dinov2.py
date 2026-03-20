#!/usr/bin/env python3
"""
Fine-tune DINOv2 ViT-B/14 for e-commerce image aesthetic scoring.

Backbone: facebook/dinov2-base (HuggingFace) — ViT-B/14, 86M params.
Trains on scoring/stage2_validation_and_split/train_4000.csv and val_1000.csv.
Metrics: SRCC, PLCC, MAE.  Best checkpoint saved by val SRCC.

Checkpoint saved to: checkpoints/dinov2_vitb14_best.pt
Does NOT overwrite checkpoints/dinov3_vitb16_best.pt.

Usage (from repo root):
    python scoring/stage3_vit_finetune/train_dinov2.py
    python scoring/stage3_vit_finetune/train_dinov2.py --freeze-backbone --unfreeze-last-n 4
    python scoring/stage3_vit_finetune/train_dinov2.py --epochs 30 --lr 2e-4 --batch-size 32
"""

import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from scipy.stats import pearsonr, spearmanr
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Allow running from repo root or from this file's directory
sys.path.insert(0, str(Path(__file__).parent))
from dataset import ProductImageDataset
from model_dinov2 import DINOv2Scorer


# ── Custom transforms ─────────────────────────────────────────────────────────

class ScaleImage:
    """Downscale so longest side ≤ max_px. Never upscales."""
    def __init__(self, max_px=332):
        self.max_px = max_px

    def __call__(self, img):
        img.thumbnail((self.max_px, self.max_px), resample=Image.LANCZOS)
        return img


class PadToSquare:
    """Pad to square with a solid fill colour. Preserves product aspect ratio."""
    def __init__(self, fill=255):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        s = max(w, h)
        pl, pt = (s - w) // 2, (s - h) // 2
        return ImageOps.expand(img, border=(pl, pt, s - w - pl, s - h - pt), fill=self.fill)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
SCORING_DIR = ROOT / "scoring"

BG60K_ZIP_DIR = DATA_DIR / "ICASSP2025-BG60k"
ECOM_IMG_DIR = DATA_DIR / "Ecommerce_118K" / "train" / "train"

TRAIN_CSV = SCORING_DIR / "stage2_validation_and_split" / "train_4000.csv"
VAL_CSV = SCORING_DIR / "stage2_validation_and_split" / "val_1000.csv"

CKPT_DIR = Path(__file__).parent / "checkpoints"

# ── Transforms ────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    ScaleImage(332),
    PadToSquare(fill=255),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    ScaleImage(332),
    PadToSquare(fill=255),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── Metrics ───────────────────────────────────────────────────────────────────


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    srcc, _ = spearmanr(preds, targets)
    plcc, _ = pearsonr(preds, targets)
    mae = float(np.mean(np.abs(preds - targets)))
    return {"srcc": float(srcc), "plcc": float(plcc), "mae": mae}


# ── Train / eval loops ────────────────────────────────────────────────────────


def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for images, scores in tqdm(loader, desc="  train", leave=False):
        images, scores = images.to(device), scores.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), scores)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for images, scores in tqdm(loader, desc="  val  ", leave=False):
        images, scores = images.to(device), scores.to(device)
        preds = model(images)
        total_loss += criterion(preds, scores).item() * len(images)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(scores.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


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

    # Data
    train_ds = ProductImageDataset(TRAIN_CSV, BG60K_ZIP_DIR, ECOM_IMG_DIR, train_transform)
    val_ds = ProductImageDataset(VAL_CSV, BG60K_ZIP_DIR, ECOM_IMG_DIR, val_transform)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    # Model
    model = DINOv2Scorer(
        pretrained=args.pretrained, dropout=args.dropout,
        freeze_backbone=args.freeze_backbone, unfreeze_last_n=args.unfreeze_last_n,
    ).to(device)
    print(f"Backbone trainable params: {sum(p.numel() for p in model.backbone.parameters() if p.requires_grad):,}")

    # Differential LR: backbone at 10x lower than regression head
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = AdamW([
        {"params": backbone_params,          "lr": args.lr * 0.1},
        {"params": model.head.parameters(),  "lr": args.lr},
    ], weight_decay=args.weight_decay)

    criterion = nn.SmoothL1Loss()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    CKPT_DIR.mkdir(exist_ok=True)
    best_ckpt = CKPT_DIR / "dinov2_vitb14_best.pt"

    mlflow.set_experiment("dinov2_aesthetic_scorer")
    with mlflow.start_run(run_name=f"dinov2-ecom_lr{args.lr}_bs{args.batch_size}"):
        mlflow.log_params(vars(args))

        best_srcc = -1.0
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()

            print(
                f"Epoch {epoch:3d}/{args.epochs}"
                f"  train_loss={train_loss:.4f}"
                f"  val_loss={val['loss']:.4f}"
                f"  SRCC={val['srcc']:.4f}"
                f"  PLCC={val['plcc']:.4f}"
                f"  MAE={val['mae']:.4f}"
            )
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val["loss"],
                "val_srcc":   val["srcc"],
                "val_plcc":   val["plcc"],
                "val_mae":    val["mae"],
            }, step=epoch)

            if val["srcc"] > best_srcc:
                best_srcc = val["srcc"]
                patience_counter = 0
                torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                            "metrics": val, "args": vars(args)}, best_ckpt)
                print(f"  -> best SRCC={best_srcc:.4f}  checkpoint saved")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping (patience={args.patience}) at epoch {epoch}")
                    break

        print(f"\nBest val SRCC: {best_srcc:.4f}")
        mlflow.log_metric("best_val_srcc", best_srcc)
        print(f"Checkpoint: {best_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINOv2 ViT-B/14 aesthetic scorer fine-tuning")
    parser.add_argument("--pretrained",       default="facebook/dinov2-base",
                        help="HuggingFace model ID (default: facebook/dinov2-base)")
    parser.add_argument("--epochs",           type=int,   default=30)
    parser.add_argument("--batch-size",       type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=2e-4)
    parser.add_argument("--weight-decay",     type=float, default=1e-4)
    parser.add_argument("--dropout",          type=float, default=0.1)
    parser.add_argument("--patience",         type=int,   default=7)
    parser.add_argument("--num-workers",      type=int,   default=4,
                        help="Workers for DataLoader. Set 0 if zip loading causes issues.")
    parser.add_argument("--freeze-backbone",  action="store_true", default=False,
                        help="Freeze backbone; combine with --unfreeze-last-n for partial fine-tuning")
    parser.add_argument("--unfreeze-last-n",  type=int,   default=4,
                        help="When --freeze-backbone: unfreeze last N transformer blocks (default 4)")
    main(parser.parse_args())
