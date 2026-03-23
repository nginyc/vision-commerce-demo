#!/usr/bin/env python3
"""
Batch background replacement using Flux.1-Fill-dev (diffusers / CUDA) for
black_prod_output samples.

Reads sample_cases_50.csv to get per-image class labels, then processes every
image/mask pair in black_prod_output/sample_image + black_prod_output/sample_mask.

Mask format: grayscale L-mode, white = background (fill), black = product (keep).
Already in Flux Fill convention — no inversion needed.

Usage:
    python batch_infer_black_prod.py \
        [--data-dir  black_prod_output] \
        [--output-dir black_prod_inpainted] \
        [--steps 28] [--guidance 30] [--seed 42] [--dry-run]

    # Reduce VRAM with bfloat16 (default) or enable sequential CPU offload:
        [--dtype bf16|fp16|fp32] [--offload]

Already-completed images are skipped automatically (resumable).
"""

import argparse
import csv
import os
import sys

# Parse --gpu early so CUDA_VISIBLE_DEVICES is set before torch initialises CUDA.
_gpu_idx = "1"
for _i, _arg in enumerate(sys.argv):
    if _arg in ("--gpu", "-g") and _i + 1 < len(sys.argv):
        _gpu_idx = sys.argv[_i + 1]
    elif _arg.startswith("--gpu="):
        _gpu_idx = _arg.split("=", 1)[1]
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_idx

from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from prompts import get_prompt


# ── mask / image helpers ──────────────────────────────────────────────────────

def load_image_and_mask(image_path: str, mask_path: str, invert: bool = False, multiple: int = 16) -> tuple[Image.Image, Image.Image]:
    """Load image (RGB) and mask (Flux convention: white=fill, black=keep).
    Pass invert=True if the mask has the opposite convention (white=product, black=background).
    Resize both to the nearest multiple of `multiple` (required by VAE)."""
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    w, h = img.size
    new_w = (w // multiple) * multiple or multiple
    new_h = (h // multiple) * multiple or multiple
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

    if invert:
        import numpy as np
        mask = Image.fromarray((255 - np.array(mask)).astype(np.uint8), mode="L")

    return img, mask.convert("RGB")


# ── data loading ──────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> dict[str, str]:
    """Return {filename_stem: new_label} from sample_cases_50.csv."""
    mapping: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["file"].strip()
            label = row["new_label"].strip()
            stem = Path(fname).stem
            mapping[stem] = label
    return mapping


def collect_pairs(data_dir: str) -> list[tuple[str, str, str]]:
    """Return list of (img_id, image_path, mask_path) sorted by numeric id."""
    imgs_dir = os.path.join(data_dir, "sample_image")
    masks_dir = os.path.join(data_dir, "sample_mask")
    pairs = []
    for fname in os.listdir(imgs_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_id = Path(fname).stem
        mask_fname = f"{img_id}_mask.png"
        mask_path = os.path.join(masks_dir, mask_fname)
        if os.path.exists(mask_path):
            pairs.append((img_id, os.path.join(imgs_dir, fname), mask_path))
        else:
            print(f"[warn] no mask found for {fname}, skipping")
    pairs.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    return pairs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch Flux.1-Fill-dev background replacement (CUDA/diffusers)"
    )
    parser.add_argument("--data-dir",   default="black_prod_output",
                        help="Directory with sample_image/, sample_mask/, sample_cases_50.csv (default: black_prod_output)")
    parser.add_argument("--output-dir", default="black_prod_inpainted",
                        help="Output directory (default: black_prod_inpainted)")
    parser.add_argument("--csv",        default=None,
                        help="Path to CSV file (default: <data-dir>/sample_cases_50.csv)")
    parser.add_argument("--model",      default="black-forest-labs/FLUX.1-Fill-dev",
                        help="HuggingFace model ID (default: black-forest-labs/FLUX.1-Fill-dev)")
    parser.add_argument("--steps",    type=int,   default=28,   help="Inference steps (default: 28)")
    parser.add_argument("--guidance", type=float, default=30.0, help="Guidance scale (default: 30)")
    parser.add_argument("--seed",     type=int,   default=42,   help="RNG seed (default: 42)")
    parser.add_argument("--dtype",    default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="Model dtype (default: bf16)")
    parser.add_argument("--offload",  action="store_true",
                        help="Enable sequential CPU offload to reduce VRAM usage")
    parser.add_argument("--invert-mask", action="store_true",
                        help="Invert mask before passing to Flux (use if white=product, black=background)")
    parser.add_argument("--gpu", "-g", type=int, default=1,
                        help="Physical GPU index to use (default: 1)")
    parser.add_argument("--lora-path", default=None,
                        help="Path to LoRA checkpoint directory or .safetensors file")
    parser.add_argument("--append-prompt", default=None,
                        help="String appended to every prompt (e.g. dreambooth trigger token)")
    parser.add_argument("--prompt", default=None,
                        help="Use this fixed prompt for all images instead of category-based prompts")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print what would be processed without running inference")
    args = parser.parse_args()

    csv_path = args.csv or os.path.join(args.data_dir, "sample_cases_50.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    label_map = load_csv(csv_path)
    pairs = collect_pairs(args.data_dir)

    if not pairs:
        print("No image/mask pairs found. Check --data-dir path.")
        return

    if args.dry_run:
        for img_id, img_path, mask_path in pairs:
            label = label_map.get(img_id, "unknown")
            prompt = args.prompt if args.prompt else get_prompt(label)
            if args.append_prompt:
                prompt = f"{args.append_prompt} {prompt}"
            out = os.path.join(args.output_dir, f"{img_id}.png")
            print(f"[{img_id}] label={label!r}  in_csv={img_id in label_map}")
            print(f"         prompt={prompt!r}")
            print(f"         output={out}")
        return

    # ── load pipeline ──────────────────────────────────────────────────────
    from diffusers import FluxFillPipeline

    device = "cuda:0"  # index 0 within the restricted view (physical GPU 1)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading {args.model} (dtype={args.dtype}, device=GPU{args.gpu})...")
    pipe = FluxFillPipeline.from_pretrained(args.model, torch_dtype=dtype)

    if args.offload:
        pipe.enable_sequential_cpu_offload(gpu_id=0)
        print("Sequential CPU offload enabled (GPU 1).")
    else:
        pipe = pipe.to(device)

    if args.lora_path:
        print(f"Loading LoRA weights from {args.lora_path}...")
        pipe.load_lora_weights(args.lora_path)

    generator = torch.Generator(device).manual_seed(args.seed)

    # ── inference loop ─────────────────────────────────────────────────────
    for img_id, img_path, mask_path in tqdm(pairs, desc="Generating"):
        out_path = os.path.join(args.output_dir, f"{img_id}.png")
        if os.path.exists(out_path):
            tqdm.write(f"[skip] {img_id} already done")
            continue

        label = label_map.get(img_id, "unknown")
        if args.prompt:
            prompt = args.prompt
        else:
            if label == "unknown":
                tqdm.write(f"[warn] {img_id} not in CSV, using default prompt")
            prompt = get_prompt(label)
        if args.append_prompt:
            prompt = f"{args.append_prompt} {prompt}"

        image, fill_mask = load_image_and_mask(img_path, mask_path, invert=args.invert_mask)
        width, height = image.size

        try:
            result = pipe(
                prompt=prompt,
                image=image,
                mask_image=fill_mask,
                height=height,
                width=width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generator,
            ).images[0]
            result.save(out_path)
            tqdm.write(f"[done] {img_id} ({label!r}) → {out_path}")
        except Exception as exc:
            tqdm.write(f"[error] {img_id}: {exc}")


if __name__ == "__main__":
    main()
