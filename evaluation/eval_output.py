import os
import argparse
import csv
import json
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"  # physical GPU 1 via CUDA_VISIBLE_DEVICES=1
    return "cpu"


def find_pairs(orig_dir: Path, gen_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return [(id, orig_path, gen_path)] for every matched pair."""
    pairs = []
    for gen_path in sorted(gen_dir.glob("*.png")):
        img_id = gen_path.stem
        orig_path = orig_dir / f"{img_id}.png"
        if not orig_path.exists():
            # try .jpg fallback
            orig_path = orig_dir / f"{img_id}.jpg"
        if orig_path.exists():
            pairs.append((img_id, orig_path, gen_path))
    return pairs


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# ── CLIP similarity ───────────────────────────────────────────────────────────

def clip_image_features(images: list[Image.Image], processor, model, device: str, batch_size: int = 32) -> torch.Tensor:
    """Return L2-normalised CLIP image embeddings, shape (N, D)."""
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            # Use vision_model + visual_projection for compatibility with transformers v5
            vis_out = model.vision_model(pixel_values=inputs["pixel_values"])
            feats = model.visual_projection(vis_out.pooler_output)
            feats = F.normalize(feats, dim=-1)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


def compute_clip_similarities(
    orig_imgs: list[Image.Image],
    gen_imgs: list[Image.Image],
    model_name: str,
    device: str,
) -> list[float]:
    print(f"\nLoading CLIP model '{model_name}' on {device}...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    print("Extracting CLIP features for original images...")
    orig_feats = clip_image_features(orig_imgs, processor, model, device)
    print("Extracting CLIP features for generated images...")
    gen_feats = clip_image_features(gen_imgs, processor, model, device)

    # Per-pair cosine similarity (embeddings already normalised → dot product)
    sims = (orig_feats * gen_feats).sum(dim=-1).tolist()
    return sims


# ── FID ───────────────────────────────────────────────────────────────────────

# FID InceptionV3 expects uint8 tensors in range [0, 255], shape (N, 3, H, W)
_to_tensor_uint8 = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),                      # float32 [0, 1]
    transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),
])


def compute_fid(orig_imgs: list[Image.Image], gen_imgs: list[Image.Image], device: str) -> float:
    fid_device = device
    print(f"\nComputing FID on {fid_device} (InceptionV3)...")

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(fid_device)

    def feed(images: list[Image.Image], real: bool, desc: str):
        batch_size = 16
        for i in tqdm(range(0, len(images), batch_size), desc=desc, leave=False):
            batch = images[i : i + batch_size]
            tensors = torch.stack([_to_tensor_uint8(img) for img in batch]).to(fid_device)
            fid.update(tensors, real=real)

    feed(orig_imgs, real=True,  desc="  FID orig")
    feed(gen_imgs,  real=False, desc="  FID gen ")
    return fid.compute().item()


# ── per-directory evaluation ──────────────────────────────────────────────────

def eval_one(
    orig_dir: Path,
    gen_dir: Path,
    clip_model_name: str,
    device: str,
    no_fid: bool,
    top_k: int,
    save_csv: str | None,
    save_json: str | None,
) -> dict:
    """Evaluate a single gen_dir against orig_dir. Returns summary dict."""
    pairs = find_pairs(orig_dir, gen_dir)
    if not pairs:
        print(f"  [warn] No matching pairs found in '{gen_dir}', skipping.")
        return {}

    print(f"\n  {len(pairs)} matched pairs")
    ids       = [p[0] for p in pairs]
    orig_imgs = [load_rgb(p[1]) for p in tqdm(pairs, desc="  Loading originals", leave=False)]
    gen_imgs  = [load_rgb(p[2]) for p in tqdm(pairs, desc="  Loading generated", leave=False)]

    sims = compute_clip_similarities(orig_imgs, gen_imgs, clip_model_name, device)
    mean_sim = sum(sims) / len(sims)
    min_sim  = min(sims)
    max_sim  = max(sims)

    fid_score = None
    if not no_fid:
        fid_score = compute_fid(orig_imgs, gen_imgs, device)

    # ── best / worst ──
    ranked = sorted(zip(ids, sims), key=lambda x: x[1])
    print(f"\n  Worst {top_k} (lowest CLIP):")
    for img_id, score in ranked[:top_k]:
        print(f"    {img_id:<10}  {score:.4f}")
    print(f"\n  Best {top_k} (highest CLIP):")
    for img_id, score in ranked[-top_k:][::-1]:
        print(f"    {img_id:<10}  {score:.4f}")

    if save_csv:
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "clip_similarity"])
            for img_id, score in zip(ids, sims):
                writer.writerow([img_id, f"{score:.6f}"])
        print(f"\n  Per-image CSV: {save_csv}")

    summary = {
        "pairs": len(pairs),
        "clip_similarity": {"mean": mean_sim, "min": min_sim, "max": max_sim},
        "fid": fid_score,
        "clip_model": clip_model_name,
    }

    if save_json:
        with open(save_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary JSON: {save_json}")

    return summary


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated images: CLIP similarity + FID")
    parser.add_argument("--orig-dir",    default="black_prod_output/sample_image",
                        help="Reference images directory (default: black_prod_output/sample_image)")
    parser.add_argument("--gen-dir",     default=None,
                        help="Single generated images directory to evaluate")
    parser.add_argument("--input",       default=None,
                        help="Parent directory whose immediate subdirectories are each evaluated")
    parser.add_argument("--clip-model",  default="openai/clip-vit-large-patch14",
                        help="HuggingFace CLIP model ID")
    parser.add_argument("--top-k",       type=int, default=5,
                        help="Show best and worst N pairs by CLIP similarity")
    parser.add_argument("--save-csv",    default=None, help="Save per-image results to CSV (single-dir mode only)")
    parser.add_argument("--save-json",   default=None, help="Save summary to JSON (single-dir mode only)")
    parser.add_argument("--no-fid",      action="store_true", help="Skip FID computation")
    args = parser.parse_args()

    if not args.input and not args.gen_dir:
        parser.error("Provide either --input or --gen-dir.")

    device = pick_device()
    print(f"Device: {device}")

    orig_dir = Path(args.orig_dir)

    # ── multi-directory mode ──────────────────────────────────────────────────
    if args.input:
        input_dir = Path(args.input)
        child_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
        if not child_dirs:
            print(f"No subdirectories found in '{input_dir}'.")
            return

        print(f"\nEvaluating {len(child_dirs)} subdirectories in '{input_dir}'")
        print(f"Reference : {orig_dir}\n")

        all_results: list[tuple[str, dict]] = []
        for child in child_dirs:
            print(f"\n{'─' * 60}")
            print(f"Run: {child.name}")
            print(f"{'─' * 60}")
            summary = eval_one(
                orig_dir, child, args.clip_model, device,
                args.no_fid, args.top_k, None, None,
            )
            if summary:
                all_results.append((child.name, summary))

        # ── comparison table ──
        print(f"\n{'=' * 70}")
        print("COMPARISON TABLE")
        print(f"{'=' * 70}")
        header = f"{'Directory':<35}  {'CLIP mean':>9}  {'CLIP min':>8}  {'CLIP max':>8}"
        if not args.no_fid:
            header += f"  {'FID':>8}"
        print(header)
        print("─" * len(header))
        for name, s in all_results:
            cs = s["clip_similarity"]
            row = f"{name:<35}  {cs['mean']:>9.4f}  {cs['min']:>8.4f}  {cs['max']:>8.4f}"
            if not args.no_fid:
                fid_val = s["fid"]
                row += f"  {fid_val:>8.2f}" if fid_val is not None else f"  {'N/A':>8}"
            print(row)
        print(f"{'=' * 70}")

        if args.save_json:
            out = {name: s for name, s in all_results}
            with open(args.save_json, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\nAll summaries saved to: {args.save_json}")
        return

    # ── single-directory mode ─────────────────────────────────────────────────
    gen_dir = Path(args.gen_dir)
    print(f"Reference : {orig_dir}")
    print(f"Generated : {gen_dir}")

    summary = eval_one(
        orig_dir, gen_dir, args.clip_model, device,
        args.no_fid, args.top_k, args.save_csv, args.save_json,
    )
    if not summary:
        return

    cs = summary["clip_similarity"]
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"  Pairs evaluated  : {summary['pairs']}")
    print(f"  CLIP similarity  : mean={cs['mean']:.4f}  min={cs['min']:.4f}  max={cs['max']:.4f}")
    if summary["fid"] is not None:
        print(f"  FID score        : {summary['fid']:.2f}  (lower is better)")
    print("=" * 50)


if __name__ == "__main__":
    main()
