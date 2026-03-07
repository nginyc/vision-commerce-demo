#!/usr/bin/env python3
"""
VLM-based e-commerce background quality scorer.

Phase 1 — Validation run (400 images):
  Scores 200 random images from each dataset using Qwen2-VL-2B via mlx-vlm.
  Results saved to scoring/vlm_scores_validation.csv for manual inspection.

Usage:
  pip install mlx-vlm tqdm pandas pillow
  python scoring/score_images.py
  python scoring/score_images.py --full   # score all images (slow, run overnight)
"""

import argparse
import io
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────

SEED      = 42
N_SAMPLE  = 200          # images per dataset in validation mode
MODEL_ID  = "mlx-community/Qwen2-VL-2B-Instruct-4bit"

ROOT       = Path(__file__).parent.parent          # /Users/XiaoJun/avm
DATA_DIR   = ROOT / "data"

BG60K_DIR  = DATA_DIR / "ICASSP2025-BG60k"
BG60K_ZIPS = [BG60K_DIR / f"bg60k_imgs_{i}.zip" for i in range(10)]
BG60K_INFO = BG60K_DIR / "bg60k_info.txt"

ECOM_DIR   = DATA_DIR / "Ecommerce_118K" / "train" / "train"
ECOM_CSV   = DATA_DIR / "Ecommerce_118K" / "train.csv"

ECOM_CATEGORY_NAMES = {
     0: "women dresses",          1: "women abaya",
     2: "clothing and t-shirt",   3: "women sweaters and hoodies",
     4: "women jeans",            5: "rings",
     6: "earring",                7: "cap",
     8: "wallet",                 9: "travel",
    10: "phone cover",           11: "phone",
    12: "clock",                 13: "feeding and nursing",
    14: "rice cooker",           15: "coffee",
    16: "shoes",                 17: "heels",
    18: "home and living",       19: "data storage",
    20: "chair",                 21: "sports",
    22: "sports and outdoors",   23: "gloves",
    24: "watches",               25: "belt",
    26: "audio",                 27: "toy cars",
    28: "jacket",                29: "men fashion",
    30: "sneakers",              31: "snacks and sweets",
    32: "personal protective equipment",
    33: "disinfectants and sanitizers",
    34: "makeup",                35: "perfume",
    36: "household cleaning supplies",
    37: "laptop",
    38: "food storage and tableware",
    39: "home decor",
    40: "bathroom accessories",  41: "couch and cushions",
}

SCORING_PROMPT = """\
You are an expert e-commerce image quality assessor. Evaluate this product image across four dimensions.

IMPORTANT: Focus on the OVERALL image, not just the background.

--- DIMENSION 1: Background Cleanliness (1-10) ---
Does the background distract from the product?
Consider: Is there clutter, random objects, retail store shelving, or an
irrelevant environment visible?
10 = completely clean and intentional background (plain, studio, or styled lifestyle)
5  = mildly distracting but acceptable
1  = severely cluttered, messy retail/home environment, or irrelevant setting

--- DIMENSION 2: Text & Watermark Penalty (1-10) ---
Is there any seller-added text, watermarks, promotional banners, or price
tags overlaid on the image (NOT counting text that is physically printed on
the product packaging itself)?
10 = absolutely no seller text overlays
5  = minor watermark or small URL
1  = large promotional text, price banners, or multiple text overlays

--- DIMENSION 3: Product Prominence (1-10) ---
Is the product clearly the hero of the image?
Consider: Is the product centred or well-composed? Is it fully visible (not
cut off at edges)? Does it occupy an appropriate portion of the frame
(roughly 30-70% of image area)? Is it in sharp focus?
10 = product is perfectly framed, fully visible, sharp, and clearly the subject
5  = product is visible but poorly framed or partially obscured
1  = product is hard to find, cut off, blurry, or buried among other items

--- DIMENSION 4: Commercial Appeal (1-10) ---
Would this image make a customer want to click and buy the product?
Consider: overall lighting quality, colour harmony, whether the image tells
a story about the product, and whether it looks professional.
Note: a clean plain background CAN score highly here. A lifestyle scene
CAN also score highly if well-executed. Judge on overall feel, not background style.
10 = immediately compelling, professional, purchase-inspiring
5  = acceptable but uninspiring
1  = off-putting, amateurish, would reduce purchase confidence

--- ROUTING DECISION ---
Based on the above, does this image need background replacement?
Answer YES if ANY of the following are true:
- Background cleanliness score <= 5
- The background is a cluttered real-world environment (retail store, messy room)
- The background contains unrelated objects that distract from the product
Answer NO if:
- The background is clean (plain, studio, or well-executed lifestyle)
- The image's main problem is NOT the background (e.g. text overlay on clean background)

--- OUTPUT ---
Respond ONLY with valid JSON, no extra text, no markdown:
{
  "background_description": "<one sentence describing what you see behind the product>",
  "background_class": "plain|lifestyle|cluttered",
  "background_cleanliness": <1-10>,
  "text_watermark_score": <1-10>,
  "product_prominence": <1-10>,
  "commercial_appeal": <1-10>,
  "overall_score": <weighted average, you calculate: (cleanliness*0.25 + text*0.25 + prominence*0.25 + appeal*0.25)>,
  "needs_background_replacement": <true|false>,
  "primary_failure_reason": "<if overall_score < 6, state the single biggest problem in one sentence, else null>"
}
"""

# ── Model ────────────────────────────────────────────────────────────────────

def load_model():
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    print(f"Loading {MODEL_ID} (downloads on first run) …")
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)
    print("Model ready.")
    return model, processor, config


def score_pil_image(model, processor, config, img: Image.Image) -> dict:
    """Score a single PIL image. Returns {"score": int, "bg_class": str, "reason": str}."""
    from mlx_vlm import generate, apply_chat_template

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.convert("RGB").save(tmp.name, "JPEG", quality=92)
        tmp_path = tmp.name

    try:
        # Embed the image token into the prompt — required for the model to see the image
        formatted_prompt = apply_chat_template(
            processor, config, SCORING_PROMPT, num_images=1
        )
        raw = generate(
            model, processor,
            formatted_prompt,
            image=tmp_path,
            max_tokens=400,
            temp=0.0,
            verbose=False,
        )
        # GenerationResult object — extract text string
        if hasattr(raw, "text"):
            raw = raw.text
        else:
            raw = str(raw)
        # Extract JSON from response (model may prefix with text)
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return {"score": -1, "reason": f"no_json: {raw[:120]}"}
        parsed = json.loads(raw[start:end])
        return {
            "score":    round(float(parsed["overall_score"]), 2),
            "bg_class": parsed.get("background_class", ""),
            "reason":   parsed.get("primary_failure_reason") or parsed.get("background_description", ""),
        }
    except Exception as e:
        return {"score": -1, "reason": f"error: {e}"}
    finally:
        Path(tmp_path).unlink(missing_ok=True)

# ── Data loading ─────────────────────────────────────────────────────────────

def build_bg60k_zip_index():
    """Map basename (e.g. '0.png') → (zip_path, internal_name) for all bg60k zips."""
    index = {}
    for zp in BG60K_ZIPS:
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.endswith(".png"):
                    index[name.split("/")[-1]] = (zp, name)
    return index


def load_bg60k_sample(n: int, seed: int, exclude: set = None):
    info_df = pd.read_csv(BG60K_INFO, sep="\t", header=None, names=["filename", "category"])
    if exclude:
        info_df = info_df[~info_df["filename"].isin(exclude)]
    sample_df = info_df.sample(min(n, len(info_df)), random_state=seed)
    zip_index = build_bg60k_zip_index()

    rows = []
    for _, row in sample_df.iterrows():
        fname  = row["filename"]
        entry  = zip_index.get(fname)
        if entry is None:
            continue
        zp, internal = entry
        with zipfile.ZipFile(zp) as z:
            data = z.read(internal)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        rows.append({
            "dataset":  "bg60k",
            "filename": fname,
            "category": row["category"],
            "image":    img,
        })
    return rows


def load_ecommerce_sample(n: int, seed: int):
    csv_df    = pd.read_csv(ECOM_CSV)
    sample_df = csv_df.sample(n, random_state=seed)

    rows = []
    for _, row in sample_df.iterrows():
        cat_id   = int(row["category"])
        cat_name = ECOM_CATEGORY_NAMES.get(cat_id, str(cat_id))
        path     = ECOM_DIR / str(cat_id).zfill(2) / row["filename"]
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        rows.append({
            "dataset":  "ecommerce118k",
            "filename": row["filename"],
            "category": cat_name,
            "image":    img,
        })
    return rows


def load_all_bg60k():
    info_df   = pd.read_csv(BG60K_INFO, sep="\t", header=None, names=["filename", "category"])
    zip_index = build_bg60k_zip_index()

    rows = []
    for _, row in info_df.iterrows():
        fname  = row["filename"]
        entry  = zip_index.get(fname)
        if entry is None:
            continue
        zp, internal = entry
        with zipfile.ZipFile(zp) as z:
            data = z.read(internal)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        rows.append({
            "dataset":  "bg60k",
            "filename": fname,
            "category": row["category"],
            "image":    img,
        })
    return rows


def load_all_ecommerce():
    csv_df = pd.read_csv(ECOM_CSV)
    rows   = []
    for _, row in csv_df.iterrows():
        cat_id   = int(row["category"])
        cat_name = ECOM_CATEGORY_NAMES.get(cat_id, str(cat_id))
        path     = ECOM_DIR / str(cat_id).zfill(2) / row["filename"]
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        rows.append({
            "dataset":  "ecommerce118k",
            "filename": row["filename"],
            "category": cat_name,
            "image":    img,
        })
    return rows

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Score all images instead of validation sample")
    parser.add_argument("--bg60k", type=int, metavar="N",
                        help="Score N random bg60k images and append to validation CSV")
    args = parser.parse_args()

    out_csv = Path(__file__).parent / (
        "vlm_scores_full.csv" if args.full else "vlm_scores_validation.csv"
    )

    # Load existing results (always, so we can append / skip duplicates)
    done = set()
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        done = set(zip(existing["dataset"], existing["filename"]))
        print(f"Existing rows: {len(done)} — will append new results.")
        results = existing.to_dict("records")
    else:
        results = []

    # Load images
    if args.bg60k:
        done_bg60k = {fname for ds, fname in done if ds == "bg60k"}
        print(f"Sampling {args.bg60k} random images from bg60k (excluding {len(done_bg60k)} already scored) …")
        all_rows = load_bg60k_sample(args.bg60k, SEED, exclude=done_bg60k)
    elif args.full:
        print("Loading all bg60k images …")
        bg_rows = load_all_bg60k()
        print("Loading all ecommerce118k images …")
        ec_rows = load_all_ecommerce()
        all_rows = bg_rows + ec_rows
    else:
        print(f"Sampling {N_SAMPLE} images from each dataset …")
        bg_rows = load_bg60k_sample(N_SAMPLE, SEED)
        ec_rows = load_ecommerce_sample(N_SAMPLE, SEED)
        all_rows = bg_rows + ec_rows

    pending  = [r for r in all_rows if (r["dataset"], r["filename"]) not in done]
    print(f"Images to score: {len(pending)}")

    model, processor, config = load_model()

    for item in tqdm(pending, desc="Scoring"):
        scored = score_pil_image(model, processor, config, item["image"])
        results.append({
            "dataset":  item["dataset"],
            "filename": item["filename"],
            "category": item["category"],
            "score":    scored["score"],
            "bg_class": scored.get("bg_class", ""),
            "reason":   scored["reason"],
        })
        # Save after every image so we can resume if interrupted
        pd.DataFrame(results).to_csv(out_csv, index=False)

    # Summary
    df = pd.DataFrame(results)
    print("\n=== Score distribution by dataset ===")
    print(df.groupby("dataset")["score"].describe().round(2))

    valid = df[df["score"] > 0]
    print("\n=== Score bins ===")
    bins   = [0, 3, 7, 10]
    labels = ["low (1-3)", "medium (4-7)", "high (8-10)"]
    valid  = valid.copy()
    valid["band"] = pd.cut(valid["score"], bins=bins, labels=labels)
    print(valid.groupby(["dataset", "band"], observed=True).size().unstack(fill_value=0))

    errors = (df["score"] == -1).sum()
    if errors:
        print(f"\nParse errors: {errors} — check vlm_parse_errors in CSV")

    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
