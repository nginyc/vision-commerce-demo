#!/usr/bin/env python3
"""
SEA-LION (Gemma-SEA-LION-v4-27B-IT) label generator for ViT fine-tuning.

Samples ~2600 images from ICASSP2025-BG60k evenly across all 42 mapped categories
(ceil(2600/42) = 62 per category; categories with fewer images take all available).

Category IDs follow the same shopee mapping as ecommerce118k, except:
  - ID 1 (women abaya) is absent (China dataset, no abaya products)
  - IDs 42 (laptop) and 43 (food) are extra categories present in this dataset

Images are spread across 10 zip archives (bg60k_imgs_0.zip … bg60k_imgs_9.zip).
Image N.png lives in bg60k_imgs_{N % 10}.zip under path bg60k_imgs_{N % 10}/{N}.png.
The script reads images directly from the zips — no extraction needed.

Output: scoring/vlm_scores_sealion_bg60k2600_labels.csv

Usage:
  pip install openai python-dotenv tqdm pandas pillow
  python scoring/score_sealion_bg60k2600_labels.py
  python scoring/score_sealion_bg60k2600_labels.py --n-total 2688 --seed 42
"""

import argparse
import base64
import io
import json
import math
import os
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID = "aisingapore/Gemma-SEA-LION-v4-27B-IT"
BASE_URL  = "https://api.sea-lion.ai/v1"

ROOT    = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
BG60K_DIR = DATA_DIR / "ICASSP2025-BG60k"
BG60K_MAPPING_CSV = BG60K_DIR / "60k_shopee_mapping_corrected.csv"

# Shopee category names shared with ecommerce118k (ID 1 absent in this dataset).
# IDs 42 and 43 are extra categories from the BG60k mapping.
BG60K_CATEGORY_NAMES = {
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
    42: "laptop",                43: "food",
}

# ── Prompt ────────────────────────────────────────────────────────────────────

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

# ── Model ─────────────────────────────────────────────────────────────────────

def load_client():
    from openai import OpenAI
    api_key = os.getenv("SEALION_API_KEY")
    if not api_key:
        raise ValueError("SEALION_API_KEY not found in .env")
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    print(f"SEA-LION client ready. Model: {MODEL_ID}")
    return client


def score_pil_image(client, img: Image.Image) -> dict:
    """Score a single PIL image. Returns all four dimension scores + overall."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SCORING_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                        }},
                    ],
                }
            ],
            max_tokens=400,
            temperature=0,
        )
        raw = response.choices[0].message.content
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return {"overall_score": -1, "reason": f"no_json: {raw[:120]}"}
        parsed = json.loads(raw[start:end])

        overall = round(float(parsed["overall_score"]), 2)
        if not (1 <= overall <= 10):
            return {"overall_score": -1, "reason": f"out_of_range: {overall}"}

        return {
            "overall_score":          overall,
            "background_cleanliness": parsed.get("background_cleanliness", -1),
            "text_watermark_score":   parsed.get("text_watermark_score", -1),
            "product_prominence":     parsed.get("product_prominence", -1),
            "commercial_appeal":      parsed.get("commercial_appeal", -1),
            "bg_class":               parsed.get("background_class", ""),
            "needs_bg_replacement":   parsed.get("needs_background_replacement", None),
            "reason":                 parsed.get("primary_failure_reason") or parsed.get("background_description", ""),
        }
    except Exception as e:
        return {"overall_score": -1, "reason": f"error: {e}"}

# ── Zip image loader ──────────────────────────────────────────────────────────

def _open_zip_handles() -> dict[int, zipfile.ZipFile]:
    """Open all 10 bg60k zip archives. Keys are 0–9."""
    handles = {}
    for i in range(10):
        zpath = BG60K_DIR / f"bg60k_imgs_{i}.zip"
        if not zpath.exists():
            raise FileNotFoundError(f"Missing zip: {zpath}")
        handles[i] = zipfile.ZipFile(zpath, "r")
    return handles


def _load_image_from_zips(filename: str, zip_handles: dict) -> Image.Image | None:
    """
    Load image N.png from the appropriate zip (N % 10).
    Returns a PIL Image or None if not found.
    """
    stem = Path(filename).stem        # e.g. "42" from "42.png"
    try:
        n = int(stem)
    except ValueError:
        return None
    idx = n % 10
    inner_path = f"bg60k_imgs_{idx}/{filename}"
    zf = zip_handles.get(idx)
    if zf is None:
        return None
    try:
        data = zf.read(inner_path)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except KeyError:
        return None

# ── Sampling ──────────────────────────────────────────────────────────────────

def build_stratified_sample(n_total: int, seed: int) -> pd.DataFrame:
    """
    Sample evenly across all pred_category_id groups in the mapping CSV.
    n_per_cat = ceil(n_total / n_cats). Groups with fewer images take all available.
    ID 1 (women abaya) is absent from this dataset — that is expected.
    """
    mapping = pd.read_csv(BG60K_MAPPING_CSV)
    n_cats    = mapping["pred_category_id"].nunique()
    n_per_cat = math.ceil(n_total / n_cats)

    sampled_parts = []
    for cat_id, group in mapping.groupby("pred_category_id"):
        take = min(n_per_cat, len(group))
        sampled_parts.append(group.sample(take, random_state=seed))

    sample = pd.concat(sampled_parts).reset_index(drop=True)
    print(f"Stratified sample: {len(sample)} images across {n_cats} categories "
          f"({n_per_cat} per category target)")
    return sample


def load_images_from_sample(sample_df: pd.DataFrame,
                             zip_handles: dict) -> list:
    rows = []
    missing = 0
    for _, row in sample_df.iterrows():
        cat_id   = int(row["pred_category_id"])
        cat_name = BG60K_CATEGORY_NAMES.get(cat_id, str(cat_id))
        filename = row["image"]
        img = _load_image_from_zips(filename, zip_handles)
        if img is None:
            missing += 1
            continue
        rows.append({
            "dataset":     "bg60k",
            "filename":    filename,
            "description": row.get("description", ""),
            "category_id": cat_id,
            "category":    cat_name,
            "image":       img,
        })
    if missing:
        print(f"Warning: {missing} image files not found in zips — skipped.")
    return rows

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-total", type=int, default=2600,
                        help="Total images to score across all categories (default: 2600)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    args = parser.parse_args()

    out_csv = Path(__file__).parent / "vlm_scores_sealion_bg60k2600_labels.csv"

    # Resume from checkpoint
    done = set()
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        done = set(existing["filename"])
        print(f"Resuming — {len(done)} images already scored.")
        results = existing.to_dict("records")
    else:
        results = []

    sample_df = build_stratified_sample(args.n_total, args.seed)
    pending_df = sample_df[~sample_df["image"].isin(done)]
    print(f"Total sampled: {len(sample_df)}  |  Already done: {len(done)}  |  To score: {len(pending_df)}")

    zip_handles = _open_zip_handles()
    try:
        all_rows  = load_images_from_sample(pending_df, zip_handles)
        print(f"Images loaded from zips: {len(all_rows)}")

        client = load_client()

        for item in tqdm(all_rows, desc="Scoring"):
            scored = score_pil_image(client, item["image"])
            results.append({
                "dataset":                item["dataset"],
                "filename":               item["filename"],
                "description":            item["description"],
                "category_id":            item["category_id"],
                "category":               item["category"],
                "overall_score":          scored.get("overall_score", -1),
                "background_cleanliness": scored.get("background_cleanliness", -1),
                "text_watermark_score":   scored.get("text_watermark_score", -1),
                "product_prominence":     scored.get("product_prominence", -1),
                "commercial_appeal":      scored.get("commercial_appeal", -1),
                "bg_class":               scored.get("bg_class", ""),
                "needs_bg_replacement":   scored.get("needs_bg_replacement", ""),
                "reason":                 scored.get("reason", ""),
            })
            pd.DataFrame(results).to_csv(out_csv, index=False)
    finally:
        for zf in zip_handles.values():
            zf.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    df    = pd.DataFrame(results)
    valid = df[df["overall_score"] > 0]
    errors = (df["overall_score"] == -1).sum()

    print(f"\n=== Scored: {len(valid)} valid  |  Errors: {errors} ===")

    print("\n=== Overall score distribution ===")
    print(valid["overall_score"].describe().round(2))

    print("\n=== Score bands ===")
    bins   = [0, 3, 7, 10]
    labels = ["low (1-3)", "mid (4-7)", "high (8-10)"]
    valid  = valid.copy()
    valid["band"] = pd.cut(valid["overall_score"], bins=bins, labels=labels)
    print(valid["band"].value_counts().sort_index())

    print("\n=== Images per category (top 10 by count) ===")
    print(valid.groupby("category")["overall_score"]
               .agg(["count", "mean"])
               .rename(columns={"count": "n", "mean": "mean_score"})
               .round(2)
               .sort_values("n", ascending=False)
               .head(10))

    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
