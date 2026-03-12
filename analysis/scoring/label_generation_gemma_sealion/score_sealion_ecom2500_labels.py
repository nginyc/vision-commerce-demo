#!/usr/bin/env python3
"""
SEA-LION (Gemma-SEA-LION-v4-27B-IT) label generator for ViT fine-tuning.

Samples ~2500 images from ecommerce118k evenly across all 42 categories
(ceil(2500/42) = 60 per category; categories with fewer images take all available).

All four dimension scores are saved alongside the overall score so the
fine-tuning dataset can be used for single-output regression, multi-output
regression, or classification depending on the downstream experiment.

Output: scoring/vlm_scores_sealion_ecom2500_labels.csv

Usage:
  pip install openai python-dotenv tqdm pandas pillow
  python scoring/score_sealion_ecom2500_labels.py
  python scoring/score_sealion_ecom2500_labels.py --n-total 2500 --seed 42
"""

import argparse
import base64
import io
import json
import math
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID  = "aisingapore/Gemma-SEA-LION-v4-27B-IT"
BASE_URL  = "https://api.sea-lion.ai/v1"

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
ECOM_DIR  = DATA_DIR / "Ecommerce_118K" / "train" / "train"
ECOM_CSV  = DATA_DIR / "Ecommerce_118K" / "train.csv"

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
        # Validate score is within expected range
        if not (1 <= overall <= 10):
            return {"overall_score": -1, "reason": f"out_of_range: {overall}"}

        return {
            "overall_score":        overall,
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

# ── Sampling ──────────────────────────────────────────────────────────────────

def build_stratified_sample(n_total: int, seed: int) -> pd.DataFrame:
    """
    Sample evenly across all 42 ecommerce118k categories.
    n_per_cat = ceil(n_total / 42). Categories with fewer images take all available.
    """
    csv_df    = pd.read_csv(ECOM_CSV)
    n_cats    = csv_df["category"].nunique()
    n_per_cat = math.ceil(n_total / n_cats)

    sampled_parts = []
    for cat_id, group in csv_df.groupby("category"):
        take = min(n_per_cat, len(group))
        sampled_parts.append(group.sample(take, random_state=seed))

    sample = pd.concat(sampled_parts).reset_index(drop=True)
    print(f"Stratified sample: {len(sample)} images across {n_cats} categories "
          f"({n_per_cat} per category target)")
    return sample


def load_images_from_sample(sample_df: pd.DataFrame) -> list:
    rows = []
    missing = 0
    for _, row in sample_df.iterrows():
        cat_id   = int(row["category"])
        cat_name = ECOM_CATEGORY_NAMES.get(cat_id, str(cat_id))
        path     = ECOM_DIR / str(cat_id).zfill(2) / row["filename"]
        if not path.exists():
            missing += 1
            continue
        img = Image.open(path).convert("RGB")
        rows.append({
            "dataset":    "ecommerce118k",
            "filename":   row["filename"],
            "category_id": cat_id,
            "category":   cat_name,
            "image":      img,
        })
    if missing:
        print(f"Warning: {missing} image files not found on disk — skipped.")
    return rows

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-total", type=int, default=2500,
                        help="Total images to score across all categories (default: 2500)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    args = parser.parse_args()

    out_csv = Path(__file__).parent / "vlm_scores_sealion_ecom2500_labels.csv"

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
    all_rows  = load_images_from_sample(sample_df)
    pending   = [r for r in all_rows if r["filename"] not in done]
    print(f"Total loaded: {len(all_rows)}  |  To score: {len(pending)}")

    client = load_client()

    for item in tqdm(pending, desc="Scoring"):
        scored = score_pil_image(client, item["image"])
        results.append({
            "dataset":                item["dataset"],
            "filename":               item["filename"],
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
