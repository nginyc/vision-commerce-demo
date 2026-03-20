#!/usr/bin/env python3
"""
Score Pool B (300 human-annotated images) using SEA-LION as judge.
Results are saved with human scores so SRCC can be compared directly
against the fine-tuned DINOv2 model.

Usage:
    python scoring/scoring_validation_llm_judges/score_sealion_pool_b.py
"""

import base64
import io
import json
import os
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID = "aisingapore/Gemma-SEA-LION-v4-27B-IT"
BASE_URL = "https://api.sea-lion.ai/v1"

ROOT         = Path(__file__).parent.parent.parent
DATA_DIR     = ROOT / "data"
SCORING_DIR  = ROOT / "scoring"

BG60K_DIR    = DATA_DIR / "ICASSP2025-BG60k"
BG60K_ZIPS   = [BG60K_DIR / f"bg60k_imgs_{i}.zip" for i in range(10)]
ECOM_DIR     = DATA_DIR / "Ecommerce_118K" / "train" / "train"

POOL_B_XLSX  = SCORING_DIR / "stage1_annotation" / "pool_b_300_images.xlsx"
OUT_CSV      = Path(__file__).parent / "sealion_pool_b_results.csv"

# ── Prompt (identical to the validation script for a fair comparison) ─────────

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

# ── Image loading ─────────────────────────────────────────────────────────────


def _build_zip_index():
    index = {}
    for zp in BG60K_ZIPS:
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.endswith(".png"):
                    index[name.split("/")[-1]] = (zp, name)
    return index


def load_pool_b_images(df, zip_index):
    rows = []
    for _, row in df.iterrows():
        dataset = row["dataset"].lower()
        fname   = row["filename"]
        cat_id  = row["category_id"]
        try:
            if dataset == "bg60k":
                entry = zip_index.get(fname)
                if entry is None:
                    print(f"  WARN: {fname} not found in zips, skipping")
                    continue
                zp, internal = entry
                with zipfile.ZipFile(zp) as z:
                    data = z.read(internal)
                img = Image.open(io.BytesIO(data)).convert("RGB")
            else:
                path = ECOM_DIR / str(int(cat_id)).zfill(2) / fname
                img  = Image.open(path).convert("RGB")
            rows.append({
                "pool_b_id":    row["pool_b_id"],
                "dataset":      dataset,
                "filename":     fname,
                "category_id":  cat_id,
                "human_score":  row["human_score"],
                "image":        img,
            })
        except Exception as e:
            print(f"  WARN: could not load {fname}: {e}")
    return rows


# ── SEA-LION scoring ──────────────────────────────────────────────────────────


def load_client():
    from openai import OpenAI
    api_key = os.getenv("SEALION_API_KEY")
    if not api_key:
        raise ValueError("SEALION_API_KEY not found in .env")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def score_image(client, img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": SCORING_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=400,
            temperature=0,
        )
        raw   = response.choices[0].message.content
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return {"sealion_score": -1, "reason": f"no_json: {raw[:120]}"}
        parsed = json.loads(raw[start:end])
        return {
            "sealion_score": round(float(parsed["overall_score"]), 2),
            "reason": parsed.get("primary_failure_reason") or parsed.get("background_description", ""),
        }
    except Exception as e:
        return {"sealion_score": -1, "reason": f"error: {e}"}


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    # Load Pool B metadata
    df = pd.read_excel(POOL_B_XLSX).rename(columns={
        "Pool B ID":            "pool_b_id",
        "Dataset":              "dataset",
        "Filename":             "filename",
        "Category ID":          "category_id",
        "[HUMAN] Overall Score":"human_score",
    })
    print(f"Pool B: {len(df)} images")

    # Resume from partial run
    done = set()
    results = []
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        done     = set(existing["pool_b_id"])
        results  = existing.to_dict("records")
        print(f"Resuming — {len(done)} already scored.")

    # Load images
    print("Building zip index …")
    zip_index = _build_zip_index()
    all_rows  = load_pool_b_images(df, zip_index)
    pending   = [r for r in all_rows if r["pool_b_id"] not in done]
    print(f"To score: {len(pending)}")

    client = load_client()
    print(f"SEA-LION client ready. Model: {MODEL_ID}\n")

    for item in tqdm(pending, desc="SEA-LION scoring"):
        scored = score_image(client, item["image"])
        results.append({
            "pool_b_id":     item["pool_b_id"],
            "dataset":       item["dataset"],
            "filename":      item["filename"],
            "human_score":   item["human_score"],
            "sealion_score": scored["sealion_score"],
            "reason":        scored["reason"],
        })
        pd.DataFrame(results).to_csv(OUT_CSV, index=False)

    # Final metrics
    res_df  = pd.DataFrame(results)
    valid   = res_df[res_df["sealion_score"] > 0].copy()
    errors  = (res_df["sealion_score"] == -1).sum()

    preds   = valid["sealion_score"].values
    targets = valid["human_score"].values

    srcc, _ = spearmanr(preds, targets)
    plcc, _ = pearsonr(preds, targets)
    mae     = float(np.mean(np.abs(preds - targets)))

    print("\n" + "=" * 45)
    print("Pool B — SEA-LION vs human annotations")
    print("=" * 45)
    print(f"  SRCC : {srcc:.4f}")
    print(f"  PLCC : {plcc:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  N    : {len(valid)}  (errors: {errors})")
    print("=" * 45)
    print(f"\nResults saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
