#!/usr/bin/env python3
"""
SEA-LION (Gemma-SEA-LION-v4-27B-IT) e-commerce background quality scorer.

Scores the same images listed in vlm_scores_gpt-4o_validation.csv so results
can be compared directly against GPT-4o and Qwen2-VL-2B.

Usage:
  pip install openai python-dotenv tqdm pandas pillow
  python scoring/score_gemma-sea-lion-v4-27B.py
  python scoring/score_gemma-sea-lion-v4-27B.py --match-csv scoring/vlm_scores_gpt-4o_validation.csv --n 50
"""

import argparse
import base64
import io
import json
import os
import zipfile
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

BG60K_DIR  = DATA_DIR / "ICASSP2025-BG60k"
BG60K_ZIPS = [BG60K_DIR / f"bg60k_imgs_{i}.zip" for i in range(10)]
BG60K_INFO = BG60K_DIR / "bg60k_info.txt"

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
# Same prompt as GPT-4o for a fair comparison.

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
    """Score a single PIL image using SEA-LION. Returns {"score": float, "bg_class": str, "reason": str}."""
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
            return {"score": -1, "bg_class": "", "reason": f"no_json: {raw[:120]}"}
        parsed = json.loads(raw[start:end])
        return {
            "score":    round(float(parsed["overall_score"]), 2),
            "bg_class": parsed.get("background_class", ""),
            "reason":   parsed.get("primary_failure_reason") or parsed.get("background_description", ""),
        }
    except Exception as e:
        return {"score": -1, "bg_class": "", "reason": f"error: {e}"}

# ── Data loading ──────────────────────────────────────────────────────────────

def build_bg60k_zip_index():
    index = {}
    for zp in BG60K_ZIPS:
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.endswith(".png"):
                    index[name.split("/")[-1]] = (zp, name)
    return index


def load_from_gpt_csv(match_csv: Path):
    """Load exactly the images listed in the GPT-4o validation CSV."""
    ref_df    = pd.read_csv(match_csv)
    zip_index = build_bg60k_zip_index()
    bg_info   = pd.read_csv(BG60K_INFO, sep="\t", header=None, names=["filename", "category"])
    bg_cat    = dict(zip(bg_info["filename"], bg_info["category"]))
    ecom_df   = pd.read_csv(ECOM_CSV).set_index("filename")

    rows = []

    for _, ref in ref_df[ref_df["dataset"] == "bg60k"].iterrows():
        fname = ref["filename"]
        entry = zip_index.get(fname)
        if entry is None:
            continue
        zp, internal = entry
        with zipfile.ZipFile(zp) as z:
            data = z.read(internal)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        rows.append({
            "dataset":  "bg60k",
            "filename": fname,
            "category": bg_cat.get(fname, ref.get("category", "")),
            "image":    img,
        })

    for _, ref in ref_df[ref_df["dataset"] == "ecommerce118k"].iterrows():
        fname = ref["filename"]
        if fname not in ecom_df.index:
            continue
        cat_id   = int(ecom_df.loc[fname, "category"])
        cat_name = ECOM_CATEGORY_NAMES.get(cat_id, str(cat_id))
        path     = ECOM_DIR / str(cat_id).zfill(2) / fname
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        rows.append({
            "dataset":  "ecommerce118k",
            "filename": fname,
            "category": cat_name,
            "image":    img,
        })

    return rows

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--match-csv",
        default=str(Path(__file__).parent / "vlm_scores_gpt-4o_validation.csv"),
        metavar="CSV",
        help="Score the same images listed in this CSV (default: vlm_scores_gpt-4o_validation.csv)",
    )
    args = parser.parse_args()

    match_csv = Path(args.match_csv)
    if not match_csv.exists():
        raise FileNotFoundError(f"Match CSV not found: {match_csv}")

    out_csv = Path(__file__).parent / "vlm_scores_sealion_validation.csv"

    # Resume from checkpoint
    done = set()
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        done = set(zip(existing["dataset"], existing["filename"]))
        print(f"Resuming — {len(done)} images already scored.")
        results = existing.to_dict("records")
    else:
        results = []

    print(f"Loading images from {match_csv.name} …")
    all_rows = load_from_gpt_csv(match_csv)
    pending  = [r for r in all_rows if (r["dataset"], r["filename"]) not in done]
    print(f"Total: {len(all_rows)}  |  To score: {len(pending)}")

    client = load_client()

    for item in tqdm(pending, desc="Scoring"):
        scored = score_pil_image(client, item["image"])
        results.append({
            "dataset":  item["dataset"],
            "filename": item["filename"],
            "category": item["category"],
            "score":    scored["score"],
            "bg_class": scored["bg_class"],
            "reason":   scored["reason"],
        })
        pd.DataFrame(results).to_csv(out_csv, index=False)

    df    = pd.DataFrame(results)
    valid = df[df["score"] > 0]

    print("\n=== Score distribution by dataset ===")
    print(valid.groupby("dataset")["score"].describe().round(2))

    print("\n=== Score bins ===")
    bins   = [0, 4, 7, 10]
    labels = ["low (1-4)", "mid (5-7)", "high (8-10)"]
    valid  = valid.copy()
    valid["band"] = pd.cut(valid["score"], bins=bins, labels=labels)
    print(valid.groupby(["dataset", "band"], observed=True).size().unstack(fill_value=0))

    errors = (df["score"] == -1).sum()
    if errors:
        print(f"\nParse errors: {errors}")

    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
