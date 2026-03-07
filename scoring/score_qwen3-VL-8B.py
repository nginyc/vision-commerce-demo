#!/usr/bin/env python3
"""
Qwen3-VL-8B (4-bit quantised via MLX) e-commerce background quality scorer.

Scores the same images listed in vlm_scores_gpt-4o_validation.csv so results
can be compared directly against GPT-4o, Qwen2-VL-2B, and SEA-LION.

The 4-bit quantised 7B model requires ~4-5 GB of unified memory, well within
the 16 GB available on an Apple M4 MacBook Air.

Usage:
  pip install mlx-vlm tqdm pandas pillow
  python scoring/score_qwen3-VL-8B.py
  python scoring/score_qwen3-VL-8B.py --match-csv scoring/vlm_scores_gpt-4o_validation.csv
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

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID  = "mlx-community/Qwen3-VL-8B-Instruct-4bit"

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
# Same prompt as all other scorers for a fair comparison.

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

def load_model():
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    print(f"Loading {MODEL_ID} (downloads ~4 GB on first run) …")
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)
    print("Model ready.")
    return model, processor, config


def score_pil_image(model, processor, config, img: Image.Image) -> dict:
    """Score a single PIL image. Returns {"score": float, "bg_class": str, "reason": str}."""
    from mlx_vlm import generate, apply_chat_template

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.convert("RGB").save(tmp.name, "JPEG", quality=92)
        tmp_path = tmp.name

    try:
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
        if hasattr(raw, "text"):
            raw = raw.text
        else:
            raw = str(raw)
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
    finally:
        Path(tmp_path).unlink(missing_ok=True)

# ── Data loading ──────────────────────────────────────────────────────────────

def build_bg60k_zip_index():
    index = {}
    for zp in BG60K_ZIPS:
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.endswith(".png"):
                    index[name.split("/")[-1]] = (zp, name)
    return index


def load_from_gpt_csv(match_csv: Path) -> list:
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

    out_csv = Path(__file__).parent / "vlm_scores_qwen3-VL-8B_validation.csv"

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

    model, processor, config = load_model()

    for item in tqdm(pending, desc="Scoring"):
        scored = score_pil_image(model, processor, config, item["image"])
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
    bins   = [0, 3, 7, 10]
    labels = ["low (1-3)", "mid (4-7)", "high (8-10)"]
    valid  = valid.copy()
    valid["band"] = pd.cut(valid["score"], bins=bins, labels=labels)
    print(valid.groupby(["dataset", "band"], observed=True).size().unstack(fill_value=0))

    errors = (df["score"] == -1).sum()
    if errors:
        print(f"\nParse errors: {errors}")

    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
