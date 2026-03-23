# Evaluation

Two scripts for running batch inference and evaluating the results.

## 1. Batch Inference (`batch_inference.py`)

Runs FLUX.1-Fill-dev background replacement on all image/mask pairs in a data directory.

**Input directory layout:**
```
<data-dir>/
  sample_image/       # product images
  sample_mask/        # masks (white=background/fill, black=product/keep)
  sample_cases_50.csv # maps filenames to product category labels
```

### Baseline (no LoRA)

```bash
uv run batch_inference.py \
  --data-dir black_prod_output \
  --output-dir black_prod_inpainted/flux_baseline \
  --model black-forest-labs/FLUX.1-Fill-dev
```

### LoRA — fixed DreamBooth prompt

```bash
uv run batch_inference.py \
  --data-dir black_prod_output \
  --output-dir black_prod_inpainted/flux_finetuned_fixed_prompt \
  --model black-forest-labs/FLUX.1-Fill-dev \
  --lora-path /path/to/checkpoint-2500/pytorch_lora_weights.safetensors \
  --prompt "A JD background"
```

### LoRA — category-specific prompt with DreamBooth trigger appended

```bash
uv run batch_inference.py \
  --data-dir black_prod_output \
  --output-dir black_prod_inpainted/flux_finetuned \
  --model black-forest-labs/FLUX.1-Fill-dev \
  --lora-path /path/to/checkpoint-2500/pytorch_lora_weights.safetensors \
  --append-prompt "A JD background"
```

### Key options

| Flag | Default | Description |
|---|---|---|
| `--model` | `black-forest-labs/FLUX.1-Fill-dev` | HuggingFace model ID or local path |
| `--lora-path` | — | LoRA `.safetensors` file or checkpoint directory |
| `--prompt` | — | Fixed prompt for all images (overrides CSV-based prompts) |
| `--append-prompt` | — | String appended to each category prompt (e.g. DreamBooth trigger) |
| `--steps` | `28` | Number of inference steps |
| `--guidance` | `30.0` | Guidance scale |
| `--gpu` | `1` | Physical GPU index |
| `--dtype` | `bf16` | Model dtype (`bf16`, `fp16`, `fp32`) |
| `--offload` | — | Enable sequential CPU offload to reduce VRAM |
| `--dry-run` | — | Print what would run without doing inference |

Already-completed images are skipped automatically, so runs are resumable.

---

## 2. Evaluate Outputs (`eval_output.py`)

Computes CLIP image similarity and FID between original and generated images.

### Single directory

```bash
uv run eval_output.py \
  --orig-dir black_prod_output/sample_image \
  --gen-dir black_prod_inpainted/flux_baseline \
  --save-csv results_baseline.csv \
  --save-json results_baseline.json
```

### Compare multiple directories at once

```bash
uv run eval_output.py \
  --orig-dir black_prod_output/sample_image \
  --input black_prod_inpainted/
```

Evaluates each subdirectory of `--input` against `--orig-dir` and prints a summary table.

### Key options

| Flag | Default | Description |
|---|---|---|
| `--orig-dir` | `black_prod_output/sample_image` | Reference (original) images |
| `--gen-dir` | — | Single output directory to evaluate |
| `--input` | — | Parent directory; evaluates all subdirectories |
| `--clip-model` | `openai/clip-vit-large-patch14` | CLIP model for similarity scoring |
| `--top-k` | `5` | Number of best/worst pairs to print |
| `--no-fid` | — | Skip FID computation (faster) |
| `--save-csv` | — | Save per-image CLIP scores to CSV |
| `--save-json` | — | Save summary metrics to JSON |
