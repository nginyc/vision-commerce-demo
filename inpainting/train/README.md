# Training: DreamBooth LoRA Fine-tuning

Fine-tunes [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) for background inpainting using DreamBooth LoRA via `accelerate`.

## Setup

Set the following environment variables before running:

```bash
export MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
export INSTANCE_DIR="/path/to/training/images"
export MASK_DIR="/path/to/training/masks"
export OUTPUT_DIR="/path/to/output"
export CHECKPOINT_DIR=""  # leave empty to train from scratch, or set to resume
```

## Usage

```bash
CUDA_VISIBLE_DEVICES=1 HF_HOME=/path/to/model/cache accelerate launch train_dreambooth_inpaint_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --mask_data_dir=$MASK_DIR \
  --output_dir=$OUTPUT_DIR \
  --resume_from_checkpoint=$CHECKPOINT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="A JD background" \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --checkpointing_steps=100 \
  --max_train_steps=2500 \
  --validation_prompt="A JD background" \
  --validation_epochs=25 \
  --validation_image="" \
  --validation_mask="" \
  --seed="0"
```

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--instance_prompt` | `"A JD background"` | DreamBooth trigger token/phrase |
| `--resolution` | `512` | Training image resolution |
| `--max_train_steps` | `2500` | Total training steps |
| `--checkpointing_steps` | `100` | Save a checkpoint every N steps |
| `--optimizer` | `prodigy` | Optimizer (prodigy requires `learning_rate=1.`) |
| `--mixed_precision` | `bf16` | Use bf16 mixed precision |

The trained LoRA weights are saved as `pytorch_lora_weights.safetensors` under `$OUTPUT_DIR/checkpoint-<step>/`.
