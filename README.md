# AVM

![](./public/demo.png)

## Setup

### Prerequisites

- **[pyenv](https://github.com/pyenv/pyenv)**
- **[uv](https://github.com/astral-sh/uv)**

### Getting Started

Install Python dependencies and set up virtual environment:
```sh
pyenv install
pyenv exec python -m venv ./.venv
. .venv/bin/activate
uv sync
```

Download fine-tuned model weights into the `models/` directory:
- **Fine-tuned DINOv2 model** for image scoring at `models/dinov2_vitb14_best.pth` from https://drive.google.com/file/d/1TXZJwTTe7lCsapk2lRHIYmP97424Xw0A/view?usp=sharing
- **Fine-tuned SAM3 model** for mask segmentation at `models/sam3_foreground_best-004.pth` from https://drive.google.com/file/d/12BNSW0Gk3Cu6Cd5jE1TUm_Wqr4XcAaZt/view?usp=sharing


## Web Demo

Running the web demo with Streamlit + Hugging Face model:

```sh
. .venv/bin/activate
export HF_TOKEN=<your_hf_token> # or `hf auth login`
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Model Training and Evaluation

The folders `scoring/`, `segmentation/` and `inpainting/` contain code for training and evaluating the image scoring, mask segmentation, and background inpainting models, respectively. 