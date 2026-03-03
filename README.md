# AVM

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

## Web Demo

Running the web demo with Streamlit + Hugging Face model:

```sh
. .venv/bin/activate
export HF_TOKEN=<your_hf_token> # or `hf auth login`
streamlit run app.py
```

Then open the local URL shown in the terminal.