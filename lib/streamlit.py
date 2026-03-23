from __future__ import annotations

from typing import Final

import streamlit as st

from lib.score import ImageScoring

SCORE_LABELS: Final[dict[str, str]] = {
    "background_cleanliness": "Background Cleanliness",
    "text_watermark_score": "Text and Watermark",
    "product_prominence": "Product Prominence",
    "commercial_appeal": "Commercial Appeal",
}

SCORE_HELP_TEXT: Final[dict[str, str]] = {
    "background_cleanliness": "How clean and distraction-free the scene is.",
    "text_watermark_score": "How free the image is from seller overlays and watermarks.",
    "product_prominence": "How clearly the product is framed and visible.",
    "commercial_appeal": "How likely the image is to drive shopper clicks.",
}


def inject_styles() -> None:
    st.markdown(
        """
<style>
    :root {
        --avm-bg: #f3f6fb;
        --avm-surface: #ffffff;
        --avm-text: #0f172a;
        --avm-muted: #4b5563;
        --avm-border: #dbe3ee;
        --avm-brand: #ef4444;
        --avm-brand-hover: #dc2626;
        --avm-radius: 10px;
    }

    .stApp {
        background: var(--avm-bg);
    }

    [data-testid="block-container"] {
        max-width: 1120px !important;
        padding-top: 1.7rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.01em;
        font-weight: 650 !important;
        color: var(--avm-text);
        margin-bottom: 0.45rem;
    }

    p:not([data-testid="stBaseButton-primary"] p), .stCaption {
        color: var(--avm-muted);
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--avm-border);
        background: #f6f8fc;
    }

    [data-testid="stImage"] img {
        border-radius: var(--avm-radius);
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed #c9d4e5;
        border-radius: 10px;
        background: #f9fbff;
        padding-top: 0.55rem;
        padding-bottom: 0.55rem;
    }

    div[data-baseweb="select"] > div,
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input {
        border-radius: 8px !important;
        border-color: var(--avm-border) !important;
        min-height: 2.2rem;
    }

    .stTextArea textarea {
        min-height: 6.3rem;
    }

    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.92rem;
    }

    [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primary"] * {
        color: #ffffff !important;
    }

    .avm-stepper {
        position: relative;
        margin-top: 0.8rem;
        margin-bottom: 0.7rem;
        padding-top: 0.2rem;
    }

    .avm-stepper-line,
    .avm-stepper-line-fill {
        position: absolute;
        top: 13px;
        height: 2px;
        border-radius: 999px;
        left: 6%;
        width: 88%;
    }

    .avm-stepper-line {
        background: #dbe1ea;
    }

    .avm-stepper-line-fill {
        background: var(--avm-brand);
        transition: width 0.2s ease;
    }

    .avm-stepper-grid {
        position: relative;
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.6rem;
    }

    .avm-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        gap: 0.35rem;
    }

    .avm-step-dot {
        width: 1.7rem;
        height: 1.7rem;
        border-radius: 999px;
        border: 1.5px solid #cbd5e1;
        background: #ffffff;
        color: #64748b;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.77rem;
        line-height: 1;
    }

    .avm-step-label {
        color: #64748b;
        font-weight: 600;
        font-size: 0.84rem;
    }

    .avm-step.done .avm-step-dot {
        background: var(--avm-brand);
        border-color: var(--avm-brand);
        color: #ffffff;
    }

    .avm-step.done .avm-step-label,
    .avm-step.active .avm-step-label {
        color: var(--avm-brand);
        font-weight: 700;
    }

    .avm-step.active .avm-step-dot {
        border-color: var(--avm-brand);
        color: var(--avm-brand);
        box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.15);
    }

    .avm-analysis-callout {
        border-radius: 10px;
        padding: 0.82rem 0.95rem;
        border: 1px solid;
        margin-bottom: 0.82rem;
    }

    .avm-analysis-callout-title {
        margin: 0;
        font-weight: 700;
        font-size: 0.96rem;
    }

    .avm-analysis-callout-message {
        margin: 0.22rem 0 0;
        color: #334155;
        font-size: 0.92rem;
    }

    .avm-analysis-callout.recommended {
        background: #fff1f2;
        border-color: #fda4af;
    }

    .avm-analysis-callout.not-needed {
        background: #ecfdf3;
        border-color: #86efac;
    }

    .avm-analysis-callout.unknown {
        background: #fff8e8;
        border-color: #fcd34d;
    }

    .avm-metric-card {
        background: var(--avm-surface);
        border: 1px solid var(--avm-border);
        border-radius: 10px;
        padding: 0.6rem 0.72rem;
        margin-bottom: 0.3rem;
    }

    .avm-metric-label {
        font-size: 0.82rem;
        color: var(--avm-muted);
        margin-bottom: 0.1rem;
    }

    .avm-metric-value {
        font-size: 1.7rem;
        line-height: 1.1;
        font-weight: 600;
        color: var(--avm-text);
        margin-bottom: 0.35rem;
    }

    .avm-band {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .avm-band-ok {
        background: #dcfce7;
        color: #15803d;
    }

    .avm-band-warn {
        background: #fef3c7;
        color: #b45309;
    }

    .avm-band-unavail {
        background: #f1f5f9;
        color: #64748b;
    }

    @media (max-width: 980px) {
        [data-testid="block-container"] {
            padding-top: 1rem;
            padding-bottom: 1.35rem;
        }

        .avm-step-label {
            font-size: 0.73rem;
        }

        .avm-stepper-line,
        .avm-stepper-line-fill {
            top: 11px;
        }

        .avm-step-dot {
            width: 1.5rem;
            height: 1.5rem;
            font-size: 0.68rem;
        }
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def render_progress_stepper(stages: tuple[str, ...], current_stage: int) -> None:
    total_stages = len(stages)
    safe_stage = max(1, min(current_stage, total_stages))
    progress_percent = 0.0 if total_stages <= 1 else ((safe_stage - 1) / (total_stages - 1)) * 100

    step_items: list[str] = []
    for index, step_name in enumerate(stages, start=1):
        if index < safe_stage:
            step_state = "done"
            dot_text = "✓"
        elif index == safe_stage:
            step_state = "active"
            dot_text = str(index)
        else:
            step_state = "todo"
            dot_text = str(index)

        step_items.append(
            f"<div class='avm-step {step_state}'><div class='avm-step-dot'>{dot_text}</div><div class='avm-step-label'>Step {index}: {step_name}</div></div>"
        )

    st.markdown(
        f"<div class='avm-stepper'><div class='avm-stepper-line'></div><div class='avm-stepper-line-fill' style='width: {progress_percent:.2f}%'></div><div class='avm-stepper-grid'>{''.join(step_items)}</div></div>",
        unsafe_allow_html=True,
    )


def _format_score(value: float) -> str:
    if value < 0:
        return "N/A"
    return f"{value:.1f}/10"


def _score_band(value: float) -> str:
    if value < 0:
        return "Unavailable"
    if value >= 8:
        return "Acceptable"
    return "Needs improvement"


def render_score_metric(label: str, value: float, caption: str | None = None) -> None:
    band = _score_band(value)
    if band == "Acceptable":
        badge_class = "avm-band-ok"
        badge_icon = "✓"
    elif band == "Needs improvement":
        badge_class = "avm-band-warn"
        badge_icon = "⚠"
    else:
        badge_class = "avm-band-unavail"
        badge_icon = "—"
    st.markdown(
        f"<div class='avm-metric-card'>"
        f"<div class='avm-metric-label'>{label}</div>"
        f"<div class='avm-metric-value'>{_format_score(value)}</div>"
        f"<span class='avm-band {badge_class}'>{badge_icon} {band}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def score_payload_from_exception(exc: Exception) -> ImageScoring:
    return {
        "overall_score": -1,
        "background_cleanliness": -1,
        "text_watermark_score": -1,
        "product_prominence": -1,
        "commercial_appeal": -1,
        "bg_class": "",
        "reason": f"error: {exc}",
    }
