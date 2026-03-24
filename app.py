import gc
import hashlib
import io
import os
from pathlib import Path
from typing import Callable, Final, TypeAlias, TypeVar

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from lib.inpaint import (
    CATEGORY_PROMPTS,
    DEFAULT_MODEL_ID as INPAINT_DEFAULT_MODEL_ID,
    MODELS as INPAINT_MODELS,
    build_model as build_inpaint_model,
    detect_device,
    get_prompt,
    inpaint_background,
    get_model_config_defaults as get_inpaint_config_defaults,
)
from lib.segment import (
    DEFAULT_MODEL_ID as SEGMENT_DEFAULT_MODEL_ID,
    MODELS as SEGMENT_MODELS,
    build_model as build_segment_model,
    get_model_config_defaults as get_segment_config_defaults,
    normalize_category,
    create_mask_image,
)
from lib.score import (
    DEFAULT_MODEL_ID as SCORE_DEFAULT_MODEL_ID,
    MODELS as SCORE_MODELS,
    DINOv2WithSealionScoringModel,
    ImageScoring,
    OpenAIScoringModel,
)
from lib.streamlit import (
    SCORE_HELP_TEXT,
    SCORE_LABELS,
    inject_styles,
    render_progress_stepper,
    render_score_metric,
    score_payload_from_exception,
)

load_dotenv()

SCORING_BASE_URL = os.getenv("SEALION_BASE_URL", "https://api.sea-lion.ai/v1")
SCORING_API_KEY_ENV = "SEALION_API_KEY"
DINOV2_CHECKPOINT_PATH = str(Path(__file__).parent / "models" / "dinov2_vitb14_best.pt")
INPAINT_LORA_PATH = str(Path(__file__).parent / "models" / "flux1_fill.pytorch_lora_weights.safetensors")


# --- Session State Keys ---
class Keys:
    STAGE = "stage"
    ORIGINAL_IMAGE = "original_image"
    UPLOADED_IMAGE_DIGEST = "uploaded_image_digest"
    IMAGE_SCORING_RESULT = "image_scoring_result"
    IS_ANALYZING_IMAGE = "is_analyzing_image"
    IMAGE_ANALYSIS_ERROR = "image_analysis_error"
    LOADED_SCORING_MODEL_KEY = "loaded_scoring_model_key"
    LOADED_SEGMENT_MODEL_KEY = "loaded_segment_model_key"
    LOADED_INPAINT_MODEL_KEY = "loaded_inpaint_model_key"
    MASK_IMAGE = "mask_image"
    CATEGORY = "category"
    SEED = "seed"
    OUTPUT_IMAGE = "output_image"
    IS_GENERATING_MASK = "is_generating_mask"
    IS_GENERATING_BACKGROUND = "is_generating_background"
    MASK_GENERATION_ERROR = "mask_generation_error"


# --- Cached Models ---
@st.cache_resource(show_spinner=False)  # type: ignore
def get_cached_scoring_model(model_id: str, base_url: str):
    api_key = os.getenv(SCORING_API_KEY_ENV, "").strip()
    if not api_key:
        raise ValueError(f"{SCORING_API_KEY_ENV} is required to analyze image quality.")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return OpenAIScoringModel(client=client, model_id=model_id)


@st.cache_resource(show_spinner=False)  # type: ignore
def get_cached_dinov2_scoring_model(_model_id: str, device: str):
    api_key = os.getenv(SCORING_API_KEY_ENV, "").strip()
    if not api_key:
        raise ValueError(f"{SCORING_API_KEY_ENV} is required to analyze image quality.")
    client = OpenAI(api_key=api_key, base_url=SCORING_BASE_URL)
    sealion = OpenAIScoringModel(client=client, model_id=SCORE_MODELS["sealionv4"])
    return DINOv2WithSealionScoringModel(sealion_model=sealion, checkpoint_path=DINOV2_CHECKPOINT_PATH, device=device)


@st.cache_resource(show_spinner=False)  # type: ignore
def get_cached_inpaint_model(model_id: str, device: str, lora_path: str | None = None):
    return build_inpaint_model(model_id, device, lora_path=lora_path)


@st.cache_resource(show_spinner=False)  # type: ignore
def get_cached_segment_model(model_id: str, device: str):
    return build_segment_model(model_id, device)


def _free_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _release_model(cache_fn, session_key: str) -> None:
    """Unload a specific cached model and free its memory."""
    cache_fn.clear()
    st.session_state.pop(session_key, None)
    _free_memory()


def clear_model_resources() -> None:
    get_cached_scoring_model.clear()
    get_cached_dinov2_scoring_model.clear()
    get_cached_inpaint_model.clear()
    get_cached_segment_model.clear()
    st.session_state.pop(Keys.LOADED_SCORING_MODEL_KEY, None)
    st.session_state.pop(Keys.LOADED_SEGMENT_MODEL_KEY, None)
    st.session_state.pop(Keys.LOADED_INPAINT_MODEL_KEY, None)
    _free_memory()


_M = TypeVar("_M")


def get_model(cache_fn: Callable[..., _M], model_key: tuple, session_key: str) -> _M:
    """Load a cached model, clearing resources first if the model key changed."""
    previous = st.session_state.get(session_key)
    if previous is not None and previous != model_key:
        clear_model_resources()
    model = cache_fn(*model_key)
    st.session_state[session_key] = model_key
    return model


# --- Model Index Helpers ---
INPAINT_DEFAULT_MODEL_INDEX = next(
    (index for index, model_id in enumerate(INPAINT_MODELS.values()) if model_id == INPAINT_DEFAULT_MODEL_ID),
    0,
)

SEGMENT_DEFAULT_MODEL_INDEX = next(
    (index for index, model_id in enumerate(SEGMENT_MODELS.values()) if model_id == SEGMENT_DEFAULT_MODEL_ID),
    0,
)

SCORE_DEFAULT_MODEL_INDEX = next(
    (index for index, model_id in enumerate(SCORE_MODELS.values()) if model_id == SCORE_DEFAULT_MODEL_ID),
    0,
)


# --- Page Config ---
st.set_page_config(page_title="VisionCommerce (Demo)", layout="wide", initial_sidebar_state="expanded")
inject_styles()

ConfigValue: TypeAlias = float | int
ModelConfig: TypeAlias = dict[str, ConfigValue]
SessionStateDefaults: TypeAlias = dict[str, object]

SESSION_RESET_KEYS: Final[tuple[str, ...]] = (
    Keys.STAGE,
    Keys.ORIGINAL_IMAGE,
    Keys.UPLOADED_IMAGE_DIGEST,
    Keys.IMAGE_SCORING_RESULT,
    Keys.IS_ANALYZING_IMAGE,
    Keys.IMAGE_ANALYSIS_ERROR,
    Keys.LOADED_SCORING_MODEL_KEY,
    Keys.MASK_IMAGE,
    Keys.CATEGORY,
    Keys.SEED,
    Keys.OUTPUT_IMAGE,
    Keys.IS_GENERATING_MASK,
    Keys.IS_GENERATING_BACKGROUND,
    Keys.MASK_GENERATION_ERROR,
)
STAGES: Final[tuple[str, ...]] = (
    "Upload Image",
    "Analyze Image",
    "Mask Background",
    "Generate Background",
    "Final Image",
)

# --- Session State Initialization ---
default_state: SessionStateDefaults = {
    Keys.STAGE: 1,
    Keys.ORIGINAL_IMAGE: None,
    Keys.UPLOADED_IMAGE_DIGEST: None,
    Keys.IMAGE_SCORING_RESULT: None,
    Keys.IS_ANALYZING_IMAGE: False,
    Keys.IMAGE_ANALYSIS_ERROR: None,
    Keys.LOADED_SCORING_MODEL_KEY: None,
    Keys.MASK_IMAGE: None,
    Keys.CATEGORY: list(CATEGORY_PROMPTS.keys())[0],
    Keys.SEED: 67,
    Keys.OUTPUT_IMAGE: None,
    Keys.IS_GENERATING_MASK: False,
    Keys.IS_GENERATING_BACKGROUND: False,
    Keys.MASK_GENERATION_ERROR: None,
}


def initialize_session_state(defaults: SessionStateDefaults) -> None:
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_session_state() -> None:
    for key in SESSION_RESET_KEYS:
        st.session_state.pop(key, None)


initialize_session_state(default_state)


# --- Helper Functions ---
def set_stage(stage: int) -> None:
    st.session_state[Keys.STAGE] = stage


def run_image_scoring(image: Image.Image) -> ImageScoring:
    try:
        if selected_scoring_model_id == SCORE_MODELS["dinov2-finetuned-with-sealionv4"]:
            model_key = (selected_scoring_model_id, detect_device())
            model = get_model(get_cached_dinov2_scoring_model, model_key, Keys.LOADED_SCORING_MODEL_KEY)
        else:
            model_key = (selected_scoring_model_id, SCORING_BASE_URL)
            model = get_model(get_cached_scoring_model, model_key, Keys.LOADED_SCORING_MODEL_KEY)
        return model.score_image(image)
    except Exception as exc:
        return score_payload_from_exception(exc)


def render_config_sliders(defaults: ModelConfig, key_prefix: str) -> ModelConfig:
    config: ModelConfig = {}
    config_settings: dict[str, tuple[str, float, float, float]] = {
        "strength": ("Strength", 0.0, 1.0, 0.05),
        "guidance": ("Guidance", 1.0, 20.0, 0.5),
        "guidance_scale": ("Guidance Scale", 1.0, 20.0, 0.5),
        "num_inference_steps": ("Inference Steps", 1.0, 80.0, 1.0),
        "mask_threshold": ("Mask Threshold", 0.0, 1.0, 0.05),
        "score_threshold": ("Score Threshold", 0.0, 1.0, 0.05),
    }
    for key, (label, min_val, max_val, step) in config_settings.items():
        if key in defaults:
            default_value = defaults[key]
            value_type = int if isinstance(default_value, int) else float
            config[key] = st.slider(
                label,
                min_value=value_type(min_val),
                max_value=value_type(max_val),
                value=value_type(default_value),
                step=value_type(step),
                key=f"{key_prefix}_{key}",
            )
    return config


# --- Sidebar ---
with st.sidebar:
    st.title("Settings")
    st.caption("Advanced controls for analysis, masking and background generation.")

    with st.expander("Analysis Settings", expanded=False):
        selected_scoring_model_label = st.selectbox(
            "Scoring Model",
            options=list(SCORE_MODELS.keys()),
            index=SCORE_DEFAULT_MODEL_INDEX,
        )
        selected_scoring_model_id = SCORE_MODELS[selected_scoring_model_label]

    with st.expander("Masking Settings", expanded=False):
        selected_segment_model_label = st.selectbox(
            "Segmentation Model",
            options=list(SEGMENT_MODELS.keys()),
            index=SEGMENT_DEFAULT_MODEL_INDEX,
        )
        selected_segment_model_id = SEGMENT_MODELS[selected_segment_model_label]
        segment_config = render_config_sliders(get_segment_config_defaults(selected_segment_model_id), "segment")

    with st.expander("Background Settings", expanded=False):
        selected_inpaint_model_label = st.selectbox(
            "Inpainting Model",
            options=list(INPAINT_MODELS.keys()),
            index=INPAINT_DEFAULT_MODEL_INDEX,
        )
        selected_inpaint_model_id = INPAINT_MODELS[selected_inpaint_model_label]
        st.session_state[Keys.SEED] = int(
            st.number_input(
                "Generation Seed",
                min_value=0,
                max_value=999999,
                value=int(st.session_state.get(Keys.SEED, 67)),
                step=1,
                key="sidebar_seed",
            )
        )
        inpaint_config = render_config_sliders(get_inpaint_config_defaults(selected_inpaint_model_id), "inpaint")

    st.divider()
    if st.button("↻ Start Over", use_container_width=True, type="secondary"):
        reset_session_state()
        st.rerun()


# ==========================================
# Stage Render Functions
# ==========================================

def render_stage_upload() -> None:
    st.header("Upload Product Image")
    st.write("Start by uploading the original image of your product. For best results, use a clear photo of the product on a simple background.")

    up_col1, up_col2 = st.columns([1, 1], gap="large")
    with up_col1:
        image_file = st.file_uploader("Upload Product Image", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")

    if image_file:
        image_bytes = image_file.getvalue()
        image_digest = hashlib.sha256(image_bytes).hexdigest()
        if st.session_state.get(Keys.UPLOADED_IMAGE_DIGEST) != image_digest:
            st.session_state[Keys.UPLOADED_IMAGE_DIGEST] = image_digest
            st.session_state[Keys.ORIGINAL_IMAGE] = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.session_state[Keys.MASK_IMAGE] = None
            st.session_state[Keys.OUTPUT_IMAGE] = None
            st.session_state[Keys.IMAGE_SCORING_RESULT] = None
            st.session_state[Keys.IMAGE_ANALYSIS_ERROR] = None
            st.session_state[Keys.IS_ANALYZING_IMAGE] = False

        with up_col2:
            st.image(st.session_state[Keys.ORIGINAL_IMAGE], caption="Uploaded Image", use_container_width=True)

            st.write("")  # Spacing
            if st.button("Next: Analyze Image", type="primary", use_container_width=True):
                set_stage(2)
                st.rerun()

    else:
        st.info("👆 Please upload an original product image to continue.")


def render_stage_analyze() -> None:
    st.header("Analyze Image")
    st.write("We evaluate image readiness for e-commerce and recommend whether background replacement is needed.")

    if st.session_state.get(Keys.ORIGINAL_IMAGE) is None:
        st.warning("No image found. Please upload an image first.")
        if st.button("Back to Upload", use_container_width=True):
            set_stage(1)
            st.rerun()
        return

    analysis_col1, analysis_col2 = st.columns([1, 1], gap="large")
    with analysis_col1:
        st.image(st.session_state[Keys.ORIGINAL_IMAGE], caption="Uploaded Image", use_container_width=True)

    scoring_result = st.session_state.get(Keys.IMAGE_SCORING_RESULT)
    is_analyzing_image = bool(st.session_state.get(Keys.IS_ANALYZING_IMAGE, False))

    if scoring_result is None and not is_analyzing_image:
        st.session_state[Keys.IS_ANALYZING_IMAGE] = True
        st.rerun()

    if is_analyzing_image:
        with st.spinner("Analyzing uploaded image quality..."):
            analyzed = run_image_scoring(st.session_state[Keys.ORIGINAL_IMAGE])
            st.session_state[Keys.IMAGE_SCORING_RESULT] = analyzed
            if analyzed.get("overall_score", -1) < 0:
                st.session_state[Keys.IMAGE_ANALYSIS_ERROR] = str(analyzed.get("reason", "Scoring failed"))
            else:
                st.session_state[Keys.IMAGE_ANALYSIS_ERROR] = None
            st.session_state[Keys.IS_ANALYZING_IMAGE] = False
            _release_model(get_cached_scoring_model, Keys.LOADED_SCORING_MODEL_KEY)
            _release_model(get_cached_dinov2_scoring_model, Keys.LOADED_SCORING_MODEL_KEY)
        st.rerun()

    with analysis_col2:
        scoring = st.session_state.get(Keys.IMAGE_SCORING_RESULT)
        if scoring is None:
            st.info("Preparing analysis...")
        else:
            overall_score = float(scoring.get("overall_score", -1))

            if overall_score < 0:
                replacement_title = "Replacement decision unavailable"
                replacement_message = "Scoring could not determine replacement need. Re-analyze or continue to masking."
                replacement_class = "unknown"
            elif overall_score >= 8:
                replacement_title = "Background is acceptable"
                replacement_message = "No major background issue detected. You can still continue to generate alternatives."
                replacement_class = "not-needed"
            else:
                replacement_title = "Background replacement recommended"
                replacement_message = "The current background likely hurts listing quality. Continue to masking and generate a cleaner scene."
                replacement_class = "recommended"

            st.markdown(
                f"""
                <div class='avm-analysis-callout {replacement_class}'>
                    <p class='avm-analysis-callout-title'>{replacement_title}</p>
                    <p class='avm-analysis-callout-message'>{replacement_message}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            render_score_metric("Overall Score", overall_score, "Overall image readiness for marketplace listing.")

            reason = str(scoring.get("reason", "") or "No additional notes.")
            if overall_score < 0:
                st.error(f"Analysis issue: {reason}")
            else:
                st.info(reason)

    st.subheader("Breakdown")
    metric_names = [
        "background_cleanliness",
        "text_watermark_score",
        "product_prominence",
        "commercial_appeal",
    ]
    breakdown_col1, breakdown_col2 = st.columns(2, gap="large")
    scoring_payload = st.session_state.get(Keys.IMAGE_SCORING_RESULT)
    if scoring_payload is not None:
        for index, metric_name in enumerate(metric_names):
            score_value = float(scoring_payload.get(metric_name, -1))
            with breakdown_col1 if index % 2 == 0 else breakdown_col2:
                render_score_metric(SCORE_LABELS[metric_name], score_value, SCORE_HELP_TEXT[metric_name])

    analysis_error = st.session_state.get(Keys.IMAGE_ANALYSIS_ERROR)
    if analysis_error:
        st.warning(str(analysis_error))

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button("← Back", use_container_width=True):
            set_stage(1)
            st.rerun()
    with btn_col2:
        if st.button("Re-analyze", use_container_width=True):
            st.session_state[Keys.IMAGE_SCORING_RESULT] = None
            st.session_state[Keys.IMAGE_ANALYSIS_ERROR] = None
            st.session_state[Keys.IS_ANALYZING_IMAGE] = True
            st.rerun()
    with btn_col3:
        if st.button("Continue to Masking →", type="primary", use_container_width=True):
            set_stage(3)
            st.rerun()


def render_stage_mask() -> None:
    st.header("Profile Product")
    st.write("Select the product category to automatically map and separate the product from its background.")

    img_col, info_col = st.columns([1, 1], gap="large")
    with img_col:
        st.image(st.session_state[Keys.ORIGINAL_IMAGE], caption="Original Image", use_container_width=True)

    with info_col:
        category_options = list(CATEGORY_PROMPTS.keys())
        current_category = st.session_state[Keys.CATEGORY]
        default_idx = category_options.index(current_category) if current_category in category_options else 0
        category = st.selectbox(
            "Product Category",
            options=category_options,
            index=default_idx,
            key="stage2_category",
            help="Choose the category that best describes your product for accurate AI masking.",
        )
        st.session_state[Keys.CATEGORY] = category

        st.markdown("---")
        st.markdown("Ready to separate your product from the background using the selected category.")

        mask_error = st.session_state.get(Keys.MASK_GENERATION_ERROR)
        if mask_error:
            st.error(str(mask_error))
            st.session_state[Keys.MASK_GENERATION_ERROR] = None

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("← Back", use_container_width=True):
                set_stage(2)
                st.rerun()
        with btn_col2:
            is_generating_mask = bool(st.session_state.get(Keys.IS_GENERATING_MASK, False))
            mask_button_clicked = st.button(
                "Generating Mask..." if is_generating_mask else "Generate Mask",
                type="primary",
                use_container_width=True,
                disabled=is_generating_mask,
                key="generate_mask_button",
            )

        if mask_button_clicked and not is_generating_mask:
            st.session_state[Keys.IS_GENERATING_MASK] = True
            st.rerun()

        if is_generating_mask:
            try:
                device_used = detect_device()
                model_key = (selected_segment_model_id.strip(), device_used)
                model = get_model(get_cached_segment_model, model_key, Keys.LOADED_SEGMENT_MODEL_KEY)

                mask = model.generate_mask(
                    image=st.session_state[Keys.ORIGINAL_IMAGE],
                    prompt=normalize_category(category),
                    config=segment_config,
                )

                if mask is not None:
                    mask_img = create_mask_image(image=st.session_state[Keys.ORIGINAL_IMAGE], mask=mask)
                    st.session_state[Keys.MASK_IMAGE] = mask_img
                    st.session_state[Keys.IS_GENERATING_MASK] = False
                    _release_model(get_cached_segment_model, Keys.LOADED_SEGMENT_MODEL_KEY)
                    set_stage(4)
                    st.rerun()

                st.session_state[Keys.MASK_GENERATION_ERROR] = (
                    "Failed to generate mask. Try adjusting thresholds in the sidebar or change category."
                )
            finally:
                st.session_state[Keys.IS_GENERATING_MASK] = False

            st.rerun()


def render_stage_inpaint() -> None:
    st.header("Generate Background")
    st.write("Verify the mask and customize the background prompt before creating the final image.")

    content_col1, content_col2 = st.columns([1, 1], gap="large")

    with content_col1:
        st.subheader("Image Previews")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.write("**Original**")
            st.image(st.session_state[Keys.ORIGINAL_IMAGE], use_container_width=True)
        with img_col2:
            st.write("**Mask**")
            if st.session_state[Keys.MASK_IMAGE]:
                st.image(st.session_state[Keys.MASK_IMAGE], use_container_width=True)

        st.write("")  # Spacing
        if st.button("← Back to Profile", use_container_width=True):
            st.session_state.pop(Keys.OUTPUT_IMAGE, None)
            set_stage(3)
            st.rerun()

    with content_col2:
        st.subheader("Final Touches")
        is_fixed_prompt = selected_inpaint_model_id == "flux-finetuned-fixed-prompt"
        prompt = st.text_area(
            "Background Prompt",
            value="A JD background" if is_fixed_prompt else get_prompt(st.session_state[Keys.CATEGORY].strip()),
            disabled=is_fixed_prompt,
            height=120,
            help="Describe the exact background you want for your product. You can update this based on the required aesthetics.",
        )

        final_mask_image = st.session_state[Keys.MASK_IMAGE]
        with st.expander("Upload Override Mask (Optional)"):
            st.caption("Upload an RGBA mask to override the AI-generated mask. Transparent areas become the background; opaque areas are kept as product.")
            override_mask_file = st.file_uploader("Override Mask image", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")

            if override_mask_file:
                final_mask_image = Image.open(override_mask_file).convert("RGBA")
                st.image(final_mask_image, caption="Overridden Mask", width=300)

        st.write("")  # Spacing
        is_generating_background = bool(st.session_state.get(Keys.IS_GENERATING_BACKGROUND, False))
        gen_pressed = st.button(
            "Generating Background..." if is_generating_background else "Generate Background",
            type="primary",
            use_container_width=True,
            disabled=is_generating_background,
            key="generate_background_button",
        )

    if gen_pressed and not is_generating_background:
        st.session_state[Keys.IS_GENERATING_BACKGROUND] = True
        st.rerun()

    if is_generating_background:
        device_used = detect_device()
        model_key = (selected_inpaint_model_id.strip(), device_used, INPAINT_LORA_PATH)
        model = get_model(get_cached_inpaint_model, model_key, Keys.LOADED_INPAINT_MODEL_KEY)

        st.session_state[Keys.OUTPUT_IMAGE] = inpaint_background(
            model,
            st.session_state[Keys.ORIGINAL_IMAGE],
            final_mask_image,
            prompt.strip(),
            seed=st.session_state[Keys.SEED],
            config=inpaint_config,
        )

        st.session_state[Keys.IS_GENERATING_BACKGROUND] = False
        _release_model(get_cached_inpaint_model, Keys.LOADED_INPAINT_MODEL_KEY)
        set_stage(5)
        st.rerun()


def render_stage_result() -> None:
    st.header("Final Result")
    st.write("Your product image is ready with the generated background.")

    if st.session_state.get(Keys.OUTPUT_IMAGE) is None:
        st.error("No image found. Please go back and generate one.")
        if st.button("← Back to Settings"):
            set_stage(4)
            st.rerun()
        return

    _, res_col, _ = st.columns([1, 4, 1])
    with res_col:
        st.image(st.session_state[Keys.OUTPUT_IMAGE], use_container_width=True)

    st.divider()

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        st.caption("Tip: Right-click the image to save it to your device.")
    with btn_col2:
        if st.button("← Back to Prompt & Mask", use_container_width=True):
            set_stage(4)
            st.rerun()
    with btn_col3:
        if st.button("↻ Generate Another", type="primary", use_container_width=True):
            reset_session_state()
            st.rerun()

    st.write("")  # Spacing

    st.subheader("Inputs Used")
    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        st.write(f"**Product Category:** {st.session_state[Keys.CATEGORY]}")
    with meta_col2:
        st.write(f"**Generation Seed:** {st.session_state[Keys.SEED]}")

    prev_col1, prev_col2 = st.columns(2)
    with prev_col1:
        st.write("**Original Image**")
        st.image(st.session_state[Keys.ORIGINAL_IMAGE], use_container_width=True)
    with prev_col2:
        st.write("**Generated Mask**")
        if st.session_state[Keys.MASK_IMAGE]:
            st.image(st.session_state[Keys.MASK_IMAGE], use_container_width=True)


# --- Main Content Area ---
with st.container():
    st.title("VisionCommerce")
    st.markdown("Generate consistent, **studio-quality** backgrounds for your product images. Designed for e-commerce platforms like Shopee and Lazada.")

    render_progress_stepper(STAGES, int(st.session_state[Keys.STAGE]))

    st.divider()

    STAGE_RENDERERS: dict[int, Callable[[], None]] = {
        1: render_stage_upload,
        2: render_stage_analyze,
        3: render_stage_mask,
        4: render_stage_inpaint,
        5: render_stage_result,
    }
    STAGE_RENDERERS[int(st.session_state[Keys.STAGE])]()
