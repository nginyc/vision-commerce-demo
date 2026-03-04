import gc
from typing import Final, TypeAlias

import streamlit as st
from PIL import Image

# Inpaint imports
from avm.inpaint import (
    CATEGORY_PROMPTS,
    DEFAULT_MODEL_ID as INPAINT_DEFAULT_MODEL_ID,
    MODELS as INPAINT_MODELS,
    build_model as build_inpaint_model,
    detect_device,
    get_prompt,
    inpaint_background,
    get_model_config_defaults as get_inpaint_config_defaults,
)

# Segment imports
from avm.segment import (
    DEFAULT_MODEL_ID as SEGMENT_DEFAULT_MODEL_ID,
    MODELS as SEGMENT_MODELS,
    build_model as build_segment_model,
    get_model_config_defaults as get_segment_config_defaults,
    normalize_category,
    create_mask_image,
)

@st.cache_resource(show_spinner=False) # type: ignore
def get_cached_inpaint_model(model_id: str, device: str):
    return build_inpaint_model(model_id, device)

@st.cache_resource(show_spinner=False) # type: ignore
def get_cached_segment_model(model_id: str, device: str):
    return build_segment_model(model_id, device)

def clear_model_resources() -> None:
    get_cached_inpaint_model.clear()
    get_cached_segment_model.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

INPAINT_DEFAULT_MODEL_INDEX = next(
    (index for index, model_id in enumerate(INPAINT_MODELS.values()) if model_id == INPAINT_DEFAULT_MODEL_ID),
    0,
)

SEGMENT_DEFAULT_MODEL_INDEX = next(
    (index for index, model_id in enumerate(SEGMENT_MODELS.values()) if model_id == SEGMENT_DEFAULT_MODEL_ID),
    0,
)

st.set_page_config(page_title="AVM Web Demo", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for premium look
st.markdown("""
<style>
    /* Limit the max width of the main container */
    [data-testid="block-container"] {
        max-width: 1000px !important;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    /* Add subtle shadow to images */
    [data-testid="stImage"] img {
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    /* Refine typography */
    h1, h2, h3 {
        font-weight: 600 !important;
        color: #111827;
    }
    /* Mute unselected step indicators */
    .avm-stepper {
        position: relative;
        margin-top: 1.25rem;
        margin-bottom: 1rem;
        padding-top: 0.25rem;
    }
    .avm-stepper-line,
    .avm-stepper-line-fill {
        position: absolute;
        top: 18px;
        height: 2px;
        border-radius: 999px;
        left: 6%;
        width: 88%;
    }
    .avm-stepper-line {
        background: #e5e7eb;
    }
    .avm-stepper-line-fill {
        background: #ef4444;
        transition: width 0.2s ease;
    }
    .avm-stepper-grid {
        position: relative;
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
    }
    .avm-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        gap: 0.5rem;
    }
    .avm-step-dot {
        width: 2.2rem;
        height: 2.2rem;
        border-radius: 999px;
        border: 2px solid #d1d5db;
        background: #ffffff;
        color: #6b7280;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.95rem;
        line-height: 1;
    }
    .avm-step-label {
        color: #6b7280;
        font-weight: 600;
        font-size: 1.05rem;
    }
    .avm-step.done .avm-step-dot {
        background: #ef4444;
        border-color: #ef4444;
        color: #ffffff;
    }
    .avm-step.done .avm-step-label,
    .avm-step.active .avm-step-label {
        color: #ef4444;
        font-weight: 700;
    }
    .avm-step.active .avm-step-dot {
        border-color: #ef4444;
        color: #ef4444;
        box-shadow: 0 0 0 4px rgba(239, 68, 68, 0.15);
    }
</style>
""", unsafe_allow_html=True)

ConfigValue: TypeAlias = float | int
ModelConfig: TypeAlias = dict[str, ConfigValue]
SessionStateDefaults: TypeAlias = dict[str, object]
SESSION_RESET_KEYS: Final[tuple[str, ...]] = (
    "stage",
    "original_image",
    "mask_image",
    "category",
    "seed",
    "output_image",
    "is_generating_mask",
    "is_generating_background",
    "mask_generation_error",
)
STAGES: Final[tuple[str, ...]] = (
    "Upload Image",
    "Mask Product",
    "Generate Background",
    "Final Result"
)

# --- Session State Initialization ---
default_state: SessionStateDefaults = {
    "stage": 1,
    "original_image": None,
    "mask_image": None,
    "category": list(CATEGORY_PROMPTS.keys())[0],
    "seed": 42,
    "is_generating_mask": False,
    "is_generating_background": False,
    "mask_generation_error": None,
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
    st.session_state.stage = stage


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
                key=f"{key_prefix}_{key}"
            )
    return config

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Generation Settings")
    st.caption("Advanced controls for mask and background generation.")
    
    st.session_state.seed = int(
        st.number_input(
            "Generation Seed",
            min_value=0,
            max_value=999999,
            value=int(st.session_state.get("seed", 42)),
            step=1,
            key="sidebar_seed",
        )
    )
    
    with st.expander("Masking Settings", expanded=False):
        selected_segment_model_label = st.selectbox(
            "Segmentation Model",
            options=list(SEGMENT_MODELS.keys()),
            index=SEGMENT_DEFAULT_MODEL_INDEX,
            disabled=True,
            help="Only SAM3 is available for mask generation at this time."
        )
        selected_segment_model_id = SEGMENT_MODELS[selected_segment_model_label]
        segment_config_defaults = get_segment_config_defaults(selected_segment_model_id)
        segment_config = render_config_sliders(segment_config_defaults, "segment")
        
    with st.expander("Background Settings", expanded=False):
        selected_inpaint_model_label = st.selectbox(
            "Inpainting Model",
            options=list(INPAINT_MODELS.keys()),
            index=INPAINT_DEFAULT_MODEL_INDEX,
        )
        selected_inpaint_model_id = INPAINT_MODELS[selected_inpaint_model_label]
        inpaint_config_defaults = get_inpaint_config_defaults(selected_inpaint_model_id)
        inpaint_config = render_config_sliders(inpaint_config_defaults, "inpaint")
        
    st.divider()
    if st.button("↻ Start Over", use_container_width=True, type="secondary"):
        reset_session_state()
        st.rerun()

# --- Main Content Area ---
# Using CSS max-width instead of column wrapping
main_col = st.container()

with main_col:
    st.title("Autonomous Visual Merchandiser ✨")
    st.markdown("Generate consistent, **studio-quality** backgrounds for your product images. Designed for e-commerce platforms like Shopee and Lazada.")
    
    # Progress visualizer
    stages = STAGES
    render_progress_stepper(stages, int(st.session_state.stage))
            
    st.divider()

    # ==========================================
    # STAGE 1: Upload Original Image
    # ==========================================
    if st.session_state.stage == 1:
        st.header("Upload Product Image")
        st.write("Start by uploading the original image of your product. For best results, use a clear photo of the product on a simple background.")
            
        up_col1, up_col2 = st.columns([1, 1], gap="large")
        with up_col1:
            image_file = st.file_uploader("Upload Product Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        
        if image_file:
            st.session_state.original_image = Image.open(image_file).convert("RGB")
            with up_col2:
                st.image(st.session_state.original_image, caption="Uploaded Image", use_container_width=True)
                
                st.write("") # Spacing
                if st.button("Next: Profile & Mask ✨", type="primary", use_container_width=True):
                    set_stage(2)
                    st.rerun()
                
        else:
            st.info("👆 Please upload an original product image to continue.")

    # ==========================================
    # STAGE 2: Generate Mask
    # ==========================================
    elif st.session_state.stage == 2:
        st.header("Profile Product")
        st.write("Select the product category to automatically map and separate the product from its background.")
        
        img_col, info_col = st.columns([1, 1], gap="large")
        with img_col:
            st.image(st.session_state.original_image, caption="Original Image", use_container_width=True)
        
        with info_col:
            category_options = list(CATEGORY_PROMPTS.keys())
            default_idx = category_options.index(st.session_state.category) if st.session_state.category in category_options else 0
            category = st.selectbox(
                "Product Category",
                options=category_options,
                index=default_idx,
                key="stage2_category",
                help="Choose the category that best describes your product for accurate AI masking."
            )
            st.session_state.category = category
            
            st.markdown("---")
            st.markdown("Ready to separate your product from the background using the selected category.")

            mask_error = st.session_state.get("mask_generation_error")
            if mask_error:
                st.error(str(mask_error))
                st.session_state.mask_generation_error = None
            
            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                if st.button("← Back", use_container_width=True):
                    set_stage(1)
                    st.rerun()
            with btn_col2:
                is_generating_mask = bool(st.session_state.get("is_generating_mask", False))
                mask_button_clicked = st.button(
                    "Generating Mask..." if is_generating_mask else "Generate Mask ✨",
                    type="primary",
                    use_container_width=True,
                    disabled=is_generating_mask,
                    key="generate_mask_button",
                )

            if mask_button_clicked and not is_generating_mask:
                st.session_state.is_generating_mask = True
                st.rerun()

            if is_generating_mask:
                try:
                    device_used = detect_device()
                    model_key = (selected_segment_model_id.strip(), device_used)

                    previous_model_key = st.session_state.get("loaded_segment_model_key")
                    if previous_model_key is not None and previous_model_key != model_key:
                        clear_model_resources()

                    model = get_cached_segment_model(*model_key)
                    st.session_state["loaded_segment_model_key"] = model_key

                    mask = model.generate_mask(
                        image=st.session_state.original_image,
                        prompt=normalize_category(category),
                        config=segment_config,
                    )

                    if mask is not None:
                        mask_img = create_mask_image(image=st.session_state.original_image, mask=mask)
                        st.session_state.mask_image = mask_img
                        st.session_state.is_generating_mask = False
                        set_stage(3)
                        st.rerun()

                    st.session_state.mask_generation_error = (
                        "Failed to generate mask. Try adjusting thresholds in the sidebar or change category."
                    )
                finally:
                    st.session_state.is_generating_mask = False

                st.rerun()

    # ==========================================
    # STAGE 3: Background Generation
    # ==========================================
    elif st.session_state.stage == 3:
        st.header("Generate Background")
        st.write("Verify the mask and customize the background prompt before creating the final image.")
        
        content_col1, content_col2 = st.columns([1, 1], gap="large")
        
        with content_col1:
            st.subheader("Image Previews")
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.write("**Original**")
                st.image(st.session_state.original_image, use_container_width=True)
            with img_col2:
                st.write("**Mask**")
                if st.session_state.mask_image:
                    st.image(st.session_state.mask_image, use_container_width=True)
                    
            st.write("") # Spacing
            if st.button("← Back to Profile", use_container_width=True):
                st.session_state.pop("output_image", None)
                set_stage(2)
                st.rerun()
                
        with content_col2:
            st.subheader("Final Touches")
            prompt = st.text_area(
                "Background Prompt",
                value=get_prompt(st.session_state.category.strip()),
                height=120,
                help="Describe the exact background you want for your product. You can update this based on the required aesthetics."
            )
            
            with st.expander("Upload Override Mask (Optional)"):
                st.caption("Upload an RGBA mask to override the AI-generated mask. Transparent areas become the background; opaque areas are kept as product.")
                override_mask_file = st.file_uploader("Override Mask image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
                
                final_mask_image = st.session_state.mask_image
                if override_mask_file:
                    final_mask_image = Image.open(override_mask_file).convert("RGBA")
                    st.image(final_mask_image, caption="Overridden Mask", width=300)
            
            st.write("") # Spacing
            is_generating_background = bool(st.session_state.get("is_generating_background", False))
            gen_pressed = st.button(
                "Generating Background..." if is_generating_background else "Generate Background 🎨",
                type="primary",
                use_container_width=True,
                disabled=is_generating_background,
                key="generate_background_button",
            )

        if gen_pressed and not is_generating_background:
            st.session_state.is_generating_background = True
            st.rerun()

        if is_generating_background:
            device_used = detect_device()
            model_key = (selected_inpaint_model_id.strip(), device_used)

            previous_model_key = st.session_state.get("loaded_inpaint_model_key")
            if previous_model_key is not None and previous_model_key != model_key:
                clear_model_resources()

            clear_model_resources() # Free memory

            model = get_cached_inpaint_model(*model_key)
            st.session_state["loaded_inpaint_model_key"] = model_key

            st.session_state.output_image = inpaint_background(
                model,
                st.session_state.original_image,
                final_mask_image,
                prompt.strip(),
                seed=st.session_state.seed,
                config=inpaint_config
            )

            st.session_state.is_generating_background = False
            set_stage(4)
            st.rerun()

    # ==========================================
    # STAGE 4: Final Result
    # ==========================================
    elif st.session_state.stage == 4:
        st.header("Final Result")
        st.write("Your product image is ready with the generated background.")
        if st.session_state.get("output_image") is not None:
            _, res_col, _ = st.columns([1, 4, 1])
            with res_col:
                st.image(st.session_state.output_image, use_container_width=True)
                
            st.divider()
            
            # Action Buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                st.caption("Tip: Right-click the image to save it to your device.")
            with btn_col2:
                if st.button("← Back to Prompt & Mask", use_container_width=True):
                    set_stage(3)
                    st.rerun()
            with btn_col3:
                if st.button("↻ Generate Another", type="primary", use_container_width=True):
                    reset_session_state()
                    st.rerun()
            
            st.write("") # Spacing

            st.subheader("Inputs Used")
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.write(f"**Product Category:** {st.session_state.category}")
            with meta_col2:
                st.write(f"**Generation Seed:** {st.session_state.seed}")

            prev_col1, prev_col2 = st.columns(2)
            with prev_col1:
                st.write("**Original Image**")
                st.image(st.session_state.original_image, use_container_width=True)
            with prev_col2:
                st.write("**Generated Mask**")
                if st.session_state.mask_image:
                    st.image(st.session_state.mask_image, use_container_width=True)
        else:
            st.error("No image found. Please go back and generate one.")
            if st.button("← Back to Settings"):
                set_stage(3)
                st.rerun()
