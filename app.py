import gc
import streamlit as st
from PIL import Image
from avm.inpainting import (
    CATEGORY_PROMPTS,
    DEFAULT_MODEL_ID,
    MODELS,
    build_model,
    detect_device,
    get_prompt,
    inpaint_background,
    get_model_config_defaults,
)

@st.cache_resource(show_spinner=False)
def get_cached_model(model_id: str, device: str):
    return build_model(model_id, device)

def clear_model_resources() -> None:
    get_cached_model.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

DEFAULT_MODEL_INDEX = next(
    (index for index, model_id in enumerate(MODELS.values()) if model_id == DEFAULT_MODEL_ID),
    0,
)

st.set_page_config(page_title="AVM Web Demo", layout="wide")
st.title("AVM Web Demo")
st.caption("Generate consistent, studio-quality merchandising backgrounds for your product images on Shopee and Lazada.")

with st.sidebar:
    category_options = list(CATEGORY_PROMPTS.keys())
    default_category_index = 0
    category = st.selectbox(
        "Product Category",
        options=category_options,
        index=default_category_index,
    )
    seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)

    with st.expander("Advanced Options", expanded=False):
        selected_model_label = st.selectbox(
            "Model",
            options=list(MODELS.keys()),
            index=DEFAULT_MODEL_INDEX,
        )
        selected_model_lookup = {label: model_id for label, model_id in MODELS.items()}
        selected_model_id = selected_model_lookup[selected_model_label]
        prompt = st.text_area(
            "Prompt",
            value=get_prompt(category.strip()),
            height=120,
        )
        config_defaults = get_model_config_defaults(selected_model_id)

        config: dict[str, int | float] = {}
        if 'strength' in config_defaults:
            config['strength'] = st.slider(
                "Strength",
                min_value=0.0,
                max_value=1.0,
                value=float(config_defaults['strength']),
                step=0.05,
            )
        if 'guidance' in config_defaults:
            config['guidance'] = st.slider(
                "Guidance",
                min_value=1.0,
                max_value=20.0,
                value=float(config_defaults['guidance']),
                step=0.5,
            )
        if 'guidance_scale' in config_defaults:
            config['guidance_scale'] = st.slider(
                "Guidance Scale",
                min_value=1.0,
                max_value=20.0,
                value=float(config_defaults['guidance_scale']),
                step=0.5,
            )
        if 'num_inference_steps' in config_defaults:
            config['num_inference_steps'] = st.slider(
                "Inference Steps",
                min_value=1,
                max_value=80,
                value=int(config_defaults['num_inference_steps']),
                step=1,
            )

image_file = st.file_uploader("Background image", type=["png", "jpg", "jpeg"])
mask_file = st.file_uploader("Mask image", type=["png", "jpg", "jpeg"])
st.caption("Upload an RGBA mask with alpha channel. Transparent areas are generated; opaque areas are kept.")

if image_file and mask_file:
    input_image = Image.open(image_file).convert("RGB")
    mask_image = Image.open(mask_file).convert("RGBA")
    rendered_prompt = prompt.strip() or get_prompt(category.strip())

    col1, col2 = st.columns(2)
    with col1:
        st.image(input_image, caption="Background")
    with col2:
        st.image(mask_image, caption="Mask")

    if st.button("Generate", type="primary"):
        with st.spinner(f"Loading {selected_model_label} and generating..."):
            device_used = detect_device()
            model_key = (selected_model_id.strip(), device_used)
            previous_model_key = st.session_state.get("loaded_model_key")
            if previous_model_key is not None and previous_model_key != model_key:
                clear_model_resources()

            model = get_cached_model(*model_key)
            st.session_state["loaded_model_key"] = model_key
            output_image = inpaint_background(
                model,
                input_image,
                mask_image,
                rendered_prompt,
                seed=seed,
                config=config
            )

        st.success(f"Device: {device_used}")
        st.text_area("Prompt used", value=prompt, height=70)
        st.image(output_image, caption="Generated image")
else:
    st.info("Upload both a background image and a mask image to start.")
