import sys
import random
from pathlib import Path

import streamlit as st

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.p18_nano_banana_core_model.utils.data import find_dataset_items
from src.p18_nano_banana_core_model.processes import run_masked_outpaint
from src.p18_nano_banana_core_model.types import OutpaintModel


st.set_page_config(page_title="3) Masked Outpaint", layout="wide")
st.title("3) Masked Outpaint — Advanced Controls")

items = find_dataset_items()
st.sidebar.header("Dataset selection")
st.sidebar.write(f"Items available: {len(items)}")

mode = st.sidebar.radio("Mode", ["Single", "Random N"], horizontal=True)
stems = [it.stem for it in items]
choice = None
batch_count = 0
if mode == "Single":
    choice = st.sidebar.selectbox("Choose one item", stems if stems else ["<none>"])
else:
    batch_count = st.sidebar.number_input("How many random items?", min_value=1, max_value=max(1, len(items)), value=min(4, len(items) or 1), step=1)

prefer_rewritten = st.sidebar.checkbox("Use rewritten prompt (0_Prompts)", value=True)
extra_instruction = st.sidebar.text_area("Extra instruction (optional)", value="")
negative_prompt = st.sidebar.text_input("Negative prompt (optional)", value="")
seed = st.sidebar.number_input("Seed (optional)", min_value=0, max_value=2_147_483_647, value=0, step=1)
use_seed = st.sidebar.checkbox("Use seed", value=False)

feather = st.sidebar.slider("Mask feather (px)", min_value=0, max_value=64, value=8)
invert_mask = st.sidebar.checkbox("Invert mask (advanced)", value=False)
bg_transparent = st.sidebar.checkbox("Transparent background (canvas)", value=True)

model = OutpaintModel.IMAGEN_EDIT

if not items:
    st.warning("No dataset items found.")
else:
    if mode == "Single":
        cols = st.columns([1, 1])
        if items and choice:
            item = next((it for it in items if it.stem == choice), None)
            if item is not None:
                with cols[0]:
                    st.subheader("Inputs")
                    st.write(f"Stem: {item.stem}")
                    st.image(item.packshot_path, caption="Packshot", use_container_width=True)
                    if item.original_generation_path and Path(item.original_generation_path).exists():
                        st.image(item.original_generation_path, caption="Original generation (reference)", use_container_width=True)

                with cols[1]:
                    st.subheader("Generation")
                    if st.button("Generate", type="primary"):
                        res = run_masked_outpaint(
                            item,
                            prefer_rewritten=prefer_rewritten,
                            feather=int(feather),
                            invert_mask=bool(invert_mask),
                            background_fill_rgba=None if bg_transparent else (255, 255, 255, 255),
                            negative_prompt=negative_prompt or None,
                            model=model,
                            seed=int(seed) if use_seed else None,
                            extra_instruction=extra_instruction or None,
                        )
                        st.image(res.image, caption="Result", use_container_width=True)
                        with st.expander("Prompt used"):
                            st.code(res.prompt_used or "")
                    else:
                        st.info("Click Generate to run the process.")
    else:
        st.subheader("Batch Generation")
        if st.button("Generate Batch", type="primary"):
            sampled = random.sample(items, int(batch_count))
            for item in sampled:
                st.markdown(f"#### {item.stem}")
                cols = st.columns([1, 1, 1])
                with cols[0]:
                    st.image(item.packshot_path, caption="Packshot", use_container_width=True)
                res = run_masked_outpaint(
                    item,
                    prefer_rewritten=prefer_rewritten,
                    feather=int(feather),
                    invert_mask=bool(invert_mask),
                    background_fill_rgba=None if bg_transparent else (255, 255, 255, 255),
                    negative_prompt=negative_prompt or None,
                    model=model,
                    seed=int(seed) if use_seed else None,
                    extra_instruction=extra_instruction or None,
                )
                with cols[1]:
                    st.image(res.image, caption="Result", use_container_width=True)
                if item.original_generation_path and Path(item.original_generation_path).exists():
                    with cols[2]:
                        st.image(item.original_generation_path, caption="Original generation", use_container_width=True)
