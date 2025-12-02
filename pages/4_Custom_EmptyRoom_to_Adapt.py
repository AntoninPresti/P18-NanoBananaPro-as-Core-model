import random
import sys
from pathlib import Path

import streamlit as st

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.p18_nano_banana_core_model.utils.data import find_dataset_items
from src.p18_nano_banana_core_model.processes import run_custom_emptyroom_adapt

try:
    from streamlit_image_comparison import image_comparison as _image_comparison
except Exception:
    _image_comparison = None


def show_comparison(img1, img2, *, label1: str, label2: str, key: str):
    if _image_comparison is not None:
        _image_comparison(img1=img1, img2=img2, label1=label1, label2=label2)
    else:
        st.caption("Comparison widget not available — showing static images.")
        c1, c2 = st.columns(2)
        with c1:
            st.image(img1, caption=label1, use_container_width=True)
        with c2:
            st.image(img2, caption=label2, use_container_width=True)


st.set_page_config(page_title="4) Custom EmptyRoom → Adapt", layout="wide")
st.title("4) Custom EmptyRoom Draft → Gemini Adapt")

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
    batch_count = st.sidebar.number_input(
        "How many random items?",
        min_value=1,
        max_value=max(1, len(items)),
        value=min(4, len(items) or 1),
        step=1,
    )

prefer_rewritten = st.sidebar.checkbox("Use rewritten prompt (0_Prompts)", value=True)
negative_prompt = st.sidebar.text_input("Negative prompt (optional)", value="")
mask_feather = st.sidebar.slider("Mask feather (px)", min_value=0, max_value=32, value=6)


def run_and_show(item):
    with st.spinner("Generating (Custom model → Gemini adapt)..."):
        res = run_custom_emptyroom_adapt(
            item,
            prefer_rewritten=prefer_rewritten,
            negative_prompt=negative_prompt or None,
            mask_feather=int(mask_feather),
        )
    # Show initial canvas
    if getattr(res, "initial_canvas_image", None) is not None:
        st.image(
            res.initial_canvas_image,
            caption="Initial canvas (packshot placed)",
            use_container_width=True,
        )
    # Show intermediate (custom model)
    if getattr(res, "sketch_image", None) is not None:
        st.image(
            res.sketch_image,
            caption="Step 1 — Custom model (empty room draft)",
            use_container_width=True,
        )
    # Show Before/After: pre-repaste vs final
    if getattr(res, "pre_repaste_image", None) is not None:
        show_comparison(
            res.pre_repaste_image,
            res.image,
            label1="Before repaste",
            label2="After repaste",
            key=f"cmp_custom_{item.stem}",
        )
    else:
        st.image(res.image, caption="Result (after repaste)", use_container_width=True)
    if res.metadata.get("saved_dir"):
        st.caption(f"Saved to: {res.metadata.get('saved_dir')}")
    with st.expander("Prompts used"):
        st.code(res.prompt_used or "")


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
                        st.image(
                            item.original_generation_path,
                            caption="Original generation (reference)",
                            use_container_width=True,
                        )
                with cols[1]:
                    st.subheader("Generation")
                    if st.button("Generate", type="primary"):
                        run_and_show(item)
                    else:
                        st.info("Click Generate to run the process.")
    else:
        st.subheader("Batch Generation")
        if st.button("Generate Batch", type="primary"):
            sampled = random.sample(items, int(batch_count))
            for item in sampled:
                st.markdown(f"#### {item.stem}")
                cols = st.columns([1, 2])
                with cols[0]:
                    st.image(item.packshot_path, caption="Packshot", use_container_width=True)
                with cols[1]:
                    run_and_show(item)
