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
from src.p18_nano_banana_core_model.processes import run_sketch_to_photo
import importlib

try:
    _sic = importlib.import_module("streamlit_image_comparison")
    _image_comparison = getattr(_sic, "image_comparison", None)
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


st.set_page_config(page_title="2) Sketch → Photoreal", layout="wide")
st.title("2) Sketch → Photoreal")

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

sketch_prefix = st.sidebar.text_area(
    "Sketch Step: instruction prefix",
    value="Take this product and create a sketch/line-art of the following scene:",
)
photo_prompt = st.sidebar.text_area(
    "Photo Step: instruction",
    value=(
        "Make this sketch photorealistic. Keep exactly the same scene and layout; do not move any elements."
    ),
)
mask_feather = st.sidebar.slider(
    "Mask feather (px)", min_value=0, max_value=32, value=6
)

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
                    st.image(
                        item.packshot_path, caption="Packshot", use_container_width=True
                    )
                    if (
                        item.original_generation_path
                        and Path(item.original_generation_path).exists()
                    ):
                        st.image(
                            item.original_generation_path,
                            caption="Original generation (reference)",
                            use_container_width=True,
                        )

                with cols[1]:
                    st.subheader("Generation")
                    if st.button("Generate", type="primary"):
                        res = run_sketch_to_photo(
                            item,
                            prefer_rewritten=prefer_rewritten,
                            sketch_prompt_prefix=sketch_prefix,
                            photo_prompt=photo_prompt,
                            negative_prompt=negative_prompt or None,
                            mask_feather=int(mask_feather),
                        )
                        # Show initial canvas that is sent to the model at Step 1
                        if getattr(res, "initial_canvas_image", None) is not None:
                            st.image(
                                res.initial_canvas_image,
                                caption="Initial canvas (input to Step 1)",
                                use_container_width=True,
                            )
                        # Show intermediate sketch (step 1)
                        if getattr(res, "sketch_image", None) is not None:
                            st.image(
                                res.sketch_image,
                                caption="Sketch (step 1)",
                                use_container_width=True,
                            )
                        # Show step 2 Before/After (pre-repaste vs final)
                        if getattr(res, "pre_repaste_image", None) is not None:
                            show_comparison(
                                res.pre_repaste_image,
                                res.image,
                                label1="Before repaste",
                                label2="After repaste",
                                key=f"cmp_sketch_{item.stem}",
                            )
                        else:
                            st.image(
                                res.image,
                                caption="Result (after repaste)",
                                use_container_width=True,
                            )
                        if res.metadata.get("saved_dir"):
                            st.caption(f"Saved to: {res.metadata.get('saved_dir')}")
                        with st.expander("Prompts used"):
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
                    st.image(
                        item.packshot_path, caption="Packshot", use_container_width=True
                    )
                res = run_sketch_to_photo(
                    item,
                    prefer_rewritten=prefer_rewritten,
                    sketch_prompt_prefix=sketch_prefix,
                    photo_prompt=photo_prompt,
                    negative_prompt=negative_prompt or None,
                    mask_feather=int(mask_feather),
                )
                with cols[1]:
                    # Show initial canvas first
                    if getattr(res, "initial_canvas_image", None) is not None:
                        st.image(
                            res.initial_canvas_image,
                            caption="Initial canvas (input to Step 1)",
                            use_container_width=True,
                        )
                    # Show intermediate sketch first
                    if getattr(res, "sketch_image", None) is not None:
                        st.image(
                            res.sketch_image,
                            caption="Sketch (step 1)",
                            use_container_width=True,
                        )
                    # Then the comparison for step 2
                    if getattr(res, "pre_repaste_image", None) is not None:
                        show_comparison(
                            res.pre_repaste_image,
                            res.image,
                            label1="Before repaste",
                            label2="After repaste",
                            key=f"cmp_sketch_batch_{item.stem}",
                        )
                    else:
                        st.image(
                            res.image,
                            caption="Result (after repaste)",
                            use_container_width=True,
                        )
                    if res.metadata.get("saved_dir"):
                        st.caption(f"Saved to: {res.metadata.get('saved_dir')}")
                if (
                    item.original_generation_path
                    and Path(item.original_generation_path).exists()
                ):
                    with cols[2]:
                        st.image(
                            item.original_generation_path,
                            caption="Original generation",
                            use_container_width=True,
                        )
