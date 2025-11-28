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
from src.p18_nano_banana_core_model.processes import (
    run_simple_prompt,
    run_sketch_to_photo,
    run_masked_outpaint,
)

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


st.set_page_config(page_title="9) Compare Processes", layout="wide")
st.title("9) Compare Processes — Side-by-side")

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
        value=min(3, len(items) or 1),
        step=1,
    )

prefer_rewritten = st.sidebar.checkbox("Use rewritten prompt (0_Prompts)", value=True)
negative_prompt = st.sidebar.text_input("Negative prompt (optional)", value="")
seed = st.sidebar.number_input(
    "Seed (optional)", min_value=0, max_value=2_147_483_647, value=0, step=1
)
use_seed = st.sidebar.checkbox("Use seed", value=False)

# Process-specific knobs
mask_feather = st.sidebar.slider(
    "Mask feather (for masked/sketch processes)", min_value=0, max_value=64, value=8
)
extra_guardrails = st.sidebar.text_area("Simple Prompt: extra instructions", value="")
sketch_prefix = st.sidebar.text_input(
    "Sketch Step: instruction prefix",
    value="Take this product and create a sketch/line-art of the following scene:",
)
sketch_guardrails = st.sidebar.text_input(
    "Sketch Step: guardrails",
    value="In the sketch, the product must remain at exactly the same position within the square I sent.",
)
photo_prompt = st.sidebar.text_input(
    "Photo Step: instruction",
    value="Make this sketch photorealistic. Keep exactly the same scene and layout; do not move any elements.",
)


def run_all(item):
    res_simple = run_simple_prompt(
        item,
        prefer_rewritten=prefer_rewritten,
        extra_guardrails=extra_guardrails or None,
        negative_prompt=negative_prompt or None,
        seed=int(seed) if use_seed else None,
    )
    res_sketch = run_sketch_to_photo(
        item,
        prefer_rewritten=prefer_rewritten,
        sketch_prompt_prefix=sketch_prefix,
        photo_prompt=photo_prompt,
        negative_prompt=negative_prompt or None,
        seed=int(seed) if use_seed else None,
        mask_feather=int(mask_feather),
    )
    res_masked = run_masked_outpaint(
        item,
        prefer_rewritten=prefer_rewritten,
        feather=int(mask_feather),
        invert_mask=False,
        background_fill_rgba=None,
        negative_prompt=negative_prompt or None,
        seed=int(seed) if use_seed else None,
        extra_instruction=None,
    )
    return res_simple, res_sketch, res_masked


if not items:
    st.warning("No dataset items found.")
else:
    if mode == "Single":
        if items and choice:
            item = next((it for it in items if it.stem == choice), None)
            if item is not None:
                st.subheader(f"Item: {item.stem}")
                if st.button("Generate All", type="primary"):
                    res_simple, res_sketch, res_masked = run_all(item)
                    cols = st.columns([1, 1, 1, 1, 1])
                    with cols[0]:
                        st.image(
                            item.packshot_path,
                            caption="Packshot",
                            use_container_width=True,
                        )
                    with cols[1]:
                        if getattr(res_simple, "pre_repaste_image", None) is not None:
                            show_comparison(
                                res_simple.pre_repaste_image,
                                res_simple.image,
                                label1="Before repaste",
                                label2="After repaste",
                                key=f"cmp_compare_simple_{item.stem}",
                            )
                        else:
                            st.image(
                                res_simple.image,
                                caption="Simple Prompt (after repaste)",
                                use_container_width=True,
                            )
                    with cols[2]:
                        if getattr(res_sketch, "pre_repaste_image", None) is not None:
                            show_comparison(
                                res_sketch.pre_repaste_image,
                                res_sketch.image,
                                label1="Before repaste",
                                label2="After repaste",
                                key=f"cmp_compare_sketch_{item.stem}",
                            )
                        else:
                            st.image(
                                res_sketch.image,
                                caption="Sketch→Photo (after repaste)",
                                use_container_width=True,
                            )
                    with cols[3]:
                        if getattr(res_masked, "pre_repaste_image", None) is not None:
                            show_comparison(
                                res_masked.pre_repaste_image,
                                res_masked.image,
                                label1="Before repaste",
                                label2="After repaste",
                                key=f"cmp_compare_masked_{item.stem}",
                            )
                        else:
                            st.image(
                                res_masked.image,
                                caption="Masked Outpaint (after repaste)",
                                use_container_width=True,
                            )
                    with cols[4]:
                        if (
                            item.original_generation_path
                            and Path(item.original_generation_path).exists()
                        ):
                            st.image(
                                item.original_generation_path,
                                caption="Original ref",
                                use_container_width=True,
                            )
                        else:
                            st.empty()
                    with st.expander("Prompts used"):
                        st.markdown("### Simple prompt")
                        st.code(res_simple.prompt_used or "")
                        st.markdown("### Sketch→Photo prompts")
                        st.code(res_sketch.prompt_used or "")
                        st.markdown("### Masked outpaint prompt")
                        st.code(res_masked.prompt_used or "")
                else:
                    st.info(
                        "Click 'Generate All' to run all processes on the selected item."
                    )
    else:
        st.subheader("Batch Compare")
        if st.button("Generate Batch", type="primary"):
            sampled = random.sample(items, int(batch_count))
            for item in sampled:
                st.markdown(f"#### {item.stem}")
                res_simple, res_sketch, res_masked = run_all(item)
                cols = st.columns([1, 1, 1, 1, 1])
                with cols[0]:
                    st.image(
                        item.packshot_path, caption="Packshot", use_container_width=True
                    )
                with cols[1]:
                    if getattr(res_simple, "pre_repaste_image", None) is not None:
                        show_comparison(
                            res_simple.pre_repaste_image,
                            res_simple.image,
                            label1="Before repaste",
                            label2="After repaste",
                            key=f"cmp_compare_simple_batch_{item.stem}",
                        )
                    else:
                        st.image(
                            res_simple.image,
                            caption="Simple (after repaste)",
                            use_container_width=True,
                        )
                with cols[2]:
                    if getattr(res_sketch, "pre_repaste_image", None) is not None:
                        show_comparison(
                            res_sketch.pre_repaste_image,
                            res_sketch.image,
                            label1="Before repaste",
                            label2="After repaste",
                            key=f"cmp_compare_sketch_batch_{item.stem}",
                        )
                    else:
                        st.image(
                            res_sketch.image,
                            caption="Sketch→Photo (after repaste)",
                            use_container_width=True,
                        )
                with cols[3]:
                    if getattr(res_masked, "pre_repaste_image", None) is not None:
                        show_comparison(
                            res_masked.pre_repaste_image,
                            res_masked.image,
                            label1="Before repaste",
                            label2="After repaste",
                            key=f"cmp_compare_masked_batch_{item.stem}",
                        )
                    else:
                        st.image(
                            res_masked.image,
                            caption="Masked Outpaint (after repaste)",
                            use_container_width=True,
                        )
                with cols[4]:
                    if (
                        item.original_generation_path
                        and Path(item.original_generation_path).exists()
                    ):
                        st.image(
                            item.original_generation_path,
                            caption="Original ref",
                            use_container_width=True,
                        )
                    else:
                        st.empty()
