import sys
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from PIL import Image

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.p18_nano_banana_core_model.utils.data import find_dataset_items, get_prompt_for_item
from src.p18_nano_banana_core_model.processes import (
    run_simple_prompt,
    run_sketch_to_photo,
)

# Try to import the image comparison widget
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


RUNS_ROOT = Path("data/NBpro_TrainSet/runs")
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="10) Movement Detection Run", layout="wide")
st.title("10) Movement Detection — Multi-run Orchestrator")

items = find_dataset_items()
st.sidebar.header("Run & options")
st.sidebar.write(f"Dataset items: {len(items)}")

# Choose or create a run
existing_runs = sorted([p.name for p in RUNS_ROOT.iterdir() if p.is_dir()])
mode = st.sidebar.radio("Run mode", ["Create new", "Load existing"], horizontal=True)
if mode == "Create new":
    run_name = st.sidebar.text_input("New run name", value="movement_run")
else:
    run_name = st.sidebar.selectbox(
        "Select existing run", existing_runs if existing_runs else ["<none>"]
    )
if not run_name:
    st.stop()

run_dir = RUNS_ROOT / run_name
run_dir.mkdir(parents=True, exist_ok=True)
def _list_runs_for(item_stem: str, process_name: str) -> List[Path]:
    """Return list of timestamp subfolders for a given item/process in the current run."""
    base = run_dir / process_name / item_stem
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def _load_images_from_run_dir(dir_path: Path) -> Tuple[Image.Image | None, Image.Image | None]:
    pre = dir_path / "before_repaste.png"
    fin = dir_path / "final.png"
    pre_img = Image.open(pre).convert("RGBA") if pre.exists() else None
    fin_img = Image.open(fin).convert("RGBA") if fin.exists() else None
    return pre_img, fin_img


def _ensure_n_results_for_item(
    item,
    base_dir: Path,
    *,
    prefer_rewritten: bool,
    negative_prompt: str | None,
    mask_feather: int,
    target_n: int,
):
    # Process 1: simple_prompt
    sp_dirs = _list_runs_for(item.stem, "simple_prompt")
    missing = max(0, target_n - len(sp_dirs))
    for _ in range(missing):
        run_simple_prompt(
            item,
            prefer_rewritten=prefer_rewritten,
            negative_prompt=negative_prompt,
            save_base_dir=str(base_dir),
        )
    # Process 2: sketch_to_photo
    s2_dirs = _list_runs_for(item.stem, "sketch_to_photo")
    missing2 = max(0, target_n - len(s2_dirs))
    for _ in range(missing2):
        run_sketch_to_photo(
            item,
            prefer_rewritten=prefer_rewritten,
            photo_prompt=(
                "Make this sketch photorealistic. Keep exactly the same scene and layout; do not move any elements."
            ),
            negative_prompt=negative_prompt,
            mask_feather=mask_feather,
            save_base_dir=str(base_dir),
        )


if not items:
    st.warning("No dataset items found.")
    st.stop()

# Options (placed after helper definitions)
st.sidebar.caption(f"Run dir: {run_dir}")
prefer_rewritten = st.sidebar.checkbox("Use rewritten prompt (0_Prompts)", value=True)
mask_feather = st.sidebar.slider(
    "Mask feather (for Sketch→Photo)", min_value=0, max_value=32, value=6
)
negative_prompt = st.sidebar.text_input("Negative prompt (optional)", value="")

shots_per_process = st.sidebar.number_input(
    "Generations per process", min_value=1, max_value=8, value=4, step=1
)

cols_top = st.columns([1, 1])
with cols_top[0]:
    if st.button("Generate missing for ALL packshots", type="primary"):
        with st.spinner("Generating missing runs for all items…"):
            for it in items:
                _ensure_n_results_for_item(
                    it,
                    run_dir,
                    prefer_rewritten=prefer_rewritten,
                    negative_prompt=negative_prompt or None,
                    mask_feather=int(mask_feather),
                    target_n=int(shots_per_process),
                )
with cols_top[1]:
    st.caption(
        "This tool saves all outputs into the selected run directory and can resume a run by loading existing generations."
    )

for item in items:
    # Heading and prompt caption
    st.markdown(f"## {item.stem}")
    prompt_text = get_prompt_for_item(item, prefer_rewritten=prefer_rewritten)
    st.caption(prompt_text)

    # Ensure we have up to N results already, but do not force-generate unless user clicked the button
    # Offer a per-item button too
    if st.button(f"Generate missing for {item.stem}"):
        with st.spinner(f"Generating for {item.stem}…"):
            _ensure_n_results_for_item(
                item,
                run_dir,
                prefer_rewritten=prefer_rewritten,
                negative_prompt=negative_prompt or None,
                mask_feather=int(mask_feather),
                target_n=int(shots_per_process),
            )

    # Reload directory listings after potential generation
    sp_dirs = _list_runs_for(item.stem, "simple_prompt")[-int(shots_per_process) :]
    s2_dirs = _list_runs_for(item.stem, "sketch_to_photo")[-int(shots_per_process) :]

    # Display 4 comparisons for process 1
    st.markdown("#### Process 1 — Simple Prompt")
    if not sp_dirs:
        st.info("No runs yet for Simple Prompt; click the 'Generate missing' button above.")
    else:
        row_cols = st.columns(len(sp_dirs))
        for col, d in zip(row_cols, sp_dirs):
            with col:
                pre, fin = _load_images_from_run_dir(d)
                if pre is not None and fin is not None:
                    show_comparison(
                        pre,
                        fin,
                        label1="Before repaste",
                        label2="After repaste",
                        key=f"cmp_run_sp_{item.stem}_{d.name}",
                    )
                elif fin is not None:
                    st.image(fin, caption="After repaste", use_container_width=True)
                else:
                    st.empty()

    # Display 4 comparisons for process 2
    st.markdown("#### Process 2 — Sketch → Photoreal")
    if not s2_dirs:
        st.info("No runs yet for Sketch→Photo; click the 'Generate missing' button above.")
    else:
        row_cols = st.columns(len(s2_dirs))
        for col, d in zip(row_cols, s2_dirs):
            with col:
                pre, fin = _load_images_from_run_dir(d)
                if pre is not None and fin is not None:
                    show_comparison(
                        pre,
                        fin,
                        label1="Before repaste",
                        label2="After repaste",
                        key=f"cmp_run_s2_{item.stem}_{d.name}",
                    )
                elif fin is not None:
                    st.image(fin, caption="After repaste", use_container_width=True)
                else:
                    st.empty()

    st.divider()
    st.markdown("## Next packshot")
