import sys
from pathlib import Path

import streamlit as st


# Ensure src is importable when running `streamlit run Home.py`
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.p18_nano_banana_core_model.utils.data import find_dataset_items
from src.p18_nano_banana_core_model.genai_client import has_api_key


st.set_page_config(page_title="Nano Banana Pro — Outpaint R&D", layout="wide")

st.title("Nano Banana Pro — Outpaint R&D Harness")

if not has_api_key():
    st.warning(
        "GOOGLE_API_KEY not found or google-genai not available. The app will run in MOCK mode and display placeholder backgrounds instead of real generations.",
        icon="⚠️",
    )

st.markdown(
    "Use the pages in the left sidebar to explore different processes, tune parameters, and compare results side-by-side."
)

with st.expander("Dataset overview", expanded=True):
    items = find_dataset_items()
    st.write(f"Found {len(items)} dataset items in data/NBpro_TrainSet.")
    if items:
        st.write("First 10 stems:")
        st.code(
            ", ".join([it.stem for it in items[:10]]) + (" ..." if len(items) > 10 else "")
        )

st.markdown("""
Pages:
- 1) Simple Prompt — direct masked outpaint with anti-drift guardrails.
- 2) Sketch → Photoreal — generate a sketch, then convert to realistic without moving the product.
- 3) Masked Outpaint — more tunables for the mask and background.
- 9) Compare Processes — run all processes on one or many images and view side-by-side.
""")

st.info("Tip: prompts in 0_Prompts_original and 0_Prompts are auto-cleaned to remove [square bracket] notes.")
