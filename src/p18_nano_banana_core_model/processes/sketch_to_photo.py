from __future__ import annotations

from typing import Optional

from PIL import Image

from ..genai_client import edit_image
from ..image_utils import ensure_size, load_image, make_outpaint_mask, paste_packshot
from ..types import DatasetItem, ProcessResult
from ..utils.data import get_prompt_for_item
from ..utils.prompt import add_do_not_move_guardrails, strip_square_brackets
from ..utils.saver import save_generation


def run_sketch_to_photo(
    item: DatasetItem,
    *,
    prefer_rewritten: bool = True,
    sketch_prompt_prefix: str = (
        "Fill this image with a sketch/line-art of the following scene:\n"
    ),
    photo_prompt: str = (
        "Make this sketch photorealistic. Keep exactly the same scene and layout; do not move any elements."
    ),
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    mask_feather: int = 6,
) -> ProcessResult:
    """Process 2 — Two-step: sketch around fixed product, then photorealistic render.

    Both steps use the same protected packshot mask. The second step converts
    the first step's background to a realistic rendering without moving elements.
    """
    base_prompt = get_prompt_for_item(item, prefer_rewritten=prefer_rewritten)
    base_prompt = strip_square_brackets(base_prompt)
    # Build Step 1 prompt using the SAME guardrail helper as the Simple Prompt process
    step1_core = f"{sketch_prompt_prefix.strip()} {base_prompt}"
    step1_prompt = add_do_not_move_guardrails(step1_core)

    cfg = item.config
    size = (int(cfg.width or 1024), int(cfg.height or 1024))
    pack_xy = (cfg.packshot_top_left_pos_x, cfg.packshot_top_left_pos_y)
    pack_wh = (cfg.packshot_width, cfg.packshot_height)

    # Compose base canvas (packshot on transparent)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    pack = load_image(cfg.packshot_url or item.packshot_path)
    pack = ensure_size(pack, pack_wh)
    canvas.paste(pack, pack_xy, pack)

    mask = make_outpaint_mask(
        size, pack_xy, pack_wh, feather=mask_feather, invert=False
    )

    # Step 1: Sketch background generation
    sketch_img = edit_image(
        base_image=canvas,
        mask=mask,
        prompt=step1_prompt,
        size=size,
        seed=seed if seed is not None else cfg.seed,
        negative_prompt=negative_prompt or cfg.negative_prompt,
    )

    # Step 2: Photorealistic rendering from sketch
    step2_prompt = photo_prompt.strip()
    photo_img = edit_image(
        base_image=sketch_img,
        mask=mask,
        prompt=step2_prompt,
        size=size,
        seed=seed if seed is not None else cfg.seed,
        negative_prompt=negative_prompt or cfg.negative_prompt,
    )

    final_img = paste_packshot(photo_img, pack, pack_xy)
    result = ProcessResult(
        image=final_img,
        sketch_image=sketch_img,
        pre_repaste_image=photo_img,
        initial_canvas_image=canvas,
        prompt_used=f"STEP1: {step1_prompt}\n\nSTEP2: {step2_prompt}",
        negative_prompt_used=negative_prompt or cfg.negative_prompt,
        seed=seed if seed is not None else cfg.seed,
        size=size,
        metadata={"process": "sketch_to_photo"},
    )
    saved_dir = save_generation(
        item=item,
        process_name="sketch_to_photo",
        result=result,
        params={
            "prefer_rewritten": prefer_rewritten,
            "mask_feather": mask_feather,
            "negative_prompt": negative_prompt or cfg.negative_prompt,
            "using_mask": True,
            "sketch_prompt_prefix": sketch_prompt_prefix,
            "photo_prompt": photo_prompt,
        },
    )
    result.metadata["saved_dir"] = str(saved_dir)
    return result
