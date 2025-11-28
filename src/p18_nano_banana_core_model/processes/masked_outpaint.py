from __future__ import annotations

from typing import Optional

from PIL import Image

from ..genai_client import edit_image
from ..image_utils import ensure_size, load_image, make_outpaint_mask, paste_packshot
from ..types import DatasetItem, ProcessResult
from ..utils.data import get_prompt_for_item
from ..utils.prompt import add_do_not_move_guardrails
from ..utils.saver import save_generation


def run_masked_outpaint(
    item: DatasetItem,
    *,
    prefer_rewritten: bool = True,
    feather: int = 8,
    invert_mask: bool = False,
    background_fill_rgba: tuple[int, int, int, int] | None = (255, 255, 255, 0),
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    extra_instruction: Optional[str] = None,
) -> ProcessResult:
    """Process 3 — Explicit masked outpaint with tunable mask softness and options."""
    prompt = get_prompt_for_item(item, prefer_rewritten=prefer_rewritten)
    prompt = add_do_not_move_guardrails(prompt)
    if extra_instruction:
        prompt = f"{prompt}\n\n{extra_instruction.strip()}"

    cfg = item.config
    size = (int(cfg.width or 1024), int(cfg.height or 1024))
    pack_xy = (cfg.packshot_top_left_pos_x, cfg.packshot_top_left_pos_y)
    pack_wh = (cfg.packshot_width, cfg.packshot_height)

    # Base canvas
    if background_fill_rgba is None:
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    else:
        canvas = Image.new("RGBA", size, background_fill_rgba)

    pack = load_image(cfg.packshot_url or item.packshot_path)
    pack = ensure_size(pack, pack_wh)
    canvas.alpha_composite(pack, pack_xy)

    mask = make_outpaint_mask(
        size, pack_xy, pack_wh, feather=feather, invert=invert_mask
    )

    edited = edit_image(
        base_image=canvas,
        mask=mask,
        prompt=prompt,
        size=size,
        seed=seed if seed is not None else cfg.seed,
        negative_prompt=negative_prompt or cfg.negative_prompt,
    )

    final_img = paste_packshot(edited, pack, pack_xy)
    result = ProcessResult(
        image=final_img,
        pre_repaste_image=edited,
        prompt_used=prompt,
        negative_prompt_used=negative_prompt or cfg.negative_prompt,
        seed=seed if seed is not None else cfg.seed,
        size=size,
        metadata={
            "process": "masked_outpaint",
            "feather": feather,
            "invert_mask": invert_mask,
        },
    )
    # Persist generation
    saved_dir = save_generation(
        item=item,
        process_name="masked_outpaint",
        result=result,
        params={
            "prefer_rewritten": prefer_rewritten,
            "feather": feather,
            "invert_mask": invert_mask,
            "background_fill_rgba": background_fill_rgba,
            "negative_prompt": negative_prompt or cfg.negative_prompt,
            "using_mask": True,
        },
    )
    result.metadata["saved_dir"] = str(saved_dir)
    return result
