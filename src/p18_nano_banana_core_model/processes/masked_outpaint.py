from __future__ import annotations

from typing import Optional, Dict, Any

from PIL import Image

from ..types import DatasetItem, ProcessResult, OutpaintModel
from ..utils.data import get_prompt_for_item
from ..utils.prompt import add_do_not_move_guardrails
from ..image_utils import load_image, ensure_size, make_outpaint_mask, paste_packshot
from ..genai_client import edit_image


def run_masked_outpaint(
    item: DatasetItem,
    *,
    prefer_rewritten: bool = True,
    feather: int = 8,
    invert_mask: bool = False,
    background_fill_rgba: tuple[int, int, int, int] | None = (255, 255, 255, 0),
    negative_prompt: Optional[str] = None,
    model: OutpaintModel = OutpaintModel.IMAGEN_EDIT,
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

    mask = make_outpaint_mask(size, pack_xy, pack_wh, feather=feather, invert=invert_mask)

    edited = edit_image(
        base_image=canvas,
        mask=mask,
        prompt=prompt,
        size=size,
        model=str(model),
        seed=seed if seed is not None else cfg.seed,
        negative_prompt=negative_prompt or cfg.negative_prompt,
    )

    final_img = paste_packshot(edited, pack, pack_xy)
    return ProcessResult(
        image=final_img,
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
