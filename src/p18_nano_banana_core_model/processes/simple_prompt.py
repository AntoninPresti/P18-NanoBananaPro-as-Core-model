from __future__ import annotations

from typing import Optional

from PIL import Image

from ..genai_client import edit_image
from ..image_utils import ensure_size, load_image, make_outpaint_mask, paste_packshot
from ..types import DatasetItem, ProcessResult
from ..utils.data import get_prompt_for_item
from ..utils.prompt import add_do_not_move_guardrails
from ..utils.saver import save_generation


def run_simple_prompt(
    item: DatasetItem,
    *,
    prefer_rewritten: bool = True,
    add_guardrails_text: bool = True,
    extra_guardrails: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
) -> ProcessResult:
    """Process 1 — Simple prompt + anti-drift guardrails, masked outpaint via Imagen Edit."""
    prompt = get_prompt_for_item(item, prefer_rewritten=prefer_rewritten)
    if add_guardrails_text:
        sketch_prompt_prefix: str = (
            "Fill the white space around this product to create the following scene: "
        )

        prompt = f"{sketch_prompt_prefix.strip()}\n{prompt}"
        prompt = add_do_not_move_guardrails(prompt)
    if extra_guardrails:
        prompt = f"{prompt}\n\n{extra_guardrails.strip()}"

    cfg = item.config
    size = (int(cfg.width or 1024), int(cfg.height or 1024))
    pack_xy = (cfg.packshot_top_left_pos_x, cfg.packshot_top_left_pos_y)
    pack_wh = (cfg.packshot_width, cfg.packshot_height)

    # Compose base canvas with packshot
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    pack = load_image(cfg.packshot_url or item.packshot_path)
    pack = ensure_size(pack, pack_wh)
    canvas.alpha_composite(pack, pack_xy)

    # Mask protecting the packshot
    mask = make_outpaint_mask(size, pack_xy, pack_wh, feather=4, invert=False)

    # Edit background
    edited = edit_image(
        base_image=canvas,
        mask=mask,
        prompt=prompt,
        size=size,
        seed=seed if seed is not None else cfg.seed,
        negative_prompt=negative_prompt or cfg.negative_prompt,
    )

    # Re-paste original packshot
    final_img = paste_packshot(edited, pack, pack_xy)

    result = ProcessResult(
        image=final_img,
        pre_repaste_image=edited,
        prompt_used=prompt,
        negative_prompt_used=negative_prompt or cfg.negative_prompt,
        seed=seed if seed is not None else cfg.seed,
        size=size,
        metadata={"process": "simple_prompt"},
    )
    saved_dir = save_generation(
        item=item,
        process_name="simple_prompt",
        result=result,
        params={
            "prefer_rewritten": prefer_rewritten,
            "extra_guardrails": extra_guardrails,
            "negative_prompt": negative_prompt or cfg.negative_prompt,
            "using_mask": True,
        },
    )
    result.metadata["saved_dir"] = str(saved_dir)
    return result
