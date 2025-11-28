from __future__ import annotations

from typing import Optional

from PIL import Image

from ..types import DatasetItem, ProcessResult, OutpaintModel
from ..utils.data import get_prompt_for_item
from ..utils.prompt import strip_square_brackets
from ..image_utils import load_image, ensure_size, make_outpaint_mask, paste_packshot
from ..genai_client import edit_image


def run_sketch_to_photo(
    item: DatasetItem,
    *,
    prefer_rewritten: bool = True,
    sketch_prompt_prefix: str = (
        "Prends ce produit et crée un sketch/line-art de la scène suivante: "
    ),
    sketch_guardrails: str = (
        "Dans le sketch, la position du produit doit rester exactement la même dans le carré envoyé."
    ),
    photo_prompt: str = (
        "Rends ce sketch photoréaliste, en gardant exactement la même scène et la même disposition."
    ),
    negative_prompt: Optional[str] = None,
    model_edit: OutpaintModel = OutpaintModel.IMAGEN_EDIT,
    seed: Optional[int] = None,
    mask_feather: int = 6,
) -> ProcessResult:
    """Process 2 — Two-step: sketch around fixed product, then photorealistic render.

    Both steps use the same protected packshot mask. The second step converts
    the first step's background to a realistic rendering without moving elements.
    """
    base_prompt = get_prompt_for_item(item, prefer_rewritten=prefer_rewritten)
    base_prompt = strip_square_brackets(base_prompt)
    step1_prompt = f"{sketch_prompt_prefix.strip()} {base_prompt}\n\n{sketch_guardrails.strip()}"

    cfg = item.config
    size = (int(cfg.width or 1024), int(cfg.height or 1024))
    pack_xy = (cfg.packshot_top_left_pos_x, cfg.packshot_top_left_pos_y)
    pack_wh = (cfg.packshot_width, cfg.packshot_height)

    # Compose base canvas (packshot on transparent)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    pack = load_image(cfg.packshot_url or item.packshot_path)
    pack = ensure_size(pack, pack_wh)
    canvas.paste(pack, pack_xy, pack)

    mask = make_outpaint_mask(size, pack_xy, pack_wh, feather=mask_feather, invert=False)

    # Step 1: Sketch background generation
    sketch_img = edit_image(
        base_image=canvas,
        mask=mask,
        prompt=step1_prompt,
        size=size,
        model=str(model_edit),
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
        model=str(model_edit),
        seed=seed if seed is not None else cfg.seed,
        negative_prompt=negative_prompt or cfg.negative_prompt,
    )

    final_img = paste_packshot(photo_img, pack, pack_xy)
    return ProcessResult(
        image=final_img,
        prompt_used=f"STEP1: {step1_prompt}\n\nSTEP2: {step2_prompt}",
        negative_prompt_used=negative_prompt or cfg.negative_prompt,
        seed=seed if seed is not None else cfg.seed,
        size=size,
        metadata={"process": "sketch_to_photo"},
    )
