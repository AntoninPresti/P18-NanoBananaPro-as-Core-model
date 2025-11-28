from __future__ import annotations

from typing import Optional, Dict, Any

from PIL import Image

from ..types import DatasetItem, ProcessResult, OutpaintModel
from ..utils.data import get_prompt_for_item
from ..utils.prompt import add_do_not_move_guardrails
from ..outpaint import outpaint_generation


def run_simple_prompt(
    item: DatasetItem,
    *,
    prefer_rewritten: bool = True,
    add_guardrails_text: bool = True,
    extra_guardrails: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    model: OutpaintModel = OutpaintModel.IMAGEN_EDIT,
    seed: Optional[int] = None,
) -> ProcessResult:
    """Process 1 — Simple prompt + anti-drift guardrails, masked outpaint via Imagen Edit."""
    prompt = get_prompt_for_item(item, prefer_rewritten=prefer_rewritten)
    if add_guardrails_text:
        prompt = add_do_not_move_guardrails(prompt)
    if extra_guardrails:
        prompt = f"{prompt}\n\n{extra_guardrails.strip()}"

    cfg = item.config
    img = outpaint_generation(
        generation_width=int(cfg.width or 1024),
        generation_height=int(cfg.height or 1024),
        packshot_width=cfg.packshot_width,
        packshot_height=cfg.packshot_height,
        packshot_top_left_pos_x=cfg.packshot_top_left_pos_x,
        packshot_top_left_pos_y=cfg.packshot_top_left_pos_y,
        packshot_url=cfg.packshot_url or item.packshot_path,
        prompt=prompt,
        negative_prompt=negative_prompt or cfg.negative_prompt,
        model=model,
        packshot_type=None,
        seed=seed if seed is not None else cfg.seed,
    )

    return ProcessResult(
        image=img,
        prompt_used=prompt,
        negative_prompt_used=negative_prompt or cfg.negative_prompt,
        seed=seed if seed is not None else cfg.seed,
        size=(int(cfg.width or 1024), int(cfg.height or 1024)),
        metadata={"process": "simple_prompt"},
    )
