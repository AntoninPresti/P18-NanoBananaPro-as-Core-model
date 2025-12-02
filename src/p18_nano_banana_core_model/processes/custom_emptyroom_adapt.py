from __future__ import annotations

from typing import Optional, Dict, Any

import streamlit
from PIL import Image

from ..types import DatasetItem, ProcessResult
from ..utils.data import get_prompt_for_item
from ..utils.prompt import add_do_not_move_guardrails, strip_square_brackets
from ..image_utils import (
    load_image,
    ensure_size,
    make_outpaint_mask,
    paste_packshot,
)
from ..genai_client import edit_image
from ..utils.saver import save_generation


def _call_presti_empty_room(
    *,
    packshot_url: str,
    generation_width: int,
    generation_height: int,
    packshot_width: int,
    packshot_height: int,
    packshot_top_left_pos_x: int,
    packshot_top_left_pos_y: int,
    prompt: str,
    packshot_type: Optional[str] = None,
    model_name: str = "exp31_a",
    team_id: int = 3714,
) -> str:
    """Call the Presti internal endpoint to obtain an initial empty-room draft image.

    Returns the URL of the generated image.
    """
    # Lazy import to avoid hard dependency outside of this process
    from ai_projects.scripts_101.call_presti_endpoint import call_presti_endpoint

    endpoint = "/outpaint_generation"
    payload: Dict[str, Any] = {
        "packshot_url": packshot_url,
        "generation_width": int(generation_width),
        "generation_height": int(generation_height),
        "packshot_width": int(packshot_width),
        "packshot_height": int(packshot_height),
        "packshot_top_left_pos_x": int(packshot_top_left_pos_x),
        "packshot_top_left_pos_y": int(packshot_top_left_pos_y),
        "prompt": prompt,
        "model": model_name,
        "packshot_type": packshot_type or "",
        "team_id": int(team_id),
    }

    response_data = call_presti_endpoint(endpoint=endpoint, payload=payload)
    response_data = response_data.get("generation_output", {})
    if not isinstance(response_data, dict) or "output_url" not in response_data:
        raise RuntimeError("Invalid response from Presti endpoint: missing output_url")
    return str(response_data["output_url"])


def run_custom_emptyroom_adapt(
    item: DatasetItem,
    *,
    prefer_rewritten: bool = True,
    negative_prompt: Optional[str] = None,
    mask_feather: int = 6,
    save_base_dir: Optional[str] = None,
) -> ProcessResult:
    """Process 4 — Custom Empty-Room Draft → Gemini Adapt

    Step 1: Call the in-house model (exp31_a) to generate an empty-room image with the packshot fixed.
    Step 2: Use Gemini Edit to adapt the room to the prompt (walls/props), protecting the packshot.
    """
    # Prompt used for both: we keep strong guardrails when adapting with Gemini (step 2)
    base_prompt = get_prompt_for_item(item, prefer_rewritten=prefer_rewritten)
    base_prompt = strip_square_brackets(base_prompt)

    cfg = item.config
    size = (int(cfg.width or 1024), int(cfg.height or 1024))
    pack_xy = (cfg.packshot_top_left_pos_x, cfg.packshot_top_left_pos_y)
    pack_wh = (cfg.packshot_width, cfg.packshot_height)

    # Build the initial canvas used as context (packshot composited on transparent)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    pack = load_image(cfg.packshot_url or item.packshot_path)
    pack = ensure_size(pack, pack_wh)
    canvas.alpha_composite(pack, pack_xy)

    # Step 1: Call Presti endpoint (custom model).
    # Important: Explicitly request an EMPTY ROOM draft so the custom model doesn't follow
    # the full scene prompt. We keep packshot fixed by geometry; the text must insist on
    # empty room (no props/furniture) and only carry style/lighting cues from the prompt.
    empty_room_prompt = (
        "Create a clean, EMPTY ROOM around the product. Visible wall and windows behind the product."
        "Neutral walls and floor, coherent perspective. "
        "You may take only lighting, color palette, and general mood cues from this scene description: "
        f"{base_prompt}"
    )
    packshot_type = cfg.raw.get("packshot_type") if isinstance(cfg.raw, dict) else None
    presti_output_url = _call_presti_empty_room(
        packshot_url=(cfg.packshot_url or item.packshot_path),
        generation_width=size[0],
        generation_height=size[1],
        packshot_width=pack_wh[0],
        packshot_height=pack_wh[1],
        packshot_top_left_pos_x=pack_xy[0],
        packshot_top_left_pos_y=pack_xy[1],
        prompt=empty_room_prompt,
        packshot_type=packshot_type,
        model_name="exp31_a",
        team_id=3714,
    )
    # Load intermediate as RGBA
    intermediate_img = load_image(presti_output_url)
    # Ensure it matches canvas size
    if intermediate_img.size != size:
        intermediate_img = intermediate_img.resize(size, Image.LANCZOS).convert("RGBA")

    # Step 2: Adapt with Gemini Edit; protect packshot with mask and re-paste afterwards.
    mask = make_outpaint_mask(
        size, pack_xy, pack_wh, feather=int(mask_feather), invert=False
    )

    # Compose the original packshot on top of the intermediate to make sure it's exactly aligned
    step2_base = paste_packshot(intermediate_img, pack, pack_xy)

    step2_prompt = add_do_not_move_guardrails(
        "Adapt the room to match the desired scene: update walls, floors, lighting, and props as required. **Keep the crib in the exact same position and perspective within the image**"
        f"\n\nTarget scene: {base_prompt}"
    )
    adapted_img = edit_image(
        base_image=step2_base,
        mask=mask,
        prompt=step2_prompt,
        size=size,
        seed=cfg.seed,  # Gemini seed is best-effort/ignored in many cases
        negative_prompt=negative_prompt or cfg.negative_prompt,
    )

    # Final repaste to guarantee exact packshot pixels
    final_img = paste_packshot(adapted_img, pack, pack_xy)

    # Package result
    result = ProcessResult(
        image=final_img,
        pre_repaste_image=adapted_img,
        sketch_image=intermediate_img,  # save the step-1 draft as sketch.png for consistency
        initial_canvas_image=canvas,
        prompt_used=f"STEP1 (Presti exp31_a — empty room): {empty_room_prompt}\n\nSTEP2 (Gemini adapt): {step2_prompt}",
        negative_prompt_used=negative_prompt or cfg.negative_prompt,
        seed=cfg.seed,
        size=size,
        metadata={
            "process": "custom_emptyroom_adapt",
            "presti_model": "exp31_a",
            "presti_output_url": presti_output_url,
            "presti_prompt_used": empty_room_prompt,
            "mask_feather": int(mask_feather),
        },
    )

    saved_dir = save_generation(
        item=item,
        process_name="custom_emptyroom_adapt",
        result=result,
        params={
            "prefer_rewritten": prefer_rewritten,
            "negative_prompt": negative_prompt or cfg.negative_prompt,
            "using_mask": True,
            "mask_feather": int(mask_feather),
            "presti": {
                "endpoint": "/outpaint_generation",
                "model": "exp31_a",
                "team_id": 3714,
                "prompt": empty_room_prompt,
            },
        },
        base_dir=save_base_dir,
    )
    result.metadata["saved_dir"] = str(saved_dir)
    return result
