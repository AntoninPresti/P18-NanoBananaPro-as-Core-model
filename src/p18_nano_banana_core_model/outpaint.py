from __future__ import annotations

from typing import Optional

from PIL import Image

from .genai_client import edit_image
from .image_utils import ensure_size, load_image, make_outpaint_mask, paste_packshot


def outpaint_generation(
    *,
    generation_width: int,
    generation_height: int,
    packshot_width: int,
    packshot_height: int,
    packshot_top_left_pos_x: int,
    packshot_top_left_pos_y: int,
    packshot_url: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    model: Optional[str] = None,
    packshot_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Pure-Python equivalent of the FastAPI endpoint signature.

    - Builds a canvas of (generation_width, generation_height)
    - Loads the packshot at the exact coordinates and size
    - Builds a mask to protect the packshot and request Imagen Edit for background
    - Re-pastes the original packshot as a final step for perfect alignment
    Returns a RGBA PIL.Image.
    """
    size = (int(generation_width), int(generation_height))
    pack_xy = (int(packshot_top_left_pos_x), int(packshot_top_left_pos_y))
    pack_wh = (int(packshot_width), int(packshot_height))

    # Base canvas is transparent; packshot will be composited for edit
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))

    # Load and place packshot
    pack = load_image(packshot_url)
    pack = ensure_size(pack, pack_wh)
    canvas.alpha_composite(pack, pack_xy)

    # Create mask: protect packshot (black), allow background edits (white)
    mask = make_outpaint_mask(size, pack_xy, pack_wh, feather=4, invert=False)

    # Edit using Imagen Edit with guardrail prompt
    edited = edit_image(
        base_image=canvas,
        mask=mask,
        prompt=prompt,
        size=size,
        seed=seed,
        negative_prompt=negative_prompt,
    )

    # Re-paste original packshot to ensure no drift
    final_img = paste_packshot(edited, pack, pack_xy)
    return final_img
