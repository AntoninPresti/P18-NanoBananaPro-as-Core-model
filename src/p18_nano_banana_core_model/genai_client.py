from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from google.genai.types import GenerateContentResponse
from PIL import Image

try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except Exception:  # pragma: no cover - optional at runtime for mock mode
    genai = None  # fallback to mock when library is not available
    types = None  # type: ignore


GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"


def _env_api_key() -> Optional[str]:
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    # optional .env support if python-dotenv is installed
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
        return os.getenv("GOOGLE_API_KEY")
    except Exception:
        return None


def has_api_key() -> bool:
    return bool(_env_api_key()) and genai is not None


def _get_client():
    if genai is None:
        return None
    key = _env_api_key()
    if not key:
        return None
    # The new client auto-reads env key if not provided, but we set it explicitly.
    return genai.Client(api_key=key)


def _closest_aspect_ratio_label(width: int, height: int) -> str:
    # Allowed labels per docs
    candidates: Dict[str, float] = {
        "1:1": 1.0,
        "2:3": 2 / 3,
        "3:2": 3 / 2,
        "3:4": 3 / 4,
        "4:3": 4 / 3,
        "4:5": 4 / 5,
        "5:4": 5 / 4,
        "9:16": 9 / 16,
        "16:9": 16 / 9,
        "21:9": 21 / 9,
    }
    r = width / height if height else 1.0
    # Pick label with minimal absolute relative error
    label = min(candidates.keys(), key=lambda k: abs(candidates[k] - r))
    return label


def _resolution_label(width: int, height: int) -> str:
    # Simple heuristic
    longer = max(width, height)
    if longer <= 1024:
        return "1K"
    if longer <= 2048:
        return "2K"
    return "4K"


def _build_image_config(size: Tuple[int, int]):
    if types is None:
        return None
    w, h = size
    return types.ImageConfig(
        aspect_ratio=_closest_aspect_ratio_label(w, h),
        image_size=_resolution_label(w, h),
    )


def _extract_first_image_from_parts(parts) -> Image.Image:
    # Iterate through parts to find the first image result
    for part in getattr(parts, "__iter__", lambda: [])():
        try:
            img = part.as_image()
        except Exception:
            img = None
        if img is not None:
            # part.as_image() returns PIL.Image
            return img._pil_image.convert("RGBA")
    # Some responses expose parts as a simple list
    for part in parts or []:
        try:
            img = part.as_image()
        except Exception:
            img = None
        if img is not None:
            return img._pil_image.convert("RGBA")
    raise RuntimeError("No image found in response parts.")


def generate_image(
    prompt: str,
    size: Tuple[int, int] = (1024, 1024),
    model: str = GEMINI_IMAGE_MODEL,
    seed: Optional[int] = None,
    safety_filter_level: Optional[
        str
    ] = None,  # unused in this path; kept for API compat
    extra: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    """Text-to-image generation using Gemini 3 Pro Image Preview.

    The Google GenAI v2 client uses models.generate_content(). We always route to
    GEMINI_IMAGE_MODEL as required.
    """
    client = _get_client()
    if client is None or types is None:
        from .image_utils import draw_placeholder_background

        return draw_placeholder_background(size, text="MOCK GENERATE")

    # Build config
    img_cfg = _build_image_config(size)
    gen_cfg = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=img_cfg,
    )
    # NOTE: seed is not currently exposed in this high-level API config; ignored if provided
    _ = seed  # keep signature compatible
    _ = safety_filter_level

    resp: GenerateContentResponse = client.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=[prompt],
        config=gen_cfg,
    )
    # Normalize output to requested canvas size (safety against model returning a different size)
    img = _extract_first_image_from_parts(resp.parts)
    target_size = size
    if target_size and img.size != target_size:
        img = img.resize(target_size, Image.LANCZOS).convert("RGBA")
    else:
        img = img.convert("RGBA")
    return img


def edit_image(
    base_image: Image.Image,
    mask: Optional[Image.Image],
    prompt: str,
    size: Optional[Tuple[int, int]] = None,
    seed: Optional[int] = None,
    negative_prompt: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    """Image edit/outpaint using Gemini 3 Pro Image Preview.

    We provide the base image (with the packshot already placed) as part of the
    contents and strongly instruct the model to keep the product fixed. If a mask
    is provided, we also include it as another image part — while not formally
    documented as a separate "mask" part in the public example, many clients pass
    auxiliary images alongside the prompt.

    Mask convention: white = editable, black = protected.
    """
    client = _get_client()
    if client is None or types is None:
        from .image_utils import draw_placeholder_background

        target_size = size or base_image.size
        return draw_placeholder_background(target_size, text="MOCK EDIT")

    # Resize the working image if a target size is requested
    work = base_image
    if size and work.size != size:
        work = work.resize(size, Image.LANCZOS)
    work = work.convert("RGBA")

    # Combine prompts if negative prompt provided
    if negative_prompt:
        prompt = f"{prompt.strip()}\n\nNegative prompt (constraints): {negative_prompt.strip()}"

    contents: List[Any] = [prompt, work]
    if mask is not None:
        # Ensure mask is single-channel
        contents.append(mask.convert("L"))

    img_cfg = _build_image_config(work.size)
    gen_cfg = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=img_cfg,
    )
    _ = seed  # not exposed via this method; kept for API compat
    _ = extra

    resp = client.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=contents,
        config=gen_cfg,
    )
    # Normalize output to canvas size to guarantee perfect repaste alignment
    img = _extract_first_image_from_parts(resp.parts)
    target_size = size or base_image.size
    if img.size != target_size:
        img = img.resize(target_size, Image.LANCZOS).convert("RGBA")
    else:
        img = img.convert("RGBA")
    return img
