from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def load_image(path_or_url: str, timeout: int = 20) -> Image.Image:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = requests.get(path_or_url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
    else:
        img = Image.open(path_or_url)
    return img.convert("RGBA")


def ensure_size(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    if img.size == size:
        return img
    return img.resize(size, Image.LANCZOS)


def make_outpaint_mask(
    canvas_size: Tuple[int, int],
    packshot_xy: Tuple[int, int],
    packshot_wh: Tuple[int, int],
    feather: int = 8,
    invert: bool = False,
) -> Image.Image:
    """
    Returns an L-mode mask where white=editable and black=protected by default.
    If invert=True, flips semantics.
    """
    w, h = canvas_size
    mask = Image.new("L", (w, h), color=255)  # editable everywhere
    draw = ImageDraw.Draw(mask)
    x, y = packshot_xy
    pw, ph = packshot_wh
    draw.rectangle([x, y, x + pw, y + ph], fill=0)  # protect packshot
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    if invert:
        mask = Image.eval(mask, lambda v: 255 - v)
    return mask


def paste_packshot(canvas: Image.Image, packshot: Image.Image, xy: Tuple[int, int]) -> Image.Image:
    out = canvas.copy()
    out.alpha_composite(packshot, xy)
    return out


def draw_placeholder_background(size: Tuple[int, int], text: str = "NO API KEY - MOCK") -> Image.Image:
    w, h = size
    bg = Image.new("RGBA", (w, h), (240, 240, 240, 255))
    draw = ImageDraw.Draw(bg)
    # Simple center text
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    tw, th = draw.textlength(text, font=font), 12
    draw.text(((w - tw) / 2, (h - th) / 2), text, fill=(120, 120, 120, 255), font=font)
    return bg
