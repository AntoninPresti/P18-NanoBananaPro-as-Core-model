from __future__ import annotations

from enum import StrEnum
from typing import Optional, Tuple, Literal, Dict, Any
from pydantic import BaseModel


class OutpaintModel(StrEnum):
    # Aliases to keep naming consistent with internal jargon
    GEMINI_3_PRO_PREVIEW = "gemini-3-pro-preview"  # text model (prompt helper)
    IMAGEN_GENERATE = "imagen-3.0-generate-002"    # text-to-image
    IMAGEN_EDIT = "imagen-3.0-edit-001"            # image edit/outpaint


class PackshotConfig(BaseModel):
    packshot_width: int
    packshot_height: int
    packshot_top_left_pos_x: int
    packshot_top_left_pos_y: int
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    packshot_url: Optional[str] = None
    output_url: Optional[str] = None
    # keep raw payload for debugging
    raw: Dict[str, Any] = {}


class DatasetItem(BaseModel):
    stem: str
    packshot_path: str
    prompt_original_path: Optional[str] = None
    prompt_rewritten_path: Optional[str] = None
    original_generation_path: Optional[str] = None
    config_path: str
    config: PackshotConfig


class ProcessResult(BaseModel):
    image: Any  # PIL.Image.Image, but keep Any to avoid optional runtime deps here
    pre_repaste_image: Any | None = None  # image before re-pasting the original packshot
    metadata: Dict[str, Any] = {}
    prompt_used: Optional[str] = None
    negative_prompt_used: Optional[str] = None
    seed: Optional[int] = None
    size: Tuple[int, int] | None = None
