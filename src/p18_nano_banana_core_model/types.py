from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel


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
    pre_repaste_image: Any | None = (
        None  # image before re-pasting the original packshot
    )
    sketch_image: Any | None = None  # for 2-step processes: intermediate sketch
    initial_canvas_image: Any | None = (
        None  # the base canvas sent to the model at step 1
    )
    metadata: Dict[str, Any] = {}
    prompt_used: Optional[str] = None
    negative_prompt_used: Optional[str] = None
    seed: Optional[int] = None
    size: Tuple[int, int] | None = None
