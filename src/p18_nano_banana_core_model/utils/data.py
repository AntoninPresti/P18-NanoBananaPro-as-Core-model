from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .prompt import read_prompt_file_cleaned
from ..types import DatasetItem, PackshotConfig


DATASET_ROOT = Path("data/NBpro_TrainSet")


def load_packshot_config(json_path: Path) -> PackshotConfig:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    cfg = PackshotConfig(
        packshot_width=int(payload["packshot_width"]),
        packshot_height=int(payload["packshot_height"]),
        packshot_top_left_pos_x=int(payload["packshot_top_left_pos_x"]),
        packshot_top_left_pos_y=int(payload["packshot_top_left_pos_y"]),
        width=int(payload.get("width") or payload.get("generation_width") or 1024),
        height=int(payload.get("height") or payload.get("generation_height") or 1024),
        seed=(payload.get("seed") if payload.get("seed") is not None else None),
        negative_prompt=payload.get("negative_prompt"),
        packshot_url=payload.get("packshot_url"),
        output_url=payload.get("output_url"),
        raw=payload,
    )
    return cfg


def find_dataset_items(root: Path = DATASET_ROOT) -> List[DatasetItem]:
    packs_dir = root / "0_Packshots"
    cfg_dir = root / "0_Packshots Config"
    prom_dir = root / "0_Prompts"
    prom_orig_dir = root / "0_Prompts_original"
    ref_dir = root / "0_Original generations"

    items: List[DatasetItem] = []
    for png_path in sorted(packs_dir.glob("*.png")):
        stem = png_path.stem
        cfg_path = cfg_dir / f"{stem}.json"
        if not cfg_path.exists():
            continue
        cfg = load_packshot_config(cfg_path)
        item = DatasetItem(
            stem=stem,
            packshot_path=str(png_path),
            prompt_original_path=str(prom_orig_dir / f"{stem}.txt"),
            prompt_rewritten_path=str(prom_dir / f"{stem}.txt"),
            original_generation_path=str(ref_dir / f"{stem}.png"),
            config_path=str(cfg_path),
            config=cfg,
        )
        items.append(item)
    return items


def get_prompt_for_item(item: DatasetItem, prefer_rewritten: bool = True) -> str:
    # Prefer rewritten prompt, fallback to original
    if prefer_rewritten and item.prompt_rewritten_path:
        p = Path(item.prompt_rewritten_path)
        if p.exists():
            text = read_prompt_file_cleaned(p)
            if text:
                return text
    if item.prompt_original_path:
        p = Path(item.prompt_original_path)
        if p.exists():
            return read_prompt_file_cleaned(p)
    return ""
