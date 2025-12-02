from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from PIL import Image

from ..types import DatasetItem, ProcessResult


DATASET_ROOT = Path("data/NBpro_TrainSet")


def _unique_run_dir(base: Path) -> Path:
    """Return a unique directory path under `base` using timestamp and index suffix.

    Never overwrites: if the directory exists, append _2, _3, ...
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    run_dir = base / ts
    if not run_dir.exists():
        return run_dir
    idx = 2
    while True:
        candidate = base / f"{ts}_{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def save_generation(
    *,
    item: DatasetItem,
    process_name: str,
    result: ProcessResult,
    params: Dict[str, Any] | None = None,
    base_dir: Optional[Path | str] = None,
) -> Path:
    """Persist images and metadata for a single generation run.

    Layout:
    data/NBpro_TrainSet/GENERATIONS/<process>/<stem>/<timestamp>/
      - final.png
      - before_repaste.png (optional)
      - meta.json

    Returns the directory path created.
    """
    params = dict(params or {})
    root = Path(base_dir) if base_dir is not None else DATASET_ROOT
    proc_dir = root / process_name / item.stem
    proc_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _unique_run_dir(proc_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save images
    final_path = run_dir / "final.png"
    before_repaste_path = run_dir / "before_repaste.png"
    sketch_path = run_dir / "sketch.png"

    img: Image.Image = result.image
    img.save(final_path, format="PNG")

    has_pre = False
    has_sketch = False
    if getattr(result, "pre_repaste_image", None) is not None:
        pre_img: Image.Image = result.pre_repaste_image  # type: ignore
        pre_img.save(before_repaste_path, format="PNG")
        has_pre = True

    # Save sketch image if present (for Sketch → Photoreal process)
    if getattr(result, "sketch_image", None) is not None:
        sk_img: Image.Image = result.sketch_image  # type: ignore
        sk_img.save(sketch_path, format="PNG")
        has_sketch = True

    cfg = item.config
    meta = {
        "stem": item.stem,
        "process": process_name,
        "timestamp": datetime.now().isoformat(),
        "prompt_used": result.prompt_used,
        "negative_prompt": result.negative_prompt_used,
        "seed": result.seed,
        "size": result.size,
        "model": params.get("model"),
        "paths": {
            "final": str(final_path.name),
            "before_repaste": str(before_repaste_path.name) if has_pre else None,
            "sketch": str(sketch_path.name) if has_sketch else None,
        },
        "packshot": {
            "path": item.packshot_path,
            "packshot_url": cfg.packshot_url,
            "width": cfg.packshot_width,
            "height": cfg.packshot_height,
            "top_left_x": cfg.packshot_top_left_pos_x,
            "top_left_y": cfg.packshot_top_left_pos_y,
            "canvas_width": int(cfg.width or 1024),
            "canvas_height": int(cfg.height or 1024),
        },
        "parameters": params,
    }

    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir
