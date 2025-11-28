from __future__ import annotations

import re
from pathlib import Path

BRACKETS_RE = re.compile(r"\[[^\]]*\]")


def strip_square_brackets(text: str) -> str:
    """Remove any content within square brackets, inclusive, and normalize whitespace."""
    cleaned = BRACKETS_RE.sub("", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def read_prompt_file_cleaned(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    content = p.read_text(encoding="utf-8", errors="ignore")
    return strip_square_brackets(content)


def add_do_not_move_guardrails(prompt: str) -> str:
    guard = (
        "Important: Do not move, resize, rotate, or alter the product. "
        "Keep the product exactly at the same coordinates and size. "
        "Only generate the background scene around it."
    )
    if prompt:
        return f"{prompt.strip()}\n\n{guard}"
    return guard
