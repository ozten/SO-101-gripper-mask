"""Shared file-enumeration helpers."""

from __future__ import annotations

import re
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def natural_key(name: str) -> tuple:
    """Sort key treating digit runs numerically (2 before 10)."""
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def enumerate_images(directory: Path) -> list[Path]:
    """Image files in `directory` (case-insensitive extension match), numerically sorted."""
    files = [
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: natural_key(p.name))
