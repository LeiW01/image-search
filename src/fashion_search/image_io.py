"""统一读取图片并计算感知哈希。"""

from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError


class ImageDecodeError(ValueError):
    """源文件不能被完整解码为图片。"""


def load_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as source:
            source.load()
            return source.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageDecodeError(f"无法解码图片 {path}: {exc}") from exc


def compute_phash(image: Image.Image) -> str:
    """返回 8×8 pHash（64 bit，即16位十六进制）。"""
    return str(imagehash.phash(image, hash_size=8))


def phash_distance(left: str, right: str) -> int:
    return imagehash.hex_to_hash(left) - imagehash.hex_to_hash(right)
