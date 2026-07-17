"""按约定目录结构扫描商品展示图。"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from .domain import ImageRecord

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
IMAGE_NAMESPACE = uuid.UUID("2166b4c4-b5ef-49e8-bd14-792c7738ad62")


def _digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_display_images(
    root: Path, *, compute_digest: bool = True, limit: int | None = None
) -> tuple[list[ImageRecord], list[str]]:
    """扫描 ``{shop}/{product}/display``，忽略 detail 与其他目录。"""
    root = root.expanduser().resolve()
    if not root.is_dir():
        return [], [f"图片根目录不存在或不是目录：{root}"]
    if limit is not None and limit <= 0:
        raise ValueError("扫描数量限制必须大于 0")

    records: list[ImageRecord] = []
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            relative = path.relative_to(root)
            parts = relative.parts
            if len(parts) < 4 or parts[2] != "display":
                continue
            stat = path.stat()
            relative_text = relative.as_posix()
            records.append(
                ImageRecord(
                    image_id=str(uuid.uuid5(IMAGE_NAMESPACE, relative_text)),
                    shop_id=parts[0],
                    product_id=parts[1],
                    path=path,
                    relative_path=relative_text,
                    modified_ns=stat.st_mtime_ns,
                    size_bytes=stat.st_size,
                    source_digest=_digest(path) if compute_digest else "",
                )
            )
            if limit is not None and len(records) >= limit:
                break
        except (OSError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
    return records, errors
