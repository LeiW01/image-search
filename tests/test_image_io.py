from pathlib import Path

import pytest
from PIL import Image

from fashion_search.image_io import ImageDecodeError, compute_phash, load_rgb, phash_distance


def test_load_rgb_converts_rgba(tmp_path: Path) -> None:
    path = tmp_path / "rgba.png"
    Image.new("RGBA", (16, 16), (255, 0, 0, 128)).save(path)

    image = load_rgb(path)

    assert image.mode == "RGB"
    assert image.size == (16, 16)


def test_phash_is_stable_and_has_16_hex_characters() -> None:
    image = Image.new("RGB", (32, 32), "blue")

    first = compute_phash(image)
    second = compute_phash(image)

    assert first == second
    assert len(first) == 16
    assert phash_distance(first, second) == 0


def test_corrupt_image_has_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not-an-image")

    with pytest.raises(ImageDecodeError, match="无法解码图片"):
        load_rgb(path)
