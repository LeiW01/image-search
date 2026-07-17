from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fashion_search.config import Settings
from fashion_search.encoders import (
    DinoV2Encoder,
    FashionSiglipEncoder,
    l2_normalize,
    preferred_device,
)


def test_l2_normalize_returns_unit_rows() -> None:
    vectors = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)

    normalized = l2_normalize(vectors)

    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0])


def test_l2_normalize_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="零向量"):
        l2_normalize(np.zeros((1, 3), dtype=np.float32))


def test_preferred_device_is_supported() -> None:
    assert preferred_device() in {"mps", "cpu"}


@pytest.mark.model
def test_real_models_return_normalized_768_vectors(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (224, 224), (80, 120, 180)).save(image_path)
    settings = Settings.default(tmp_path)

    fashion = FashionSiglipEncoder(settings).encode([image_path])
    dino = DinoV2Encoder(settings).encode([image_path])

    assert fashion.shape == (1, 768)
    assert dino.shape == (1, 768)
    np.testing.assert_allclose(np.linalg.norm(fashion, axis=1), [1.0], atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(dino, axis=1), [1.0], atol=1e-4)
