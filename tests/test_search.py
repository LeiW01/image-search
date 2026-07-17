from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fashion_search.config import Settings
from fashion_search.domain import ImageCandidate
from fashion_search.search import SearchService


class FakeEncoder:
    def encode(self, paths):
        return np.array([[1.0, 0.0]], dtype=np.float32)


class FakeCalibrator:
    threshold = 0.8

    def predict(self, features):
        return np.asarray(features)[:, :2].mean(axis=1)


class FakeStore:
    def __init__(self, candidates):
        self.candidates = candidates

    def search_images(self, *args, **kwargs):
        return self.candidates


def _candidate(image_id: str, product: str, fashion: float, dino: float, hamming: int):
    return ImageCandidate(
        image_id=image_id,
        shop_id="shop",
        product_id=product,
        path=Path(f"/{image_id}.jpg"),
        phash="0" * 16,
        fashion_score=fashion,
        dino_score=dino,
        hamming_distance=hamming,
    )


def test_search_prioritizes_same_products_and_deduplicates_products(tmp_path: Path) -> None:
    query = tmp_path / "query.jpg"
    Image.new("RGB", (32, 32), "red").save(query)
    candidates = [
        _candidate("p1-a", "p1", 0.99, 0.98, 0),
        _candidate("p1-b", "p1", 0.97, 0.96, 1),
        _candidate("p2-a", "p2", 0.85, 0.40, 20),
    ]
    settings = replace(Settings.default(tmp_path), same_limit=5, similar_limit=20)
    service = SearchService(
        settings, FakeStore(candidates), FakeEncoder(), FakeEncoder(), FakeCalibrator()
    )

    result = service.search(query, fashion_weight=0.8)

    assert [item.product_id for item in result.same_products] == ["p1"]
    assert [item.product_id for item in result.similar_products] == ["p2"]


def test_search_rejects_invalid_weight(tmp_path: Path) -> None:
    service = SearchService(
        Settings.default(tmp_path), FakeStore([]), FakeEncoder(), FakeEncoder(), FakeCalibrator()
    )
    with pytest.raises(ValueError, match="0 到 1"):
        service.search(tmp_path / "unused.jpg", fashion_weight=1.1)
