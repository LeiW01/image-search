"""索引与检索流程共享的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ImageRecord:
    image_id: str
    shop_id: str
    product_id: str
    path: Path
    relative_path: str
    modified_ns: int
    size_bytes: int
    source_digest: str

    @property
    def product_key(self) -> str:
        return f"{self.shop_id}:{self.product_id}"


@dataclass(frozen=True, slots=True)
class ImageFeatures:
    record: ImageRecord
    phash: str
    dino_vector: np.ndarray
    fashion_vector: np.ndarray


@dataclass(frozen=True, slots=True)
class ImageCandidate:
    image_id: str
    shop_id: str
    product_id: str
    path: Path
    phash: str
    dino_score: float
    fashion_score: float
    hamming_distance: int
    similarity_score: float = 0.0


@dataclass(frozen=True, slots=True)
class ProductResult:
    shop_id: str
    product_id: str
    representative_path: Path
    score: float
    reason: str
    same_probability: float | None = None


@dataclass(frozen=True, slots=True)
class SearchResponse:
    same_products: list[ProductResult] = field(default_factory=list)
    similar_products: list[ProductResult] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class IndexStats:
    scanned: int
    indexed: int
    skipped: int
    failed: int
    deleted: int = 0
    errors: tuple[str, ...] = ()
