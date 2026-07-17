"""Qdrant Local 双命名向量存储。"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from .config import Settings
from .domain import ImageCandidate, ImageFeatures
from .image_io import phash_distance


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(settings.qdrant_path))

    def close(self) -> None:
        self.client.close()

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.settings.collection_name):
            return
        vector = models.VectorParams(size=self.settings.vector_size, distance=models.Distance.COSINE)
        self.client.create_collection(
            collection_name=self.settings.collection_name,
            vectors_config={"fashion": vector, "dino": vector},
        )

    def _payload(self, feature: ImageFeatures) -> dict[str, object]:
        record = feature.record
        return {
            "image_id": record.image_id,
            "shop_id": record.shop_id,
            "product_id": record.product_id,
            "product_key": record.product_key,
            "relative_path": record.relative_path,
            "phash": feature.phash,
            "source_size": record.size_bytes,
            "source_mtime_ns": record.modified_ns,
            "source_digest": record.source_digest,
            "fashion_revision": self.settings.fashion_revision,
            "dino_revision": self.settings.dino_revision,
        }

    def upsert(self, features: Iterable[ImageFeatures]) -> None:
        points = [
            models.PointStruct(
                id=feature.record.image_id,
                vector={
                    "fashion": feature.fashion_vector.tolist(),
                    "dino": feature.dino_vector.tolist(),
                },
                payload=self._payload(feature),
            )
            for feature in features
        ]
        if not points:
            return
        self.ensure_collection()
        for attempt in range(3):
            try:
                self.client.upsert(self.settings.collection_name, points=points, wait=True)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.1 * (2**attempt))

    def _scroll_all(self, *, with_vectors: bool = False):
        if not self.client.collection_exists(self.settings.collection_name):
            return []
        records = []
        offset = None
        while True:
            page, offset = self.client.scroll(
                self.settings.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            records.extend(page)
            if offset is None:
                return records

    def indexed_payloads(self) -> dict[str, dict[str, object]]:
        return {
            str(point.payload["image_id"]): dict(point.payload)
            for point in self._scroll_all()
            if point.payload and "image_id" in point.payload
        }

    def all_points(self, *, with_vectors: bool = True):
        return self._scroll_all(with_vectors=with_vectors)

    def delete_missing(self, valid_image_ids: set[str]) -> int:
        existing = set(self.indexed_payloads())
        missing = existing - valid_image_ids
        if missing:
            self.client.delete(
                self.settings.collection_name,
                points_selector=models.PointIdsList(points=sorted(missing)),
                wait=True,
            )
        return len(missing)

    def _query(self, name: str, vector: np.ndarray, top_k: int):
        if not self.client.collection_exists(self.settings.collection_name):
            return []
        response = self.client.query_points(
            self.settings.collection_name,
            query=np.asarray(vector, dtype=np.float32).tolist(),
            using=name,
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )
        return response.points

    def search_images(
        self,
        fashion_vector: np.ndarray,
        dino_vector: np.ndarray,
        *,
        top_k: int,
        query_phash: str | None = None,
        exclude_image_id: str | None = None,
    ) -> list[ImageCandidate]:
        merged = {}
        for name, vector in (("fashion", fashion_vector), ("dino", dino_vector)):
            for point in self._query(name, vector, top_k):
                payload = point.payload or {}
                image_id = str(payload.get("image_id", point.id))
                if image_id != exclude_image_id:
                    merged[image_id] = point

        # pHash 只滚动轻量 payload，命中后才按 point id 读取双向量。
        # 当前本地规模可接受线性比较；百万级应替换成专用二进制指纹索引。
        if query_phash and self.client.collection_exists(self.settings.collection_name):
            near_ids = []
            for point in self._scroll_all(with_vectors=False):
                payload = point.payload or {}
                image_id = str(payload.get("image_id", point.id))
                stored = payload.get("phash")
                if (
                    image_id not in merged
                    and image_id != exclude_image_id
                    and stored
                    and phash_distance(query_phash, str(stored)) <= 8
                ):
                    near_ids.append(point.id)
            if near_ids:
                for point in self.client.retrieve(
                    self.settings.collection_name,
                    ids=near_ids,
                    with_payload=True,
                    with_vectors=True,
                ):
                    payload = point.payload or {}
                    merged[str(payload.get("image_id", point.id))] = point

        candidates = []
        fashion_query = np.asarray(fashion_vector, dtype=np.float32)
        dino_query = np.asarray(dino_vector, dtype=np.float32)
        for image_id, point in merged.items():
            payload = point.payload or {}
            vectors = point.vector or {}
            fashion = np.asarray(vectors["fashion"], dtype=np.float32)
            dino = np.asarray(vectors["dino"], dtype=np.float32)
            stored_phash = str(payload.get("phash", "0000000000000000"))
            candidates.append(
                ImageCandidate(
                    image_id=image_id,
                    shop_id=str(payload["shop_id"]),
                    product_id=str(payload["product_id"]),
                    path=self.settings.image_root / str(payload["relative_path"]),
                    phash=stored_phash,
                    dino_score=float(np.dot(dino_query, dino)),
                    fashion_score=float(np.dot(fashion_query, fashion)),
                    hamming_distance=phash_distance(query_phash, stored_phash) if query_phash else 64,
                )
            )
        candidates.sort(key=lambda item: max(item.fashion_score, item.dino_score), reverse=True)
        return candidates
