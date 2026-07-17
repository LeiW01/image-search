from dataclasses import replace
from pathlib import Path

import numpy as np

from fashion_search.config import Settings
from fashion_search.domain import ImageFeatures, ImageRecord
from fashion_search.vector_store import VectorStore


def _feature(tmp_path: Path, image_id: str, product_id: str, vector: list[float]) -> ImageFeatures:
    path = tmp_path / f"{image_id}.jpg"
    path.write_bytes(b"x")
    record = ImageRecord(
        image_id=image_id,
        shop_id="shop",
        product_id=product_id,
        path=path,
        relative_path=f"shop/{product_id}/display/{image_id}.jpg",
        modified_ns=1,
        size_bytes=1,
        source_digest=f"digest-{image_id}",
    )
    values = np.asarray(vector, dtype=np.float32)
    return ImageFeatures(record=record, phash="0000000000000000", dino_vector=values, fashion_vector=values)


def test_qdrant_round_trip_named_vectors_search_and_delete(tmp_path: Path) -> None:
    settings = replace(
        Settings.default(tmp_path),
        vector_size=4,
        qdrant_path=tmp_path / "qdrant",
        collection_name="test_products",
    )
    first = _feature(tmp_path, "11111111-1111-1111-1111-111111111111", "p1", [1, 0, 0, 0])
    second = _feature(tmp_path, "22222222-2222-2222-2222-222222222222", "p2", [0, 1, 0, 0])
    store = VectorStore(settings)
    store.ensure_collection()
    store.upsert([first, second])

    payloads = store.indexed_payloads()
    assert set(payloads) == {first.record.image_id, second.record.image_id}
    candidates = store.search_images(
        first.fashion_vector,
        first.dino_vector,
        top_k=1,
        query_phash="0000000000000000",
    )
    assert candidates[0].product_id == "p1"
    assert {candidate.image_id for candidate in candidates} == {
        first.record.image_id,
        second.record.image_id,
    }

    store.close()
    reopened = VectorStore(settings)
    assert len(reopened.indexed_payloads()) == 2
    assert reopened.delete_missing({first.record.image_id}) == 1
    assert set(reopened.indexed_payloads()) == {first.record.image_id}
    reopened.close()
