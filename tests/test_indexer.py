from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from fashion_search.config import Settings
from fashion_search.indexer import Indexer
from fashion_search.vector_store import VectorStore


class FakeEncoder:
    def __init__(self) -> None:
        self.encoded = 0

    def encode(self, paths: list[Path]) -> np.ndarray:
        self.encoded += len(paths)
        rows = []
        for index, _ in enumerate(paths):
            row = np.zeros(4, dtype=np.float32)
            row[index % 4] = 1
            rows.append(row)
        return np.stack(rows)


def test_indexer_is_incremental_tolerates_corrupt_images_and_deletes_missing(tmp_path: Path) -> None:
    display = tmp_path / "images" / "shop" / "product" / "display"
    display.mkdir(parents=True)
    valid = display / "valid.jpg"
    broken = display / "broken.jpg"
    Image.new("RGB", (32, 32), "red").save(valid)
    broken.write_bytes(b"broken")
    settings = replace(
        Settings.default(tmp_path),
        image_root=tmp_path / "images",
        qdrant_path=tmp_path / "qdrant",
        state_dir=tmp_path / "state",
        collection_name="index_test",
        vector_size=4,
        batch_size=2,
    )
    fashion = FakeEncoder()
    dino = FakeEncoder()
    store = VectorStore(settings)
    indexer = Indexer(settings, store, fashion, dino)

    first = indexer.run()
    second = indexer.run()

    assert (first.scanned, first.indexed, first.failed) == (2, 1, 1)
    assert second.indexed == 0
    assert second.skipped == 1
    assert len(store.indexed_payloads()) == 1
    assert (settings.state_dir / "index-report.json").is_file()

    valid.unlink()
    third = indexer.run()
    assert third.deleted == 1
    assert store.indexed_payloads() == {}
    store.close()
