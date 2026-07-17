"""可增量、局部失败可恢复的建索引流程。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .config import Settings
from .domain import ImageFeatures, ImageRecord, IndexStats
from .image_io import compute_phash, load_rgb
from .scanner import scan_display_images
from .vector_store import VectorStore


class Encoder(Protocol):
    def encode(self, paths: list[Path]): ...


class Indexer:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        fashion_encoder: Encoder,
        dino_encoder: Encoder,
    ) -> None:
        self.settings = settings
        self.store = store
        self.fashion_encoder = fashion_encoder
        self.dino_encoder = dino_encoder

    def _unchanged(self, record: ImageRecord, payload: dict[str, object] | None) -> bool:
        return bool(
            payload
            and payload.get("source_digest") == record.source_digest
            and payload.get("fashion_revision") == self.settings.fashion_revision
            and payload.get("dino_revision") == self.settings.dino_revision
        )

    def _process_batch(self, records: list[ImageRecord], errors: list[str]) -> int:
        if not records:
            return 0
        valid: list[tuple[ImageRecord, str]] = []
        for record in records:
            try:
                valid.append((record, compute_phash(load_rgb(record.path))))
            except Exception as exc:
                errors.append(f"{record.relative_path}: {exc}")
        if not valid:
            return 0

        paths = [record.path for record, _ in valid]
        try:
            fashion_vectors = self.fashion_encoder.encode(paths)
            dino_vectors = self.dino_encoder.encode(paths)
            features = [
                ImageFeatures(
                    record=record,
                    phash=phash,
                    fashion_vector=fashion_vectors[index],
                    dino_vector=dino_vectors[index],
                )
                for index, (record, phash) in enumerate(valid)
            ]
            self.store.upsert(features)
            return len(features)
        except Exception as exc:
            if len(valid) == 1:
                errors.append(f"{valid[0][0].relative_path}: 向量计算失败：{exc}")
                return 0
            middle = len(valid) // 2
            only_records = [record for record, _ in valid]
            return self._process_batch(only_records[:middle], errors) + self._process_batch(
                only_records[middle:], errors
            )

    def run(self, *, limit: int | None = None) -> IndexStats:
        records, scan_errors = scan_display_images(self.settings.image_root, limit=limit)
        errors = list(scan_errors)
        existing = self.store.indexed_payloads()
        pending = [record for record in records if not self._unchanged(record, existing.get(record.image_id))]
        skipped = len(records) - len(pending)
        indexed = 0
        for start in range(0, len(pending), self.settings.batch_size):
            indexed += self._process_batch(pending[start : start + self.settings.batch_size], errors)
        # 试跑只看部分图片，不能把不在试跑集合中的已有 point 当成已删除原图。
        deleted = (
            self.store.delete_missing({record.image_id for record in records})
            if limit is None
            else 0
        )
        stats = IndexStats(
            scanned=len(records),
            indexed=indexed,
            skipped=skipped,
            failed=len(errors),
            deleted=deleted,
            errors=tuple(errors[:100]),
        )
        self.settings.state_dir.mkdir(parents=True, exist_ok=True)
        report = self.settings.state_dir / "index-report.json"
        report.write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2), encoding="utf-8")
        return stats
