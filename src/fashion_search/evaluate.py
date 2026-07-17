"""基于同商品多展示图的留一图自动评测。"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np


DEFAULT_CONFIGURATIONS = {
    "FashionSigLIP": 1.0,
    "DINOv2": 0.0,
    "融合0.5": 0.5,
    "融合0.7": 0.7,
    "融合0.8": 0.8,
    "融合0.9": 0.9,
}


def compute_recall_at_k(expected: list[str], ranked: list[list[str]], k: int) -> float:
    if not expected:
        return 0.0
    hits = sum(target in results[:k] for target, results in zip(expected, ranked, strict=True))
    return hits / len(expected)


def evaluate_dataset(service, records, output_path: Path, configurations=None):
    configurations = configurations or DEFAULT_CONFIGURATIONS
    counts = Counter(record.product_key for record in records)
    queries = [record for record in records if counts[record.product_key] >= 2]
    report = {}
    for name, weight in configurations.items():
        expected = []
        rankings = []
        latencies = []
        for record in queries:
            started = time.perf_counter()
            response = service.search(
                record.path,
                exclude_image_id=record.image_id,
                fashion_weight=weight,
            )
            latencies.append((time.perf_counter() - started) * 1000)
            ordered = response.same_products + response.similar_products
            rankings.append([f"{item.shop_id}:{item.product_id}" for item in ordered])
            expected.append(record.product_key)
        report[name] = {
            "recall@1": compute_recall_at_k(expected, rankings, 1),
            "recall@5": compute_recall_at_k(expected, rankings, 5),
            "recall@10": compute_recall_at_k(expected, rankings, 10),
            "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "query_count": len(queries),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
