"""双模型召回、同款判断和商品级聚合。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import Settings
from .domain import ImageCandidate, ProductResult, SearchResponse
from .image_io import compute_phash, load_rgb


class SearchService:
    def __init__(self, settings, store, fashion_encoder, dino_encoder, calibrator) -> None:
        self.settings: Settings = settings
        self.store = store
        self.fashion_encoder = fashion_encoder
        self.dino_encoder = dino_encoder
        self.calibrator = calibrator

    @staticmethod
    def _aggregate(
        candidates: list[ImageCandidate],
        scores: np.ndarray,
        *,
        reason: str,
        probabilities: np.ndarray | None = None,
    ) -> list[ProductResult]:
        groups: dict[str, list[tuple[ImageCandidate, float, float | None]]] = defaultdict(list)
        for index, candidate in enumerate(candidates):
            key = f"{candidate.shop_id}:{candidate.product_id}"
            probability = None if probabilities is None else float(probabilities[index])
            groups[key].append((candidate, float(scores[index]), probability))

        products = []
        for matches in groups.values():
            matches.sort(key=lambda item: item[1], reverse=True)
            top = matches[:3]
            best_candidate, best_score, best_probability = top[0]
            aggregate_score = 0.7 * best_score + 0.3 * float(
                np.mean([score for _, score, _ in top])
            )
            products.append(
                ProductResult(
                    shop_id=best_candidate.shop_id,
                    product_id=best_candidate.product_id,
                    representative_path=best_candidate.path,
                    score=aggregate_score,
                    reason=reason,
                    same_probability=best_probability,
                )
            )
        products.sort(key=lambda item: item.score, reverse=True)
        return products

    def search(
        self,
        image_path: Path,
        *,
        exclude_image_id: str | None = None,
        fashion_weight: float = 0.8,
    ) -> SearchResponse:
        if not 0.0 <= fashion_weight <= 1.0:
            raise ValueError("FashionSigLIP 权重必须在 0 到 1 之间")

        image = load_rgb(image_path)
        query_phash = compute_phash(image)
        fashion = self.fashion_encoder.encode([image_path])[0]
        dino = self.dino_encoder.encode([image_path])[0]
        candidates = self.store.search_images(
            fashion,
            dino,
            top_k=self.settings.image_top_k,
            query_phash=query_phash,
            exclude_image_id=exclude_image_id,
        )
        if not candidates:
            return SearchResponse()

        feature_rows = np.asarray(
            [
                [
                    candidate.fashion_score,
                    candidate.dino_score,
                    1.0 - candidate.hamming_distance / 64.0,
                ]
                for candidate in candidates
            ],
            dtype=np.float32,
        )
        probabilities = self.calibrator.predict(feature_rows)
        same_candidates = [
            candidate
            for candidate, probability in zip(candidates, probabilities, strict=True)
            if probability >= self.calibrator.threshold
        ]
        same_probabilities = np.asarray(
            [
                probability
                for probability in probabilities
                if probability >= self.calibrator.threshold
            ],
            dtype=np.float32,
        )
        same = self._aggregate(
            same_candidates,
            same_probabilities,
            reason="高置信同款",
            probabilities=same_probabilities,
        )[: self.settings.same_limit]
        same_keys = {f"{item.shop_id}:{item.product_id}" for item in same}

        similar_candidates = [
            candidate
            for candidate in candidates
            if f"{candidate.shop_id}:{candidate.product_id}" not in same_keys
        ]
        similar_scores = np.asarray(
            [
                fashion_weight * candidate.fashion_score
                + (1.0 - fashion_weight) * candidate.dino_score
                for candidate in similar_candidates
            ],
            dtype=np.float32,
        )
        similar = self._aggregate(
            similar_candidates, similar_scores, reason="风格与视觉相似"
        )[: self.settings.similar_limit]
        return SearchResponse(same_products=same, similar_products=similar)
