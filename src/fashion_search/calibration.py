"""用商品目录标签校准“同款概率”。"""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from .image_io import phash_distance


def pair_features(
    fashion_left: np.ndarray,
    dino_left: np.ndarray,
    phash_left: str,
    fashion_right: np.ndarray,
    dino_right: np.ndarray,
    phash_right: str,
) -> np.ndarray:
    return np.asarray(
        [
            float(np.dot(fashion_left, fashion_right)),
            float(np.dot(dino_left, dino_right)),
            1.0 - phash_distance(phash_left, phash_right) / 64.0,
        ],
        dtype=np.float32,
    )


def select_high_precision_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_precision: float = 0.95,
) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    acceptable = []
    for threshold in np.unique(probabilities):
        predicted = probabilities >= threshold
        count = int(predicted.sum())
        if not count:
            continue
        precision = float(labels[predicted].sum() / count)
        if precision >= target_precision:
            acceptable.append(float(threshold))
    return min(acceptable) if acceptable else 1.0


class SameProductCalibrator:
    def __init__(self, target_precision: float = 0.95) -> None:
        self.target_precision = target_precision
        self.model = LogisticRegression(class_weight="balanced", random_state=42)
        self.threshold = 1.0

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        validation_features: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
    ) -> "SameProductCalibrator":
        self.model.fit(features, labels)
        check_features = features if validation_features is None else validation_features
        check_labels = labels if validation_labels is None else validation_labels
        probabilities = self.model.predict_proba(check_features)[:, 1]
        self.threshold = select_high_precision_threshold(
            check_labels, probabilities, self.target_precision
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.atleast_2d(features))[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "SameProductCalibrator":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("校准文件类型不正确")
        return loaded


def build_training_pairs(points, max_positive_per_product: int = 20, seed: int = 42):
    """由 Qdrant points 构造同商品正样本与数量相近的跨商品负样本。"""
    groups = defaultdict(list)
    for point in points:
        payload = point.payload or {}
        groups[str(payload.get("product_key"))].append(point)

    rng = random.Random(seed)
    positive_pairs = []
    for group in groups.values():
        pairs = list(itertools.combinations(group, 2))
        rng.shuffle(pairs)
        positive_pairs.extend(pairs[:max_positive_per_product])

    all_points = [point for group in groups.values() for point in group]
    negative_pairs = []
    attempts = 0
    while len(negative_pairs) < len(positive_pairs) and attempts < max(100, len(positive_pairs) * 20):
        attempts += 1
        left, right = rng.sample(all_points, 2)
        if left.payload.get("product_key") != right.payload.get("product_key"):
            negative_pairs.append((left, right))

    features = []
    labels = []
    for label, pairs in ((1, positive_pairs), (0, negative_pairs)):
        for left, right in pairs:
            features.append(
                pair_features(
                    np.asarray(left.vector["fashion"]),
                    np.asarray(left.vector["dino"]),
                    str(left.payload["phash"]),
                    np.asarray(right.vector["fashion"]),
                    np.asarray(right.vector["dino"]),
                    str(right.payload["phash"]),
                )
            )
            labels.append(label)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int8)
