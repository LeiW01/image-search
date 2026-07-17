from pathlib import Path

import numpy as np

from fashion_search.calibration import (
    SameProductCalibrator,
    pair_features,
    select_high_precision_threshold,
)


def test_pair_features_combine_two_cosines_and_phash() -> None:
    features = pair_features(
        np.array([1.0, 0.0]),
        np.array([0.8, 0.6]),
        "0000000000000000",
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        "000000000000000f",
    )

    np.testing.assert_allclose(features, [1.0, 0.6, 60 / 64])


def test_threshold_prioritizes_precision() -> None:
    threshold = select_high_precision_threshold(
        np.array([1, 0, 1]), np.array([0.9, 0.8, 0.7]), target_precision=0.95
    )
    assert threshold == 0.9
    assert select_high_precision_threshold(np.array([0, 0]), np.array([0.9, 0.8])) == 1.0


def test_calibrator_round_trip(tmp_path: Path) -> None:
    features = np.array(
        [[0.99, 0.98, 1.0], [0.95, 0.96, 0.9], [0.2, 0.1, 0.0], [0.3, 0.2, 0.1]]
    )
    labels = np.array([1, 1, 0, 0])
    calibrator = SameProductCalibrator().fit(features, labels)
    before = calibrator.predict(features)
    path = tmp_path / "calibrator.joblib"
    calibrator.save(path)

    restored = SameProductCalibrator.load(path)

    np.testing.assert_allclose(restored.predict(features), before)
    assert restored.threshold == calibrator.threshold
