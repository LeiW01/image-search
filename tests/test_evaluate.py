from pathlib import Path

from fashion_search.domain import ImageRecord, ProductResult, SearchResponse
from fashion_search.evaluate import compute_recall_at_k, evaluate_dataset


def _record(image_id: str, product_id: str) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        shop_id="shop",
        product_id=product_id,
        path=Path(f"/{image_id}.jpg"),
        relative_path=f"shop/{product_id}/display/{image_id}.jpg",
        modified_ns=1,
        size_bytes=1,
        source_digest=image_id,
    )


def test_compute_recall_at_k() -> None:
    expected = ["a", "c"]
    ranked = [["a", "b"], ["x", "c"]]
    assert compute_recall_at_k(expected, ranked, 1) == 0.5
    assert compute_recall_at_k(expected, ranked, 5) == 1.0


def test_evaluation_only_uses_multi_image_products_and_excludes_query(tmp_path: Path) -> None:
    excluded = []

    class FakeService:
        def search(self, path, *, exclude_image_id, fashion_weight):
            excluded.append(exclude_image_id)
            result = ProductResult("shop", "p1", Path("/result.jpg"), 1.0, "test")
            return SearchResponse(similar_products=[result])

    records = [_record("a", "p1"), _record("b", "p1"), _record("c", "p2")]
    output = tmp_path / "evaluation.json"

    report = evaluate_dataset(FakeService(), records, output, configurations={"融合0.8": 0.8})

    assert set(excluded) == {"a", "b"}
    assert report["融合0.8"]["recall@1"] == 1.0
    assert report["融合0.8"]["query_count"] == 2
    assert output.is_file()
