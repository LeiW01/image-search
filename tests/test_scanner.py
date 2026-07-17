from pathlib import Path

from fashion_search.scanner import scan_display_images


def test_scan_only_display_images_and_extract_product_ids(tmp_path: Path) -> None:
    display = tmp_path / "shop-a" / "product-1" / "display"
    detail = tmp_path / "shop-a" / "product-1" / "detail"
    nested = display / "nested"
    nested.mkdir(parents=True)
    detail.mkdir(parents=True)
    (display / "front.jpg").write_bytes(b"front")
    (nested / "side.webp").write_bytes(b"side")
    (display / "notes.txt").write_text("ignore")
    (detail / "long.jpg").write_bytes(b"detail")

    records, errors = scan_display_images(tmp_path)

    assert errors == []
    assert [record.relative_path for record in records] == [
        "shop-a/product-1/display/front.jpg",
        "shop-a/product-1/display/nested/side.webp",
    ]
    assert {record.product_key for record in records} == {"shop-a:product-1"}
    assert len({record.image_id for record in records}) == 2
    assert all(record.source_digest for record in records)


def test_image_ids_are_stable(tmp_path: Path) -> None:
    display = tmp_path / "shop" / "product" / "display"
    display.mkdir(parents=True)
    (display / "a.png").write_bytes(b"same-content")

    first, _ = scan_display_images(tmp_path)
    second, _ = scan_display_images(tmp_path)

    assert first[0].image_id == second[0].image_id
    assert first[0].source_digest == second[0].source_digest


def test_missing_root_is_reported_without_raising(tmp_path: Path) -> None:
    records, errors = scan_display_images(tmp_path / "missing")

    assert records == []
    assert len(errors) == 1
