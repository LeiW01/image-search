from pathlib import Path

from fashion_search.config import Settings


def test_default_settings_are_local_and_pinned(tmp_path: Path) -> None:
    settings = Settings.default(tmp_path)

    assert settings.image_root == Path("/Users/wanglei/work/simarket/data/images")
    assert settings.state_dir == tmp_path / "state"
    assert settings.qdrant_path == tmp_path / "state" / "qdrant"
    assert settings.fashion_revision == "c56244cc94f92419e8369fa71efdaf403b124ce8"
    assert settings.dino_revision == "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
    assert settings.vector_size == 768
    assert settings.bind_host == "127.0.0.1"

