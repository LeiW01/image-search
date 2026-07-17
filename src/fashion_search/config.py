"""项目运行配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """集中保存本地原型的路径、模型与检索参数。"""

    project_root: Path
    image_root: Path
    state_dir: Path
    qdrant_path: Path
    fashion_weights_path: Path
    dino_local_dir: Path
    collection_name: str
    fashion_model: str
    fashion_revision: str
    dino_model: str
    dino_revision: str
    vector_size: int
    batch_size: int
    image_top_k: int
    same_limit: int
    similar_limit: int
    bind_host: str
    bind_port: int

    @classmethod
    def default(cls, project_root: Path) -> "Settings":
        root = project_root.expanduser().resolve()
        state_dir = root / "state"
        return cls(
            project_root=root,
            image_root=Path("/Users/wanglei/work/simarket/data/images"),
            state_dir=state_dir,
            qdrant_path=state_dir / "qdrant",
            fashion_weights_path=(
                state_dir
                / "models"
                / "marqo-fashionSigLIP"
                / "open_clip_model.safetensors"
            ),
            dino_local_dir=state_dir / "models" / "dinov2-base",
            collection_name="fashion_products_v1",
            fashion_model="Marqo/marqo-fashionSigLIP",
            fashion_revision="c56244cc94f92419e8369fa71efdaf403b124ce8",
            dino_model="facebook/dinov2-base",
            dino_revision="f9e44c814b77203eaa57a6bdbbd535f21ede1415",
            vector_size=768,
            batch_size=16,
            image_top_k=200,
            same_limit=5,
            similar_limit=20,
            bind_host="127.0.0.1",
            bind_port=7860,
        )
