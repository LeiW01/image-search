"""FashionSigLIP 与 DINOv2 图片编码器。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from .config import Settings
from .image_io import load_rgb


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("向量必须是二维数组")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("不能归一化零向量")
    return values / norms


def preferred_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class FashionSiglipEncoder:
    """使用 Marqo 固定版本的 FashionSigLIP 输出图片向量。"""

    def __init__(self, settings: Settings, device: str | None = None) -> None:
        self.settings = settings
        self.device = device or preferred_device()
        self._model = None
        self._preprocess = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import open_clip

        weights = hf_hub_download(
            repo_id=self.settings.fashion_model,
            filename="open_clip_model.safetensors",
            revision=self.settings.fashion_revision,
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP",
            pretrained=weights,
            device=self.device,
            force_image_size=224,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            image_interpolation="bicubic",
            image_resize_mode="squash",
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess

    def _encode_once(self, paths: Sequence[Path]) -> np.ndarray:
        self._load()
        assert self._model is not None and self._preprocess is not None
        batch = torch.stack([self._preprocess(load_rgb(path)) for path in paths]).to(self.device)
        with torch.inference_mode():
            output = self._model.encode_image(batch)
        return l2_normalize(output.float().cpu().numpy())

    def encode(self, paths: Sequence[Path]) -> np.ndarray:
        if not paths:
            return np.empty((0, self.settings.vector_size), dtype=np.float32)
        try:
            return self._encode_once(paths)
        except RuntimeError:
            if self.device != "mps" or self._model is None:
                raise
            self.device = "cpu"
            self._model = self._model.to("cpu")
            return self._encode_once(paths)


class DinoV2Encoder:
    """使用固定版本 DINOv2 ViT-B/14 的 CLS 特征。"""

    def __init__(self, settings: Settings, device: str | None = None) -> None:
        self.settings = settings
        self.device = device or preferred_device()
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoImageProcessor, AutoModel

        self._processor = AutoImageProcessor.from_pretrained(
            self.settings.dino_model,
            revision=self.settings.dino_revision,
        )
        self._model = AutoModel.from_pretrained(
            self.settings.dino_model,
            revision=self.settings.dino_revision,
            use_safetensors=True,
        ).to(self.device)
        self._model.eval()

    def _encode_once(self, paths: Sequence[Path]) -> np.ndarray:
        self._load()
        assert self._processor is not None and self._model is not None
        images = [load_rgb(path) for path in paths]
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self._model(**inputs).last_hidden_state[:, 0]
        return l2_normalize(output.float().cpu().numpy())

    def encode(self, paths: Sequence[Path]) -> np.ndarray:
        if not paths:
            return np.empty((0, self.settings.vector_size), dtype=np.float32)
        try:
            return self._encode_once(paths)
        except RuntimeError:
            if self.device != "mps" or self._model is None:
                raise
            self.device = "cpu"
            self._model = self._model.to("cpu")
            return self._encode_once(paths)
