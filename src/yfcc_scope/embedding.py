# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .settings import DINOV3_REPO_DIR
from .types import EmbeddingType

_device = "cuda" if torch.cuda.is_available() else "cpu"


def _autocast_context():
    if _device == "cuda":
        return torch.autocast(_device)
    return contextlib.nullcontext()


class LazyModel:
    """Singleton base class for lazily-loaded embedding models.

    Instantiating a concrete subclass returns a shared instance per subclass
    (the first instantiation triggers _load). Each subclass shadows
    _instance with its own class attribute, so ClipModel and DINOv3Model
    are independent singletons.
    """

    _instance: LazyModel | None = None
    _lock = threading.Lock()

    model: Any
    preprocess: Callable
    tokenizer: Callable | None

    def __new__(cls) -> LazyModel:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._load()
                    cls._instance = instance
        return cls._instance

    def _load(self) -> None:
        raise NotImplementedError

    def _encode_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _encode_text(self, text_input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def image_features(self, image: bytes) -> np.ndarray:
        img = Image.open(BytesIO(image)).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0).to(_device)
        with torch.no_grad(), _autocast_context():
            feat = self._encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.float().cpu().numpy().astype(np.float16, copy=False)

    def text_features(self, texts: list[str]) -> np.ndarray:
        assert self.tokenizer is not None
        text_input = self.tokenizer(texts).to(_device)
        with torch.no_grad(), _autocast_context():
            feat = self._encode_text(text_input)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.float().cpu().numpy().astype(np.float16, copy=False)


class ClipModel(LazyModel):
    def _load(self) -> None:
        import open_clip

        model, self.preprocess, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model.eval()
        self.model = model.to(_device)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def _encode_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(img_tensor)

    def _encode_text(self, text_input: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(text_input)


class DINOv3Model(LazyModel):
    def _load(self) -> None:
        from torchvision.transforms import v2

        repo_dir = Path(DINOV3_REPO_DIR)
        checkpoint = (
            repo_dir / "checkpoints" / "dinov3_vits16plus_pretrain_lvd1689m.pth"
        )
        model = torch.hub.load(
            str(repo_dir),
            "dinov3_vits16plus",
            source="local",
            weights=str(checkpoint),
        )
        model.eval()
        self.model = model.to(_device)
        self.preprocess = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((256, 256), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def _encode_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(img_tensor)


def image_model(embedding_type: EmbeddingType) -> LazyModel:
    if embedding_type.name.startswith("dinov3"):
        return DINOv3Model()
    return ClipModel()
