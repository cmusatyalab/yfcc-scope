# SPDX-FileCopyrightText: 2025, 2026 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

from __future__ import annotations

import contextlib
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .settings import DINOV3_REPO_DIR

_device = "cuda" if torch.cuda.is_available() else "cpu"

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_dino_model = None
_dino_preprocess = None


def _autocast_context():
    if _device == "cuda":
        return torch.autocast(_device)
    return contextlib.nullcontext()


def _load_clip_model():
    global _clip_model, _clip_preprocess, _clip_tokenizer

    if _clip_model is not None:
        return

    import open_clip

    _clip_model, _clip_preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    _clip_model.eval()
    _clip_model = _clip_model.to(_device)
    _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")


def clip_text_features(texts):
    _load_clip_model()

    text_input = _clip_tokenizer(texts).to(_device)
    with torch.no_grad(), _autocast_context():
        text_feat = _clip_model.encode_text(text_input)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat.float().cpu().numpy().astype(np.float16, copy=False)


def clip_image_features(image_bytes):
    _load_clip_model()

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_tensor = _clip_preprocess(img).unsqueeze(0).to(_device)
    with torch.no_grad(), _autocast_context():
        img_feat = _clip_model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    return img_feat.float().cpu().numpy().astype(np.float16, copy=False)


def _load_dino_model():
    global _dino_model, _dino_preprocess

    if _dino_model is not None:
        return

    from torchvision.transforms import v2

    repo_dir = Path(DINOV3_REPO_DIR)
    _dino_model = torch.hub.load(
        str(repo_dir),
        "dinov3_vits16plus",
        source="local",
        weights=str(repo_dir / "checkpoints/dinov3_vits16plus_pretrain_lvd1689m.pth"),
    )
    _dino_preprocess = v2.Compose(
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
    _dino_model.eval()
    _dino_model = _dino_model.to(_device)


def dino_image_features(image_bytes):
    _load_dino_model()

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_tensor = _dino_preprocess(img).unsqueeze(0).to(_device)
    with torch.no_grad(), _autocast_context():
        img_feat = _dino_model(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    return img_feat.float().cpu().numpy().astype(np.float16, copy=False)
