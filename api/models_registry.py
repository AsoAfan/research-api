"""Singleton loaders + inference helpers for the four classifiers.

Architectures reconstructed locally (NOT imported from resnet101.py / vgg19.py,
which run training as a side-effect of import). Class order is alphabetical to
match the ImageFolder ordering used during training.
"""
from __future__ import annotations

import pathlib
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# --- Windows-pickled YOLO checkpoint shim (see eval_yolo.py) ---
sys.modules.setdefault("pathlib._local", pathlib)
if sys.platform != "win32":
    class _PortableWindowsPath(pathlib.PurePosixPath):
        pass
    pathlib.WindowsPath = _PortableWindowsPath  # type: ignore[attr-defined]
    if not hasattr(pathlib, "_local"):
        pathlib._local = pathlib  # type: ignore[attr-defined]
    pathlib._local.WindowsPath = _PortableWindowsPath  # type: ignore[attr-defined]

from ultralytics import YOLO  # noqa: E402  (must come after shim)


CLASSES = ["bulge", "extrusion", "normal", "protrusion"]
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESNET_WEIGHTS = REPO_ROOT / "models" / "resnet101" / "2_best_resnet101.pth"
VGG_WEIGHTS = REPO_ROOT / "models" / "vgg" / "2_best_vgg19.pth"
YOLO22_WEIGHTS = REPO_ROOT / "models" / "yolo" / "train-22" / "weights" / "best.pt"
YOLO26_WEIGHTS = REPO_ROOT / "models" / "yolo" / "train-26" / "weights" / "best.pt"


# ResNet101 inference transform — matches eval_resnet.py
RESNET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# VGG19 inference transform — matches eval_vgg.py (no normalize, no resize)
# The crops are already 224x224, so ToTensor is enough.
VGG_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
])


class ResNet101_Model(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = models.resnet101(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class SeqSelfAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = torch.bmm(Q, K.transpose(1, 2)) / (x.size(-1) ** 0.5)
        attn = self.softmax(attn)
        return torch.bmm(attn, V)


class VGG19_Attention(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        vgg = models.vgg19(weights=None)
        self.features = nn.Sequential(*list(vgg.features.children())[:28])
        self.attention = SeqSelfAttention(512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        B, C, _, _ = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.attention(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# Registry — load once at startup
# --------------------------------------------------------------------------- #
_registry: Dict[str, object] = {}


def _to_pil(crop_uint8: np.ndarray) -> Image.Image:
    """A disc crop is a 2D uint8 grayscale array. Convert to a 3-channel PIL image."""
    if crop_uint8.ndim == 2:
        rgb = cv2.cvtColor(crop_uint8, cv2.COLOR_GRAY2RGB)
    elif crop_uint8.ndim == 3 and crop_uint8.shape[2] == 3:
        rgb = crop_uint8
    else:
        raise ValueError(f"Unexpected crop shape: {crop_uint8.shape}")
    return Image.fromarray(rgb)


def _normalize_crop_to_uint8(crop: np.ndarray) -> np.ndarray:
    """The cropped slices come straight from NIfTI as float; normalize to uint8."""
    if crop.dtype == np.uint8:
        return crop
    arr = crop.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn) * 255.0
    return arr.astype(np.uint8)


def load_all() -> Dict[str, object]:
    if _registry:
        return _registry

    print(f"[models_registry] device={DEVICE}")

    # ResNet101
    resnet = ResNet101_Model(NUM_CLASSES).to(DEVICE)
    resnet.load_state_dict(torch.load(str(RESNET_WEIGHTS), map_location=DEVICE))
    resnet.eval()
    _registry["resnet101"] = resnet
    print(f"[models_registry] loaded resnet101 from {RESNET_WEIGHTS}")

    # VGG19 + attention
    vgg = VGG19_Attention(NUM_CLASSES).to(DEVICE)
    vgg.load_state_dict(torch.load(str(VGG_WEIGHTS), map_location=DEVICE))
    vgg.eval()
    _registry["vgg19"] = vgg
    print(f"[models_registry] loaded vgg19 from {VGG_WEIGHTS}")

    # YOLO classifiers
    _registry["yolo22"] = YOLO(str(YOLO22_WEIGHTS))
    print(f"[models_registry] loaded yolo22 from {YOLO22_WEIGHTS}")

    _registry["yolo26"] = YOLO(str(YOLO26_WEIGHTS))
    print(f"[models_registry] loaded yolo26 from {YOLO26_WEIGHTS}")

    return _registry


def _yolo_class_index_map(yolo_model) -> Dict[int, str]:
    """Returns a dict mapping YOLO class index → CLASSES label."""
    names = yolo_model.names  # e.g. {0: 'bulge', 1: 'extrusion', ...}
    if isinstance(names, dict):
        return {int(k): str(v).lower() for k, v in names.items()}
    return {i: str(v).lower() for i, v in enumerate(names)}


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_torch(model: nn.Module, crop: np.ndarray, transform: transforms.Compose) -> Tuple[str, Dict[str, float], float]:
    crop_u8 = _normalize_crop_to_uint8(crop)
    img = _to_pil(crop_u8)
    x = transform(img).unsqueeze(0).to(DEVICE)
    t0 = time.time()
    logits = model(x)
    elapsed = (time.time() - t0) * 1000.0
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return CLASSES[idx], {CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}, elapsed


def predict_yolo(yolo_model, crop: np.ndarray) -> Tuple[str, Dict[str, float], float]:
    crop_u8 = _normalize_crop_to_uint8(crop)
    if crop_u8.ndim == 2:
        rgb = cv2.cvtColor(crop_u8, cv2.COLOR_GRAY2RGB)
    else:
        rgb = crop_u8
    t0 = time.time()
    results = yolo_model.predict(source=rgb, imgsz=224, verbose=False)
    elapsed = (time.time() - t0) * 1000.0
    r = results[0]
    probs_t = r.probs.data.cpu().numpy()  # tensor over yolo's own class order
    idx_to_name = _yolo_class_index_map(yolo_model)
    probs: Dict[str, float] = {c: 0.0 for c in CLASSES}
    for i, p in enumerate(probs_t):
        name = idx_to_name.get(i, "").lower()
        if name in probs:
            probs[name] = float(p)
    pred = max(probs, key=probs.get)
    return pred, probs, elapsed


def predict_one(model_name: str, crop: np.ndarray) -> Tuple[str, Dict[str, float], float]:
    reg = load_all()
    if model_name == "resnet101":
        return predict_torch(reg["resnet101"], crop, RESNET_TRANSFORM)  # type: ignore[arg-type]
    if model_name == "vgg19":
        return predict_torch(reg["vgg19"], crop, VGG_TRANSFORM)  # type: ignore[arg-type]
    if model_name == "yolo22":
        return predict_yolo(reg["yolo22"], crop)
    if model_name == "yolo26":
        return predict_yolo(reg["yolo26"], crop)
    raise ValueError(f"Unknown model: {model_name}")
