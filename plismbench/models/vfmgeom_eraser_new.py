"""PLISMBench extractors using vfmgeom saved eraser artifacts.

This file assumes that your vfmgeom repository is installed in the PLISMBench
environment, for example:

    pip install -e /home/valentin/workspaces/vfm-geom-xai

It supports both single eraser NPZ files and chained eraser NPZ files through
``vfmgeom.erasers.saved.SavedEraser``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torchvision import transforms
from vfmgeom.concept_erasure.loader import SavedEraser, SavedEraserArtifact

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module


def _resolve_eraser_path(
    eraser_path: str | Path | None,
    *,
    env_var: str = "ERASER_PATH",
) -> Path:
    resolved = eraser_path or os.environ.get(env_var)
    if resolved is None:
        raise ValueError(
            f"Missing saved eraser path. Provide eraser_path=... or set {env_var}."
        )

    path = Path(resolved).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Saved eraser not found: {path}")
    return path


def _apply_saved_eraser(
    features: torch.Tensor,
    eraser: SavedEraserArtifact,
) -> torch.Tensor:
    """Apply a saved eraser while keeping wrapper code chain/single agnostic."""
    features = features.to(dtype=eraser.reference_tensor.dtype)
    return eraser(features)


class H0MiniWithEraser(Extractor):
    """H0-mini with a saved vfmgeom eraser artifact."""

    output_dim = 768

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
        eraser_path: str | Path | None = None,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_dim = 768
        self.mixed_precision = mixed_precision
        self.normalize_output = normalize_output

        timm_kwargs: dict[str, Any] = {
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }
        feature_extractor = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=True,
            **timm_kwargs,
        )

        self.feature_extractor, self.device = prepare_module(
            feature_extractor,
            device,
            self.mixed_precision,
        )

        if self.device is None:
            self.feature_extractor = self.feature_extractor.module

        self.eraser_path = _resolve_eraser_path(eraser_path)

        model_device = next(self.feature_extractor.parameters()).device
        self.eraser = SavedEraser.load(
            self.eraser_path,
            map_location=model_device,
            dtype=torch.float32,
        )

        if self.eraser.input_dim != self.embedding_dim:
            raise ValueError(
                "Eraser/encoder dimension mismatch: "
                f"eraser expects {self.eraser.input_dim}, "
                f"H0-mini outputs {self.embedding_dim}."
            )

    @property  # type: ignore[override]
    def transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> np.ndarray:
        model_device = next(self.feature_extractor.parameters()).device
        images = images.to(model_device)

        autocast_enabled = self.mixed_precision and model_device.type == "cuda"
        with torch.autocast(
            device_type=model_device.type,
            enabled=autocast_enabled,
        ):
            last_hidden_state = self.feature_extractor(images)
            features = last_hidden_state[:, 0]

        features = _apply_saved_eraser(features, self.eraser)

        if self.normalize_output:
            features = F.normalize(features, dim=-1)

        return features.detach().cpu().numpy().astype(np.float32)


class HOptimus1WithEraser(Extractor):
    """H-optimus-1 with a saved vfmgeom eraser artifact."""

    output_dim = 1536

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
        eraser_path: str | Path | None = None,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_dim = 1536
        self.mixed_precision = mixed_precision
        self.normalize_output = normalize_output

        timm_kwargs: dict[str, Any] = {
            "init_values": 1e-5,
            "dynamic_img_size": False,
        }
        feature_extractor = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1",
            pretrained=True,
            **timm_kwargs,
        )

        self.feature_extractor, self.device = prepare_module(
            feature_extractor,
            device,
            self.mixed_precision,
        )

        if self.device is None:
            self.feature_extractor = self.feature_extractor.module

        self.eraser_path = _resolve_eraser_path(eraser_path)

        model_device = next(self.feature_extractor.parameters()).device
        self.eraser = SavedEraser.load(
            self.eraser_path,
            map_location=model_device,
            dtype=torch.float32,
        )

        if self.eraser.input_dim != self.embedding_dim:
            raise ValueError(
                "Eraser/encoder dimension mismatch: "
                f"eraser expects {self.eraser.input_dim}, "
                f"H-optimus-1 outputs {self.embedding_dim}."
            )

    @property  # type: ignore[override]
    def transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> np.ndarray:
        model_device = next(self.feature_extractor.parameters()).device
        images = images.to(model_device)

        autocast_enabled = self.mixed_precision and model_device.type == "cuda"
        with torch.autocast(
            device_type=model_device.type,
            enabled=autocast_enabled,
        ):
            features = self.feature_extractor(images)

        features = _apply_saved_eraser(features, self.eraser)

        if self.normalize_output:
            features = F.normalize(features, dim=-1)

        return features.detach().cpu().numpy().astype(np.float32)
