"""H0-mini extractor with a saved paired-delta eraser."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module


@dataclass(frozen=True)
class SavedLinearEraser:
    """Affine eraser reconstructed from the NPZ produced by the fitting CLI.

    Supports both explicit full matrices and implicit low-rank residuals:

        P = I - proj_left @ proj_right
    """

    P: Tensor | None
    proj_left: Tensor | None
    proj_right: Tensor | None
    bias: Tensor | None
    metadata: dict[str, Any]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SavedLinearEraser":
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Paired-delta eraser not found: {path}")

        with np.load(path, allow_pickle=False) as data:
            P = cls._load_tensor(data, "P", map_location, dtype)
            proj_left = cls._load_tensor(data, "proj_left", map_location, dtype)
            proj_right = cls._load_tensor(data, "proj_right", map_location, dtype)
            bias = cls._load_tensor(data, "bias", map_location, dtype)
            metadata = cls._load_metadata(data)
            available = set(data.files)

        if P is None and (proj_left is None or proj_right is None):
            raise KeyError(
                "Invalid eraser file. Expected either 'P', or both "
                f"'proj_left' and 'proj_right'. Available keys: {sorted(available)}"
            )

        if P is not None and (P.ndim != 2 or P.shape[0] != P.shape[1]):
            raise ValueError(f"Expected square P, got shape {tuple(P.shape)}.")

        if proj_left is not None and proj_right is not None:
            if proj_left.ndim != 2 or proj_right.ndim != 2:
                raise ValueError("proj_left and proj_right must be matrices.")
            if proj_left.shape[1] != proj_right.shape[0]:
                raise ValueError(
                    "Low-rank factor mismatch: "
                    f"{tuple(proj_left.shape)} vs {tuple(proj_right.shape)}."
                )
            if proj_left.shape[0] != proj_right.shape[1]:
                raise ValueError("Low-rank factors do not define a square map.")

        return cls(
            P=P,
            proj_left=proj_left,
            proj_right=proj_right,
            bias=bias,
            metadata=metadata,
        )

    @staticmethod
    def _load_tensor(
        data: np.lib.npyio.NpzFile,
        name: str,
        map_location: torch.device | str | None,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if name not in data.files:
            return None
        tensor = torch.from_numpy(np.asarray(data[name])).to(dtype=dtype)
        return tensor.to(map_location) if map_location is not None else tensor

    @staticmethod
    def _load_metadata(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
        if "metadata_json" not in data.files:
            return {}
        raw = data["metadata_json"]
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(str(raw))

    @property
    def input_dim(self) -> int:
        if self.P is not None:
            return int(self.P.shape[0])
        assert self.proj_left is not None
        return int(self.proj_left.shape[0])

    def apply_linear(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, got {x.shape[-1]}."
            )

        if self.P is not None:
            return x @ self.P.mH

        if self.proj_left is None or self.proj_right is None:
            raise RuntimeError("Missing low-rank eraser factors.")

        correction = (x @ self.proj_right.mH) @ self.proj_left.mH
        return x - correction

    def transform_delta(self, delta: Tensor) -> Tensor:
        return self.apply_linear(delta)

    def __call__(self, x: Tensor) -> Tensor:
        centered = x - self.bias if self.bias is not None else x
        output = self.apply_linear(centered)
        return output + self.bias if self.bias is not None else output

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SavedLinearEraser":
        def move(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.to(device=device, dtype=dtype)

        return SavedLinearEraser(
            P=move(self.P),
            proj_left=move(self.proj_left),
            proj_right=move(self.proj_right),
            bias=move(self.bias),
            metadata=self.metadata,
        )


def _resolve_eraser_path(
    eraser_path: str | Path | None,
    *,
    env_var: str = "ERASER_PATH",
) -> Path:
    resolved = eraser_path or os.environ.get(env_var)
    if resolved is None:
        raise ValueError(
            "Missing paired-delta eraser path. Provide eraser_path=... "
            f"or set {env_var}."
        )

    path = Path(resolved).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Paired-delta eraser not found: {path}")
    return path


class H0MiniWithEraser(Extractor):
    """H0-mini with a saved PCA or soft paired-delta eraser."""

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
        self.eraser = SavedLinearEraser.load(
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

        reference = (
            self.eraser.P if self.eraser.P is not None else self.eraser.proj_left
        )
        assert reference is not None
        features = features.to(dtype=reference.dtype)
        features = self.eraser(features)

        if self.normalize_output:
            features = F.normalize(features, dim=-1)

        return features.detach().cpu().numpy().astype(np.float32)


class HOptimus1WithEraser(Extractor):
    """H-Optimus-1 model developped by Bioptimus available on Hugging-Face (1).

    You will need to be granted access to be able to use this model.

    .. note::
        (1) https://huggingface.co/bioptimus/H-optimus-1

    Parameters
    ----------
    device: int | list[int] | None = DEFAULT_DEVICE,
        Compute resources to use.
        If None, will use all available GPUs.
        If -1, extraction will run on CPU.
    mixed_precision: bool = True
        Whether to use mixed_precision.

    """

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
        eraser_path: str | Path | None = None,
        normalize_output: bool = False,
    ):
        super().__init__()
        self.output_dim = 1536
        self.embedding_dim = 1536
        self.mixed_precision = mixed_precision
        self.normalize_output = normalize_output

        timm_kwargs: dict[str, Any] = {
            "init_values": 1e-5,
            "dynamic_img_size": False,
        }
        feature_extractor = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1", pretrained=True, **timm_kwargs
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
        self.eraser = SavedLinearEraser.load(
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

    @property  # type: ignore
    def transform(self) -> transforms.Compose:
        """Transform method to apply element wise."""
        return transforms.Compose(
            [
                transforms.ToTensor(),  # swap axes and normalize
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> np.ndarray:
        """Compute and return features.

        Parameters
        ----------
        images: torch.Tensor
            Input of size (n_tiles, n_channels, dim_x, dim_y).

        Returns
        -------
            torch.Tensor: Tensor of size (n_tiles, features_dim).
        """
        model_device = next(self.feature_extractor.parameters()).device
        images = images.to(model_device)

        autocast_enabled = self.mixed_precision and model_device.type == "cuda"
        with torch.autocast(
            device_type=model_device.type,
            enabled=autocast_enabled,
        ):
            features = self.feature_extractor(images)

        reference = (
            self.eraser.P if self.eraser.P is not None else self.eraser.proj_left
        )
        assert reference is not None
        features = features.to(dtype=reference.dtype)
        features = self.eraser(features)

        if self.normalize_output:
            features = F.normalize(features, dim=-1)

        return features.detach().cpu().numpy().astype(np.float32)
