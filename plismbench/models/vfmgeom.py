"""Models adapted with LEACE."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torchvision import transforms
from vfmgeom.concept_erasure.leace import LeaceEraser

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module


def _resolve_leace_path(
    leace_path: str | Path | None,
    *,
    env_var: str,
    default_path: str | Path | None = None,
) -> Path:
    """
    Resolve LEACE eraser path.

    Priority:
        1. explicit leace_path argument
        2. model-specific environment variable
        3. generic LEACE_ERASER_PATH environment variable
        4. default_path
    """
    resolved = (
        leace_path
        or os.environ.get(env_var)
        or os.environ.get("LEACE_ERASER_PATH")
        or default_path
    )

    if resolved is None:
        raise ValueError(
            "Missing LEACE eraser path. Provide leace_path=..., "
            f"set {env_var}, or set LEACE_ERASER_PATH."
        )

    resolved = Path(resolved).expanduser()

    if not resolved.exists():
        raise FileNotFoundError(f"LEACE eraser not found: {resolved}")

    return resolved


class LeaceMixin:
    """Mixin to load and apply a LEACE eraser to extracted features."""

    def _init_leace(
        self,
        leace_path: str | Path | None,
        *,
        env_var: str,
        default_path: str | Path | None = None,
        normalize_output: bool | None = None,
    ) -> None:
        self.leace_path = _resolve_leace_path(
            leace_path,
            env_var=env_var,
            default_path=default_path,
        )

        self.leace_eraser = LeaceEraser.load(
            self.leace_path,
            map_location=self.device,
        ).to(self.device)

        if normalize_output is None:
            normalize_output = os.environ.get("LEACE_NORMALIZE_OUTPUT", "0") == "1"

        self.normalize_output = normalize_output

    def _apply_leace(self, features: torch.Tensor) -> torch.Tensor:
        features = self.leace_eraser(features)

        if self.normalize_output:
            features = F.normalize(features, dim=-1)

        return features

    def _to_numpy(self, features: torch.Tensor) -> np.ndarray:
        return features.detach().cpu().numpy()


class H0MiniWithLeace(Extractor):
    """H0-mini with a LEACE projection applied to CLS-token features."""

    DEFAULT_LEACE_PATH = (
        "/home/valentin/workspaces/vfm-geom-xai/output/leace/"
        "scanner_scorpion/0.5-mpp/h0mini/weights.pt"
    )

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
        pca_path: str | Path | None = None,
        normalize_output: bool | None = None,
        pca_components: int = 64,
    ):
        super().__init__()

        self.output_dim = 768
        self.mixed_precision = mixed_precision

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

        self._init_pca(
            pca_path,
            env_var="H0MINI_LEACE_ERASER_PATH",
            default_path=self.DEFAULT_LEACE_PATH,
            normalize_output=normalize_output,
        )

    def _init_pca(
        self,
        pca_path: str | Path | None,
        *,
        env_var: str,
        default_path: str | Path | None = None,
        normalize_output: bool | None = None,
    ) -> None:
        pca_path_str = str(pca_path or os.environ.get(env_var) or default_path)
        resolved_path = Path(pca_path_str).resolve()

        pca_data = np.load(resolved_path)
        self.pca_components = torch.from_numpy(pca_data["components"]).to(self.device)

        if normalize_output is None:
            normalize_output = os.environ.get("LEACE_NORMALIZE_OUTPUT", "0") == "1"

        self.normalize_output = normalize_output

    @property  # type: ignore
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
        images = images.to(self.device)

        last_hidden_state = self.feature_extractor(images)
        features = last_hidden_state[:, 0]  # CLS token

        features = self._apply_leace(features)

        return self._to_numpy(features)


class UNI2hWithLeace(LeaceMixin, Extractor):
    """UNI2-h with a LEACE projection applied to extracted features."""

    DEFAULT_LEACE_PATH = (
        "/home/valentin/workspaces/vfm-geom-xai/output/uni2_224_scorpion_scanner.pt"
    )

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
        leace_path: str | Path | None = None,
        normalize_output: bool | None = None,
    ):
        super().__init__()

        self.output_dim = 1536
        self.mixed_precision = mixed_precision

        timm_kwargs: dict[str, Any] = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }

        feature_extractor = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
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

        self._init_leace(
            leace_path,
            env_var="UNI2H_LEACE_ERASER_PATH",
            default_path=self.DEFAULT_LEACE_PATH,
            normalize_output=normalize_output,
        )

    @property  # type: ignore
    def transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> np.ndarray:
        images = images.to(self.device)

        features = self.feature_extractor(images)
        features = self._apply_leace(features)

        return self._to_numpy(features)


class HOptimus1WithLeace(LeaceMixin, Extractor):
    """H-Optimus-1 with a LEACE projection applied to extracted features."""

    DEFAULT_LEACE_PATH = (
        "/home/valentin/workspaces/vfm-geom-xai/output/"
        "h-optimus-1_224_scorpion_scanner.pt"
    )

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
        leace_path: str | Path | None = None,
        normalize_output: bool | None = None,
    ):
        super().__init__()

        self.output_dim = 1536
        self.mixed_precision = mixed_precision

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

        self._init_leace(
            leace_path,
            env_var="HOPTIMUS1_LEACE_ERASER_PATH",
            default_path=self.DEFAULT_LEACE_PATH,
            normalize_output=normalize_output,
        )

    @property  # type: ignore
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
        images = images.to(self.device)

        features = self.feature_extractor(images)
        features = self._apply_leace(features)

        return self._to_numpy(features)


class H0MiniWithPCAProj(Extractor):
    """H0-mini with a PCA scanner-delta projection applied to CLS-token features.

    Two modes are supported:

    1. same_dim=True:
        Project away the first `rank` PCA components and keep the original
        embedding dimension.

    2. same_dim=False:
        Rotate into the PCA basis and keep only the remainder coordinates,
        reducing dimensionality from D to D - rank.
    """

    DEFAULT_PCA_PATH = (
        "/home/valentin/workspaces/vfm-geom-xai/"
        "output/scorpion_analysis/paired_scanner_delta_pca_group_mean/"
        "h0-mini_scorpion_224px_0p5mpp_cls/"
        "fold_projectors/paired_scanner_delta_pca_fold0.npz"
    )

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
        pca_path: str | Path | None = None,
        rank: int = 16,
        same_dim: bool = True,
        normalize_output: bool = False,
    ):
        super().__init__()

        self.embedding_dim = 768
        self.rank = rank
        self.same_dim = same_dim
        self.normalize_output = normalize_output
        self.mixed_precision = mixed_precision

        self.output_dim = self.embedding_dim if same_dim else self.embedding_dim - rank

        if self.output_dim <= 0:
            raise ValueError(
                f"Invalid output_dim={self.output_dim}. "
                f"rank={rank} must be smaller than embedding_dim={self.embedding_dim}."
            )

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

        self._init_pca_projection(pca_path)

    def _resolve_pca_path(self, pca_path: str | Path | None) -> Path:
        if pca_path is not None:
            return Path(pca_path)

        env_path = os.environ.get("H0MINI_PCA_PROJECTOR_PATH")
        if env_path is not None:
            return Path(env_path)

        return Path(self.DEFAULT_PCA_PATH)

    def _init_pca_projection(self, pca_path: str | Path | None) -> None:
        pca_path = self._resolve_pca_path(pca_path)

        if not pca_path.exists():
            raise FileNotFoundError(f"PCA projector not found: {pca_path}")

        data = np.load(pca_path, allow_pickle=True)

        if "components" not in data:
            raise KeyError(f"Missing 'components' in PCA projector: {pca_path}")

        components = data["components"].astype(np.float32)

        if components.ndim != 2:
            raise ValueError(
                f"Expected PCA components with shape (n_components, dim), "
                f"got {components.shape}."
            )

        if components.shape[1] != self.embedding_dim:
            raise ValueError(
                f"PCA component dimension mismatch. "
                f"Expected {self.embedding_dim}, got {components.shape[1]}."
            )

        if self.rank > components.shape[0]:
            raise ValueError(
                f"Requested rank={self.rank}, but projector only contains "
                f"{components.shape[0]} components."
            )

        if not self.same_dim and components.shape[0] < self.embedding_dim:
            raise ValueError(
                "same_dim=False requires a full PCA basis with D components. "
                f"Got only {components.shape[0]} components for D={self.embedding_dim}."
            )

        self.pca_path = pca_path

        self.pca_components = torch.from_numpy(components)



    @property  # type: ignore
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

    def _apply_pca_projection(self, features: torch.Tensor) -> torch.Tensor:
        components = self.pca_components.to(
            device=features.device,
            dtype=features.dtype,
        )

        if self.same_dim:
            # Remove first `rank` scanner-delta directions.
            C = components[: self.rank]  # (rank, dim)
            return features - (features @ C.T) @ C

        # Rotate into PCA basis and keep the orthogonal complement coordinates.
        C_rem = components[self.rank :]  # (dim-rank, dim)
        return features @ C_rem.T

    def _normalize(self, features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return features / (features.norm(dim=1, keepdim=True) + eps)

    def _to_numpy(self, features: torch.Tensor) -> np.ndarray:
        return features.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> np.ndarray:
        images = images.to(self.device)

        last_hidden_state = self.feature_extractor(images)
        features = last_hidden_state[:, 0]  # CLS token

        features = self._apply_pca_projection(features)

        if self.normalize_output:
            features = self._normalize(features)

        return self._to_numpy(features)
