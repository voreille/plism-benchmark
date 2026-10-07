"""Models adapted with LEACE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from sipe.model.arch import H0_mini_for_Adversarial
from sipe.model.builder import build_sipe_extractor
from torchvision import transforms

from plismbench.models.extractor import Extractor
from plismbench.models.utils import DEFAULT_DEVICE, prepare_module


class H0MiniGAP(Extractor):
    """H0-mini model developped by Owkin & Bioptimus available on Hugging-Face (1).

    You will need to be granted access to be able to use this model.

    .. note::
        (1) https://huggingface.co/bioptimus/H0-mini

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
    ):
        super().__init__()
        self.output_dim = 768
        self.mixed_precision = mixed_precision

        timm_kwargs: dict[str, Any] = {
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }
        feature_extractor = timm.create_model(
            "hf-hub:bioptimus/H0-mini", pretrained=True, **timm_kwargs
        )
        self.num_prefix_tokens = feature_extractor.num_prefix_tokens

        self.feature_extractor, self.device = prepare_module(
            feature_extractor,
            device,
            self.mixed_precision,
        )
        if self.device is None:
            self.feature_extractor = self.feature_extractor.module

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
        last_hidden_state = self.feature_extractor(images.to(self.device))
        patch_features = last_hidden_state[:, self.num_prefix_tokens :]
        features = patch_features.mean(dim=1)  # global average pooling
        return features.cpu().numpy()


class SIPEH0Mini(Extractor):
    """SIPE model based on H0-mini, returning disentangled features.

    By default, returns concat(stain_vector, GAP(content_map)),
    giving a 768-dimensional feature vector.
    """

    def __init__(
        self,
        device: int | list[int] | None = DEFAULT_DEVICE,
        mixed_precision: bool = False,
        checkpoint_dir: str
        | Path = "/mnt/nas6/data/BigPicture_CBIR/model_weights/SIPE-50k-Curriculum/checkpoints/ckpt_from_epoch_65",
        feature_mode: str = "content",
    ):
        super().__init__()

        self.output_dim = 768
        self.mixed_precision = mixed_precision

        if device == -1:
            torch_device = "cpu"
        elif isinstance(device, int):
            torch_device = f"cuda:{device}"
        elif device is None:
            torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif isinstance(device, list):
            torch_device = f"cuda:{device[0]}"
        else:
            torch_device = device

        self.feature_extractor = build_sipe_extractor(
            checkpoint_dir=checkpoint_dir,
            device=torch_device,
            mixed_precision=mixed_precision,
            feature_mode=feature_mode,
        )

    @property  # type: ignore
    def transform(self) -> transforms.Compose:
        return self.feature_extractor.transform

    def __call__(self, images: torch.Tensor) -> np.ndarray:
        return self.feature_extractor(images)
