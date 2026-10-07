"""Tests for :func:`plismbench.models.init_extractor` with external factories."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from plismbench.models import init_extractor
from plismbench.models.extractor import Extractor


class _DummyExtractor(Extractor):
    """Minimal extractor returning a constant feature vector."""

    def __init__(self, device: int | list[int] | None, output_dim: int = 4):
        super().__init__()
        self.device = device
        self.output_dim = output_dim

    def __call__(self, images: torch.Tensor) -> np.ndarray:
        """Return zeros of shape (n_tiles, output_dim)."""
        return np.zeros((images.shape[0], self.output_dim), dtype=np.float32)


def make_dummy(device: int | list[int] | None, **kwargs) -> _DummyExtractor:
    """Build a dummy extractor (external factory)."""
    return _DummyExtractor(device=device, **kwargs)


def make_not_an_extractor(device: int | list[int] | None) -> object:
    """Return an object that is not an extractor."""
    return object()


def test_init_external_extractor() -> None:
    """External ``module:factory`` extractors receive device and kwargs."""
    model = init_extractor(
        f"{__name__}:make_dummy", device=-1, extractor_kwargs={"output_dim": 7}
    )
    assert isinstance(model, _DummyExtractor)
    assert model.device == -1
    assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 7)


def test_init_external_extractor_wrong_type() -> None:
    """External factories must return an :class:`Extractor`."""
    with pytest.raises(TypeError):
        init_extractor(f"{__name__}:make_not_an_extractor", device=-1)
