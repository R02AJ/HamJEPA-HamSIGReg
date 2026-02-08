from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch


class MultivariateTest(torch.nn.Module):
    """
    Base class for multivariate statistical tests.

    This repository historically used a misspelled base class name
    ("MultivariatetTest"). We keep a backwards-compatible alias below.

    Provides:
      - numpy -> torch conversion
      - float casting
      - shape validation for (N, D) inputs
      - optional dim checking

    Contract:
      - prepare_data(x) returns a float Tensor with shape (N, D)
    """

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def prepare_data(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(x)}")

        if not x.is_floating_point():
            x = x.float()

        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (N, D), got shape {tuple(x.shape)}")

        if self.dim is not None and x.shape[1] != self.dim:
            raise ValueError(f"Expected D={self.dim}, got D={x.shape[1]}")

        return x


# Backwards compatible alias for old typo
MultivariatetTest = MultivariateTest
