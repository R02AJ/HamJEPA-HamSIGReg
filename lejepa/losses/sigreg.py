from __future__ import annotations

from typing import Literal, Optional

import torch

from ..multivariate.slicing import SlicingUnivariateTest
from ..univariate.eps_pulley import EppsPulley


class SIGReg(torch.nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (SIGReg) convenience wrapper:
        SIGReg = SlicingUnivariateTest(EppsPulley)

    Input:  (..., N, D)
    Output: scalar (by default reduction='mean')

    Speed knobs (use these to get your epoch time down):
      - num_slices: reduce from 1000 -> 256 -> 128
      - subsample: compute on only e.g. 128-512 samples per step
      - refresh_interval: reuse A for e.g. 4-16 steps
      - n_points: reduce 17 -> 9 (coarser quadrature)
    """

    def __init__(
        self,
        *,
        num_slices: int = 256,
        t_max: float = 3.0,
        t_range: Optional[tuple[float, float]] = None,
        n_points: int = 17,
        weight_type: Literal["gaussian", "uniform"] = "gaussian",
        ddp_sync: bool = True,
        force_fp32: bool = True,
        refresh_interval: int = 1,
        subsample: Optional[int] = None,
        reduction: Literal["mean", "sum", None] = "mean",
        clip_value: Optional[float] = None,
    ) -> None:
        super().__init__()

        uni = EppsPulley(
            t_max=t_max,
            t_range=t_range,
            n_points=n_points,
            weight_type=weight_type,
            ddp_sync=ddp_sync,
            force_fp32=force_fp32,
        )

        self.test = SlicingUnivariateTest(
            univariate_test=uni,
            num_slices=num_slices,
            reduction=reduction,
            clip_value=clip_value,
            refresh_interval=refresh_interval,
            # If ddp_sync=True inside EppsPulley, we MUST synchronize projections.
            sync_projections_across_gpus=True if ddp_sync else False,
            subsample=subsample,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim < 2:
            raise ValueError(f"Expected z with shape (..., N, D), got {tuple(z.shape)}")
        return self.test(z)
