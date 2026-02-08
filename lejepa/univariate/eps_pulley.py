from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch
from torch import distributed as dist

from .base import UnivariateTest


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_reduce_inplace(x: torch.Tensor, op: dist.ReduceOp) -> torch.Tensor:
    if _dist_is_initialized():
        dist.all_reduce(x, op=op)
    return x


class EppsPulley(UnivariateTest):
    """
    Fast Epps-Pulley statistic for testing whether projected samples follow N(0,1).

    Input shape:  (..., N, K) or (N, K) or (N,)
    Output shape: (..., K)

    DDP correctness:
      If ddp_sync=True, we compute a global empirical characteristic function by
      summing cos/sin totals across ranks and dividing by global N. This is only
      valid if all ranks used the same random projection directions.
    """

    def __init__(
        self,
        t_max: float = 3.0,
        n_points: int = 17,
        integration: Literal["trapezoid"] = "trapezoid",
        weight_type: Literal["gaussian", "uniform"] = "gaussian",
        ddp_sync: bool = True,
        force_fp32: bool = True,
        *,
        # backwards-compat for older tests
        t_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__()

        # Back-compat: accept symmetric t_range = (-t_max, +t_max)
        if t_range is not None:
            lo, hi = float(t_range[0]), float(t_range[1])
            if not math.isfinite(lo) or not math.isfinite(hi):
                raise ValueError(f"t_range must be finite, got {t_range}")
            if abs(lo + hi) > 1e-6 * max(1.0, abs(lo), abs(hi)):
                raise ValueError(
                    f"t_range must be symmetric around 0, got {t_range}. Use t_max instead."
                )
            t_max = max(abs(lo), abs(hi))

        if t_max <= 0 or not math.isfinite(float(t_max)):
            raise ValueError(f"t_max must be positive finite, got {t_max}")
        if n_points < 3 or n_points % 2 != 1:
            raise ValueError(f"n_points must be odd and >=3, got {n_points}")
        if integration != "trapezoid":
            raise ValueError("Only trapezoid integration is implemented.")
        if weight_type not in ("gaussian", "uniform"):
            raise ValueError("weight_type must be 'gaussian' or 'uniform'.")

        self.integration = integration
        self.n_points = int(n_points)
        self.t_max = float(t_max)
        self.weight_type = weight_type
        self.ddp_sync = bool(ddp_sync)
        self.force_fp32 = bool(force_fp32)

        # Positive t knots
        t = torch.linspace(0.0, self.t_max, self.n_points, dtype=torch.float32)

        # Trapezoid weights on [0, t_max], doubled to account for [-t_max, 0)
        dt = self.t_max / (self.n_points - 1)
        w = torch.full((self.n_points,), 2.0 * dt, dtype=torch.float32)
        w[[0, -1]] = dt

        # Target CF for N(0,1)
        phi = torch.exp(-0.5 * t.square())

        # Weight function
        if self.weight_type == "gaussian":
            w = w * phi

        self.register_buffer("t", t, persistent=True)
        self.register_buffer("phi", phi, persistent=True)
        self.register_buffer("weights", w, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(-1)  # (N, 1)
        if x.ndim < 2:
            raise ValueError(f"Expected (..., N, K) or (N,K), got {tuple(x.shape)}")

        N_local = x.size(-2)
        if N_local < 1:
            out_shape = x.shape[:-2] + (x.size(-1),)
            return x.new_zeros(out_shape, dtype=torch.float32)

        # Compute in float32 for trig stability unless explicitly disabled
        x_work = x.float() if self.force_fp32 else x

        x_t = x_work.unsqueeze(-1) * self.t  # (..., N, K, T)

        # sincos is slightly more efficient if available
        if hasattr(torch, "sincos"):
            sin_vals, cos_vals = torch.sincos(x_t)
        else:
            sin_vals = torch.sin(x_t)
            cos_vals = torch.cos(x_t)

        cos_sum = cos_vals.sum(dim=-3)  # (..., K, T)
        sin_sum = sin_vals.sum(dim=-3)

        N_global = torch.tensor(float(N_local), device=x.device, dtype=torch.float32)

        if self.ddp_sync and _dist_is_initialized():
            # NOTE: This is only correct if all ranks share the same projection matrix A.
            _all_reduce_inplace(cos_sum, dist.ReduceOp.SUM)
            _all_reduce_inplace(sin_sum, dist.ReduceOp.SUM)
            _all_reduce_inplace(N_global, dist.ReduceOp.SUM)

        N_safe = N_global.clamp_min(1.0)
        cos_mean = cos_sum / N_safe
        sin_mean = sin_sum / N_safe

        err = (cos_mean - self.phi).square() + sin_mean.square()
        stat = (err @ self.weights) * N_safe  # (..., K)

        return stat.to(dtype=torch.float32)
