from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import distributed as dist


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _broadcast_int(seed: int, device: torch.device) -> int:
    """Broadcast an integer seed from rank 0 to all ranks."""
    if not _dist_is_initialized():
        return int(seed)
    t = torch.tensor(int(seed), device=device, dtype=torch.long)
    dist.broadcast(t, src=0)
    return int(t.item())


class SlicingUnivariateTest(torch.nn.Module):
    """
    Extend a univariate goodness-of-fit statistic to multivariate data via random slicing.

    Given x in R^D, sample K random unit vectors a_k and test 1D projections:
        y_k = x * a_k

    Wrapped univariate_test must accept (..., N, K) and return (..., K).

    DDP correctness rule:
    ---------------------
    If the univariate_test performs DDP synchronization (e.g., global all-reduces),
    then all ranks MUST use the same projection matrix A. Set
        sync_projections_across_gpus=True
    in that case.

    Speed controls:
    --------------
    - refresh_interval: reuse A for multiple forwards
    - subsample: compute statistic on only `subsample` samples along N dimension
    """

    def __init__(
        self,
        univariate_test: torch.nn.Module,
        num_slices: int,
        *,
        reduction: Literal["mean", "sum", None] = "mean",
        sampler: Literal["gaussian"] = "gaussian",
        clip_value: Optional[float] = None,
        refresh_interval: int = 1,
        sync_projections_across_gpus: bool = True,
        subsample: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_slices < 1:
            raise ValueError(f"num_slices must be >= 1, got {num_slices}")
        if reduction not in ("mean", "sum", None):
            raise ValueError(f"reduction must be 'mean', 'sum', or None, got {reduction}")
        if sampler != "gaussian":
            raise ValueError(f"Unsupported sampler='{sampler}'. Only 'gaussian' is implemented.")
        if refresh_interval < 1:
            raise ValueError(f"refresh_interval must be >= 1, got {refresh_interval}")
        if subsample is not None and subsample < 1:
            raise ValueError(f"subsample must be >= 1, got {subsample}")

        self.univariate_test = univariate_test
        self.num_slices = int(num_slices)
        self.reduction = reduction
        self.sampler = sampler
        self.clip_value = clip_value
        self.refresh_interval = int(refresh_interval)
        self.sync_projections_across_gpus = bool(sync_projections_across_gpus)
        self.subsample = subsample

        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

        # Cached projection matrix (not in state_dict).
        self.register_buffer("_A", torch.empty(0), persistent=False)
        self._A_device: Optional[torch.device] = None

        # Cached generator per-device.
        self._generator: Optional[torch.Generator] = None
        self._generator_device: Optional[torch.device] = None

    def _get_generator(self, device: torch.device) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        return self._generator

    def _need_new_A(self, D: int, device: torch.device) -> bool:
        if self._A.numel() == 0:
            return True
        if self._A_device != device:
            return True
        if self._A.shape != (D, self.num_slices):
            return True
        step = int(self.global_step.item())
        return (step % self.refresh_interval) == 0

    @torch.no_grad()
    def _resample_A(self, D: int, device: torch.device, dtype: torch.dtype) -> None:
        if not self._need_new_A(D, device):
            return

        # Deterministic seed derived from step counter.
        seed = int(self.global_step.item())

        # If syncing projections across GPUs, broadcast the seed from rank 0.
        if self.sync_projections_across_gpus and _dist_is_initialized():
            seed = _broadcast_int(seed, device=device)
        else:
            # Make per-rank seeds different (deterministically) when not syncing.
            if _dist_is_initialized():
                seed = seed + 1_000_003 * dist.get_rank()

        g = self._get_generator(device)
        g.manual_seed(seed)

        A = torch.randn((D, self.num_slices), device=device, dtype=dtype, generator=g)
        A = A / A.norm(p=2, dim=0).clamp_min(1e-12)
        self._A = A
        self._A_device = device

    def _maybe_subsample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Subsample along the sample dimension (-2) if requested.
        x shape: (..., N, D)
        """
        if self.subsample is None:
            return x
        N = x.size(-2)
        if N <= self.subsample:
            return x

        device = x.device
        g = self._get_generator(device)
        # Seed subsampling deterministically from step counter, but allow per-rank variation
        # if projections are not synchronized.
        seed = int(self.global_step.item()) + 97
        if (not self.sync_projections_across_gpus) and _dist_is_initialized():
            seed = seed + 1_000_003 * dist.get_rank()
        g.manual_seed(seed)

        idx = torch.randperm(N, device=device, generator=g)[: self.subsample]
        # Index into the N dimension (-2)
        return x.index_select(dim=-2, index=idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., N, D)
        returns:
            scalar if reduction in {'mean','sum'} else (..., num_slices)
        """
        if x.ndim < 2:
            raise ValueError(f"Expected x with shape (..., N, D), got {tuple(x.shape)}")

        # Optional subsampling for speed
        x = self._maybe_subsample(x)

        D = x.size(-1)
        device = x.device
        dtype = x.dtype

        self._resample_A(D, device=device, dtype=dtype)
        # Increment after using for seeding
        self.global_step.add_(1)

        proj = x @ self._A  # (..., N, K)
        stats = self.univariate_test(proj)  # (..., K)

        if self.clip_value is not None:
            stats = torch.where(stats < self.clip_value, torch.zeros_like(stats), stats)

        if self.reduction == "mean":
            return stats.mean()
        if self.reduction == "sum":
            return stats.sum()
        return stats
