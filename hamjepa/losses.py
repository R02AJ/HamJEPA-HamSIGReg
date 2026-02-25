from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist

from .predictor import HamiltonianFlowPredictor


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_reduce_inplace(x: torch.Tensor, op: dist.ReduceOp) -> torch.Tensor:
    if _dist_is_initialized():
        dist.all_reduce(x, op=op)
    return x


def _broadcast_int(seed: int, device: torch.device) -> int:
    if not _dist_is_initialized():
        return int(seed)
    t = torch.tensor(int(seed), device=device, dtype=torch.long)
    dist.broadcast(t, src=0)
    return int(t.item())


class HamiltonianConsistencyLoss(nn.Module):
    """
    Flow z_a through HamiltonianFlowPredictor, match to z_b.

    loss = mse(flow(z_a), z_b) + energy_weight * mse(H(z_a), H(flow(z_a)))

    detach_target=True is usually the safe default; if you turn it off, you are
    explicitly allowing the target branch to move to make prediction easier.
    """

    def __init__(
        self,
        predictor: HamiltonianFlowPredictor,
        *,
        detach_target: bool = True,
        energy_weight: float = 0.0,
        match: Literal["q", "qp"] = "q",
        p_weight: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.detach_target = bool(detach_target)
        self.energy_weight = float(energy_weight)
        self.match = match
        self.p_weight = float(p_weight)
        self.bidirectional = bool(bidirectional)

    def _match_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d = self.predictor.d
        if self.match == "q":
            loss = F.mse_loss(pred[..., :d], target[..., :d])
            if self.p_weight > 0:
                loss = loss + self.p_weight * F.mse_loss(pred[..., d:], target[..., d:])
            return loss
        if self.match == "qp":
            return F.mse_loss(pred, target)
        raise ValueError(f"Unknown match='{self.match}' (use 'q' or 'qp')")

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        if z_a.shape != z_b.shape:
            raise ValueError(f"Shape mismatch: {tuple(z_a.shape)} vs {tuple(z_b.shape)}")

        target_ab = z_b.detach() if self.detach_target else z_b
        pred_ab = self.predictor(z_a, direction=1)
        loss = self._match_loss(pred_ab, target_ab)

        if self.energy_weight > 0:
            d = self.predictor.d
            q_a, p_a = z_a[..., :d], z_a[..., d:]
            q_p, p_p = pred_ab[..., :d], pred_ab[..., d:]
            H_a = self.predictor.H(q_a, p_a).detach()
            H_p = self.predictor.H(q_p, p_p)
            loss = loss + self.energy_weight * F.mse_loss(H_p, H_a)

        if self.bidirectional:
            target_ba = z_a.detach() if self.detach_target else z_a
            pred_ba = self.predictor(z_b, direction=-1)
            loss_ba = self._match_loss(pred_ba, target_ba)
            if self.energy_weight > 0:
                d = self.predictor.d
                q_b, p_b = z_b[..., :d], z_b[..., d:]
                q_pb, p_pb = pred_ba[..., :d], pred_ba[..., d:]
                H_b = self.predictor.H(q_b, p_b).detach()
                H_pb = self.predictor.H(q_pb, p_pb)
                loss_ba = loss_ba + self.energy_weight * F.mse_loss(H_pb, H_b)
            loss = 0.5 * (loss + loss_ba)

        return loss


class PhaseSpaceEnergyBudget(nn.Module):
    """
    Fixed-units scale control (immune to learnable-H gauge cheating).

    Keeps E[||q||^2]/d ~ q_target and E[||p||^2]/d ~ p_target.
    Works on z shaped [B,2d] or [V,B,2d].
    """

    def __init__(
        self,
        state_dim: int,
        *,
        q_target: float = 1.0,
        p_target: float = 1.0,
        ddp_sync: bool = True,
    ) -> None:
        super().__init__()
        if state_dim % 2 != 0:
            raise ValueError(f"state_dim must be even, got {state_dim}")
        self.state_dim = int(state_dim)
        self.d = self.state_dim // 2
        self.q_target = float(q_target)
        self.p_target = float(p_target)
        self.ddp_sync = bool(ddp_sync)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zf = z.reshape(-1, self.state_dim).float()
        q = zf[:, : self.d]
        p = zf[:, self.d :]

        q2_sum = q.square().sum()
        p2_sum = p.square().sum()
        count = torch.tensor(float(q.numel()), device=zf.device, dtype=torch.float32)

        if self.ddp_sync and _dist_is_initialized():
            _all_reduce_inplace(q2_sum, dist.ReduceOp.SUM)
            _all_reduce_inplace(p2_sum, dist.ReduceOp.SUM)
            _all_reduce_inplace(count, dist.ReduceOp.SUM)

        q2_mean = q2_sum / count.clamp_min(1.0)
        p2_mean = p2_sum / count.clamp_min(1.0)

        return (q2_mean - self.q_target) ** 2 + (p2_mean - self.p_target) ** 2


class VarianceFloor(nn.Module):
    """
    Minimal anti-collapse term (variance floor only; no whitening).

    L = mean_i relu(std_floor - std_i)^2
    Uses fp32 stats and optional DDP all-reduce for stability.
    """

    def __init__(
        self,
        dim: int,
        *,
        std_floor: float = 0.2,
        eps: float = 1e-4,
        ddp_sync: bool = True,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if std_floor <= 0:
            raise ValueError("std_floor must be > 0")
        self.dim = int(dim)
        self.std_floor = float(std_floor)
        self.eps = float(eps)
        self.ddp_sync = bool(ddp_sync)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.reshape(-1, self.dim).float()
        B = xf.size(0)
        if B < 2:
            return xf.new_zeros(())

        s1 = xf.sum(dim=0)
        s2 = xf.square().sum(dim=0)
        n = torch.tensor(float(B), device=xf.device, dtype=torch.float32)

        if self.ddp_sync and _dist_is_initialized():
            _all_reduce_inplace(s1, dist.ReduceOp.SUM)
            _all_reduce_inplace(s2, dist.ReduceOp.SUM)
            _all_reduce_inplace(n, dist.ReduceOp.SUM)

        mean = s1 / n.clamp_min(1.0)
        var = s2 / n.clamp_min(1.0) - mean.square()
        std = torch.sqrt(torch.clamp(var, min=0.0) + self.eps)

        penalty = torch.relu(self.std_floor - std).square()
        return penalty.mean()


class ProjectedLogDetFloor(nn.Module):
    """
    Random-project features to k dims, compute cov, enforce:
      - logdet_per_dim >= logdet_floor
      - pr_norm >= pr_norm_floor (optional)
      - eigmax_frac <= eigmax_frac_ceiling (optional)

    pr_norm = PR / k, where PR = tr(C)^2 / tr(C^2)
    eigmax_frac = max_eig / tr(C)
    """

    def __init__(
        self,
        dim: int,
        proj_dim: int = 128,
        logdet_floor: float = -1.5,
        eps: float = 1e-4,
        ddp_sync: bool = False,
        refresh_interval: int = 32,
        pr_norm_floor: Optional[float] = None,
        eigmax_frac_ceiling: Optional[float] = None,
        pr_weight: float = 1.0,
        eigmax_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.proj_dim = int(proj_dim)
        self.logdet_floor = float(logdet_floor)
        self.eps = float(eps)
        self.ddp_sync = bool(ddp_sync)
        self.refresh_interval = int(refresh_interval)

        self.pr_norm_floor = None if pr_norm_floor is None else float(pr_norm_floor)
        self.eigmax_frac_ceiling = (
            None if eigmax_frac_ceiling is None else float(eigmax_frac_ceiling)
        )
        self.pr_weight = float(pr_weight)
        self.eigmax_weight = float(eigmax_weight)

        self.register_buffer("_R", torch.empty(0), persistent=False)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long), persistent=True)

        # last stats (for logging)
        self.register_buffer("last_logdet_per_dim", torch.zeros(()), persistent=False)
        self.register_buffer("last_logdet_per_dim_pd", torch.zeros(()), persistent=False)
        self.register_buffer("last_pr", torch.zeros(()), persistent=False)
        self.register_buffer("last_pr_norm", torch.zeros(()), persistent=False)
        self.register_buffer("last_eigmax_frac", torch.zeros(()), persistent=False)
        self.register_buffer("last_tr", torch.zeros(()), persistent=False)

    @torch.no_grad()
    def _resample_R(self, device: torch.device, k: int) -> None:
        R = torch.randn(self.dim, k, device=device, dtype=torch.float32)
        q, _ = torch.linalg.qr(R, mode="reduced")  # (dim,k), orthonormal columns
        self._R = q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"Expected (B,{self.dim}), got {tuple(x.shape)}")

        xf = x.float()
        device = xf.device
        B = xf.shape[0]
        if B < 2:
            return xf.new_zeros(())

        k = min(self.proj_dim, self.dim, B - 1)

        step = int(self.global_step.item())
        need_resample = (
            self._R.numel() == 0
            or self._R.device != device
            or self._R.shape[0] != self.dim
            or self._R.shape[1] != k
            or (self.refresh_interval > 0 and (step % self.refresh_interval == 0))
        )
        if need_resample:
            self._resample_R(device=device, k=k)
        self.global_step.add_(1)

        # center then project
        xc = xf - xf.mean(dim=0, keepdim=True)
        y = xc @ self._R  # (B,k)

        # compute cov via sufficient stats (DDP-safe)
        sum_y = y.sum(dim=0)          # (k,)
        sum_yy = y.T @ y              # (k,k)
        n = torch.tensor(float(B), device=device, dtype=torch.float32)

        if self.ddp_sync and _dist_is_initialized():
            _all_reduce_inplace(sum_y, dist.ReduceOp.SUM)
            _all_reduce_inplace(sum_yy, dist.ReduceOp.SUM)
            _all_reduce_inplace(n, dist.ReduceOp.SUM)

        n = n.clamp_min(2.0)
        mean_y = sum_y / n
        cov = (sum_yy - n * torch.outer(mean_y, mean_y)) / (n - 1.0)
        cov = 0.5 * (cov + cov.T)  # symmetrize
        cov = cov + self.eps * torch.eye(k, device=device, dtype=cov.dtype)

        # logdet + PD logdet
        sign, logabsdet = torch.linalg.slogdet(cov)
        logdet = logabsdet
        logdet_pd = torch.where(sign > 0.0, logabsdet, logabsdet.new_tensor(-1e4))

        logdet_per_dim = logdet / float(k)
        logdet_pd_per_dim = logdet_pd / float(k)

        # PR + eigmax
        tr = cov.trace()
        tr2 = (cov * cov).sum()
        pr = (tr * tr) / (tr2 + 1e-12)
        pr_norm = pr / float(k)

        eigs = torch.linalg.eigvalsh(cov)
        eigmax_frac = eigs[-1] / (eigs.sum() + 1e-12)

        # stash
        self.last_logdet_per_dim.copy_(logdet_per_dim.detach())
        self.last_logdet_per_dim_pd.copy_(logdet_pd_per_dim.detach())
        self.last_pr.copy_(pr.detach())
        self.last_pr_norm.copy_(pr_norm.detach())
        self.last_eigmax_frac.copy_(eigmax_frac.detach())
        self.last_tr.copy_(tr.detach())

        # losses
        loss = xf.new_zeros(())
        loss = loss + torch.relu(self.logdet_floor - logdet_per_dim).square()

        if self.pr_norm_floor is not None:
            loss = loss + self.pr_weight * torch.relu(self.pr_norm_floor - pr_norm).square()

        if self.eigmax_frac_ceiling is not None:
            loss = loss + self.eigmax_weight * torch.relu(eigmax_frac - self.eigmax_frac_ceiling).square()

        return loss
