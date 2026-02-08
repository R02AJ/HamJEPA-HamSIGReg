"""
Learnable spectral Hamiltonian on a token grid.

Implements H_pos(phi) = U diag(lambda(phi)) U^T with fixed FFT/DCT eigenvectors
and a monotone spectral envelope, plus a stabilizing regularizer.
"""
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LearnableHConfig:
    h: int
    w: int
    d_f: int
    bins: int = 16
    init_loglambda_span: float = 1.0
    eps_ridge: float = 0.0
    gauge: Literal["logdet", "trace"] = "logdet"
    curvature_weight: float = 1e-3
    loglambda_l2_weight: float = 1e-4
    binning: Literal["linear_radius", "linear_radius_sq", "quantile_radius"] = "linear_radius"
    log_lambda_clip: float = 3.0


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    y = y.clamp_min(1e-6)
    return torch.log(torch.expm1(y))


class LearnableSpectralHamiltonian(nn.Module):
    """
    Learnable positional Hamiltonian H_pos(phi) = U diag(lambda(phi)) U^T
    with fixed Fourier basis (implemented via FFT) and learnable monotone
    radial envelope lambda(phi).

    Applies H(phi)^{1/2} = H_pos(phi)^{1/2} x I_{d_f} across token positions
    independently for each feature channel, without forming H explicitly.

    Input shapes supported:
      - (n, P, d_f) where P=h*w
      - (n, h, w, d_f)
      - (n, d_f, h, w)
    Output shape matches input shape.
    """

    def __init__(self, cfg: LearnableHConfig):
        super().__init__()
        assert cfg.h > 0 and cfg.w > 0 and cfg.d_f > 0 and cfg.bins > 0
        self.cfg = cfg
        self.P = cfg.h * cfg.w

        # Parameters for monotone bin values g_1,...,g_B
        # g_1 is free, deltas are positive via softplus.
        self.g1 = nn.Parameter(torch.zeros(()))  # scalar
        if cfg.bins > 1 and float(cfg.init_loglambda_span) > 0.0:
            span = min(float(cfg.init_loglambda_span), 0.9 * float(cfg.log_lambda_clip))
            g_init = torch.linspace(-span, span, cfg.bins, dtype=torch.float32)
            deltas = g_init[1:] - g_init[:-1]
            self.g1 = nn.Parameter(g_init[0].clone())
            self.raw_deltas = nn.Parameter(_inv_softplus(deltas))
        else:
            self.raw_deltas = nn.Parameter(torch.full((cfg.bins - 1,), -2.0))  # length B-1

        bin_idx_2d = self._make_bin_index_2d(cfg.h, cfg.w, cfg.bins, cfg.binning)
        self.register_buffer("bin_idx_2d", bin_idx_2d, persistent=True)

    @staticmethod
    def _fftfreq_1d(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        k = torch.arange(n, device=device, dtype=dtype)
        k = torch.where(k <= (n // 2), k, k - n)
        return k / float(n)

    @classmethod
    def _make_bin_index_2d(
        cls, h: int, w: int, B: int, mode: str
    ) -> torch.Tensor:
        device = torch.device("cpu")
        dtype = torch.float32

        fx = cls._fftfreq_1d(h, device=device, dtype=dtype)
        fy = cls._fftfreq_1d(w, device=device, dtype=dtype)

        Fx = fx[:, None].expand(h, w)
        Fy = fy[None, :].expand(h, w)

        if mode == "linear_radius":
            r = torch.sqrt(Fx * Fx + Fy * Fy)
        elif mode == "linear_radius_sq":
            r = (Fx * Fx + Fy * Fy)
        elif mode == "quantile_radius":
            r = torch.sqrt(Fx * Fx + Fy * Fy)
        else:
            raise ValueError(f"Unknown binning mode: {mode}")

        r_flat = r.flatten()

        if mode == "quantile_radius":
            qs = torch.linspace(0.0, 1.0, B + 1, device=device, dtype=dtype)
            edges = torch.quantile(r_flat, qs)
        else:
            r_min = float(r_flat.min().item())
            r_max = float(r_flat.max().item())
            edges = torch.linspace(r_min, r_max + 1e-12, B + 1, device=device, dtype=dtype)

        bin_idx = torch.bucketize(r, edges[1:-1], right=False)
        bin_idx = bin_idx.clamp(min=0, max=B - 1).to(torch.long)
        return bin_idx

    def g_bins(self) -> torch.Tensor:
        if self.cfg.bins == 1:
            return self.g1[None]
        deltas = F.softplus(self.raw_deltas)
        g = torch.cat([self.g1[None], self.g1[None] + torch.cumsum(deltas, dim=0)], dim=0)
        return g

    def log_lambda_2d(self, device: Optional[torch.device] = None) -> torch.Tensor:
        g = self.g_bins()
        ell_tilde = g[self.bin_idx_2d]

        if self.cfg.gauge == "logdet":
            ell = ell_tilde - ell_tilde.mean()
        elif self.cfg.gauge == "trace":
            ell = ell_tilde
        else:
            raise ValueError(f"Unknown gauge: {self.cfg.gauge}")

        ell = ell.clamp(-self.cfg.log_lambda_clip, self.cfg.log_lambda_clip)
        if device is not None:
            ell = ell.to(device)
        return ell

    def lambda_2d(self, device: torch.device) -> torch.Tensor:
        ell = self.log_lambda_2d(device=device)
        lam = torch.exp(ell)

        if self.cfg.eps_ridge > 0:
            lam = lam + self.cfg.eps_ridge

        # Re-apply gauge after any ridge adjustment
        if self.cfg.gauge == "trace":
            lam = lam / lam.mean().clamp_min(1e-12)
        elif self.cfg.gauge == "logdet":
            lam = lam / torch.exp(torch.log(lam).mean()).clamp_min(1e-12)

        return lam

    def sqrtH_apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply H(phi)^{1/2} to token-grid embeddings.

        x: embeddings, shape (n,P,d_f) or (n,h,w,d_f) or (n,d_f,h,w)
        returns: same shape as x
        """
        cfg = self.cfg
        h, w, d_f = cfg.h, cfg.w, cfg.d_f

        orig_shape = x.shape
        if x.dim() == 3:
            n, P, df_in = x.shape
            assert P == h * w, f"Expected P=h*w={h*w}, got {P}"
            assert df_in == d_f, f"Expected d_f={d_f}, got {df_in}"
            x_grid = x.view(n, h, w, d_f).permute(0, 3, 1, 2).contiguous()
        elif x.dim() == 4:
            if x.shape[1] == d_f and x.shape[2] == h and x.shape[3] == w:
                x_grid = x
            else:
                n, hh, ww, df_in = x.shape
                assert hh == h and ww == w and df_in == d_f
                x_grid = x.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(f"Unsupported x shape: {x.shape}")

        orig_dtype = x_grid.dtype
        x_work = x_grid
        if x_work.dtype == torch.bfloat16:
            x_work = x_work.float()
        X = torch.fft.fft2(x_work, dim=(-2, -1), norm="ortho")

        lam = self.lambda_2d(device=x_grid.device)
        sqrt_lam = torch.sqrt(lam).to(dtype=X.real.dtype)

        X = X * sqrt_lam[None, None, :, :]

        y_grid = torch.fft.ifft2(X, dim=(-2, -1), norm="ortho").real
        if y_grid.dtype != orig_dtype:
            y_grid = y_grid.to(orig_dtype)

        if len(orig_shape) == 3:
            y = y_grid.permute(0, 2, 3, 1).contiguous().view(orig_shape)
        elif len(orig_shape) == 4:
            if orig_shape[1] == d_f and orig_shape[2] == h and orig_shape[3] == w:
                y = y_grid
            else:
                y = y_grid.permute(0, 2, 3, 1).contiguous()
        else:
            raise RuntimeError("Unreachable")
        return y

    def spectral_regularizer(self) -> torch.Tensor:
        """
        R(phi) = curvature penalty on g_b + L2 penalty on centered log-lambda.
        """
        cfg = self.cfg
        g = self.g_bins()
        reg = torch.zeros((), device=g.device, dtype=g.dtype)

        if cfg.bins >= 3 and cfg.curvature_weight > 0:
            second_diff = g[2:] - 2.0 * g[1:-1] + g[:-2]
            reg = reg + cfg.curvature_weight * (second_diff ** 2).sum()

        if cfg.loglambda_l2_weight > 0:
            ell = self.log_lambda_2d(device=g.device)
            reg = reg + cfg.loglambda_l2_weight * (ell ** 2).mean()

        return reg

    @torch.no_grad()
    def health_stats(self, device: Optional[torch.device] = None) -> dict[str, torch.Tensor]:
        """
        Quick diagnostics for whether H is stable/pathological.

        Returns tensors (on `device`) so the caller can .item() them for logging.
        """
        dev = device if device is not None else self.g1.device

        g = self.g_bins().to(dev)
        ell = self.log_lambda_2d(device=dev)
        lam = self.lambda_2d(device=dev)

        lam_min = lam.min()
        lam_max = lam.max()
        lam_cond = lam_max / lam_min.clamp_min(1e-12)

        return {
            "lam_min": lam_min,
            "lam_max": lam_max,
            "lam_cond": lam_cond,
            "g0": g[0],
            "g_last": g[-1],
            "loglam_min": ell.min(),
            "loglam_max": ell.max(),
        }
