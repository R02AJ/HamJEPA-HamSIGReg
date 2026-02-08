"""
Learnable spectral Hamiltonian on a token grid.

Implements H_pos(phi) = U diag(lambda(phi)) U^T with fixed FFT/DCT eigenvectors
and a monotone spectral envelope, plus a stabilizing regularizer.
"""
from dataclasses import dataclass
import math
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
    # Base spectrum (geometry prior)
    base: Literal["identity", "r2", "laplacian"] = "identity"
    base_ridge: float = 1.0
    base_strength: float = 4.0
    init_loglambda_span: float = 0.0
    alpha_init: float = 1.0
    beta_init: float = 1.0
    delta_clip: float = 0.0
    center_delta: bool = False
    cond_max: float = 0.0
    cond_weight: float = 0.0
    eps_ridge: float = 0.0
    gauge: Literal["logdet", "trace"] = "logdet"
    curvature_weight: float = 1e-3
    loglambda_l2_weight: float = 1e-4
    binning: Literal["linear_radius", "linear_radius_sq", "quantile_radius"] = "linear_radius"
    log_lambda_clip: float = 3.0
    span_max: float = 0.10


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

        if float(cfg.span_max) <= 0.0:
            raise ValueError("LearnableHConfig.span_max must be > 0 for bounded learnable-H.")

        # Bounded-span (trust region) parameterization for monotone g bins.
        # Total log-span Delta in (0, span_max), controlled by raw_span.
        init_span = float(getattr(cfg, "init_loglambda_span", 0.0))
        init_ratio = init_span / float(cfg.span_max)
        init_ratio = max(1e-6, min(1.0 - 1e-6, init_ratio))
        raw_span_init = math.log(init_ratio / (1.0 - init_ratio))
        self.raw_span = nn.Parameter(torch.tensor(raw_span_init, dtype=torch.float32))
        # Positive, normalized increments across bins (bins-1).
        self.raw_shape = nn.Parameter(torch.zeros((cfg.bins - 1,), dtype=torch.float32))

        bin_idx_2d = self._make_bin_index_2d(cfg.h, cfg.w, cfg.bins, cfg.binning)
        self.register_buffer("bin_idx_2d", bin_idx_2d, persistent=True)
        self.register_buffer("loglam0_2d", self._make_base_log_lambda_2d(cfg), persistent=True)
        self.register_buffer("alpha", torch.tensor(float(cfg.alpha_init)), persistent=False)
        self.register_buffer("beta", torch.tensor(float(cfg.beta_init)), persistent=False)

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
            return torch.zeros((1,), device=self.raw_span.device, dtype=self.raw_span.dtype)
        w = F.softplus(self.raw_shape) + 1e-12
        w = w / w.sum()
        span = float(self.cfg.span_max) * torch.sigmoid(self.raw_span)
        g_rest = span * torch.cumsum(w, dim=0)
        g = torch.cat(
            [torch.zeros((1,), device=g_rest.device, dtype=g_rest.dtype), g_rest],
            dim=0,
        )
        dc = float(self.cfg.delta_clip)
        if dc > 0.0:
            g = dc * torch.tanh(g / dc)
        return g

    def delta_2d(self, device: Optional[torch.device] = None) -> torch.Tensor:
        g = self.g_bins()
        delta = g[self.bin_idx_2d]
        if self.cfg.center_delta:
            delta = delta - delta.mean()
        if device is not None:
            delta = delta.to(device)
        return delta

    @classmethod
    def _make_base_log_lambda_2d(cls, cfg: LearnableHConfig) -> torch.Tensor:
        h, w = int(cfg.h), int(cfg.w)
        base = str(cfg.base).lower()
        device = torch.device("cpu")
        dtype = torch.float32

        if base == "identity":
            lam0 = torch.ones((h, w), device=device, dtype=dtype)
        else:
            fx = cls._fftfreq_1d(h, device=device, dtype=dtype)
            fy = cls._fftfreq_1d(w, device=device, dtype=dtype)
            Fx = fx[:, None].expand(h, w)
            Fy = fy[None, :].expand(h, w)
            a = float(cfg.base_ridge)
            b = float(cfg.base_strength)
            if base == "r2":
                r2 = Fx * Fx + Fy * Fy
                lam0 = a + b * r2
            elif base == "laplacian":
                lamx = 4.0 * torch.sin(math.pi * fx).pow(2)
                lamy = 4.0 * torch.sin(math.pi * fy).pow(2)
                lap = lamx[:, None].expand(h, w) + lamy[None, :].expand(h, w)
                lam0 = a + b * lap
            else:
                raise ValueError(f"Unknown base: {cfg.base}")

        # trace gauge
        lam0 = lam0 / lam0.mean().clamp_min(1e-12)
        return torch.log(lam0.clamp_min(1e-12))

    def set_schedule(self, *, alpha: Optional[float] = None, beta: Optional[float] = None) -> None:
        if alpha is not None:
            self.alpha.fill_(float(max(0.0, min(1.0, alpha))))
        if beta is not None:
            self.beta.fill_(float(max(0.0, min(1.0, beta))))

    def log_lambda_2d(self, device: Optional[torch.device] = None) -> torch.Tensor:
        delta = self.delta_2d(device=device)
        loglam0 = self.loglam0_2d.to(delta.device)
        alpha = self.alpha.to(delta.device)
        beta = self.beta.to(delta.device)

        # base + deviation with schedule
        lam_base = torch.exp(loglam0)
        lam_base = (1.0 - beta) + beta * lam_base
        lam = lam_base * torch.exp(alpha * delta)

        if self.cfg.gauge == "trace":
            lam = lam / lam.mean().clamp_min(1e-12)
        elif self.cfg.gauge == "logdet":
            lam = lam / torch.exp(torch.log(lam.clamp_min(1e-12)).mean()).clamp_min(1e-12)
        else:
            raise ValueError(f"Unknown gauge: {self.cfg.gauge}")

        ell = torch.log(lam.clamp_min(1e-12)).clamp(-self.cfg.log_lambda_clip, self.cfg.log_lambda_clip)
        return ell if device is None else ell.to(device)

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
        if x_work.dtype in (torch.float16, torch.bfloat16):
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
            delta = self.delta_2d(device=g.device)
            reg = reg + cfg.loglambda_l2_weight * (delta ** 2).mean()

        if cfg.cond_max > 0 and cfg.cond_weight > 0:
            stats = self.health_stats(device=g.device)
            excess = torch.relu(stats["lam_cond"] - cfg.cond_max)
            reg = reg + cfg.cond_weight * excess.pow(2)

        return reg

    @torch.no_grad()
    def health_stats(self, device: Optional[torch.device] = None) -> dict[str, torch.Tensor]:
        """
        Summary stats for logging/diagnostics.
        Returns lam_min/max/cond, g0/g_last, loglam_min/max.
        """
        if device is None:
            device = self.raw_span.device

        g = self.g_bins().to(device)
        g0 = g[0]
        g_last = g[-1]

        loglam = self.log_lambda_2d(device=device)
        loglam_min = loglam.min()
        loglam_max = loglam.max()

        lam = torch.exp(loglam)
        lam_min = lam.min()
        lam_max = lam.max()
        lam_cond = lam_max / lam_min.clamp_min(1e-12)

        return {
            "lam_min": lam_min,
            "lam_max": lam_max,
            "lam_cond": lam_cond,
            "g0": g0,
            "g_last": g_last,
            "loglam_min": loglam_min,
            "loglam_max": loglam_max,
        }
