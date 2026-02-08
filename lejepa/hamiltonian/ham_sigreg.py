from __future__ import annotations

from math import pi
from typing import Optional

import torch
import torch.nn as nn

from lejepa.losses import SIGReg
from .learnable_h import LearnableHConfig, LearnableSpectralHamiltonian


class HamSIGReg(nn.Module):
    """
    Apply H^{1/2} to embeddings then SIGReg.

    Efficient implementations (no dense H^{1/2}):
      - identity: no-op
      - diag: elementwise scaling
      - chain: FFT-diagonal periodic Laplacian
      - learnable: FFT2-based grid spectral envelope

    Input z: [B,d] or [V,B,d]
    """

    def __init__(
        self,
        d: int,
        kind: str = "chain",
        device: Optional[torch.device] = None,
        num_slices: int = 256,
        sigreg_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.kind = kind.lower()
        self.d = int(d)

        if self.kind == "identity":
            pass

        elif self.kind == "diag":
            v = torch.linspace(1.0, 2.0, steps=self.d, device=device)
            v = v / torch.exp(torch.log(v).mean()).clamp_min(1e-12)
            self.register_buffer("sqrt_diag", torch.sqrt(v), persistent=True)

        elif self.kind == "chain":
            self.chain_mode = str(kwargs.pop("mode", "1d")).lower()
            self.kappa = float(kwargs.pop("kappa", 1.0))
            self.eps = float(kwargs.pop("eps", 1e-3))
            self.beta = float(kwargs.pop("beta", 1.0))
            self.normalize = str(kwargs.pop("normalize", "trace")).lower()

            if self.chain_mode == "grid":
                self.h = int(kwargs.pop("h"))
                self.w = int(kwargs.pop("w"))
                self.d_f = int(kwargs.pop("d_f"))
                if self.h * self.w * self.d_f != self.d:
                    raise ValueError(f"h*w*d_f must equal d (got {self.h*self.w*self.d_f} vs {self.d}).")

                fx = torch.fft.fftfreq(self.w, device=device)
                fy = torch.fft.fftfreq(self.h, device=device)
                nu_x = 4.0 * torch.sin(torch.pi * fx).pow(2)
                nu_y = 4.0 * torch.sin(torch.pi * fy).pow(2)
                NuY, NuX = torch.meshgrid(nu_y, nu_x, indexing="ij")
                nu = NuY + NuX

                lam = self.eps + self.kappa * nu
                if self.normalize == "trace":
                    lam = lam / lam.mean()
                else:
                    lam = lam / torch.exp(torch.log(lam).mean()).clamp_min(1e-12)
                self.register_buffer("lam_grid_base", lam, persistent=True)
                self._set_chain_beta(self.beta)
            else:
                kk = torch.arange(self.d // 2 + 1, device=device, dtype=torch.float32)
                nu = 2.0 - 2.0 * torch.cos(2.0 * pi * kk / float(self.d))
                lam = self.kappa * nu + self.eps
                if self.normalize == "trace":
                    lam = lam / lam.mean()
                else:
                    lam = lam / torch.exp(torch.log(lam).mean()).clamp_min(1e-12)
                self.register_buffer("lam_rfft_base", lam, persistent=True)
                self._set_chain_beta(self.beta)

        elif self.kind == "learnable":
            h = kwargs.pop("h", None)
            w = kwargs.pop("w", None)
            d_f = kwargs.pop("d_f", None)
            if h is None or w is None or d_f is None:
                raise ValueError("Learnable H requires h, w, and d_f in ham_kwargs.")
            if int(h) * int(w) * int(d_f) != self.d:
                raise ValueError(f"Learnable H size mismatch: h*w*d_f={h*w*d_f} but d={self.d}.")
            cfg = LearnableHConfig(
                h=int(h),
                w=int(w),
                d_f=int(d_f),
                bins=int(kwargs.pop("bins", 16)),
                base=kwargs.pop("base", "identity"),
                base_ridge=float(kwargs.pop("base_ridge", 1.0)),
                base_strength=float(kwargs.pop("base_strength", 4.0)),
                init_loglambda_span=float(kwargs.pop("init_loglambda_span", 0.0)),
                alpha_init=float(kwargs.pop("alpha_init", 1.0)),
                beta_init=float(kwargs.pop("beta_init", 1.0)),
                delta_clip=float(kwargs.pop("delta_clip", 0.0)),
                center_delta=bool(kwargs.pop("center_delta", False)),
                span_max=float(kwargs.pop("span_max", 0.10)),
                cond_max=float(kwargs.pop("cond_max", 0.0)),
                cond_weight=float(kwargs.pop("cond_weight", 0.0)),
                eps_ridge=float(kwargs.pop("eps_ridge", 0.0)),
                gauge=kwargs.pop("gauge", "logdet"),
                curvature_weight=float(kwargs.pop("curvature_weight", 1e-3)),
                loglambda_l2_weight=float(kwargs.pop("loglambda_l2_weight", 1e-4)),
                log_lambda_clip=float(kwargs.pop("log_lambda_clip", 3.0)),
                binning=kwargs.pop("binning", "linear_radius"),
            )
            self.learnable_h = LearnableSpectralHamiltonian(cfg)
        else:
            raise ValueError(f"Unknown H kind: {self.kind}")

        if kwargs:
            raise ValueError(f"Unused kwargs for HamSIGReg: {list(kwargs.keys())}")

        self.sigreg = SIGReg(num_slices=num_slices, **(sigreg_kwargs or {}))
        self.register_buffer("_zero", torch.zeros(()), persistent=False)

    def transform(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply H^{1/2} to z. z: [B,d] or [V,B,d]
        """
        if z.dim() not in (2, 3) or z.size(-1) != self.d:
            raise ValueError(f"Expected z shape [B,{self.d}] or [V,B,{self.d}], got {tuple(z.shape)}")

        if self.kind == "identity":
            return z

        if self.kind == "diag":
            s = self.sqrt_diag.to(device=z.device, dtype=z.dtype)
            return z * s

        if self.kind == "chain":
            if self.chain_mode == "grid":
                orig_dtype = z.dtype
                x = z.reshape(-1, self.h, self.w, self.d_f)
                if x.dtype in (torch.float16, torch.bfloat16):
                    x = x.float()
                X = torch.fft.fft2(x, dim=(1, 2), norm="ortho")
                s = self.sqrt_lam_grid.to(device=X.device, dtype=X.real.dtype)
                X = X * s[None, :, :, None]
                y = torch.fft.ifft2(X, dim=(1, 2), norm="ortho").real
                if y.dtype != orig_dtype:
                    y = y.to(orig_dtype)
                return y.reshape(z.shape)
            else:
                orig_dtype = z.dtype
                z_work = z
                if z_work.dtype in (torch.float16, torch.bfloat16):
                    z_work = z_work.float()
                Z = torch.fft.rfft(z_work, dim=-1, norm="ortho")
                s = self.sqrt_lam_rfft.to(device=z.device, dtype=Z.real.dtype)
                Z = Z * s
                out = torch.fft.irfft(Z, n=self.d, dim=-1, norm="ortho")
                return out.to(orig_dtype) if out.dtype != orig_dtype else out

        if self.kind == "learnable":
            cfg = self.learnable_h.cfg
            z_flat = z.reshape(-1, self.d).contiguous()
            z_tokens = z_flat.view(-1, cfg.h, cfg.w, cfg.d_f)
            z_t = self.learnable_h.sqrtH_apply(z_tokens).view(-1, self.d)
            return z_t.view(z.shape)

        raise RuntimeError("Unreachable")

    def forward(self, z: torch.Tensor, global_step: int = 0) -> torch.Tensor:
        z_t = self.transform(z)
        return self.sigreg(z_t)

    def spectral_regularizer(self) -> torch.Tensor:
        if self.kind == "learnable":
            return self.learnable_h.spectral_regularizer()
        return self._zero

    def _set_chain_beta(self, beta: float) -> None:
        beta = float(max(0.0, min(1.0, beta)))
        self.beta = beta
        if self.chain_mode == "grid":
            lam = (1.0 - beta) + beta * self.lam_grid_base
            self.register_buffer("sqrt_lam_grid", torch.sqrt(lam), persistent=False)
        else:
            lam = (1.0 - beta) + beta * self.lam_rfft_base
            self.register_buffer("sqrt_lam_rfft", torch.sqrt(lam), persistent=False)

    def set_h_schedule(self, *, alpha: Optional[float] = None, beta: Optional[float] = None) -> None:
        if self.kind == "learnable":
            self.learnable_h.set_schedule(alpha=alpha, beta=beta)
        elif self.kind == "chain" and beta is not None:
            self._set_chain_beta(beta)
