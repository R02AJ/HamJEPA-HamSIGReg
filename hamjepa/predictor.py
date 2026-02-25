from __future__ import annotations

import math
import warnings
from typing import Literal

import torch
import torch.nn as nn

from .hamiltonian import QuadraticHamiltonian, LearnableHamiltonian, SeparableHamiltonian
from .integrators import integrate_hamiltonian, integrate_separable_leapfrog


def _inv_softplus(x: float) -> float:
    if x <= 0:
        raise ValueError("x must be > 0 for inverse softplus")
    return math.log(math.expm1(x))


class HamiltonianFlowPredictor(nn.Module):
    """
    Interpret z in R^{2d} as (q,p), integrate Hamiltonian dynamics, output z_T.

    Contract:
      - z0 last dimension must be even
      - output shape == input shape
    """

    def __init__(
        self,
        state_dim: int,
        *,
        hamiltonian: Literal["quadratic", "learnable", "separable"] = "learnable",
        hidden_dim: int = 256,
        depth: int = 2,
        activation: Literal["gelu", "relu", "silu"] = "gelu",
        residual_scale: float = 0.01,
        base_coeff: float = 1.0,
        method: Literal["leapfrog", "symplectic_euler"] = "leapfrog",
        steps: int = 1,
        dt: float = 0.1,
        learn_dt: bool = False,
        integrate_fp32: bool = True,
    ) -> None:
        super().__init__()
        if state_dim % 2 != 0:
            raise ValueError(f"state_dim must be even, got {state_dim}")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if dt <= 0:
            raise ValueError("dt must be > 0")

        self.state_dim = int(state_dim)
        self.d = self.state_dim // 2

        if hamiltonian == "quadratic":
            self.H = QuadraticHamiltonian()
        elif hamiltonian == "learnable":
            if method == "leapfrog":
                warnings.warn(
                    "HamiltonianFlowPredictor: 'learnable' Hamiltonian with explicit leapfrog is "
                    "not guaranteed symplectic. For MV-HJEPA use hamiltonian='separable'."
                )
            self.H = LearnableHamiltonian(
                dim=self.d,
                hidden_dim=hidden_dim,
                depth=depth,
                activation=activation,
                residual_scale=residual_scale,
                base_coeff=base_coeff,
            )
        elif hamiltonian == "separable":
            self.H = SeparableHamiltonian(
                dim=self.d,
                hidden_dim=hidden_dim,
                depth=depth,
                activation=activation,
                residual_scale=residual_scale,
                base_coeff=base_coeff,
            )
        else:
            raise ValueError(f"Unknown hamiltonian='{hamiltonian}'")

        self.method = method
        self.steps = int(steps)
        self.integrate_fp32 = bool(integrate_fp32)

        if learn_dt:
            self.raw_dt = nn.Parameter(torch.tensor(_inv_softplus(float(dt)), dtype=torch.float32))
            self.register_buffer("dt", torch.tensor(0.0), persistent=False)
        else:
            self.raw_dt = None
            self.register_buffer("dt", torch.tensor(float(dt), dtype=torch.float32), persistent=True)

    def _get_dt(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.raw_dt is None:
            return self.dt.to(device=device, dtype=dtype)
        return torch.nn.functional.softplus(self.raw_dt).to(device=device, dtype=dtype)

    def forward(self, z0: torch.Tensor, *, direction: int = 1) -> torch.Tensor:
        if z0.size(-1) != self.state_dim:
            raise ValueError(f"Expected last dim {self.state_dim}, got {z0.size(-1)}")
        if direction not in (-1, 1):
            raise ValueError(f"direction must be +1 or -1, got {direction}")

        # Integrator stability: do dynamics in fp32 even under bf16/fp16 training.
        orig_dtype = z0.dtype
        z_work = z0
        if self.integrate_fp32 and z_work.dtype in (torch.float16, torch.bfloat16):
            z_work = z_work.float()

        q0, p0 = z_work[..., : self.d], z_work[..., self.d :]
        dt = self._get_dt(device=z_work.device, dtype=z_work.dtype) * float(direction)

        if isinstance(self.H, SeparableHamiltonian) and self.method == "leapfrog":
            qT, pT = integrate_separable_leapfrog(self.H.potential, q0, p0, dt=dt, steps=self.steps)
        else:
            qT, pT = integrate_hamiltonian(self.H, q0, p0, dt=dt, steps=self.steps, method=self.method)

        zT = torch.cat([qT, pT], dim=-1)
        return zT.to(orig_dtype) if zT.dtype != orig_dtype else zT
