from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class QuadraticHamiltonian(nn.Module):
    """
    Fixed quadratic Hamiltonian:
        H(q,p) = 0.5*(||q||^2 + ||p||^2)
    """

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return 0.5 * (q.square().sum(dim=-1) + p.square().sum(dim=-1))


class LearnableHamiltonian(nn.Module):
    """
    Learnable Hamiltonian with quadratic base + scaled residual MLP:

        H_theta(q,p) = 0.5*(||q||^2 + ||p||^2) + s * f_theta([q,p])

    The residual_scale 's' is critical for stability (prevents exploding gradients
    through the integrator).

    dim is the dimension of q (and p). Total state dimension is 2*dim.
    """

    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int = 256,
        depth: int = 2,
        activation: Literal["gelu", "relu", "silu"] = "gelu",
        residual_scale: float = 0.01,
        base_coeff: float = 1.0,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if residual_scale < 0:
            raise ValueError(f"residual_scale must be >= 0, got {residual_scale}")
        if base_coeff < 0:
            raise ValueError(f"base_coeff must be >= 0, got {base_coeff}")

        self.dim = int(dim)
        self.residual_scale = float(residual_scale)
        self.base_coeff = float(base_coeff)

        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU(inplace=True)
        elif activation == "silu":
            act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation '{activation}'")

        layers = []
        in_dim = 2 * self.dim
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(act)
        layers.append(nn.Linear(hidden_dim, 1))
        self.residual = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        base = 0.5 * (q.square().sum(dim=-1) + p.square().sum(dim=-1))
        res = self.residual(torch.cat([q, p], dim=-1)).squeeze(-1)
        return base + self.residual_scale * res


class SeparableHamiltonian(nn.Module):
    """
    Separable Hamiltonian suitable for explicit leapfrog symplectic integration:
        H(q,p) = T(p) + V(q)
        T(p) = 0.5 * ||p||^2   (fixed units; no mass-matrix knob in MV)
        V(q) = 0.5 * ||q||^2 + s * f_phi(q)

    The base 0.5||q||^2 term blocks the trivial-potential loophole (dp/dt=0).
    """

    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int = 256,
        depth: int = 2,
        activation: Literal["gelu", "relu", "silu"] = "gelu",
        residual_scale: float = 0.01,
        base_coeff: float = 1.0,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if residual_scale < 0:
            raise ValueError(f"residual_scale must be >= 0, got {residual_scale}")
        if base_coeff < 0:
            raise ValueError(f"base_coeff must be >= 0, got {base_coeff}")

        self.dim = int(dim)
        self.residual_scale = float(residual_scale)
        self.base_coeff = float(base_coeff)

        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU(inplace=True)
        elif activation == "silu":
            act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation '{activation}'")

        layers = []
        for i in range(depth):
            layers.append(nn.Linear(self.dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(act)
        layers.append(nn.Linear(hidden_dim, 1))
        self.residual_V = nn.Sequential(*layers)

    def potential(self, q: torch.Tensor) -> torch.Tensor:
        base_V = 0.5 * self.base_coeff * q.square().sum(dim=-1)
        res_V = self.residual_V(q).squeeze(-1)
        return base_V + self.residual_scale * res_V

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        T = 0.5 * p.square().sum(dim=-1)
        return T + self.potential(q)
