#!/usr/bin/env python3
"""
Physics showcase: SigReg-style (non-symplectic) vs HamJEPA (symplectic)

What this is:
- "SigReg" here is a proxy for a generic one-step predictor rollout:
  x_{t+1} = x_t + dt * f(x_t)   (Explicit Euler)  -> non-symplectic, drift
- "HamJEPA" uses the Hamiltonian structure + symplectic integrator:
  (q,p)_{t+1} = Leapfrog(H, q_t, p_t, dt)         -> symplectic, stable

Why it's faithful:
- SigReg-style models learn unconstrained updates. Iterating that update is
  equivalent to rolling out a generic non-symplectic map: energy/area drift.
- HamJEPA learns/uses Hamiltonian structure and integrates symplectically:
  bounded energy error + phase-space volume preservation.

This script generates:
1) Liouville / area preservation (blob -> blob)
2) Time reversibility heatmap (return error)
3) Long-horizon invariants plot (energy vs time)
4) Symplecticity defect histogram (||J^T Ω J - Ω||)
5) Covariance stability over rollout (trace/logdet/min-eig)
6) Energy drift heatmap (fixed; nonlinear pendulum + SymLog scaling)

Run:
  python scripts/physics_showcase_sigreg_vs_hamjepa.py --out_dir assets/physics --device cpu --all

Or generate individual figures:
  python scripts/physics_showcase_sigreg_vs_hamjepa.py --out_dir assets/physics --device cuda --which heatmap
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Headless-friendly matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch

# Ensure repo root is importable when running as scripts/xxx.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --- Use your repo's HamJEPA integrator utilities ---
from hamjepa.integrators import leapfrog_step, hamiltonian_vector_field


# ----------------------------
# Systems
# ----------------------------

class PendulumHamiltonian(torch.nn.Module):
    """
    Nonlinear pendulum:
      H(q,p) = 0.5 p^2 + (1 - cos q)

    q periodic; this is deliberately nonlinear so heatmaps aren't uniform.
    """
    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # q,p: (..., d). We'll treat d=1 by default but keep sum(-1).
        kinetic = 0.5 * (p ** 2).sum(dim=-1)
        potential = (1.0 - torch.cos(q)).sum(dim=-1)
        return kinetic + potential


@dataclass
class SystemSpec:
    name: str
    H: torch.nn.Module
    q_range: Tuple[float, float]
    p_range: Tuple[float, float]
    blob_center: Tuple[float, float]
    blob_radius: float


def get_system(name: str) -> SystemSpec:
    if name == "pendulum":
        return SystemSpec(
            name="pendulum",
            H=PendulumHamiltonian(),
            q_range=(-math.pi, math.pi),
            p_range=(-2.5, 2.5),
            blob_center=(0.5, 0.0),
            blob_radius=0.35,
        )
    raise ValueError(f"Unknown system: {name}")


# ----------------------------
# Helpers: angle wrap, hull/area
# ----------------------------

def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi]."""
    return (x + math.pi) % (2 * math.pi) - math.pi


def convex_hull(points: np.ndarray) -> np.ndarray:
    """
    Monotonic chain convex hull. points: (N,2) float.
    Returns hull vertices in CCW order.
    """
    pts = np.array(points, dtype=np.float64)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]  # sort by x, then y

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull


def polygon_area(poly: np.ndarray) -> float:
    """Shoelace formula. poly: (M,2)"""
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def area_from_mapped_grid(Q: np.ndarray, P: np.ndarray) -> float:
    """
    Area estimate for a deformed structured grid by summing cell triangle areas.
    """
    assert Q.shape == P.shape and Q.ndim == 2
    n = Q.shape[0]
    assert Q.shape == (n, n)

    def tri_area(ax, ay, bx, by, cx, cy) -> float:
        return 0.5 * abs(ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))

    total = 0.0
    for i in range(n - 1):
        for j in range(n - 1):
            ax, ay = Q[i, j], P[i, j]
            bx, by = Q[i + 1, j], P[i + 1, j]
            cx, cy = Q[i, j + 1], P[i, j + 1]
            dx, dy = Q[i + 1, j + 1], P[i + 1, j + 1]
            total += tri_area(ax, ay, bx, by, cx, cy)
            total += tri_area(bx, by, dx, dy, cx, cy)
    return total


# ----------------------------
# Step functions: SigReg proxy vs HamJEPA
# ----------------------------

def sigreg_explicit_euler_step(H: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                               q: torch.Tensor,
                               p: torch.Tensor,
                               dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Non-symplectic explicit Euler on Hamiltonian vector field.
    Proxy for "generic one-step predictor rolled out repeatedly".
    """
    dq_dt, dp_dt = hamiltonian_vector_field(H, q, p)
    q_next = q + dt * dq_dt
    p_next = p + dt * dp_dt
    return q_next, p_next


def hamjepa_leapfrog_step(H: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                          q: torch.Tensor,
                          p: torch.Tensor,
                          dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symplectic leapfrog from your repo."""
    return leapfrog_step(H, q, p, dt)


# ----------------------------
# Rollout utilities
# ----------------------------

@torch.no_grad()
def energy(H: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
           q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return H(q, p)


def rollout_final(step_fn: Callable, H: Callable, q0: torch.Tensor, p0: torch.Tensor,
                  dt: torch.Tensor, steps: int,
                  wrap_q: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Roll forward steps times, detaching each step to avoid graph growth.
    We keep grad enabled inside step_fn because it uses autograd.grad, but we detach outputs.
    """
    q, p = q0, p0
    for _ in range(steps):
        q, p = step_fn(H, q, p, dt)
        q, p = q.detach(), p.detach()
        if wrap_q:
            q = wrap_angle(q)
    return q, p


def rollout_trajectory(step_fn: Callable, H: Callable, q0: torch.Tensor, p0: torch.Tensor,
                       dt: torch.Tensor, steps: int, stride: int = 1,
                       wrap_q: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns trajectories sampled every 'stride' steps.
    Shapes:
      q_traj: (T, N, d), p_traj: (T, N, d)
    """
    q, p = q0, p0
    qs = []
    ps = []
    for t in range(steps + 1):
        if t % stride == 0:
            qs.append(q.detach().clone())
            ps.append(p.detach().clone())
        if t == steps:
            break
        q, p = step_fn(H, q, p, dt)
        q, p = q.detach(), p.detach()
        if wrap_q:
            q = wrap_angle(q)
    return torch.stack(qs, dim=0), torch.stack(ps, dim=0)


# ----------------------------
# Figures
# ----------------------------

def fig_energy_drift_heatmap(spec: SystemSpec, device: torch.device, out_path: str,
                             dt: float, steps: int, grid_n: int,
                             eps_rel: float = 1e-3) -> None:
    """
    Fixed heatmap:
      - nonlinear pendulum (so it's not uniform)
      - masks H0 ~ 0 singularity (or uses epsilon)
      - SymLogNorm so HamJEPA isn't a blank panel and SigReg isn't saturated
    """
    H = spec.H.to(device)

    q_lin = torch.linspace(spec.q_range[0], spec.q_range[1], grid_n, device=device)
    p_lin = torch.linspace(spec.p_range[0], spec.p_range[1], grid_n, device=device)
    Q0, P0 = torch.meshgrid(q_lin, p_lin, indexing="xy")
    q0 = Q0.reshape(-1, 1)
    p0 = P0.reshape(-1, 1)

    dt_t = torch.tensor(dt, device=device)

    H0 = energy(H, q0, p0).reshape(-1)

    qS, pS = rollout_final(sigreg_explicit_euler_step, H, q0, p0, dt_t, steps, wrap_q=True)
    HS = energy(H, qS, pS).reshape(-1)

    qH, pH = rollout_final(hamjepa_leapfrog_step, H, q0, p0, dt_t, steps, wrap_q=True)
    HH = energy(H, qH, pH).reshape(-1)

    # Relative drift, stabilized:
    # mask points with tiny H0 to avoid dividing by ~0
    mask = (H0 > 5e-3)
    relS = torch.empty_like(H0)
    relH = torch.empty_like(H0)
    relS[:] = float("nan")
    relH[:] = float("nan")
    relS[mask] = (HS[mask] - H0[mask]) / (H0[mask] + eps_rel)
    relH[mask] = (HH[mask] - H0[mask]) / (H0[mask] + eps_rel)

    relS_img = relS.reshape(grid_n, grid_n).detach().cpu().numpy()
    relH_img = relH.reshape(grid_n, grid_n).detach().cpu().numpy()

    # Shared normalization with robust clipping
    abs_all = np.concatenate([np.abs(relS_img[np.isfinite(relS_img)]),
                              np.abs(relH_img[np.isfinite(relH_img)])])
    if abs_all.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.quantile(abs_all, 0.995))
        vmax = max(vmax, 1e-3)

    norm = mcolors.SymLogNorm(linthresh=1e-2, vmin=-vmax, vmax=vmax, base=10)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    extent = [spec.q_range[0], spec.q_range[1], spec.p_range[0], spec.p_range[1]]

    im0 = axes[0].imshow(relS_img.T, origin="lower", extent=extent, aspect="auto", norm=norm)
    axes[0].set_title(f"SigReg proxy (Explicit Euler)\nrel drift (ΔH/(H0+ε)), T={steps}")
    axes[0].set_xlabel("q0"); axes[0].set_ylabel("p0")

    im1 = axes[1].imshow(relH_img.T, origin="lower", extent=extent, aspect="auto", norm=norm)
    axes[1].set_title(f"HamJEPA (Leapfrog)\nrel drift (ΔH/(H0+ε)), T={steps}")
    axes[1].set_xlabel("q0"); axes[1].set_ylabel("p0")

    cbar = fig.colorbar(im1, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("ΔH / (H0 + ε)  (SymLog scale)")

    fig.suptitle("Energy Drift Heatmap (nonlinear pendulum, shared SymLog scale)", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def fig_liouville_blob(spec: SystemSpec, device: torch.device, out_path: str,
                       dt: float, steps: int, blob_n: int) -> None:
    """
    Blob rollout visualization with area metrics.
    Grid-area ratio is the primary metric; hull is qualitative only.
    """
    H = spec.H.to(device)
    qc, pc = spec.blob_center
    r = spec.blob_radius

    q_lin = torch.linspace(qc - r, qc + r, blob_n, device=device, dtype=torch.float64)
    p_lin = torch.linspace(pc - r, pc + r, blob_n, device=device, dtype=torch.float64)
    Q0, P0 = torch.meshgrid(q_lin, p_lin, indexing="xy")
    q0 = Q0.reshape(-1, 1)
    p0 = P0.reshape(-1, 1)

    dt_t = torch.tensor(dt, device=device, dtype=torch.float64)

    # Important: no angle wrapping for Liouville metrics (wrapping is discontinuous in phase space).
    qS, pS = rollout_final(sigreg_explicit_euler_step, H, q0, p0, dt_t, steps, wrap_q=False)
    qH, pH = rollout_final(hamjepa_leapfrog_step, H, q0, p0, dt_t, steps, wrap_q=False)

    Q0_np = Q0.detach().cpu().numpy()
    P0_np = P0.detach().cpu().numpy()
    QS_np = qS.detach().cpu().numpy().reshape(blob_n, blob_n)
    PS_np = pS.detach().cpu().numpy().reshape(blob_n, blob_n)
    QH_np = qH.detach().cpu().numpy().reshape(blob_n, blob_n)
    PH_np = pH.detach().cpu().numpy().reshape(blob_n, blob_n)

    # Primary (Liouville-relevant) metric on structured grid.
    A0_grid = area_from_mapped_grid(Q0_np, P0_np)
    AS_grid = area_from_mapped_grid(QS_np, PS_np)
    AH_grid = area_from_mapped_grid(QH_np, PH_np)

    # Secondary qualitative metric.
    pts0 = np.stack([Q0_np.ravel(), P0_np.ravel()], axis=1)
    ptsS = np.stack([QS_np.ravel(), PS_np.ravel()], axis=1)
    ptsH = np.stack([QH_np.ravel(), PH_np.ravel()], axis=1)
    hull0 = convex_hull(pts0); A0 = polygon_area(hull0)
    hullS = convex_hull(ptsS); AS = polygon_area(hullS)
    hullH = convex_hull(ptsH); AH = polygon_area(hullH)

    ratioS_grid = AS_grid / (A0_grid + 1e-12)
    ratioH_grid = AH_grid / (A0_grid + 1e-12)
    ratioS_hull = AS / (A0 + 1e-12)
    ratioH_hull = AH / (A0 + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)

    for ax, pts, hull, title in [
        (
            axes[0],
            ptsS,
            hullS,
            f"SigReg proxy (Euler)\nGrid-area ratio: {ratioS_grid:.4f}x\nHull-area ratio: {ratioS_hull:.4f}x (qual.)",
        ),
        (
            axes[1],
            ptsH,
            hullH,
            f"HamJEPA (Leapfrog)\nGrid-area ratio: {ratioH_grid:.4f}x\nHull-area ratio: {ratioH_hull:.4f}x (qual.)",
        ),
    ]:
        ax.scatter(pts0[:, 0], pts0[:, 1], s=3, alpha=0.08, label="t=0 blob")
        ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.15, label=f"t={steps}")
        ax.plot(np.r_[hull0[:, 0], hull0[0, 0]], np.r_[hull0[:, 1], hull0[0, 1]], linewidth=2, label="Hull t=0 (qual.)")
        ax.plot(np.r_[hull[:, 0], hull[0, 0]], np.r_[hull[:, 1], hull[0, 1]], linewidth=2, label="Hull t=T (qual.)")
        ax.set_aspect("equal", "box")
        ax.set_xlabel("q"); ax.set_ylabel("p")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Liouville / Phase-space Area Preservation (blob rollout, unwrapped q, T={steps})", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def fig_liouville_area_preservation(spec: SystemSpec, device: torch.device, out_path: str,
                                    dt: float, steps: int, blob_n: int) -> None:
    """
    Quantitative Liouville figure (distinct from blob/hull view):
      - grid-area ratio A(t)/A(0) over time
      - absolute area-error |A(t)/A(0)-1| over time (log scale)

    Uses the actual repo system/integrators with unwrapped q.
    """
    H = spec.H.to(device)
    qc, pc = spec.blob_center
    r = spec.blob_radius
    n = blob_n

    q_lin = torch.linspace(qc - r, qc + r, n, device=device, dtype=torch.float64)
    p_lin = torch.linspace(pc - r, pc + r, n, device=device, dtype=torch.float64)
    Q0_t, P0_t = torch.meshgrid(q_lin, p_lin, indexing="xy")
    q0 = Q0_t.reshape(-1, 1)
    p0 = P0_t.reshape(-1, 1)
    dt_t = torch.tensor(dt, dtype=torch.float64, device=device)

    # Important: no angle wrapping for Liouville diagnostics.
    qS_tr, pS_tr = rollout_trajectory(sigreg_explicit_euler_step, H, q0, p0, dt_t, steps, stride=1, wrap_q=False)
    qH_tr, pH_tr = rollout_trajectory(hamjepa_leapfrog_step, H, q0, p0, dt_t, steps, stride=1, wrap_q=False)

    Q0 = Q0_t.detach().cpu().numpy()
    P0 = P0_t.detach().cpu().numpy()
    A0 = area_from_mapped_grid(Q0, P0)

    def area_ratio_series(qt: torch.Tensor, pt: torch.Tensor) -> np.ndarray:
        T = qt.shape[0]
        out = np.zeros(T, dtype=np.float64)
        for t in range(T):
            Q = qt[t].detach().cpu().numpy().reshape(n, n)
            P = pt[t].detach().cpu().numpy().reshape(n, n)
            out[t] = area_from_mapped_grid(Q, P) / (A0 + 1e-12)
        return out

    ratioS = area_ratio_series(qS_tr, pS_tr)
    ratioH = area_ratio_series(qH_tr, pH_tr)
    t = np.arange(steps + 1) * dt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    axes[0].plot(t, ratioS, linewidth=2.4, label=f"SigReg proxy (Euler) final={ratioS[-1]:.4f}x")
    axes[0].plot(t, ratioH, linewidth=2.4, label=f"HamJEPA (Leapfrog) final={ratioH[-1]:.4f}x")
    axes[0].axhline(1.0, color="k", linewidth=1.0, alpha=0.6)
    axes[0].set_title("Grid-area ratio over rollout")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("A(t) / A(0)")
    axes[0].legend(loc="upper left", fontsize=9)

    errS = np.abs(ratioS - 1.0) + 1e-16
    errH = np.abs(ratioH - 1.0) + 1e-16
    axes[1].semilogy(t, errS, linewidth=2.4, label="SigReg proxy (Euler)")
    axes[1].semilogy(t, errH, linewidth=2.4, label="HamJEPA (Leapfrog)")
    axes[1].set_title("Absolute area error (log scale)")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("|A(t)/A(0) - 1|")
    axes[1].legend(loc="upper left", fontsize=9)

    fig.suptitle(f"Liouville Area Preservation (quantitative grid metric, unwrapped q, T={steps})", fontsize=14)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_time_reversibility_heatmap(spec: SystemSpec, device: torch.device, out_path: str,
                                   dt: float, steps: int, grid_n: int) -> None:
    """
    Forward T steps, flip momentum p<- -p, forward T steps again, flip back.
    Return error ||x_return - x0||. Plot log10 error heatmap.
    """
    H = spec.H.to(device)
    q_lin = torch.linspace(spec.q_range[0], spec.q_range[1], grid_n, device=device)
    p_lin = torch.linspace(spec.p_range[0], spec.p_range[1], grid_n, device=device)
    Q0, P0 = torch.meshgrid(q_lin, p_lin, indexing="xy")
    q0 = Q0.reshape(-1, 1)
    p0 = P0.reshape(-1, 1)
    dt_t = torch.tensor(dt, device=device)

    def return_error(step_fn: Callable) -> torch.Tensor:
        q1, p1 = rollout_final(step_fn, H, q0, p0, dt_t, steps, wrap_q=True)
        # flip momentum
        p1r = -p1
        q2, p2 = rollout_final(step_fn, H, q1, p1r, dt_t, steps, wrap_q=True)
        # flip back
        p2r = -p2

        dq = wrap_angle(q2 - q0)
        dp = (p2r - p0)
        err = torch.sqrt((dq ** 2 + dp ** 2).sum(dim=-1))
        return err

    eS = return_error(sigreg_explicit_euler_step).reshape(grid_n, grid_n).detach().cpu().numpy()
    eH = return_error(hamjepa_leapfrog_step).reshape(grid_n, grid_n).detach().cpu().numpy()

    # log scale, robust
    eS_log = np.log10(eS + 1e-12)
    eH_log = np.log10(eH + 1e-12)
    vmin = float(np.quantile(np.concatenate([eS_log.ravel(), eH_log.ravel()]), 0.02))
    vmax = float(np.quantile(np.concatenate([eS_log.ravel(), eH_log.ravel()]), 0.98))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    extent = [spec.q_range[0], spec.q_range[1], spec.p_range[0], spec.p_range[1]]

    im0 = axes[0].imshow(eS_log.T, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
    axes[0].set_title("SigReg proxy (Euler)\nlog10 return error")
    axes[0].set_xlabel("q0"); axes[0].set_ylabel("p0")

    im1 = axes[1].imshow(eH_log.T, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
    axes[1].set_title("HamJEPA (Leapfrog)\nlog10 return error")
    axes[1].set_xlabel("q0"); axes[1].set_ylabel("p0")

    cbar = fig.colorbar(im1, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("log10(return error)")

    fig.suptitle(f"Time Reversibility Test (forward T, flip p, forward T), T={steps}", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def fig_energy_vs_time(spec: SystemSpec, device: torch.device, out_path: str,
                       dt: float, steps: int, n_ic: int, seed: int) -> None:
    """
    Sample random initial conditions; plot robust energy-drift summary.
    """
    rng = np.random.default_rng(seed)
    H = spec.H.to(device)

    q0 = torch.tensor(rng.uniform(spec.q_range[0], spec.q_range[1], size=(n_ic, 1)), device=device, dtype=torch.float32)
    p0 = torch.tensor(rng.uniform(spec.p_range[0], spec.p_range[1], size=(n_ic, 1)), device=device, dtype=torch.float32)
    dt_t = torch.tensor(dt, device=device)

    # trajectories (store all steps)
    qS, pS = rollout_trajectory(sigreg_explicit_euler_step, H, q0, p0, dt_t, steps, stride=1, wrap_q=True)
    qH, pH = rollout_trajectory(hamjepa_leapfrog_step, H, q0, p0, dt_t, steps, stride=1, wrap_q=True)

    with torch.no_grad():
        ES = energy(H, qS.reshape(-1, 1), pS.reshape(-1, 1)).reshape(steps + 1, n_ic).detach().cpu().numpy()
        EH = energy(H, qH.reshape(-1, 1), pH.reshape(-1, 1)).reshape(steps + 1, n_ic).detach().cpu().numpy()

    # relative drift (avoid division by ~0)
    ES0 = np.maximum(ES[0:1, :], 1e-3)
    EH0 = np.maximum(EH[0:1, :], 1e-3)
    rS = (ES - ES[0:1, :]) / ES0
    rH = (EH - EH[0:1, :]) / EH0

    t = np.arange(steps + 1) * dt

    fig = plt.figure(figsize=(10.5, 5.8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    def summary_band(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q10 = np.quantile(r, 0.10, axis=1)
        q25 = np.quantile(r, 0.25, axis=1)
        q50 = np.quantile(r, 0.50, axis=1)
        q75 = np.quantile(r, 0.75, axis=1)
        q90 = np.quantile(r, 0.90, axis=1)
        return q10, q25, q50, q75, q90

    s10, s25, s50, s75, s90 = summary_band(rS)
    h10, h25, h50, h75, h90 = summary_band(rH)

    # Keep the main panel readable: IQR + median only.
    ax.fill_between(t, s25, s75, alpha=0.22, label="SigReg 25-75%")
    ax.plot(t, s50, linewidth=2.8, label="SigReg median")

    ax.fill_between(t, h25, h75, alpha=0.22, label="HamJEPA 25-75%")
    ax.plot(t, h50, linewidth=2.8, label="HamJEPA median")

    ax.axhline(0.0, linewidth=1)

    # Robust y-range so a few outliers do not flatten the whole figure.
    all_band = np.concatenate([s25, s75, h25, h75])
    y_lo = float(np.quantile(all_band, 0.01))
    y_hi = float(np.quantile(all_band, 0.99))
    if y_hi > y_lo:
        pad = 0.08 * (y_hi - y_lo)
        ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.set_title("Long-horizon invariants: Relative energy drift over time")
    ax.set_xlabel("time")
    ax.set_ylabel("(H(t) - H0) / (H0 + ε)")
    ax.legend(loc="upper left", ncol=2, fontsize=9)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def symplectic_defect_2d(J: torch.Tensor) -> torch.Tensor:
    """
    Compute ||J^T Ω J - Ω||_F for 2D (q,p).
    J: (2,2)
    """
    Omega = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], device=J.device, dtype=J.dtype)
    M = J.T @ Omega @ J - Omega
    return torch.sqrt((M * M).sum())


def jacobian_of_map(F: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """
    J = dF/dx for x in R^2. Uses autograd.functional.jacobian.
    """
    # Ensure x is a leaf with grad
    x = x.detach().clone().requires_grad_(True)
    J = torch.autograd.functional.jacobian(F, x, create_graph=False)
    return J


def fig_symplecticity_hist(spec: SystemSpec, device: torch.device, out_path: str,
                           dt: float, n_samples: int, seed: int) -> None:
    """
    Symplecticity defect:
      d(x) = ||J(x)^T Ω J(x) - Ω||_F
    Plots both log10 histogram and raw defect histogram with log-spaced bins.
    """
    rng = np.random.default_rng(seed)
    H = spec.H.to(device)
    dtype = torch.float64
    dt_t = torch.tensor(dt, device=device, dtype=dtype)

    def make_map(step_fn: Callable) -> Callable[[torch.Tensor], torch.Tensor]:
        def F(x: torch.Tensor) -> torch.Tensor:
            # Do not wrap q here; wrap is discontinuous and breaks Jacobians.
            q = x[0:1].view(1, 1)
            p = x[1:2].view(1, 1)
            qn, pn = step_fn(H, q, p, dt_t)
            return torch.cat([qn.view(-1), pn.view(-1)], dim=0)
        return F

    FS = make_map(sigreg_explicit_euler_step)
    FH = make_map(hamjepa_leapfrog_step)

    # sample states
    xs = np.stack([
        rng.uniform(spec.q_range[0], spec.q_range[1], size=(n_samples,)),
        rng.uniform(spec.p_range[0], spec.p_range[1], size=(n_samples,))
    ], axis=1)

    dS = np.empty(n_samples, dtype=np.float64)
    dH = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        x = torch.tensor(xs[i], device=device, dtype=dtype)
        JS = jacobian_of_map(FS, x)
        JH = jacobian_of_map(FH, x)
        dS[i] = float(symplectic_defect_2d(JS).detach().cpu().item())
        dH[i] = float(symplectic_defect_2d(JH).detach().cpu().item())

    eps = np.finfo(np.float64).eps
    dS_clip = np.clip(dS, eps, None)
    dH_clip = np.clip(dH, eps, None)

    dmax = float(max(dS_clip.max(), dH_clip.max(), eps * 10.0))
    bins_logx = np.logspace(np.log10(eps), np.log10(dmax), 60)

    logS = np.log10(dS_clip)
    logH = np.log10(dH_clip)
    vmin = float(np.floor(min(logS.min(), logH.min())))
    vmax = float(np.ceil(max(logS.max(), logH.max())))
    if vmin == vmax:
        vmax = vmin + 1.0
    bins_log10 = np.linspace(vmin, vmax, 60)

    fracS_floor = 100.0 * float(np.mean(dS <= eps))
    fracH_floor = 100.0 * float(np.mean(dH <= eps))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)

    axes[0].hist(logS, bins=bins_log10, alpha=0.60, label=f"SigReg proxy (Euler) (median={np.median(logS):.2f})")
    axes[0].hist(
        logH,
        bins=bins_log10,
        alpha=0.60,
        label=f"HamJEPA (Leapfrog) (median={np.median(logH):.2f}, <=eps {fracH_floor:.1f}%)",
    )
    axes[0].set_title("Symplecticity defect (log10 scale)")
    axes[0].set_xlabel(r"$\log_{10}\,\|J^\top \Omega J - \Omega\|_F$")
    axes[0].set_ylabel("count")
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)

    axes[1].hist(dS_clip, bins=bins_logx, alpha=0.60, label=f"SigReg proxy (Euler) (<=eps {fracS_floor:.1f}%)")
    axes[1].hist(dH_clip, bins=bins_logx, alpha=0.60, label=f"HamJEPA (Leapfrog) (<=eps {fracH_floor:.1f}%)")
    axes[1].set_xscale("log")
    axes[1].set_title("Symplecticity violation score (native units)")
    axes[1].set_xlabel(r"$\|J^\top \Omega J - \Omega\|_F$ (log x; log-spaced bins)")
    axes[1].set_ylabel("count")
    axes[1].legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.suptitle("Symplecticity (lower is better): Euler breaks it, Leapfrog preserves it", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def fig_covariance_stability(spec: SystemSpec, device: torch.device, out_path: str,
                             dt: float, steps: int, n_ic: int, seed: int,
                             stride: int = 2) -> None:
    """
    Track covariance metrics over rollout:
      - trace(Cov)
      - logdet(Cov) (with small jitter)
      - min eigenvalue
    """
    rng = np.random.default_rng(seed)
    H = spec.H.to(device)
    dt_t = torch.tensor(dt, device=device)

    q0 = torch.tensor(rng.uniform(spec.q_range[0], spec.q_range[1], size=(n_ic, 1)), device=device, dtype=torch.float32)
    p0 = torch.tensor(rng.uniform(spec.p_range[0], spec.p_range[1], size=(n_ic, 1)), device=device, dtype=torch.float32)

    qS, pS = rollout_trajectory(sigreg_explicit_euler_step, H, q0, p0, dt_t, steps, stride=stride, wrap_q=True)
    qH, pH = rollout_trajectory(hamjepa_leapfrog_step, H, q0, p0, dt_t, steps, stride=stride, wrap_q=True)

    def cov_metrics(qt: torch.Tensor, pt: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # qt, pt: (T, N, 1)
        T = qt.shape[0]
        tr = np.zeros(T)
        ld = np.zeros(T)
        mn = np.zeros(T)
        for t in range(T):
            X = torch.cat([qt[t], pt[t]], dim=-1).detach().cpu().numpy()  # (N,2)
            X = X - X.mean(axis=0, keepdims=True)
            C = (X.T @ X) / max(X.shape[0] - 1, 1)  # (2,2)
            # jitter for stability
            Cj = C + 1e-9 * np.eye(2)
            w = np.linalg.eigvalsh(Cj)
            tr[t] = float(np.trace(C))
            ld[t] = float(np.log(np.linalg.det(Cj)))
            mn[t] = float(np.min(w))
        return tr, ld, mn

    trS, ldS, mnS = cov_metrics(qS, pS)
    trH, ldH, mnH = cov_metrics(qH, pH)

    t = np.arange(trS.shape[0]) * stride * dt

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10.5), constrained_layout=True)

    axes[0].plot(t, trS, linewidth=2, label="SigReg proxy")
    axes[0].plot(t, trH, linewidth=2, label="HamJEPA")
    axes[0].set_title("Covariance trace over rollout")
    axes[0].set_xlabel("time"); axes[0].set_ylabel("trace(Cov)")
    axes[0].legend()

    axes[1].plot(t, ldS, linewidth=2, label="SigReg proxy")
    axes[1].plot(t, ldH, linewidth=2, label="HamJEPA")
    axes[1].set_title("Covariance logdet over rollout")
    axes[1].set_xlabel("time"); axes[1].set_ylabel("logdet(Cov)")
    axes[1].legend()

    axes[2].plot(t, mnS, linewidth=2, label="SigReg proxy")
    axes[2].plot(t, mnH, linewidth=2, label="HamJEPA")
    axes[2].set_title("Min eigenvalue over rollout")
    axes[2].set_xlabel("time"); axes[2].set_ylabel("min eig(Cov)")
    axes[2].legend()

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", type=str, default="pendulum", choices=["pendulum"])
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out_dir", type=str, default="assets/physics_showcase")
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--steps", type=int, default=250)

    ap.add_argument("--grid_n", type=int, default=151)
    ap.add_argument("--blob_n", type=int, default=41)
    ap.add_argument("--n_ic", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--which",
        type=str,
        default="all",
        choices=["all", "heatmap", "liouville", "liouville_area", "reversibility", "energy", "symplecticity", "cov"],
    )
    ap.add_argument("--all", action="store_true", help="Generate all figures (same as --which all).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.all:
        args.which = "all"

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    spec = get_system(args.system)

    os.makedirs(args.out_dir, exist_ok=True)

    out_map: Dict[str, str] = {
        "heatmap": os.path.join(args.out_dir, "energy_drift_heatmap.png"),
        "liouville": os.path.join(args.out_dir, "liouville_blob_area.png"),
        "liouville_area": os.path.join(args.out_dir, "liouville_area_preservation.png"),
        "reversibility": os.path.join(args.out_dir, "time_reversibility_heatmap.png"),
        "energy": os.path.join(args.out_dir, "energy_vs_time.png"),
        "symplecticity": os.path.join(args.out_dir, "symplecticity_defect_hist.png"),
        "cov": os.path.join(args.out_dir, "covariance_stability.png"),
    }

    if args.which in ("all", "heatmap"):
        print("[make] heatmap ->", out_map["heatmap"])
        fig_energy_drift_heatmap(spec, device, out_map["heatmap"], dt=args.dt, steps=args.steps, grid_n=args.grid_n)

    if args.which in ("all", "liouville"):
        print("[make] liouville ->", out_map["liouville"])
        fig_liouville_blob(spec, device, out_map["liouville"], dt=args.dt, steps=args.steps, blob_n=args.blob_n)

    if args.which in ("all", "liouville_area"):
        print("[make] liouville area-preservation ->", out_map["liouville_area"])
        fig_liouville_area_preservation(spec, device, out_map["liouville_area"], dt=args.dt, steps=args.steps, blob_n=args.blob_n)

    if args.which in ("all", "reversibility"):
        print("[make] reversibility ->", out_map["reversibility"])
        fig_time_reversibility_heatmap(spec, device, out_map["reversibility"], dt=args.dt, steps=args.steps, grid_n=args.grid_n)

    if args.which in ("all", "energy"):
        print("[make] energy ->", out_map["energy"])
        fig_energy_vs_time(spec, device, out_map["energy"], dt=args.dt, steps=args.steps, n_ic=args.n_ic, seed=args.seed)

    if args.which in ("all", "symplecticity"):
        print("[make] symplecticity ->", out_map["symplecticity"])
        fig_symplecticity_hist(spec, device, out_map["symplecticity"], dt=args.dt, n_samples=200, seed=args.seed)

    if args.which in ("all", "cov"):
        print("[make] cov ->", out_map["cov"])
        fig_covariance_stability(spec, device, out_map["cov"], dt=args.dt, steps=args.steps, n_ic=args.n_ic, seed=args.seed)

    print("\nDone. Outputs in:", args.out_dir)
    for k, v in out_map.items():
        if os.path.exists(v):
            print(" -", v)


if __name__ == "__main__":
    main()
