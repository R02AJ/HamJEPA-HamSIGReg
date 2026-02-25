#!/usr/bin/env python3
"""
Visualizing Drift: SigReg (Left) vs HamJEPA (Right)

- SigReg-style proxy: explicit Euler (non-symplectic rollout) -> drift.
- HamJEPA proxy: symplectic leapfrog using repo integrator when available.

Saves an animated GIF suitable for README use.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

cache_dir = REPO_ROOT / ".mplconfig"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("TMPDIR", str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

_LEAPFROG_STEP_SEPARABLE = None


def _try_import_repo_leapfrog() -> None:
    """Prefer repo integrator, fallback if unavailable."""
    global _LEAPFROG_STEP_SEPARABLE
    try:
        from hamjepa.integrators import leapfrog_step_separable as lf  # type: ignore

        _LEAPFROG_STEP_SEPARABLE = lf
        return
    except Exception:
        pass
    try:
        from integrators import leapfrog_step_separable as lf  # type: ignore

        _LEAPFROG_STEP_SEPARABLE = lf
        return
    except Exception:
        _LEAPFROG_STEP_SEPARABLE = None


_try_import_repo_leapfrog()


def harmonic_potential(q: torch.Tensor) -> torch.Tensor:
    return 0.5 * (q * q).sum(dim=-1)


def energy(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    return 0.5 * (q**2).sum(axis=-1) + 0.5 * (p**2).sum(axis=-1)


def rollout_explicit_euler(
    q0: np.ndarray, p0: np.ndarray, dt: float, steps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Non-symplectic baseline rollout (drift)."""
    q = q0.copy()
    p = p0.copy()
    q_hist = [q.copy()]
    p_hist = [p.copy()]
    for _ in range(steps):
        q_old = q.copy()
        p_old = p.copy()
        q = q_old + dt * p_old
        p = p_old - dt * q_old
        q_hist.append(q.copy())
        p_hist.append(p.copy())
    q_hist = np.asarray(q_hist)
    p_hist = np.asarray(p_hist)
    return q_hist, p_hist, energy(q_hist, p_hist)


def _fallback_leapfrog_step(
    q: torch.Tensor, p: torch.Tensor, dt: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Local symplectic leapfrog if repo import is unavailable."""
    p_half = p - 0.5 * dt * q
    q_next = q + dt * p_half
    p_next = p_half - 0.5 * dt * q_next
    return q_next, p_next


def rollout_leapfrog_hamjepa(
    q0: np.ndarray, p0: np.ndarray, dt: float, steps: int, device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HamJEPA-style symplectic rollout."""
    q = torch.tensor(q0, dtype=torch.float32, device=device)
    p = torch.tensor(p0, dtype=torch.float32, device=device)
    q_hist = [q.detach().cpu().numpy().copy()]
    p_hist = [p.detach().cpu().numpy().copy()]

    use_repo = _LEAPFROG_STEP_SEPARABLE is not None
    dt_t = torch.tensor(float(dt), dtype=torch.float32, device=device)

    for _ in range(steps):
        if use_repo:
            q_req = q.detach().requires_grad_(True)
            q_next, p_next = _LEAPFROG_STEP_SEPARABLE(  # type: ignore[misc]
                harmonic_potential, q_req, p.detach(), dt_t
            )
            q = q_next.detach()
            p = p_next.detach()
        else:
            q, p = _fallback_leapfrog_step(q, p, float(dt))

        q_hist.append(q.detach().cpu().numpy().copy())
        p_hist.append(p.detach().cpu().numpy().copy())

    q_hist = np.asarray(q_hist)
    p_hist = np.asarray(p_hist)
    return q_hist, p_hist, energy(q_hist, p_hist)


def make_gif(
    out_path: str,
    dt: float = 0.1,
    steps: int = 250,
    stride: int = 1,
    fps: int = 30,
    trail: int = 120,
    dpi: int = 120,
) -> None:
    q0 = np.array([1.0, 0.0], dtype=np.float32)
    p0 = np.array([0.0, 1.0], dtype=np.float32)

    q_sig, _, e_sig = rollout_explicit_euler(q0, p0, dt=dt, steps=steps)
    q_ham, _, e_ham = rollout_leapfrog_hamjepa(q0, p0, dt=dt, steps=steps)

    q_sig = q_sig[::stride]
    q_ham = q_ham[::stride]
    e_sig = e_sig[::stride]
    e_ham = e_ham[::stride]
    t = np.arange(len(q_sig), dtype=np.float32) * dt * stride

    max_r = float(np.max(np.linalg.norm(q_sig, axis=1)))
    lim = max(1.25, 1.08 * max_r)

    fig = plt.figure(figsize=(10.5, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.2], wspace=0.18, hspace=0.28)

    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, :])

    fig.suptitle("Visualizing Drift: SigReg (Left) vs HamJEPA (Right)", y=0.98)

    for ax in (ax_l, ax_r):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        theta = np.linspace(0.0, 2.0 * math.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), linestyle="--", linewidth=1.0, alpha=0.35)

    ax_l.set_title("SigReg-style rollout (Explicit Euler / non-symplectic)")
    ax_r.set_title("HamJEPA rollout (Symplectic Leapfrog)")

    (line_l,) = ax_l.plot([], [], linewidth=2.0, alpha=0.95)
    (pt_l,) = ax_l.plot([], [], marker="o", markersize=6)
    (line_r,) = ax_r.plot([], [], linewidth=2.0, alpha=0.95)
    (pt_r,) = ax_r.plot([], [], marker="o", markersize=6)

    ax_e.grid(True, alpha=0.25)
    ax_e.set_xlabel("time")
    ax_e.set_ylabel("Energy")
    (e_l,) = ax_e.plot([], [], linewidth=2.0, label="SigReg (Euler)")
    (e_r,) = ax_e.plot([], [], linewidth=2.0, label="HamJEPA (Leapfrog)")
    vline = ax_e.axvline(0.0, linewidth=1.0, alpha=0.4)

    e_min = float(min(e_sig.min(), e_ham.min()))
    e_max = float(max(e_sig.max(), e_ham.max()))
    pad = 0.06 * (e_max - e_min + 1e-9)
    ax_e.set_ylim(e_min - pad, e_max + pad)
    ax_e.set_xlim(float(t.min()), float(t.max()))
    ax_e.legend(loc="upper left", frameon=False)

    txt = ax_e.text(
        0.99,
        0.05,
        "",
        transform=ax_e.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        alpha=0.9,
    )

    def init():
        line_l.set_data([], [])
        pt_l.set_data([], [])
        line_r.set_data([], [])
        pt_r.set_data([], [])
        e_l.set_data([], [])
        e_r.set_data([], [])
        vline.set_xdata([0.0, 0.0])
        txt.set_text("")
        return line_l, pt_l, line_r, pt_r, e_l, e_r, vline, txt

    def update(i: int):
        j0 = max(0, i - trail)
        line_l.set_data(q_sig[j0 : i + 1, 0], q_sig[j0 : i + 1, 1])
        pt_l.set_data([q_sig[i, 0]], [q_sig[i, 1]])

        line_r.set_data(q_ham[j0 : i + 1, 0], q_ham[j0 : i + 1, 1])
        pt_r.set_data([q_ham[i, 0]], [q_ham[i, 1]])

        e_l.set_data(t[: i + 1], e_sig[: i + 1])
        e_r.set_data(t[: i + 1], e_ham[: i + 1])
        vline.set_xdata([t[i], t[i]])

        d_e_sig = float(e_sig[i] - e_sig[0])
        d_e_ham = float(e_ham[i] - e_ham[0])
        txt.set_text(f"DeltaE  SigReg={d_e_sig:+.3f}    HamJEPA={d_e_ham:+.3f}")
        return line_l, pt_l, line_r, pt_r, e_l, e_r, vline, txt

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(t),
        interval=1000 / fps,
        blit=True,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)

    print(f"Saved GIF: {out}")
    if _LEAPFROG_STEP_SEPARABLE is not None:
        print("Used repo leapfrog_step_separable")
    else:
        print("Used fallback leapfrog (repo import not found)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="assets/visualizing_drift_sigreg_vs_hamjepa.gif",
        help="Output .gif path.",
    )
    parser.add_argument("--dt", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--trail", type=int, default=120)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dt <= 0:
        raise ValueError("--dt must be > 0")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.fps < 1:
        raise ValueError("--fps must be >= 1")
    if args.trail < 1:
        raise ValueError("--trail must be >= 1")

    make_gif(
        out_path=args.out,
        dt=args.dt,
        steps=args.steps,
        stride=args.stride,
        fps=args.fps,
        trail=args.trail,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
