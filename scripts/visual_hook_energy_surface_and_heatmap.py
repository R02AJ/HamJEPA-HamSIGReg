#!/usr/bin/env python3
"""
Visual hook for README:
  1) 3D energy surface animation: SigReg-proxy (left) vs HamJEPA (right)
  2) Side-by-side heatmaps of relative energy drift over (q0, p0) grid

This is a toy 1D Hamiltonian system (no dataset required):
  H(q,p) = 0.5 * p^2 + 0.5 * (omega * q)^2
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import lejepa  # noqa: F401
except Exception:
    lejepa = None  # type: ignore

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation as mpl_animation
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import SymLogNorm

try:
    from hamjepa.integrators import integrate_separable_leapfrog
except Exception as exc:  # pragma: no cover - runtime fallback path
    integrate_separable_leapfrog = None  # type: ignore
    HAMJEPA_IMPORT_ERR = exc
else:
    HAMJEPA_IMPORT_ERR = None


@dataclass(frozen=True)
class HeatmapSystem:
    name: str
    energy: Callable[[np.ndarray, np.ndarray], np.ndarray]
    dVdq: Callable[[np.ndarray], np.ndarray]


def make_pendulum_system() -> HeatmapSystem:
    def H(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 0.5 * (p**2) + (1.0 - np.cos(q))

    def dVdq(q: np.ndarray) -> np.ndarray:
        return np.sin(q)

    return HeatmapSystem(name="pendulum", energy=H, dVdq=dVdq)


def make_duffing_system() -> HeatmapSystem:
    def H(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 0.5 * (p**2) + 0.5 * (q**2) + 0.25 * (q**4)

    def dVdq(q: np.ndarray) -> np.ndarray:
        return q + (q**3)

    return HeatmapSystem(name="duffing", energy=H, dVdq=dVdq)


def potential(q: torch.Tensor, omega: float) -> torch.Tensor:
    return 0.5 * (omega * q).pow(2).sum(dim=-1)


def energy(q: torch.Tensor, p: torch.Tensor, omega: float) -> torch.Tensor:
    return 0.5 * p.pow(2).sum(dim=-1) + potential(q, omega)


def step_sigreg_proxy(
    q: torch.Tensor, p: torch.Tensor, dt: float, omega: float, method: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    if method == "explicit":
        q_next = q + dt * p
        p_next = p - dt * (omega**2) * q
        return q_next, p_next

    if method == "implicit":
        den = 1.0 + (dt * omega) ** 2
        q_next = (q + dt * p) / den
        p_next = (p - dt * (omega**2) * q) / den
        return q_next, p_next

    raise ValueError(f"Unknown sigreg method: {method}")


def step_hamjepa_repo(
    q: torch.Tensor, p: torch.Tensor, dt: float, omega: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    if integrate_separable_leapfrog is None:
        raise RuntimeError(
            "Could not import hamjepa.integrators.integrate_separable_leapfrog "
            f"(error: {HAMJEPA_IMPORT_ERR})"
        )

    # Integrator requires q to be a grad-enabled leaf.
    with torch.enable_grad():
        q_req = q.detach().requires_grad_(True)
        p_det = p.detach()
        dt_t = torch.as_tensor(dt, dtype=q.dtype, device=q.device)
        q_next, p_next = integrate_separable_leapfrog(
            lambda qq: potential(qq, omega), q_req, p_det, dt_t, steps=1
        )
    return q_next.detach(), p_next.detach()


def step_hamjepa_analytic(
    q: torch.Tensor, p: torch.Tensor, dt: float, omega: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    p_half = p - 0.5 * dt * (omega**2) * q
    q_next = q + dt * p_half
    p_next = p_half - 0.5 * dt * (omega**2) * q_next
    return q_next, p_next


def step_hamjepa(
    q: torch.Tensor, p: torch.Tensor, dt: float, omega: float, backend: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    if backend == "repo":
        return step_hamjepa_repo(q, p, dt, omega)
    if backend == "analytic":
        return step_hamjepa_analytic(q, p, dt, omega)
    raise ValueError(f"Unknown ham backend: {backend}")


def rollout_single(
    q0: float,
    p0: float,
    dt: float,
    steps: int,
    omega: float,
    device: torch.device,
    sigreg_method: str,
    ham_backend: str,
) -> dict:
    q_sig = torch.tensor([[q0]], dtype=torch.float32, device=device)
    p_sig = torch.tensor([[p0]], dtype=torch.float32, device=device)
    q_ham = q_sig.clone()
    p_ham = p_sig.clone()

    qs_sig, ps_sig, hs_sig = [], [], []
    qs_ham, ps_ham, hs_ham = [], [], []

    for _ in range(steps + 1):
        qs_sig.append(float(q_sig.item()))
        ps_sig.append(float(p_sig.item()))
        hs_sig.append(float(energy(q_sig, p_sig, omega).item()))

        qs_ham.append(float(q_ham.item()))
        ps_ham.append(float(p_ham.item()))
        hs_ham.append(float(energy(q_ham, p_ham, omega).item()))

        q_sig, p_sig = step_sigreg_proxy(q_sig, p_sig, dt, omega, sigreg_method)
        q_ham, p_ham = step_hamjepa(q_ham, p_ham, dt, omega, ham_backend)

    return {
        "sig": (np.asarray(qs_sig), np.asarray(ps_sig), np.asarray(hs_sig)),
        "ham": (np.asarray(qs_ham), np.asarray(ps_ham), np.asarray(hs_ham)),
    }


def build_surface(ax, lim: float, omega: float, grid_n: int = 70) -> None:
    q = np.linspace(-lim, lim, grid_n)
    p = np.linspace(-lim, lim, grid_n)
    qq, pp = np.meshgrid(q, p, indexing="xy")
    zz = 0.5 * (pp**2 + (omega * qq) ** 2)

    ax.plot_surface(qq, pp, zz, linewidth=0, antialiased=True, alpha=0.55)
    ax.contour(qq, pp, zz, zdir="z", offset=0.0, linewidths=0.8, alpha=0.45)
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_zlabel("H(q,p)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0.0, float(zz.max()) * 1.05)
    ax.view_init(elev=28, azim=-55)


def animate_energy_surface(
    traj_sig: Tuple[np.ndarray, np.ndarray, np.ndarray],
    traj_ham: Tuple[np.ndarray, np.ndarray, np.ndarray],
    dt: float,
    stride: int,
    out_gif: Path,
    out_mp4: Path | None,
    fps: int,
    omega: float,
) -> None:
    qs_sig, ps_sig, hs_sig = traj_sig
    qs_ham, ps_ham, hs_ham = traj_ham

    idx = np.arange(0, len(qs_sig), stride, dtype=int)
    qs_sig, ps_sig, hs_sig = qs_sig[idx], ps_sig[idx], hs_sig[idx]
    qs_ham, ps_ham, hs_ham = qs_ham[idx], ps_ham[idx], hs_ham[idx]

    lim = float(
        max(
            np.max(np.abs(qs_sig)),
            np.max(np.abs(ps_sig)),
            np.max(np.abs(qs_ham)),
            np.max(np.abs(ps_ham)),
            1.0,
        )
        * 1.2
    )

    fig = plt.figure(figsize=(12, 5.4), constrained_layout=True)
    fig.suptitle("Energy Surface Drift: SigReg proxy (Left) vs HamJEPA (Right)")
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax1.set_title("SigReg proxy (non-symplectic) - drift")
    ax2.set_title("HamJEPA leapfrog (symplectic) - bounded energy")
    build_surface(ax1, lim=lim, omega=omega)
    build_surface(ax2, lim=lim, omega=omega)

    line1, = ax1.plot([], [], [], lw=2.2, color="red")
    dot1, = ax1.plot([], [], [], marker="o", markersize=7, color="red")
    txt1 = ax1.text2D(0.02, 0.98, "", transform=ax1.transAxes, va="top")

    line2, = ax2.plot([], [], [], lw=2.2, color="green")
    dot2, = ax2.plot([], [], [], marker="o", markersize=7, color="green")
    txt2 = ax2.text2D(0.02, 0.98, "", transform=ax2.transAxes, va="top")

    def init():
        line1.set_data([], [])
        line1.set_3d_properties([])
        dot1.set_data([], [])
        dot1.set_3d_properties([])
        txt1.set_text("")
        line2.set_data([], [])
        line2.set_3d_properties([])
        dot2.set_data([], [])
        dot2.set_3d_properties([])
        txt2.set_text("")
        return line1, dot1, txt1, line2, dot2, txt2

    def update(i: int):
        t = i * dt * stride

        line1.set_data(qs_sig[: i + 1], ps_sig[: i + 1])
        line1.set_3d_properties(hs_sig[: i + 1])
        dot1.set_data([qs_sig[i]], [ps_sig[i]])
        dot1.set_3d_properties([hs_sig[i]])
        txt1.set_text(f"t={t:.2f}\nH={hs_sig[i]:.4f}")

        line2.set_data(qs_ham[: i + 1], ps_ham[: i + 1])
        line2.set_3d_properties(hs_ham[: i + 1])
        dot2.set_data([qs_ham[i]], [ps_ham[i]])
        dot2.set_3d_properties([hs_ham[i]])
        txt2.set_text(f"t={t:.2f}\nH={hs_ham[i]:.4f}")
        return line1, dot1, txt1, line2, dot2, txt2

    anim = FuncAnimation(
        fig,
        update,
        frames=len(qs_sig),
        init_func=init,
        blit=False,
        interval=1000 / fps,
    )

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_gif), writer=PillowWriter(fps=fps))

    if out_mp4 is not None:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        if not mpl_animation.writers.is_available("ffmpeg"):
            print(
                "[warn] ffmpeg writer is not available; skipped MP4 export. "
                "Install ffmpeg to enable MP4 output."
            )
        else:
            try:
                mp4_writer = FFMpegWriter(
                    fps=fps,
                    codec="libx264",
                    bitrate=1800,
                    extra_args=["-pix_fmt", "yuv420p"],
                )
                anim.save(str(out_mp4), writer=mp4_writer, dpi=180)
            except Exception as exc:
                print(f"[warn] Failed to save MP4 ({out_mp4}): {exc}")
    plt.close(fig)


def step_sigreg_proxy_np(
    q: np.ndarray, p: np.ndarray, dt: float, system: HeatmapSystem
) -> Tuple[np.ndarray, np.ndarray]:
    q_next = q + dt * p
    p_next = p - dt * system.dVdq(q)
    return q_next, p_next


def step_hamjepa_leapfrog_np(
    q: np.ndarray, p: np.ndarray, dt: float, system: HeatmapSystem
) -> Tuple[np.ndarray, np.ndarray]:
    p_half = p - 0.5 * dt * system.dVdq(q)
    q_next = q + dt * p_half
    p_next = p_half - 0.5 * dt * system.dVdq(q_next)
    return q_next, p_next


def rollout_grid_np(
    q0: np.ndarray,
    p0: np.ndarray,
    steps: int,
    dt: float,
    system: HeatmapSystem,
    method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    q = q0.copy()
    p = p0.copy()
    for _ in range(steps):
        if method == "sigreg":
            q, p = step_sigreg_proxy_np(q, p, dt, system)
        elif method == "hamjepa":
            q, p = step_hamjepa_leapfrog_np(q, p, dt, system)
        else:
            raise ValueError(f"Unknown method: {method}")
    return q, p


def energy_drift_heatmap(
    system: HeatmapSystem,
    qmin: float,
    qmax: float,
    pmin: float,
    pmax: float,
    grid: int,
    dt: float,
    steps: int,
    eps_h0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_lin = np.linspace(qmin, qmax, grid, dtype=np.float64)
    p_lin = np.linspace(pmin, pmax, grid, dtype=np.float64)
    q_mesh, p_mesh = np.meshgrid(q_lin, p_lin, indexing="xy")

    h0 = system.energy(q_mesh, p_mesh)
    valid = h0 > eps_h0

    q_sig, p_sig = rollout_grid_np(q_mesh, p_mesh, steps=steps, dt=dt, system=system, method="sigreg")
    q_ham, p_ham = rollout_grid_np(q_mesh, p_mesh, steps=steps, dt=dt, system=system, method="hamjepa")

    h_sig = system.energy(q_sig, p_sig)
    h_ham = system.energy(q_ham, p_ham)

    drift_sig = np.full_like(h0, np.nan, dtype=np.float64)
    drift_ham = np.full_like(h0, np.nan, dtype=np.float64)
    drift_sig[valid] = (h_sig[valid] - h0[valid]) / h0[valid]
    drift_ham[valid] = (h_ham[valid] - h0[valid]) / h0[valid]

    return q_mesh, p_mesh, h0, drift_sig, drift_ham


def plot_heatmaps(
    q_mesh: np.ndarray,
    p_mesh: np.ndarray,
    h0: np.ndarray,
    drift_sig: np.ndarray,
    drift_ham: np.ndarray,
    out_path: Path,
    linthresh: float,
    percentile: float,
    hm_steps: int,
    dt: float,
    system_name: str,
    with_contours: bool,
) -> None:
    both = np.concatenate(
        [drift_sig[np.isfinite(drift_sig)], drift_ham[np.isfinite(drift_ham)]]
    )
    vmax = float(np.percentile(np.abs(both), percentile))
    vmax = max(vmax, linthresh * 10.0)

    norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="white")

    extent = [q_mesh.min(), q_mesh.max(), p_mesh.min(), p_mesh.max()]
    fig, axs = plt.subplots(
        1, 2, figsize=(13.0, 5.2), sharex=True, sharey=True, constrained_layout=True
    )
    fig.suptitle(
        f"Energy Drift Heatmap ({system_name}) after T={hm_steps} steps, dt={dt}",
        y=1.02,
    )

    masked_sig = np.ma.masked_invalid(drift_sig)
    masked_ham = np.ma.masked_invalid(drift_ham)

    im0 = axs[0].imshow(
        masked_sig,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        aspect="auto",
    )
    im1 = axs[1].imshow(
        masked_ham,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        aspect="auto",
    )

    axs[0].set_title("SigReg proxy (Forward Euler)\nnon-symplectic -> drift")
    axs[1].set_title("HamJEPA (Leapfrog)\nsymplectic -> bounded drift")
    for ax in axs:
        ax.set_xlabel("q0")
        ax.set_ylabel("p0")

    if with_contours:
        levels = np.linspace(np.nanmin(h0), np.nanpercentile(h0, 95), 10)
        for ax in axs:
            ax.contour(
                q_mesh, p_mesh, h0, levels=levels, colors="k", linewidths=0.55, alpha=0.18
            )

    cbar = fig.colorbar(im1, ax=axs, location="right", shrink=0.92, pad=0.02)
    cbar.set_label("Relative energy drift  DeltaH / H0   (SymLog scale)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "surface", "heatmap"], default="all")
    parser.add_argument("--out_dir", type=str, default="assets/visuals")

    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.10)

    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--q0", type=float, default=1.0)
    parser.add_argument("--p0", type=float, default=0.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--sigreg_method",
        choices=["explicit", "implicit"],
        default="implicit",
        help="SigReg proxy integrator: explicit=blow-up, implicit=collapse.",
    )
    parser.add_argument(
        "--ham_backend",
        choices=["repo", "analytic"],
        default="repo",
        help="repo uses hamjepa.integrators; analytic uses local fallback.",
    )

    parser.add_argument("--grid", type=int, default=101)
    parser.add_argument("--hm_system", choices=["pendulum", "duffing"], default="pendulum")
    parser.add_argument("--qmin", type=float, default=-np.pi)
    parser.add_argument("--qmax", type=float, default=np.pi)
    parser.add_argument("--pmin", type=float, default=-2.5)
    parser.add_argument("--pmax", type=float, default=2.5)
    parser.add_argument("--hm_steps", type=int, default=250)
    parser.add_argument("--linthresh", type=float, default=1e-3)
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--eps_h0", type=float, default=1e-6)
    parser.add_argument("--contours", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dt <= 0:
        raise ValueError("--dt must be > 0")
    if args.steps < 1 or args.hm_steps < 1:
        raise ValueError("--steps and --hm_steps must be >= 1")
    if args.grid < 11:
        raise ValueError("--grid must be >= 11")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.percentile <= 0 or args.percentile >= 100:
        raise ValueError("--percentile must be in (0, 100)")
    if args.eps_h0 <= 0:
        raise ValueError("--eps_h0 must be > 0")
    if args.qmin >= args.qmax or args.pmin >= args.pmax:
        raise ValueError("Require qmin < qmax and pmin < pmax")

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ham_backend = args.ham_backend
    if ham_backend == "repo" and integrate_separable_leapfrog is None:
        print("[warn] Failed to import hamjepa integrator, switching to analytic backend.")
        print(f"[warn] Import error: {HAMJEPA_IMPORT_ERR}")
        ham_backend = "analytic"

    if args.mode in ("all", "surface"):
        traj = rollout_single(
            q0=args.q0,
            p0=args.p0,
            dt=args.dt,
            steps=args.steps,
            omega=args.omega,
            device=device,
            sigreg_method=args.sigreg_method,
            ham_backend=ham_backend,
        )
        out_gif = out_dir / "energy_surface_sigreg_vs_hamjepa.gif"
        out_mp4 = out_dir / "energy_surface_sigreg_vs_hamjepa.mp4"
        animate_energy_surface(
            traj_sig=traj["sig"],
            traj_ham=traj["ham"],
            dt=args.dt,
            stride=args.stride,
            out_gif=out_gif,
            out_mp4=out_mp4,
            fps=args.fps,
            omega=args.omega,
        )
        print(f"Saved 3D surface GIF: {out_gif}")
        if out_mp4.exists():
            print(f"Saved 3D surface MP4: {out_mp4}")

    if args.mode in ("all", "heatmap"):
        if args.hm_system == "pendulum":
            hm_system = make_pendulum_system()
        else:
            hm_system = make_duffing_system()

        q_mesh, p_mesh, h0, drift_sig, drift_ham = energy_drift_heatmap(
            system=hm_system,
            dt=args.dt,
            steps=args.hm_steps,
            qmin=args.qmin,
            qmax=args.qmax,
            pmin=args.pmin,
            pmax=args.pmax,
            grid=args.grid,
            eps_h0=args.eps_h0,
        )
        out_png = out_dir / "energy_drift_heatmap_sigreg_vs_hamjepa.png"
        plot_heatmaps(
            q_mesh=q_mesh,
            p_mesh=p_mesh,
            h0=h0,
            drift_sig=drift_sig,
            drift_ham=drift_ham,
            out_path=out_png,
            linthresh=args.linthresh,
            percentile=args.percentile,
            hm_steps=args.hm_steps,
            dt=args.dt,
            system_name=hm_system.name,
            with_contours=args.contours,
        )
        print(f"Saved heatmap PNG: {out_png}")
        for name, arr in [("SigReg", drift_sig), ("HamJEPA", drift_ham)]:
            vals = np.abs(arr[np.isfinite(arr)])
            print(
                f"[{name}] median|DeltaH/H0|={np.median(vals):.3e}  "
                f"p90={np.percentile(vals, 90):.3e}  "
                f"p99={np.percentile(vals, 99):.3e}  "
                f"max={np.max(vals):.3e}"
            )


if __name__ == "__main__":
    main()
