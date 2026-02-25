from __future__ import annotations

from typing import Callable, Literal, Tuple

import torch


def hamiltonian_vector_field(
    H: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    p: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Hamiltonian vector field:
        dq/dt =  dH/dp
        dp/dt = -dH/dq

    H must return a scalar per sample (shape (...,) or (...,1)).
    """
    q_req = q.requires_grad_(True)
    p_req = p.requires_grad_(True)

    energy = H(q_req, p_req)
    if energy.ndim > q_req.ndim - 1:
        energy = energy.squeeze(-1)

    grad_q, grad_p = torch.autograd.grad(
        energy.sum(), (q_req, p_req), create_graph=True, retain_graph=True
    )
    dq_dt = grad_p
    dp_dt = -grad_q
    return dq_dt, dp_dt


def symplectic_euler_step(
    H: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    p: torch.Tensor,
    dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symplectic Euler:
        p_{n+1} = p_n - dt * dH/dq(q_n, p_n)
        q_{n+1} = q_n + dt * dH/dp(q_n, p_{n+1})
    """
    dq_dt, dp_dt = hamiltonian_vector_field(H, q, p)
    p_next = p + dt * dp_dt
    dq_dt_next, _ = hamiltonian_vector_field(H, q, p_next)
    q_next = q + dt * dq_dt_next
    return q_next, p_next


def leapfrog_step(
    H: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    p: torch.Tensor,
    dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Leapfrog / Velocity Verlet (2nd order symplectic):
        p_{n+1/2} = p_n - (dt/2) * dH/dq(q_n, p_n)
        q_{n+1}   = q_n + dt     * dH/dp(q_n, p_{n+1/2})
        p_{n+1}   = p_{n+1/2} - (dt/2) * dH/dq(q_{n+1}, p_{n+1/2})
    """
    _, dp_dt = hamiltonian_vector_field(H, q, p)
    p_half = p + 0.5 * dt * dp_dt
    dq_dt_half, _ = hamiltonian_vector_field(H, q, p_half)
    q_next = q + dt * dq_dt_half
    _, dp_dt_next = hamiltonian_vector_field(H, q_next, p_half)
    p_next = p_half + 0.5 * dt * dp_dt_next
    return q_next, p_next


def integrate_hamiltonian(
    H: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    p: torch.Tensor,
    dt: torch.Tensor,
    steps: int,
    method: Literal["leapfrog", "symplectic_euler"] = "leapfrog",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if steps < 1:
        return q, p

    step_fn = leapfrog_step if method == "leapfrog" else symplectic_euler_step
    q_t, p_t = q, p
    for _ in range(steps):
        q_t, p_t = step_fn(H, q_t, p_t, dt)
    return q_t, p_t


def leapfrog_step_separable(
    V: Callable[[torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    p: torch.Tensor,
    dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Velocity Verlet for separable H(q,p) = 0.5||p||^2 + V(q):
        p_{n+1/2} = p_n - (dt/2) * dV/dq(q_n)
        q_{n+1}   = q_n + dt * p_{n+1/2}
        p_{n+1}   = p_{n+1/2} - (dt/2) * dV/dq(q_{n+1})
    """
    if not q.requires_grad:
        raise RuntimeError("q must require grad to compute dV/dq.")

    Vq = V(q)
    if Vq.ndim > q.ndim - 1:
        Vq = Vq.squeeze(-1)
    (gq,) = torch.autograd.grad(Vq.sum(), (q,), create_graph=True, retain_graph=True)
    p_half = p - 0.5 * dt * gq

    q_next = q + dt * p_half

    Vq_next = V(q_next)
    if Vq_next.ndim > q_next.ndim - 1:
        Vq_next = Vq_next.squeeze(-1)
    (gq_next,) = torch.autograd.grad(Vq_next.sum(), (q_next,), create_graph=True, retain_graph=True)
    p_next = p_half - 0.5 * dt * gq_next

    return q_next, p_next


def integrate_separable_leapfrog(
    V: Callable[[torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    p: torch.Tensor,
    dt: torch.Tensor,
    steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if steps < 1:
        return q, p

    q_t, p_t = q, p
    for _ in range(steps):
        q_t, p_t = leapfrog_step_separable(V, q_t, p_t, dt)
    return q_t, p_t
