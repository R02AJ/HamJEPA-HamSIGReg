import torch

from hamjepa.predictor import HamiltonianFlowPredictor


def _omega(d: int, device=None, dtype=None) -> torch.Tensor:
    I = torch.eye(d, device=device, dtype=dtype)
    Z = torch.zeros_like(I)
    top = torch.cat([Z, I], dim=1)
    bot = torch.cat([-I, Z], dim=1)
    return torch.cat([top, bot], dim=0)


def test_separable_leapfrog_is_symplectic():
    torch.manual_seed(0)

    state_dim = 8  # 2d with d=4
    pred = HamiltonianFlowPredictor(
        state_dim=state_dim,
        hamiltonian="separable",
        hidden_dim=16,
        depth=2,
        residual_scale=0.01,
        method="leapfrog",
        steps=2,
        dt=0.1,
        learn_dt=False,
    ).double()

    x = torch.randn(1, state_dim, dtype=torch.float64, requires_grad=True)

    def f(inp):
        return pred(inp)

    J = torch.autograd.functional.jacobian(f, x)  # [1,8,1,8]
    J = J.squeeze(0).squeeze(1)  # [8,8]

    Om = _omega(state_dim // 2, device=J.device, dtype=J.dtype)
    lhs = J.T @ Om @ J
    err = (lhs - Om).abs().max().item()
    assert err < 1e-3, f"Symplectic condition violated: max|J^T Ω J - Ω|={err}"

    det = torch.det(J).item()
    assert abs(det - 1.0) < 1e-2, f"det(J)={det} not ~1"
