import importlib
import math

import torch


def _import_learnable_h():
    """
    Be robust to project structure:
    - try src.regularizers.learnable_h first
    - fallback to regularizers.learnable_h
    """
    for name in ("lejepa.hamiltonian.learnable_h",):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError("Could not import LearnableSpectralHamiltonian from lejepa.hamiltonian.learnable_h.")


mod = _import_learnable_h()
LearnableHConfig = mod.LearnableHConfig
LearnableSpectralHamiltonian = mod.LearnableSpectralHamiltonian


def test_g_bins_monotone():
    cfg = LearnableHConfig(h=8, w=8, d_f=4, bins=16, log_lambda_clip=2.5, eps_ridge=1e-4)
    H = LearnableSpectralHamiltonian(cfg)
    g = H.g_bins().detach().cpu()

    assert torch.isfinite(g).all()
    assert torch.all(g[1:] >= g[:-1]), f"g not monotone: {g}"


def test_lambda_positive_and_finite():
    cfg = LearnableHConfig(h=8, w=8, d_f=4, bins=16, log_lambda_clip=2.5, eps_ridge=1e-4)
    H = LearnableSpectralHamiltonian(cfg)
    lam = H.lambda_2d(device=torch.device("cpu")).detach()

    assert torch.isfinite(lam).all(), "lambda has NaN/Inf"
    assert (lam > 0).all(), "lambda must be strictly positive"


def test_lambda_condition_number_bounded_by_clip():
    clip = 2.5
    cfg = LearnableHConfig(h=8, w=8, d_f=4, bins=16, log_lambda_clip=clip, eps_ridge=1e-4)
    H = LearnableSpectralHamiltonian(cfg)

    ell = H.log_lambda_2d(device=torch.device("cpu")).detach()
    assert ell.abs().max().item() <= clip + 1e-6, "log-lambda exceeded clip"

    lam = H.lambda_2d(device=torch.device("cpu")).detach()
    lam_min = lam.min().clamp_min(1e-12)
    lam_max = lam.max()
    cond = (lam_max / lam_min).item()

    bound = math.exp(2.0 * clip) * 1.01
    assert cond <= bound, f"Condition number too large: cond={cond:.4f} bound={bound:.4f}"


def test_optimization_smoke_no_nans_no_explosion():
    torch.manual_seed(0)
    cfg = LearnableHConfig(
        h=8,
        w=8,
        d_f=4,
        bins=16,
        eps_ridge=1e-4,
        log_lambda_clip=2.5,
        curvature_weight=0.02,
        loglambda_l2_weight=0.02,
    )
    H = LearnableSpectralHamiltonian(cfg)

    opt = torch.optim.Adam(H.parameters(), lr=1e-2)

    for step in range(10):
        x = torch.randn(2, cfg.h, cfg.w, cfg.d_f)
        y = H.sqrtH_apply(x)

        loss = (y.square().mean()) + H.spectral_regularizer()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        assert torch.isfinite(loss).all(), f"loss became non-finite at step {step}"
        for p in H.parameters():
            assert torch.isfinite(p).all(), f"param became non-finite at step {step}"

        lam = H.lambda_2d(device=torch.device("cpu")).detach()
        assert torch.isfinite(lam).all(), f"lambda became non-finite at step {step}"
        assert (lam > 0).all(), f"lambda became non-positive at step {step}"
