import torch

from lejepa.hamiltonian.learnable_h import LearnableHConfig, LearnableSpectralHamiltonian


def _make_h():
    cfg = LearnableHConfig(
        h=8,
        w=8,
        d_f=4,
        bins=8,
        log_lambda_clip=2.0,
    )
    return LearnableSpectralHamiltonian(cfg)


def test_constant_field_is_constant():
    H = _make_h()
    x = torch.ones(2, H.cfg.h, H.cfg.w, H.cfg.d_f)
    y = H.sqrtH_apply(x)

    spatial_var = y.var(dim=(1, 2), unbiased=False)
    assert spatial_var.max().item() < 1e-6


def test_layout_equivalence():
    H = _make_h()
    x_hwdf = torch.randn(2, H.cfg.h, H.cfg.w, H.cfg.d_f)
    x_dfhw = x_hwdf.permute(0, 3, 1, 2).contiguous()

    y1 = H.sqrtH_apply(x_hwdf)
    y2 = H.sqrtH_apply(x_dfhw).permute(0, 2, 3, 1).contiguous()

    assert torch.allclose(y1, y2, atol=1e-5, rtol=1e-5)
