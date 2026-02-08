import torch

from lejepa.losses import SIGReg


def test_sigreg_gaussian_smaller_than_uniform():
    torch.manual_seed(0)
    reg = SIGReg(num_slices=128, t_max=3.0, n_points=17, ddp_sync=False)

    n, d = 512, 128
    x_gauss = torch.randn(n, d)
    x_unif = (torch.rand(n, d) - 0.5) * 2.0

    s_gauss = reg(x_gauss)
    s_unif = reg(x_unif)

    assert torch.isfinite(s_gauss)
    assert torch.isfinite(s_unif)
    assert s_gauss.item() < s_unif.item()


def test_sigreg_backward():
    torch.manual_seed(0)
    reg = SIGReg(num_slices=64, t_max=3.0, n_points=9, ddp_sync=False)
    x = torch.randn(256, 64, requires_grad=True)
    loss = reg(x)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
