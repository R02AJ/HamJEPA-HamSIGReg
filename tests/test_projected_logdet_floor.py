import torch

from hamjepa.losses import ProjectedLogDetFloor, VarianceFloor


def test_rank1_line_can_pass_variance_floor_but_fail_logdet():
    torch.manual_seed(0)
    B, D = 512, 128

    # Rank-1 "line": dense direction so per-dim std is nonzero.
    a = torch.randn(B, 1)
    u = torch.ones(D) * 0.3
    x = a * u[None, :]

    var = VarianceFloor(dim=D, std_floor=0.1, ddp_sync=False)
    l_var = var(x).item()
    assert l_var < 1e-3, f"variance floor should be ~0 here, got {l_var}"

    logdet = ProjectedLogDetFloor(dim=D, proj_dim=32, logdet_floor=-2.0, ddp_sync=False)
    l_ld = logdet(x).item()
    assert l_ld > 1e-3, f"logdet floor should trigger on rank collapse, got {l_ld}"
