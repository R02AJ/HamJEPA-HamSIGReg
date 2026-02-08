import os

import pytest
import torch

from eval.datasets.cifar100_multicrop import CIFAR100MultiCrop, MultiCropCfg


def test_cifar100_multicrop_shapes(tmp_path):
    cfg = MultiCropCfg(num_global_views=2, num_local_views=6, out_size=32)

    try:
        ds = CIFAR100MultiCrop(root=str(tmp_path), train=True, cfg=cfg, download=False)
    except RuntimeError:
        pytest.skip("CIFAR-100 data not available locally.")

    if len(ds) == 0:
        pytest.skip("CIFAR-100 dataset is empty.")

    views, fine, coarse = ds[0]
    assert isinstance(views, list)
    assert len(views) == 8
    assert isinstance(fine, int)
    assert isinstance(coarse, int)

    for v in views:
        assert isinstance(v, torch.Tensor)
        assert v.shape == (3, 32, 32)
        assert torch.isfinite(v).all()
