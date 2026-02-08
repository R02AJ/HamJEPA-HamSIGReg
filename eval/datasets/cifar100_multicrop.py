from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100

_MEAN = (0.5071, 0.4867, 0.4408)
_STD = (0.2675, 0.2565, 0.2761)


@dataclass(frozen=True)
class MultiCropCfg:
    num_global_views: int = 2
    num_local_views: int = 6

    out_size: int = 32

    global_scale: Tuple[float, float] = (0.3, 1.0)
    local_scale: Tuple[float, float] = (0.05, 0.3)

    hflip_p: float = 0.5
    cj_p: float = 0.8
    grayscale_p: float = 0.2
    blur_p: float = 0.5
    solarize_p: float = 0.2
    solarize_thresh: int = 128

    cj_brightness: float = 0.4
    cj_contrast: float = 0.4
    cj_saturation: float = 0.2
    cj_hue: float = 0.1


def _build_transform(cfg: MultiCropCfg, *, scale: Tuple[float, float]) -> transforms.Compose:
    color_jitter = transforms.ColorJitter(
        brightness=cfg.cj_brightness,
        contrast=cfg.cj_contrast,
        saturation=cfg.cj_saturation,
        hue=cfg.cj_hue,
    )
    blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(cfg.out_size, scale=scale),
            transforms.RandomHorizontalFlip(p=cfg.hflip_p),
            transforms.RandomApply([color_jitter], p=cfg.cj_p),
            transforms.RandomGrayscale(p=cfg.grayscale_p),
            transforms.RandomApply([blur], p=cfg.blur_p),
            transforms.RandomSolarize(threshold=cfg.solarize_thresh, p=cfg.solarize_p),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )


class CIFAR100MultiCrop(Dataset):
    def __init__(self, root: str, train: bool, cfg: MultiCropCfg, download: bool = True):
        super().__init__()
        try:
            self.base = CIFAR100(
                root=root,
                train=train,
                download=download,
                transform=None,
                target_type="fine",
            )
        except TypeError:
            self.base = CIFAR100(
                root=root,
                train=train,
                download=download,
                transform=None,
            )

        self.cfg = cfg
        self.global_tf = _build_transform(cfg, scale=cfg.global_scale)
        self.local_tf = _build_transform(cfg, scale=cfg.local_scale)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, fine = self.base[idx]
        views: List[torch.Tensor] = []

        for _ in range(self.cfg.num_global_views):
            views.append(self.global_tf(img))

        for _ in range(self.cfg.num_local_views):
            views.append(self.local_tf(img))

        coarse = -1
        return views, int(fine), int(coarse)
