# eval/models/encoder_resnet.py
"""
ResNet encoder with two modes:
- "global": standard GAP head producing a global embedding.
- "tokens": expose an intermediate feature map as a grid of tokens and project
            channels to a chosen d_f, then flatten to (h*w*d_f).
"""
from typing import Literal, Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int = 512,
        mode: Literal["global", "tokens"] = "global",
        token_layer: Literal["layer2", "layer3", "layer4"] = "layer3",
        token_d_f: int = 32,
        token_hw: Optional[int] = None,
        stem: str = "cifar",
        split_qp: bool = False,
    ):
        super().__init__()
        mode = mode.lower()
        token_layer = token_layer.lower()
        stem = stem.lower()
        self.mode = mode
        self.token_layer = token_layer
        self.token_d_f = token_d_f
        self.token_hw = None if token_hw is None else int(token_hw)
        self.split_qp = bool(split_qp)
        self.out_dim = out_dim

        base = resnet18(weights=None)  # no pretraining
        if stem == "cifar":
            # CIFAR-friendly stem: smaller kernel/stride and no maxpool
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()
        elif stem == "imagenet":
            # keep torchvision default stem (7x7 stride2 + maxpool)
            pass
        else:
            raise ValueError(f"Unknown stem: {stem}")

        if mode == "global":
            modules = list(base.children())[:-1]  # drop FC
            self.backbone = nn.Sequential(*modules)
            self.fc = nn.Linear(base.fc.in_features, out_dim)
            return

        if mode != "tokens":
            raise ValueError(f"Unknown encoder mode: {mode}")

        # Token mode: expose intermediate feature maps
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        ch = {"layer2": 128, "layer3": 256, "layer4": 512}[token_layer]
        self.token_proj = nn.Sequential(
            nn.Conv2d(ch, token_d_f, kernel_size=1, bias=False),
            nn.BatchNorm2d(token_d_f),
        )
        self.token_pool = (
            nn.AdaptiveAvgPool2d((self.token_hw, self.token_hw))
            if self.token_hw is not None
            else None
        )
        if self.split_qp and (token_d_f % 2 != 0):
            raise ValueError("split_qp=True requires token_d_f to be even.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "global":
            h = self.backbone(x)           # [B, C, 1, 1]
            h = torch.flatten(h, 1)        # [B, C]
            z = self.fc(h)                 # [B, out_dim]
            return z

        # token mode
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        fmap = x if self.token_layer == "layer2" else None
        x = self.layer3(x)
        fmap = x if self.token_layer == "layer3" else fmap
        x = self.layer4(x)
        fmap = x if self.token_layer == "layer4" else fmap
        assert fmap is not None

        if self.token_pool is not None:
            fmap = self.token_pool(fmap)
        t = self.token_proj(fmap)                # [B, token_d_f, h, w]
        t = t.permute(0, 2, 3, 1).contiguous()   # [B, h, w, token_d_f]
        if not self.split_qp:
            z = t.view(t.size(0), -1)            # [B, h*w*token_d_f]
        else:
            # Channel-wise split into q/p per token: token_d_f = d_fq + d_fp
            df2 = self.token_d_f // 2
            tq = t[..., :df2]                    # [B, h, w, df2]
            tp = t[..., df2:]                    # [B, h, w, df2]
            q = tq.reshape(t.size(0), -1)        # [B, h*w*df2]
            p = tp.reshape(t.size(0), -1)        # [B, h*w*df2]
            z = torch.cat([q, p], dim=1)         # [B, h*w*token_d_f]

        if z.size(1) != self.out_dim:
            raise ValueError(
                f"Token encoder out_dim mismatch: got {z.size(1)} expected {self.out_dim}"
            )
        return z
