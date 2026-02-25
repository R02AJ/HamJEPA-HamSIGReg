#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
import argparse
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)

# Project imports
from eval.models.encoder_resnet import ResNetEncoder
from lejepa.hamiltonian.ham_sigreg import HamSIGReg

# Optional: use SSL dataset for two-view alignment/uniformity on ImageNet
try:
    from eval.datasets.imagenet_multicrop import ImageNetMultiCrop, MultiCropCfg
except Exception:
    ImageNetMultiCrop = None
    MultiCropCfg = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_dump(obj: Any, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, ks: Tuple[int, int] = (1, 5)) -> Dict[str, float]:
    with torch.no_grad():
        max_k = max(ks)
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1))
        out = {}
        for k in ks:
            out[f"top{k}"] = correct[:, :k].any(dim=1).float().mean().item()
        return out


def summarize_1d(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float().cpu()
    return {
        "mean": x.mean().item(),
        "std": x.std(unbiased=False).item(),
        "p10": x.quantile(0.10).item(),
        "p50": x.quantile(0.50).item(),
        "p90": x.quantile(0.90).item(),
        "min": x.min().item(),
        "max": x.max().item(),
    }


def load_yaml_cfg(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        try:
            from omegaconf import OmegaConf  # type: ignore

            return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        except Exception as e:
            raise RuntimeError("Could not load YAML config. Install pyyaml or omegaconf.") from e


class EncoderOnly(nn.Module):
    """
    Minimal wrapper so we can load checkpoints even if they were saved from a bigger model.
    We only use encoder(x) for evaluation.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        mcfg = cfg["model"]
        self.encoder = ResNetEncoder(
            out_dim=int(mcfg["embed_dim"]),
            mode=str(mcfg["encoder_mode"]),
            token_layer=str(mcfg.get("token_layer", "layer3")),
            token_d_f=int(mcfg.get("token_d_f", 32)),
            token_hw=(int(mcfg["token_hw"]) if "token_hw" in mcfg and mcfg["token_hw"] is not None else None),
            stem=str(mcfg.get("encoder_stem", "cifar")),
            split_qp=bool(mcfg.get("split_qp", False)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def load_checkpoint_into_encoder(model: EncoderOnly, ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "encoder" in ckpt and isinstance(ckpt["encoder"], dict):
            sd = ckpt["encoder"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    if not isinstance(sd, dict):
        raise ValueError("Checkpoint did not contain a state_dict-like object.")

    candidates = [
        sd,
        _strip_prefix(sd, "encoder."),
        _strip_prefix(sd, "module."),
        _strip_prefix(sd, "module.encoder."),
        _strip_prefix(sd, "model."),
        _strip_prefix(sd, "model.encoder."),
    ]

    info: Dict[str, Any] = {}
    for i, cand in enumerate(candidates):
        missing, unexpected = model.encoder.load_state_dict(cand, strict=False)
        info[f"try_{i}_missing"] = len(missing)
        info[f"try_{i}_unexpected"] = len(unexpected)
        if len(missing) == 0 and len(unexpected) == 0:
            info["loaded_from_try"] = i
            info["ckpt"] = ckpt
            return info

    raise RuntimeError(
        "Bad checkpoint load: encoder state_dict did not match. "
        f"Missing/unexpected counts: {info}"
    )


def build_ham_module_from_cfg(cfg: Dict[str, Any], d: int, device: torch.device) -> HamSIGReg:
    reg_cfg = cfg.get("regularizer", {})
    reg_type = str(reg_cfg.get("type", "sigreg")).lower()
    if reg_type != "ham_sigreg":
        raise ValueError(f"Expected regularizer.type=ham_sigreg, got {reg_type}")

    num_slices = int(reg_cfg.get("num_slices", 256))
    t_min = reg_cfg.get("t_min", None)
    t_max = float(reg_cfg.get("t_max", 3.0))
    t_range = (t_min, t_max) if t_min is not None else None
    n_points = int(reg_cfg.get("n_points", 17))
    weight_type = str(reg_cfg.get("weight_type", "gaussian"))

    sigreg_kwargs = dict(
        t_max=t_max,
        t_range=t_range,
        n_points=n_points,
        weight_type=weight_type,
        ddp_sync=False,
    )

    ham_kind = str(reg_cfg.get("ham_kind", "diag"))
    ham_kwargs = dict(reg_cfg.get("ham_kwargs", {}) or {})

    return HamSIGReg(
        d=d,
        kind=ham_kind,
        device=device,
        num_slices=num_slices,
        sigreg_kwargs=sigreg_kwargs,
        **ham_kwargs,
    ).to(device)


def _resolve_split_roots(data_root: str, split: str) -> List[str]:
    split = split.lower()
    data_root = os.path.abspath(data_root)
    if split in ("train", "trainset", "train_set"):
        primary = os.path.join(data_root, "train")
        if os.path.isdir(primary):
            return [primary]
        parts = sorted(
            d for d in os.listdir(data_root) if d.startswith("train.") and os.path.isdir(os.path.join(data_root, d))
        )
        return [os.path.join(data_root, d) for d in parts]
    if split in ("val", "valid", "validation"):
        primary = os.path.join(data_root, "val")
        if os.path.isdir(primary):
            return [primary]
        parts = sorted(
            d for d in os.listdir(data_root) if d.startswith("val.") and os.path.isdir(os.path.join(data_root, d))
        )
        return [os.path.join(data_root, d) for d in parts]
    raise ValueError(f"Unknown split '{split}'. Expected train/val.")


def _build_imagefolder_concat(roots: List[str], tfm) -> Dataset:
    if not roots:
        raise RuntimeError("No dataset roots found for the requested split.")
    datasets_list: List[datasets.ImageFolder] = []
    ref_classes = None
    for r in roots:
        ds = datasets.ImageFolder(r, transform=tfm)
        if ref_classes is None:
            ref_classes = ds.classes
        elif ds.classes != ref_classes:
            raise RuntimeError(f"Class list mismatch between splits: {r}")
        datasets_list.append(ds)
    if len(datasets_list) == 1:
        return datasets_list[0]
    return ConcatDataset(datasets_list)


def make_eval_loaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> Tuple[DataLoader, DataLoader, int]:
    dataset = dataset.lower()

    if dataset not in ("imagenet", "imagenet1k", "imagenet-1k"):
        raise ValueError(f"Unknown dataset '{dataset}'. Use imagenet / imagenet1k.")

    tfm = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    train_roots = _resolve_split_roots(data_root, "train")
    val_roots = _resolve_split_roots(data_root, "val")
    train_ds = _build_imagefolder_concat(train_roots, tfm)
    test_ds = _build_imagefolder_concat(val_roots, tfm)
    num_classes = len(train_ds.classes) if hasattr(train_ds, "classes") else 1000

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes


class TwoViewImageFolder(Dataset):
    def __init__(self, roots: List[str], transform):
        self.base = _build_imagefolder_concat(roots, tfm=None)
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        v1 = self.transform(img)
        v2 = self.transform(img)
        return v1, v2, y


def make_two_view_loader(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    max_items: int = 5000,
) -> DataLoader:
    dataset = dataset.lower()

    if dataset not in ("imagenet", "imagenet1k", "imagenet-1k"):
        raise ValueError(f"Unknown dataset '{dataset}'. Use imagenet / imagenet1k.")

    if ImageNetMultiCrop is None or MultiCropCfg is None:
        raise RuntimeError("ImageNetMultiCrop not importable. Cannot do 2-view diagnostics on ImageNet.")

    mc_cfg = MultiCropCfg(
        num_global_views=2,
        num_local_views=0,
        out_size=image_size,
        global_scale=(0.4, 1.0),
        local_scale=(0.05, 0.4),
    )
    ds = ImageNetMultiCrop(root=data_root, split="train", cfg=mc_cfg)
    if max_items is not None and len(ds) > max_items:
        idx = torch.randperm(len(ds))[:max_items].tolist()
        ds = torch.utils.data.Subset(ds, idx)

    def collate(batch):
        v1s, v2s, ys = [], [], []
        for views, y, _ in batch:
            v1s.append(views[0])
            v2s.append(views[1])
            ys.append(y)
        return torch.stack(v1s, 0), torch.stack(v2s, 0), torch.tensor(ys, dtype=torch.long)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    transform_fn=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise ValueError("Loader must yield (x,y,...)")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        z = encoder(x).float()
        if transform_fn is not None:
            z = transform_fn(z)
        feats.append(z.cpu())
        labels.append(y.cpu())

    return torch.cat(feats, 0), torch.cat(labels, 0)


@torch.no_grad()
def knn_sweep(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    ks: List[int],
    num_classes: int,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, Any]:
    ks = sorted(list(set(ks)))
    k_max = max(ks)

    Xtr = F.normalize(train_feats.to(device), dim=1)
    Ytr = train_labels.to(device)

    Xte = F.normalize(test_feats.to(device), dim=1)
    Yte = test_labels.to(device)

    correct_top1 = {k: 0 for k in ks}
    correct_top5 = {k: 0 for k in ks}
    n_total = 0

    for i in range(0, Xte.size(0), batch_size):
        xb = Xte[i : i + batch_size]
        yb = Yte[i : i + batch_size]
        n_total += yb.numel()

        sims = xb @ Xtr.t()
        _, idx = sims.topk(k_max, dim=1, largest=True, sorted=True)
        nn_labels = Ytr[idx]

        counts = torch.zeros((xb.size(0), num_classes), device=device, dtype=torch.float32)
        ones = torch.ones((xb.size(0), 1), device=device, dtype=torch.float32)

        for j in range(k_max):
            counts.scatter_add_(1, nn_labels[:, j : j + 1], ones)
            k_now = j + 1
            if k_now in correct_top1:
                preds_top1 = counts.argmax(dim=1)
                correct_top1[k_now] += (preds_top1 == yb).sum().item()

                top5 = counts.topk(5, dim=1).indices
                correct_top5[k_now] += top5.eq(yb.view(-1, 1)).any(dim=1).sum().item()

    return {
        "k_values": ks,
        "top1": [correct_top1[k] / n_total for k in ks],
        "top5": [correct_top5[k] / n_total for k in ks],
    }


def run_linear_probe(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
) -> Dict[str, Any]:
    D = train_feats.size(1)
    clf = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, num_classes)).to(device)

    ds_tr = TensorDataset(train_feats, train_labels)
    ds_te = TensorDataset(test_feats, test_labels)
    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0)
    ld_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(ld_tr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    clf.train()
    for _ in range(epochs):
        for xb, yb in ld_tr:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True)

            logits = clf(xb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

    clf.eval()
    correct1 = 0.0
    correct5 = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in ld_te:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True)
            logits = clf(xb)
            acc = topk_accuracy(logits, yb, ks=(1, 5))
            bsz = yb.size(0)
            correct1 += acc["top1"] * bsz
            correct5 += acc["top5"] * bsz
            n += bsz

    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "top1": correct1 / n,
        "top5": correct5 / n,
    }


@torch.no_grad()
def covariance_spectrum_top(
    feats: torch.Tensor,
    *,
    max_samples: int = 5000,
    q: int = 256,
    seed: int = 0,
) -> Dict[str, Any]:
    X = feats.float()
    N, D = X.shape

    if N > max_samples:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(N, generator=g)[:max_samples]
        X = X[idx]
        N = X.size(0)

    X = X - X.mean(dim=0, keepdim=True)

    q_eff = min(q, D, max(1, N - 1))
    _, S, _ = torch.pca_lowrank(X, q=q_eff, center=False)
    eigs = (S**2) / max(1, (N - 1))

    lam = eigs.clamp_min(1e-12)
    p = lam / lam.sum()
    entropy = -(p * torch.log(p)).sum()
    eff_rank = torch.exp(entropy).item()
    part_ratio = (lam.sum() ** 2 / (lam.square().sum().clamp_min(1e-12))).item()

    return {
        "n_used": N,
        "q_used": int(q_eff),
        "eigs_top": lam.cpu().tolist(),
        "eff_rank_topq": eff_rank,
        "participation_ratio_topq": part_ratio,
    }


@torch.no_grad()
def cosine_similarity_hist(
    feats: torch.Tensor,
    *,
    num_pairs: int = 20000,
    seed: int = 0,
) -> Dict[str, Any]:
    Z = F.normalize(feats.float(), dim=1)
    N = Z.size(0)

    g = torch.Generator(device="cpu").manual_seed(seed)
    i1 = torch.randint(0, N, (num_pairs,), generator=g)
    i2 = torch.randint(0, N, (num_pairs,), generator=g)

    cos = (Z[i1] * Z[i2]).sum(dim=1)
    return {"summary": summarize_1d(cos), "values": cos.cpu()}


@torch.no_grad()
def norm_hist(feats: torch.Tensor) -> Dict[str, Any]:
    norms = feats.float().norm(dim=1)
    return {"summary": summarize_1d(norms), "values": norms.cpu()}


@torch.no_grad()
def alignment_uniformity_two_view(
    encoder: nn.Module,
    two_view_loader: DataLoader,
    device: torch.device,
    transform_fn=None,
    *,
    max_batches: int = 50,
    seed: int = 0,
) -> Dict[str, Any]:
    encoder.eval()
    zs1: List[torch.Tensor] = []
    zs2: List[torch.Tensor] = []

    for i, batch in enumerate(two_view_loader):
        if i >= max_batches:
            break
        v1, v2, _ = batch
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        z1 = encoder(v1).float()
        z2 = encoder(v2).float()
        if transform_fn is not None:
            z1 = transform_fn(z1)
            z2 = transform_fn(z2)
        z1 = z1.cpu()
        z2 = z2.cpu()
        zs1.append(z1)
        zs2.append(z2)

    Z1 = torch.cat(zs1, 0)
    Z2 = torch.cat(zs2, 0)

    Z1n = F.normalize(Z1, dim=1)
    Z2n = F.normalize(Z2, dim=1)

    align = (Z1n - Z2n).pow(2).sum(dim=1).mean().item()

    N = Z1n.size(0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_pairs = min(20000, N * 2)
    i1 = torch.randint(0, N, (num_pairs,), generator=g)
    i2 = torch.randint(0, N, (num_pairs,), generator=g)
    dist2 = (Z1n[i1] - Z1n[i2]).pow(2).sum(dim=1)
    unif = torch.log(torch.exp(-2.0 * dist2).mean().clamp_min(1e-12)).item()

    return {"alignment": align, "uniformity": unif, "n_pairs": int(num_pairs), "n_used": int(N)}


def save_plots(
    out_dir: str,
    knn: Dict[str, Any],
    eig: Dict[str, Any],
    cos_vals: torch.Tensor,
    norm_vals: torch.Tensor,
    title_prefix: str,
    file_suffix: str = "",
) -> None:
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(out_dir, "plots")
    mkdirp(plot_dir)

    suffix = f"_{file_suffix}" if file_suffix else ""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"{title_prefix} - Summary", fontsize=14)

    ax = axes[0, 0]
    ax.plot(knn["k_values"], knn["top1"], marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("kNN top-1 accuracy")
    ax.set_title("kNN sweep")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    eigs = torch.tensor(eig["eigs_top"], dtype=torch.float32)
    ax.plot(torch.arange(1, eigs.numel() + 1), eigs)
    ax.set_xlabel("component rank")
    ax.set_ylabel("cov eigenvalue (top-q)")
    ax.set_title(f"Covariance eigenspectrum (top {eigs.numel()})")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(cos_vals.numpy(), bins=80)
    ax.set_xlabel("cosine similarity (random pairs)")
    ax.set_ylabel("count")
    ax.set_title("Cosine similarity histogram")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(norm_vals.numpy(), bins=80)
    ax.set_xlabel("feature norm")
    ax.set_ylabel("count")
    ax.set_title("Feature norm histogram")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(plot_dir, f"summary{suffix}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config used for training.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt/.pth).")
    ap.add_argument("--out", type=str, required=True, help="Output folder for metrics.json + plots/")
    ap.add_argument("--dataset", type=str, default="imagenet", help="imagenet or imagenet1k")
    ap.add_argument("--data_root", type=str, default="./data", help="Dataset root")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--eval_bs", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=224)

    ap.add_argument("--knn_ks", type=str, default="1,5,10,20,50,100,200")
    ap.add_argument("--knn_batch", type=int, default=256)

    ap.add_argument("--probe_epochs", type=int, default=50)
    ap.add_argument("--probe_bs", type=int, default=256)
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--probe_wd", type=float, default=1e-6)

    ap.add_argument("--diag_max_samples", type=int, default=5000)
    ap.add_argument("--diag_q", type=int, default=256)
    ap.add_argument("--diag_pairs", type=int, default=20000)

    ap.add_argument("--coord", type=str, choices=["raw", "ham", "both"], default="raw")
    ap.add_argument("--posthoc_h_config", type=str, default=None, help="Config for post-hoc H^{1/2} eval.")
    ap.add_argument("--posthoc_h_ckpt", type=str, default=None, help="Checkpoint providing learnable H params.")
    ap.add_argument("--posthoc_h_label", type=str, default=None, help="Label for post-hoc H in plots.")

    ap.add_argument("--do_align_uniform", action="store_true", help="Compute alignment/uniformity on 2-view aug.")
    ap.add_argument("--align_max_batches", type=int, default=50)

    args = ap.parse_args()
    seed_everything(args.seed)

    mkdirp(args.out)

    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    cfg_lower = cfg_name.lower()
    if "sigreg_tokens" in cfg_lower:
        title_prefix = "SIGReg baseline"
    elif "hjepa_mv" in cfg_lower:
        title_prefix = "MV-HJEPA"
    elif "hamsigreg_learnable" in cfg_lower:
        title_prefix = "Learnable H"
    elif "hamsigreg_chain" in cfg_lower:
        title_prefix = "Chain H"
    elif "hamsigreg_diag" in cfg_lower:
        title_prefix = "Diag H"
    else:
        title_prefix = cfg_name

    cfg = load_yaml_cfg(args.config)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = EncoderOnly(cfg).to(device)
    ckpt_info = load_checkpoint_into_encoder(model, args.ckpt)
    ckpt_obj = ckpt_info.pop("ckpt", {})

    train_loader, test_loader, num_classes = make_eval_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.eval_bs,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    model_cfg = cfg.get("model", {})
    embed_dim = int(model_cfg.get("embed_dim", 512))
    split_qp = bool(model_cfg.get("split_qp", False))
    if split_qp and (embed_dim % 2 != 0):
        raise ValueError(f"split_qp requires even embed_dim, got {embed_dim}")
    q_dim = embed_dim // 2 if split_qp else embed_dim

    # --- MV-HJEPA diagnostics: evaluate q-only, p-only, and concat(q,p) ---
    qp_variants = ["q", "p", "qp"] if split_qp else ["qp"]
    default_qp_variant = "q" if split_qp else "qp"

    def _slice_qp_variant(z: torch.Tensor, variant: str) -> torch.Tensor:
        """
        z: [B, D] where D=2*q_dim if split_qp else D=q_dim.
        Returns:
          - q:  z[:, :q_dim]
          - p:  z[:, q_dim:]
          - qp: z  (concat(q,p))
        """
        if not split_qp:
            return z
        if variant == "q":
            return z[:, :q_dim]
        if variant == "p":
            return z[:, q_dim:]
        if variant in ("qp", "concat"):
            return z
        raise ValueError(f"Unknown qp_variant='{variant}'. Expected one of {qp_variants}.")

    def _compose_transform(base_fn, variant: str):
        # base_fn: optional transform (e.g., H^{1/2}). We always slice AFTER base_fn.
        if base_fn is None:
            base_fn = lambda zz: zz
        return lambda zz: _slice_qp_variant(base_fn(zz), variant)

    ham_transform = None
    ham_transform_meta = None
    reg_cfg = cfg.get("regularizer", {})
    reg_type = str(reg_cfg.get("type", "sigreg")).lower()
    if reg_type == "ham_sigreg":
        reg_module = build_ham_module_from_cfg(cfg, d=embed_dim, device=device)
        if reg_module.kind == "learnable":
            reg_state = ckpt_obj.get("regularizer", None)
            if reg_state is None:
                raise RuntimeError("Checkpoint missing regularizer state required for learnable H eval.")
            reg_module.load_state_dict(reg_state, strict=False)
        reg_module.eval()
        ham_transform = lambda z: reg_module.transform(z)
        ham_transform_meta = {"type": "ham", "source": "self", "ham_kind": reg_module.kind}

    posthoc_transform = None
    posthoc_meta = None
    posthoc_label = None
    if args.posthoc_h_config is not None:
        posthoc_cfg = load_yaml_cfg(args.posthoc_h_config)
        posthoc_model_cfg = posthoc_cfg.get("model", {})
        posthoc_embed_dim = int(posthoc_model_cfg.get("embed_dim", embed_dim))
        if posthoc_embed_dim != embed_dim:
            raise ValueError(
                f"posthoc_h_config embed_dim={posthoc_embed_dim} does not match eval embed_dim={embed_dim}"
            )
        posthoc_reg = build_ham_module_from_cfg(posthoc_cfg, d=embed_dim, device=device)
        if posthoc_reg.kind == "learnable":
            if args.posthoc_h_ckpt is None:
                raise RuntimeError("posthoc_h_ckpt required when posthoc ham_kind=learnable.")
            posthoc_ckpt = torch.load(args.posthoc_h_ckpt, map_location="cpu")
            reg_state = posthoc_ckpt.get("regularizer", None)
            if reg_state is None:
                raise RuntimeError("posthoc_h_ckpt missing regularizer state.")
            posthoc_reg.load_state_dict(reg_state, strict=False)
        posthoc_reg.eval()
        posthoc_transform = lambda z: posthoc_reg.transform(z)
        posthoc_label = args.posthoc_h_label or os.path.splitext(os.path.basename(args.posthoc_h_config))[0]
        posthoc_meta = {
            "type": "ham",
            "source": "posthoc",
            "config": args.posthoc_h_config,
            "ckpt": args.posthoc_h_ckpt,
            "ham_kind": posthoc_reg.kind,
        }

    def run_eval(coord_label: str, transform_fn, title_suffix: str, file_suffix: str, *, qp_variant: str):
        t0 = time.time()
        train_feats, train_labels = extract_features(model, train_loader, device, transform_fn=transform_fn)
        test_feats, test_labels = extract_features(model, test_loader, device, transform_fn=transform_fn)
        feat_dim = int(train_feats.size(1))
        t_extract = time.time() - t0

        ks = [int(x) for x in args.knn_ks.split(",") if x.strip()]
        knn = knn_sweep(
            train_feats=train_feats,
            train_labels=train_labels,
            test_feats=test_feats,
            test_labels=test_labels,
            ks=ks,
            num_classes=num_classes,
            device=device,
            batch_size=args.knn_batch,
        )

        probe = run_linear_probe(
            train_feats=train_feats,
            train_labels=train_labels,
            test_feats=test_feats,
            test_labels=test_labels,
            num_classes=num_classes,
            device=device,
            epochs=args.probe_epochs,
            batch_size=args.probe_bs,
            lr=args.probe_lr,
            weight_decay=args.probe_wd,
        )

        eig = covariance_spectrum_top(
            train_feats,
            max_samples=args.diag_max_samples,
            q=args.diag_q,
            seed=args.seed,
        )
        cos = cosine_similarity_hist(test_feats, num_pairs=args.diag_pairs, seed=args.seed)
        norms = norm_hist(test_feats)

        align_unif = None
        if args.do_align_uniform:
            two_view_loader = make_two_view_loader(
                dataset=args.dataset,
                data_root=args.data_root,
                batch_size=args.eval_bs,
                num_workers=args.num_workers,
                image_size=args.image_size,
                max_items=args.diag_max_samples,
            )
            align_unif = alignment_uniformity_two_view(
                model, two_view_loader, device, transform_fn=transform_fn, max_batches=args.align_max_batches, seed=args.seed
            )

        save_plots(
            args.out,
            knn=knn,
            eig=eig,
            cos_vals=cos["values"],
            norm_vals=norms["values"],
            title_prefix=f"{title_prefix} {title_suffix}".strip(),
            file_suffix=file_suffix,
        )

        return {
            "coord": coord_label,
            "qp_variant": qp_variant,
            "feature_dim": feat_dim,
            "timing": {"feature_extract_sec": t_extract},
            "knn": knn,
            "linear_probe": probe,
            "diagnostics": {
                "covariance_spectrum_topq": {
                    "n_used": eig["n_used"],
                    "q_used": eig["q_used"],
                    "eff_rank_topq": eig["eff_rank_topq"],
                    "participation_ratio_topq": eig["participation_ratio_topq"],
                    "eigs_top": eig["eigs_top"],
                },
                "cosine_similarity": cos["summary"],
                "feature_norms": norms["summary"],
            },
            "alignment_uniformity": align_unif,
        }

    # Plan which coordinate(s) will run to avoid plot filename collisions.
    planned_coords: List[str] = []
    if args.coord in ("raw", "both"):
        planned_coords.append("raw")
    if args.coord in ("ham", "both"):
        if ham_transform is not None:
            planned_coords.append("ham")
        if posthoc_transform is not None:
            planned_coords.append("posthoc")
    multi_coord = len(planned_coords) > 1

    def _base_file_suffix(coord_key: str) -> str:
        # Preserve old naming when safe; otherwise prefix by coord_key to avoid overwrites.
        if args.coord == "both" or multi_coord:
            return coord_key
        return ""

    coord_results: Dict[str, Any] = {}
    coord_qp_variants: Dict[str, Dict[str, Any]] = {}

    def _eval_coord(
        coord_key: str,
        base_transform,
        title_tag: str,
        *,
        transform_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        base_suffix = _base_file_suffix(coord_key)
        v_results: Dict[str, Any] = {}

        for v in qp_variants:
            tfn = _compose_transform(base_transform, v)
            title_suffix = f"{title_tag} | {v}" if split_qp else title_tag

            # File naming:
            # - default variant keeps legacy name (summary.png or summary_raw.png, etc.)
            # - other variants append "_p" / "_qp" (or just "p"/"qp" if base suffix is empty)
            if v == default_qp_variant:
                fs = base_suffix
            else:
                fs = f"{base_suffix}_{v}" if base_suffix else v

            res = run_eval(coord_key, tfn, title_suffix, fs, qp_variant=v)
            if transform_meta is not None:
                res["transform"] = transform_meta
            v_results[v] = res

        coord_qp_variants[coord_key] = v_results
        # Keep backward-compatible "default" result per coord.
        coord_results[coord_key] = v_results[default_qp_variant]

    if args.coord in ("raw", "both"):
        _eval_coord("raw", None, "(raw)")

    if args.coord in ("ham", "both"):
        if ham_transform is None and posthoc_transform is None:
            raise RuntimeError("No H transform available. Provide --posthoc_h_config for SIGReg baseline.")
        if ham_transform is not None:
            _eval_coord("ham", ham_transform, "(H^1/2)", transform_meta=ham_transform_meta)
        if posthoc_transform is not None:
            label = f"(H^1/2 {posthoc_label})" if posthoc_label else "(H^1/2 posthoc)"
            _eval_coord("posthoc", posthoc_transform, label, transform_meta=posthoc_meta)

    metrics = {
        "checkpoint": args.ckpt,
        "config": args.config,
        "dataset": args.dataset,
        "data_root": args.data_root,
        "num_classes": num_classes,
        "ckpt_load_info": ckpt_info,
        "coord_mode": args.coord,
        "split_qp": split_qp,
    }
    # Backward-compatible default output:
    if len(coord_results) == 1:
        metrics.update(next(iter(coord_results.values())))
    else:
        metrics["coords"] = coord_results

    # Always attach the per-variant results when split_qp=True.
    if split_qp:
        metrics["qp_eval"] = {
            "variants": qp_variants,
            "default_variant": default_qp_variant,
            "q_dim": q_dim,
            "p_dim": q_dim,
            "qp_dim": embed_dim,
        }
        if len(coord_qp_variants) == 1:
            # Convenience when only a single coordinate was evaluated (typical MV-HJEPA case).
            metrics["qp_variants"] = next(iter(coord_qp_variants.values()))
        else:
            metrics["coords_qp_variants"] = coord_qp_variants
    json_dump(metrics, os.path.join(args.out, "metrics.json"))

    print(f"[eval_suite] wrote: {args.out}/metrics.json and {args.out}/plots/*.png")


if __name__ == "__main__":
    main()
