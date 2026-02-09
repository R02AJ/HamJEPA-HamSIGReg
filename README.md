# HJEPA / SIGReg (JEPA-ML)

> IMPORTANT LICENSE NOTICE (LeJEPA upstream)
>
> This codebase contains code adapted from LeJEPA (and/or code that is a derivative work of LeJEPA).
> The upstream LeJEPA license is Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
>
> Therefore, any use of that code in this repository - namely the univariate and multivariate folders and by extenstion sigreg.py - must comply with CC BY-NC 4.0 (attribution, non-commercial use, etc.). An alternative to sigreg.py is provided in sigreg_wrapper.py, however that file was not used in the paper.
>
> - License text: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
> - https://creativecommons.org/licenses/by-nc/4.0/
>

## What this repo is

This repo is a compact research codebase for training and analyzing:

- SIGReg baselines (representation regularization / covariance objectives).
- HJEPA (Hamiltonian JEPA): a JEPA-style predictor augmented with Hamiltonian dynamics
  (symplectic integration plus a learnable potential) to encourage structured latent evolution.

Primary targets used in practice:

- CIFAR-100 (fast iteration)
- ImageNet-100 (larger scale; subject to dataset license restrictions)

The project supports:

- Univariate and multivariate coordinate modes (for example, matching only `q`, only `p`, or concatenated `(q,p)` states)
- Multi-view training (global views, optional local views / multi-crop)

## Repository layout (key files)

Core scripts and modules:

- `scripts/train_cifar_hamjepa.py` - CIFAR-100 training loop.
- `scripts/train_imagenet_hamjepa.py` - ImageNet-style training loop.
- `eval/models/encoder_resnet.py` - ResNet encoder backbone.
- `hamjepa/projector.py` - Projector MLP(s).
- `hamjepa/predictor.py` - Predictors, including Hamiltonian predictor.
- `hamjepa/hamiltonian.py` - Hamiltonian model(s), separable Hamiltonians, potential networks.
- `hamjepa/integrators.py` - Symplectic integrators (leapfrog, etc.).
- `eval/datasets/imagenet_multicrop.py` - ImageNet multi-crop dataset wrapper.

Configs:

- `configs/cifar100_hjepa_mv.yaml`
- `configs/cifar100_sigreg_tokens.yaml`
- `configs/imagenet_hjepa_mv.yaml`
- `configs/imagenet_sigreg_tokens.yaml`

Artifacts:

- `eval_runs/.../metrics.json` - evaluation outputs and diagnostics.

## Setup

### 1) Environment

Recommended:

- Python 3.10+
- PyTorch and torchvision matched to your CUDA stack

Example install:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pyyaml tqdm
```

### 2) Reproducibility / speed knobs

You can control:

- `seed` in YAML configs
- deterministic vs speed backend settings

For speed on Ampere/Hopper class GPUs:

- `torch.backends.cudnn.benchmark = True`
- TF32 enabled for matmul/conv where appropriate

## Data

### CIFAR-100

Uses torchvision CIFAR-100. No manual download needed.

### ImageNet-100

Expected folder layout:

```text
/path/to/imagenet100/
  train/
    class_001/
      img1.jpg
      img2.jpg
      ...
    class_002/
      ...
  val/
    class_001/
      ...
```

Do not redistribute ImageNet data (including ImageNet-100 subsets) unless explicitly permitted by dataset terms.


## Configuration guide (YAML)

### Data / multi-crop

Common knobs:

- `data.batch_size`, `data.num_workers`
- `data.num_global_views`, `data.num_local_views`
- crop size/scale fields in the dataset config used by each train script

### HJEPA knobs

Important dynamics knobs:

- `dt`, `steps` - integrator step size and number of steps.
- `residual_scale` - scales residual potential term. Too high can create stiff-potential behavior.
- `base_coeff` - base quadratic energy coefficient.
- `damping` (if enabled) - momentum damping.

Matching choice:

- `loss.match: q` - supervise only `q`.
- `loss.match: p` - supervise only `p`.
- `loss.match: qp` - supervise full state `(q,p)`.

In practice, choose based on stability and your objective:

- `q` is often easier to stabilize early.
- `qp` is stricter and can improve full-state consistency if training is stable.

## Interpreting logs

Typical fields:

- `pred` - predictor loss.
- `budget` - budget/constraint regularizer term.
- `logdet` - projected covariance spread regularizer.
- `[MV] ... q_pr / p_pr ... V_var ...` - multivariate diagnostics.

Useful diagnostics:

- `q_pr`, `p_pr` - participation-ratio style effective-rank proxies (higher usually means less collapse).
- `q_std_min` - minimum projected std; very low values can indicate collapse.
- `q2`, `p2` - mean squared norms; near-zero or runaway values indicate instability.
- `V_var` - batch variance of potential values; large sustained growth can indicate stiffness.

## Notes on caching and throughput

RAM-caching decoded PIL images is often memory-heavy. If you cache, prefer compressed bytes and decode on access.

On fast GPUs, throughput is commonly limited by CPU-side multi-crop augmentation. If needed, tune DataLoader workers/prefetch and profile data-time vs step-time directly.

## Disclaimer

This is research code and may have rough edges. If training becomes unstable, check:

- data pipeline throughput
- augmentation intensity
- mixed precision / backend settings
- HJEPA dynamics knobs (`residual_scale`, `base_coeff`, `dt`, `steps`, `damping`)
- regularizer strength / floor settings
