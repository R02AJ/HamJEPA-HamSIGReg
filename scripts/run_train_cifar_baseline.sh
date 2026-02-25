#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."  # repo root

python scripts/train_cifar_hamjepa.py --config configs/cifar100_sigreg_tokens.yaml "$@"
