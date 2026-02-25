#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."  # repo root

python scripts/train_imagenet_hamjepa.py --config configs/imagenet_sigreg_tokens.yaml "$@"
