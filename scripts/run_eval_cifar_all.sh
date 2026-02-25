#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."  # repo root

if [[ $# -gt 0 ]]; then
  BASE_ARGS=("$@")
else
  BASE_ARGS=(--knn_ks 1,5,10,20,50,100,200 --coord both)
fi

for pair in \
  "checkpoints/cifar100_sigreg_tokens.pth configs/cifar100_sigreg_tokens.yaml"
do
  set -- $pair
  ckpt=$1
  cfg=$2
  out_dir="eval_runs/$(basename "$ckpt" .pth)"
  echo "==> Evaluating $ckpt with $cfg -> $out_dir"
  extra_args=("${BASE_ARGS[@]}")
  if [[ "$cfg" == *"hjepa_mv"* ]]; then
    filtered=()
    skip_next=0
    for arg in "${extra_args[@]}"; do
      if [[ $skip_next -eq 1 ]]; then
        skip_next=0
        continue
      fi
      if [[ "$arg" == "--coord" ]]; then
        skip_next=1
        continue
      fi
      filtered+=("$arg")
    done
    extra_args=("${filtered[@]}")
    extra_args+=("--coord" "raw")
  elif [[ "$cfg" == *"sigreg_tokens"* ]]; then
    # SIGReg baseline: raw-only, no posthoc H transform.
    filtered=()
    skip_next=0
    for arg in "${extra_args[@]}"; do
      if [[ $skip_next -eq 1 ]]; then
        skip_next=0
        continue
      fi
      if [[ "$arg" == "--coord" ]]; then
        skip_next=1
        continue
      fi
      filtered+=("$arg")
    done
    extra_args=("${filtered[@]}")
    extra_args+=("--coord" "raw")
  fi
  python scripts/eval_suite.py --ckpt "$ckpt" --config "$cfg" --out "$out_dir" "${extra_args[@]}"
done
