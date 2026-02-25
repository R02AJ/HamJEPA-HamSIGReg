#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."  # repo root

CFG="configs/imagenet_hjepa_mv.yaml"
CKPT="checkpoints/imagenet_hjepa_mv.pth"

if [[ $# -gt 0 && "$1" != --* ]]; then
  CKPT=$1
  shift
fi

OUT_DIR=""
if [[ $# -gt 0 && "$1" != --* ]]; then
  OUT_DIR=$1
  shift
fi

HAS_OUT=0
HAS_COORD=0
HAS_DATASET=0
for arg in "$@"; do
  if [ "$arg" = "--out" ]; then
    HAS_OUT=1
  fi
  if [ "$arg" = "--coord" ]; then
    HAS_COORD=1
  fi
  if [ "$arg" = "--dataset" ]; then
    HAS_DATASET=1
  fi
done

if [ -z "$OUT_DIR" ] && [ $HAS_OUT -eq 0 ]; then
  base=$(basename "$CKPT")
  name="${base%.*}"
  OUT_DIR="eval_runs/$name"
fi

CMD=(python scripts/eval_suite_imagenet.py --ckpt "$CKPT" --config "$CFG")
if [ -n "$OUT_DIR" ] && [ $HAS_OUT -eq 0 ]; then
  CMD+=(--out "$OUT_DIR")
fi
if [ $HAS_COORD -eq 0 ]; then
  CMD+=(--coord both)
fi
if [ $HAS_DATASET -eq 0 ]; then
  CMD+=(--dataset imagenet --image_size 224)
fi
CMD+=("$@")
"${CMD[@]}"
