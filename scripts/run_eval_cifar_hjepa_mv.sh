#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."  # repo root

CFG="configs/cifar100_hjepa_mv.yaml"
CKPT="checkpoints/cifar100_hjepa_mv.pth"

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
for arg in "$@"; do
  if [ "$arg" = "--out" ]; then
    HAS_OUT=1
    break
  fi
done

if [ -z "$OUT_DIR" ] && [ $HAS_OUT -eq 0 ]; then
  base=$(basename "$CKPT")
  name="${base%.*}"
  OUT_DIR="eval_runs/$name"
fi

FILTERED=()
SKIP_NEXT=0
for arg in "$@"; do
  if [[ $SKIP_NEXT -eq 1 ]]; then
    SKIP_NEXT=0
    continue
  fi
  if [[ "$arg" == "--coord" ]]; then
    SKIP_NEXT=1
    continue
  fi
  FILTERED+=("$arg")
done
FILTERED+=("--coord" "raw")

CMD=(python scripts/eval_suite.py --ckpt "$CKPT" --config "$CFG")
if [ -n "$OUT_DIR" ] && [ $HAS_OUT -eq 0 ]; then
  CMD+=(--out "$OUT_DIR")
fi
CMD+=("${FILTERED[@]}")
"${CMD[@]}"
