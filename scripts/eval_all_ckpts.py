#!/usr/bin/env python3
import argparse
import glob
import os
import subprocess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--do_align_uniform", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ckpts = sorted(
        glob.glob(os.path.join(args.ckpt_dir, "*.pt"))
        + glob.glob(os.path.join(args.ckpt_dir, "*.pth"))
    )
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {args.ckpt_dir}")

    for ckpt in ckpts:
        name = os.path.splitext(os.path.basename(ckpt))[0]
        out = os.path.join(args.out_dir, name)
        cmd = [
            "python",
            "scripts/eval_suite.py",
            "--config",
            args.config,
            "--ckpt",
            ckpt,
            "--out",
            out,
            "--dataset",
            args.dataset,
            "--data_root",
            args.data_root,
            "--device",
            args.device,
        ]
        if args.do_align_uniform:
            cmd.append("--do_align_uniform")

        print("[eval_all_ckpts] running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
