#!/usr/bin/env python
"""
야간(nframe_*) 프레임의 열화상 위에 GT 박스를 그려 라벨 누락을 눈으로 확인하기 위한 시트.

Usage:
    python scripts/night_label_sheet.py --n 24 --seed 0 --prefix nframe_
출력: inspect_out/label_sheet_<prefix>_<k>.png (시트당 12장)
"""
import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from inspect_day_night import IMG_DIR, LBL_DIR, OUT_DIR, load_boxes, draw_boxes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="nframe_")
    ap.add_argument("--per-sheet", type=int, default=12)
    args = ap.parse_args()

    stems = sorted(f[:-4] for f in os.listdir(IMG_DIR)
                   if f.startswith(args.prefix) and f.endswith(".npy"))
    stems = [s for s in stems if os.path.exists(os.path.join(LBL_DIR, s + ".txt"))]
    picked = sorted(random.Random(args.seed).sample(stems, args.n))
    os.makedirs(OUT_DIR, exist_ok=True)

    for k in range(0, len(picked), args.per_sheet):
        chunk = picked[k:k + args.per_sheet]
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, stem in zip(axes.ravel(), chunk):
            x = np.load(os.path.join(IMG_DIR, stem + ".npy"))
            boxes = load_boxes(stem, *x.shape[:2])
            ax.imshow(x[:, :, 3], cmap="inferno")
            draw_boxes(ax, boxes)
            ax.set_title(f"{stem}  GT={len(boxes)}", fontsize=12)
        fig.tight_layout()
        path = os.path.join(OUT_DIR, f"label_sheet_{args.prefix.rstrip('_')}_{k // args.per_sheet}.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        print("saved:", path)


if __name__ == "__main__":
    main()
