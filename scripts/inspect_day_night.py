#!/usr/bin/env python
"""
주간(frame_*) / 야간(nframe_*) 샘플의 RGB / 열화상 / 이벤트 채널을 나란히 그려서 눈으로 확인.

Usage:
    python scripts/inspect_day_night.py                # 주간 3, 야간 3
    python scripts/inspect_day_night.py --n 4 --seed 1
출력: inspect_out/day_night_grid.png, inspect_out/night_rgb_sample.png
"""
import argparse
import os
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(ROOT, "data", "data", "images")
LBL_DIR = os.path.join(ROOT, "data", "data", "labels")
OUT_DIR = os.path.join(ROOT, "inspect_out")


def load_boxes(stem, h, w):
    """YOLO 라벨 -> 픽셀 (x1, y1, bw, bh) 리스트."""
    path = os.path.join(LBL_DIR, stem + ".txt")
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f.read().split("\n"):
            p = line.split()
            if len(p) != 5:
                continue
            _, xc, yc, bw, bh = map(float, p)
            boxes.append(((xc - bw / 2) * w, (yc - bh / 2) * h, bw * w, bh * h))
    return boxes


def pick(prefix, n, rng):
    stems = sorted(f[:-4] for f in os.listdir(IMG_DIR)
                   if f.startswith(prefix) and f.endswith(".npy"))
    stems = [s for s in stems if os.path.exists(os.path.join(LBL_DIR, s + ".txt"))
             and os.path.getsize(os.path.join(LBL_DIR, s + ".txt")) > 0]
    return rng.sample(stems, n)


def enhance(rgb):
    """야간 RGB 밝기 보정: LAB의 L 채널에 CLAHE."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def stretch(ch):
    """1~99 퍼센타일로 대비 늘리기 (이벤트 채널 배경이 ~37이라 그냥 보면 흐림)."""
    lo, hi = np.percentile(ch, 1), np.percentile(ch, 99.5)
    return np.clip((ch.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def draw_boxes(ax, boxes):
    for x, y, bw, bh in boxes:
        ax.add_patch(patches.Rectangle((x, y), bw, bh, fill=False, ec="lime", lw=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    samples = [("Day", s) for s in pick("frame_", args.n, rng)] + \
              [("Night", s) for s in pick("nframe_", args.n, rng)]

    cols = ["RGB", "RGB enhanced (CLAHE)", "RGB + GT", "Thermal (ch3)", "Event (ch4)"]
    fig, axes = plt.subplots(len(samples), len(cols), figsize=(4 * len(cols), 3.2 * len(samples)))

    for r, (tag, stem) in enumerate(samples):
        x = np.load(os.path.join(IMG_DIR, stem + ".npy"))
        h, w = x.shape[:2]
        rgb, th, ev = x[:, :, :3], x[:, :, 3], x[:, :, 4]
        boxes = load_boxes(stem, h, w)

        panels = [
            (rgb, None),
            (enhance(rgb), None),
            (rgb if tag == "Day" else enhance(rgb), None),
            (th, "inferno"),
            (stretch(ev), "gray"),
        ]
        for c, (img, cmap) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=(1 if c == 4 else 255) if cmap else None)
            if c >= 2:
                draw_boxes(ax, boxes)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
        axes[r, 0].set_ylabel(f"{tag}\n{stem}\nRGB mean={rgb.mean():.0f}, boxes={len(boxes)}", fontsize=9)

    fig.tight_layout()
    grid_path = os.path.join(OUT_DIR, "day_night_grid.png")
    fig.savefig(grid_path, dpi=110)
    plt.close(fig)

    # 야간 RGB 샘플 캡처: 원본 vs 보정 vs 열화상
    stem = samples[args.n][1]
    x = np.load(os.path.join(IMG_DIR, stem + ".npy"))
    boxes = load_boxes(stem, *x.shape[:2])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, img, cmap, title in [
        (axes[0], x[:, :, :3], None, f"Night RGB (raw, mean={x[:, :, :3].mean():.0f})"),
        (axes[1], enhance(x[:, :, :3]), None, "Night RGB (CLAHE) + GT"),
        (axes[2], x[:, :, 3], "inferno", "Thermal + GT"),
    ]:
        ax.imshow(img, cmap=cmap)
        if ax is not axes[0]:
            draw_boxes(ax, boxes)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(stem)
    fig.tight_layout()
    night_path = os.path.join(OUT_DIR, "night_rgb_sample.png")
    fig.savefig(night_path, dpi=120)
    plt.close(fig)

    print("saved:", grid_path)
    print("saved:", night_path)
    print("samples:", [s for _, s in samples])


if __name__ == "__main__":
    main()
