#!/usr/bin/env python
"""
시간 동기 확인용 겹쳐보기. 빨강 = 열화상(또는 이벤트) 프레임 t의 엣지, 초록 = RGB 프레임 t+k의 엣지.
두 엣지가 겹치면 노랗게 보임 -> 노란색이 많은 k가 실제로 맞는 시점.

Usage:
    python scripts/sync_overlay.py frame_03283 frame_01475
출력: inspect_out/sync_overlay.png
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from inspect_day_night import IMG_DIR, OUT_DIR  # noqa: E402
from check_sync import gray, edges  # noqa: E402


def to01(e):
    return np.clip((e - 0.5) / 2.5, 0, 1)  # 정규화된 엣지에서 강한 부분만


def main():
    stems = sys.argv[1:] or ["frame_03283", "frame_01475"]
    cols = [("thermal", 0), ("thermal", 3), ("event", 0), ("event", 1)]
    fig, axes = plt.subplots(len(stems), len(cols), figsize=(4.2 * len(cols), 3.4 * len(stems)), squeeze=False)
    for r, stem in enumerate(stems):
        prefix, t = re.match(r"(n?frame_)(\d+)", stem).groups()
        t = int(t)
        night = prefix == "nframe_"
        load = lambda d: np.load(os.path.join(IMG_DIR, f"{prefix}{t + d:05d}.npy"))
        x = load(0)
        e_th = to01(edges(x[:, :, 3]))
        e_ev = to01(edges(x[:, :, 4]))
        for c, (mod, k) in enumerate(cols):
            e_rgb = to01(edges(gray(load(k), night)))
            img = np.stack([e_th if mod == "thermal" else e_ev, e_rgb, np.zeros_like(e_rgb)], -1)
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{stem}: {mod}_t (red) vs RGB_t{k:+d} (green)", fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "sync_overlay.png")
    fig.savefig(path, dpi=100)
    print("saved:", path)


if __name__ == "__main__":
    main()
