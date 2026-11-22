#!/usr/bin/env python
"""
RGB / 열화상 / 이벤트 사이의 (1) 공간 정렬 오차와 (2) 시간 동기(프레임 lag)를 측정.

아이디어
- 공간: 같은 프레임 t에서 RGB 엣지맵과 열화상(또는 이벤트) 엣지맵을 phase correlation -> (dx, dy) 픽셀 오프셋.
  오프셋이 프레임마다 거의 일정하면 '캘리브레이션(정합) 오차', 드론 움직임에 따라 커지면 '시간 동기 오차' 쪽.
- 시간: 열화상_t 를 RGB_{t+k} (k=-3..3) 와 비교해서 가장 잘 맞는 k를 찾음. k != 0 이면 프레임 단위로 어긋난 것.
  이벤트_t 는 RGB 프레임 차이 |gray_{t+k} - gray_{t+k-1}| 와 비교.
  (드론이 움직이는 구간에서만 의미 있음 -> ego-motion 이 작은 프레임은 lag 판정에서 제외)

Usage:
    python scripts/check_sync.py --n 60 --seed 0
출력: inspect_out/sync_metrics.csv, 콘솔 요약
"""
import argparse
import csv
import os
import random
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from inspect_day_night import IMG_DIR, OUT_DIR  # noqa: E402

LAGS = list(range(-3, 4))
MIN_EGO = 1.0  # 이 픽셀 이상 움직인 프레임만 lag 판정에 사용


def gray(x, night):
    g = cv2.cvtColor(x[:, :, :3], cv2.COLOR_RGB2GRAY)
    if night:
        g = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(g)
    return g.astype(np.float32)


def edges(img):
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return norm(cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 2.0))


def norm(a):
    a = a.astype(np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def ncc(a, b):
    return float((norm(a) * norm(b)).mean())


def phase(a, b):
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(norm(a), norm(b), win)
    return dx, dy, resp


def build_sequences():
    """접두사별로 연속 번호 구간 찾기 -> {stem: (prefix, num)}"""
    byp = {}
    for f in os.listdir(IMG_DIR):
        m = re.match(r"(n?frame_)(\d+)\.npy$", f)
        if m:
            byp.setdefault(m.group(1), set()).add(int(m.group(2)))
    return byp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="주간/야간 각각 샘플 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maxlag", type=int, default=3)
    args = ap.parse_args()
    global LAGS
    LAGS = list(range(-args.maxlag, args.maxlag + 1))
    L = args.maxlag
    rng = random.Random(args.seed)
    byp = build_sequences()

    cache = {}

    def load(prefix, num):
        key = (prefix, num)
        if key not in cache:
            cache[key] = np.load(os.path.join(IMG_DIR, f"{prefix}{num:05d}.npy"))
        return cache[key]

    rows = []
    for prefix, tag in [("frame_", "day"), ("nframe_", "night")]:
        nums = byp[prefix]
        # t-4 .. t+3 이 모두 있는 anchor 만
        anchors = sorted(t for t in nums if all((t + d) in nums for d in range(-L - 1, L + 1)))
        for t in rng.sample(anchors, min(args.n, len(anchors))):
            night = tag == "night"
            x = load(prefix, t)
            g = {d: gray(load(prefix, t + d), night) for d in range(-L - 1, L + 1)}
            th, ev = x[:, :, 3].astype(np.float32), x[:, :, 4].astype(np.float32)
            e_rgb, e_th = edges(g[0]), edges(th)
            ev_s = norm(cv2.GaussianBlur(ev, (0, 0), 2.0))

            # 공간 오프셋 (같은 프레임)
            tdx, tdy, tresp = phase(e_rgb, e_th)
            edx, edy, eresp = phase(e_rgb, ev_s)
            # 드론 ego-motion (RGB t -> t+1)
            mdx, mdy, _ = phase(g[0], g[1])
            ego = float(np.hypot(mdx, mdy))

            th_scores = [ncc(e_th, edges(g[k])) for k in LAGS]
            diff = {k: np.abs(g[k] - g[k - 1]) for k in LAGS}
            ev_scores = [ncc(ev_s, cv2.GaussianBlur(diff[k], (0, 0), 2.0)) for k in LAGS]

            rows.append(dict(
                tag=tag, frame=f"{prefix}{t:05d}", ego_px=round(ego, 2),
                th_dx=round(tdx, 2), th_dy=round(tdy, 2), th_resp=round(tresp, 3),
                ev_dx=round(edx, 2), ev_dy=round(edy, 2), ev_resp=round(eresp, 3),
                th_best_lag=LAGS[int(np.argmax(th_scores))],
                ev_best_lag=LAGS[int(np.argmax(ev_scores))],
                **{f"th_ncc_{k}": round(s, 4) for k, s in zip(LAGS, th_scores)},
                **{f"ev_ncc_{k}": round(s, 4) for k, s in zip(LAGS, ev_scores)},
            ))
            # 캐시 비우기 (메모리)
            if len(cache) > 128:
                cache.clear()

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "sync_metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("saved:", path)

    for tag in ["day", "night"]:
        R = [r for r in rows if r["tag"] == tag]
        mv = [r for r in R if r["ego_px"] >= MIN_EGO]
        print(f"\n===== {tag}: n={len(R)}, moving(ego>={MIN_EGO}px)={len(mv)} =====")
        ego = np.array([r["ego_px"] for r in R])
        print(f"ego-motion px/frame: median {np.median(ego):.2f}, p90 {np.percentile(ego, 90):.2f}")
        for m in ["th", "ev"]:
            dx = np.array([r[f"{m}_dx"] for r in R]); dy = np.array([r[f"{m}_dy"] for r in R])
            resp = np.array([r[f"{m}_resp"] for r in R])
            off = np.hypot(dx, dy)
            print(f"[{m} vs RGB] offset |d| median {np.median(off):.2f}px p90 {np.percentile(off, 90):.2f}px "
                  f"| dx median {np.median(dx):+.2f} dy median {np.median(dy):+.2f} | resp median {np.median(resp):.3f}")
            if len(R) > 3:
                c = np.corrcoef(ego, off)[0, 1]
                print(f"    corr(ego-motion, offset) = {c:+.2f}  (+ 크면 시간동기 문제 쪽)")
            if mv:
                lags = [r[f"{m}_best_lag"] for r in mv]
                hist = {k: lags.count(k) for k in LAGS}
                mean_curve = np.mean([[r[f"{m}_ncc_{k}"] for k in LAGS] for r in mv], axis=0)
                print(f"    best-lag hist (moving only): {hist}")
                print("    mean NCC by lag: " + ", ".join(f"{k:+d}:{s:.3f}" for k, s in zip(LAGS, mean_curve)))


if __name__ == "__main__":
    main()
