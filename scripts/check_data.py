#!/usr/bin/env python
"""
데이터 검사 (11/28 추가). GPU 불필요, 몇 분이면 끝난다. 학습/모델과 무관하게 데이터만 본다.

  1) 파일 형식 : 낮(frame_) / 밤(nframe_) 별 이미지 크기, dtype, 채널별 값 범위
                -> 낮/밤 파일 형식이 다르면 '평가 버그' 후보
  2) 라벨      : 클래스 번호, 이미지당 박스 수, 박스 크기 (낮/밤 별)
  3) 밝기      : 이미지별 RGB 평균 분포 -> 낮/밤 이름표가 맞는지 (섞인 사진이 있는지)
  4) 차 vs 주변 : 박스 안 평균 - 박스 바로 바깥 테두리 평균 (낮/밤 별)
                - 열화상 : 차가 주변보다 밝은 박스 비율 -> 낮/밤에 뒤집히는지(반전)
                - 이벤트 : |값 - 이미지 중앙값| 을 '이벤트 활동량'으로 보고 박스 안/바깥 비교
                          -> 이벤트에 차 정보가 있는지 (회의 3번, 데이터 관점)
  5) 분할 누수 : 원본 코드 분할(seed 42)에서 test 이미지의 번호 ±1 이웃이 train 에 있는 비율 (93.9% 재계산)
                + 번호 이웃 두 장 vs 무작위 두 장의 RGB 차이 -> 이웃이 정말 거의 같은 장면인지 (회의 1번)

사용: python scripts/check_data.py --data <images/ 와 labels/ 가 있는 폴더>
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np

PREFIXES = ['frame_', 'nframe_']          # frame_ = 낮(밝음), nframe_ = 밤(어두움)
NAMES = {'frame_': 'day(frame_)', 'nframe_': 'night(nframe_)'}


def log(msg=''):
    print(msg, flush=True)


def prefix_of(f):
    for p in ['nframe_', 'frame_']:
        if f.startswith(p):
            return p
    return 'other'


def load_hwc(path):
    a = np.load(path)
    if a.ndim == 3 and a.shape[-1] != 5 and a.shape[0] == 5:
        a = np.transpose(a, (1, 2, 0))
    return a


def read_labels(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh.read().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                out.append((int(float(parts[0])), *[float(x) for x in parts[1:5]]))
            except ValueError:
                continue
    return out


def neighbor_name(f, delta):
    m = re.search(r'(\d+)(?=\.npy$)', f)
    if not m:
        return None
    n = int(m.group(1)) + delta
    if n < 0:
        return None
    return f[:m.start(1)] + str(n).zfill(len(m.group(1))) + f[m.end(1):]


def pct(a, qs=(0, 5, 50, 95, 100)):
    a = np.asarray(a, dtype=np.float64)
    if len(a) == 0:
        return {}
    return {f'p{q}': round(float(np.percentile(a, q)), 2) for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--samples', type=int, default=400, help='낮/밤 각각 내용까지 읽어볼 이미지 수')
    ap.add_argument('--pairs', type=int, default=200, help='낮/밤 각각 이웃 쌍 / 무작위 쌍 수')
    args = ap.parse_args()
    img_dir, lab_dir = os.path.join(args.data, 'images'), os.path.join(args.data, 'labels')
    rng = random.Random(0)
    out = {}

    files = sorted(f for f in os.listdir(img_dir) if f.endswith('.npy'))
    by_pre = defaultdict(list)
    for f in files:
        by_pre[prefix_of(f)].append(f)
    log(f'[파일 수] 전체 {len(files)}  ' + '  '.join(f'{NAMES.get(p, p)} {len(v)}' for p, v in by_pre.items()))

    # ---- 1) 파일 형식 (헤더는 전체, 값 범위는 샘플)
    log('\n[1] 파일 형식 (shape, dtype) - 전체 파일 헤더 기준')
    shapes = {}
    out['format'] = {}
    for p in PREFIXES + (['other'] if 'other' in by_pre else []):
        c = Counter()
        for f in by_pre.get(p, []):
            a = np.load(os.path.join(img_dir, f), mmap_mode='r')
            c[(tuple(a.shape), str(a.dtype))] += 1
            shapes[f] = a.shape
        out['format'][p] = {f'{s}|{d}': n for (s, d), n in c.most_common()}
        log(f'  {NAMES.get(p, p):15s}: ' + ', '.join(f'{s} {d} x{n}' for (s, d), n in c.most_common(5)))

    samples = {p: rng.sample(by_pre[p], min(args.samples, len(by_pre[p]))) for p in PREFIXES if by_pre.get(p)}
    log(f'\n[1] 채널별 값 범위 (샘플 낮/밤 각 {args.samples}장, 원래 값 그대로)  채널: R G B 열화상 이벤트')
    bright = {}
    out['value_range'] = {}
    for p, fs in samples.items():
        mins, maxs, means = [], [], []
        bright[p] = []
        for f in fs:
            a = load_hwc(os.path.join(img_dir, f)).astype(np.float64)
            mins.append(a.reshape(-1, a.shape[-1]).min(0))
            maxs.append(a.reshape(-1, a.shape[-1]).max(0))
            means.append(a.reshape(-1, a.shape[-1]).mean(0))
            bright[p].append(float(a[..., :3].mean()))
        mn, mx, me = np.min(mins, 0), np.max(maxs, 0), np.mean(means, 0)
        out['value_range'][p] = {'min': mn.round(2).tolist(), 'max': mx.round(2).tolist(), 'mean': me.round(2).tolist()}
        log(f'  {NAMES[p]:15s}: min {mn.round(1).tolist()}  max {mx.round(1).tolist()}  mean {me.round(1).tolist()}')

    # ---- 2) 라벨
    log('\n[2] 라벨 (전체 라벨 파일)')
    out['labels'] = {}
    for p in PREFIXES:
        cls, nbox, ws, hs, n_files, n_empty = Counter(), [], [], [], 0, 0
        for f in by_pre.get(p, []):
            lp = os.path.join(lab_dir, f.replace('.npy', '.txt'))
            if not os.path.exists(lp):
                continue
            n_files += 1
            boxes = read_labels(lp)
            if not boxes:
                n_empty += 1
                continue
            nbox.append(len(boxes))
            H, W = shapes[f][0], shapes[f][1]
            if shapes[f][0] == 5:
                H, W = shapes[f][1], shapes[f][2]
            for c, x, y, w, h in boxes:
                cls[c] += 1
                ws.append(w * W)
                hs.append(h * H)
        info = {'label_files': n_files, 'empty': n_empty, 'class_ids': dict(cls),
                'boxes_per_img': pct(nbox, (50, 95)), 'box_w_px': pct(ws, (5, 50, 95)),
                'box_h_px': pct(hs, (5, 50, 95)), 'total_boxes': int(sum(nbox))}
        out['labels'][p] = info
        log(f"  {NAMES[p]:15s}: 라벨파일 {n_files} (빈 파일 {n_empty})  클래스 {dict(cls)}  박스 {info['total_boxes']}  "
            f"이미지당 박스 {info['boxes_per_img']}  박스폭 {info['box_w_px']}  박스높이 {info['box_h_px']}")

    # ---- 3) 밝기
    log('\n[3] 이미지별 RGB 평균 밝기 분포 (0~255, 샘플)')
    out['brightness'] = {}
    for p in samples:
        b = np.array(bright[p])
        info = pct(b)
        info['n_dark_lt50'] = int((b < 50).sum())
        info['n_bright_gt80'] = int((b > 80).sum())
        out['brightness'][p] = info
        log(f"  {NAMES[p]:15s}: {pct(b)}  어두운(<50) {info['n_dark_lt50']}장, 밝은(>80) {info['n_bright_gt80']}장 / {len(b)}")

    # ---- 4) 차 vs 주변
    log('\n[4] 차(박스 안) vs 바로 바깥 테두리 (샘플, 0~255 값 기준)')
    log('    gray = RGB 평균, thermal = 열화상, event_act = |이벤트 - 이미지 중앙값| (이벤트 활동량)')
    out['contrast'] = {}
    for p, fs in samples.items():
        acc = defaultdict(list)
        for f in fs:
            boxes = read_labels(os.path.join(lab_dir, f.replace('.npy', '.txt')))
            if not boxes:
                continue
            a = load_hwc(os.path.join(img_dir, f)).astype(np.float32)
            H, W = a.shape[:2]
            chans = {'gray': a[..., :3].mean(-1), 'thermal': a[..., 3],
                     'event_act': np.abs(a[..., 4] - np.median(a[..., 4]))}
            for _c, x, y, w, h in boxes:
                x0, x1 = int(round((x - w / 2) * W)), int(round((x + w / 2) * W))
                y0, y1 = int(round((y - h / 2) * H)), int(round((y + h / 2) * H))
                x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                m = max(3, int(0.5 * max(x1 - x0, y1 - y0)))
                X0, Y0, X1, Y1 = max(x0 - m, 0), max(y0 - m, 0), min(x1 + m, W), min(y1 + m, H)
                ring = np.ones((Y1 - Y0, X1 - X0), bool)
                ring[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = False
                if ring.sum() < 10:
                    continue
                for k, ch in chans.items():
                    inside = float(ch[y0:y1, x0:x1].mean())
                    outside = float(ch[Y0:Y1, X0:X1][ring].mean())
                    acc[k + '_in'].append(inside)
                    acc[k + '_out'].append(outside)
        info = {'boxes': len(acc['gray_in'])}
        for k in ['gray', 'thermal', 'event_act']:
            i, o = np.array(acc[k + '_in']), np.array(acc[k + '_out'])
            if len(i) == 0:
                continue
            info[k] = {'in_mean': round(float(i.mean()), 2), 'out_mean': round(float(o.mean()), 2),
                       'diff_mean': round(float((i - o).mean()), 2),
                       'in_gt_out_pct': round(100.0 * float((i > o).mean()), 1),
                       'ratio_in_over_out': round(float(i.mean() / max(o.mean(), 1e-6)), 3)}
        out['contrast'][p] = info
        log(f"  {NAMES[p]:15s} (박스 {info['boxes']}개)")
        for k in ['gray', 'thermal', 'event_act']:
            if k in info:
                v = info[k]
                log(f"      {k:9s}: 안 {v['in_mean']:7.2f}  바깥 {v['out_mean']:7.2f}  차이 {v['diff_mean']:+7.2f}  "
                    f"안>바깥 박스 {v['in_gt_out_pct']:5.1f}%  안/바깥 {v['ratio_in_over_out']}")

    # ---- 5) 분할 누수 + 이웃 유사도
    log('\n[5] 분할 누수 (원본 코드와 같은 분할: 라벨 있는 이미지, test_size=0.2, random_state=42)')
    try:
        from sklearn.model_selection import train_test_split
        valid = [f for f in files if read_labels(os.path.join(lab_dir, f.replace('.npy', '.txt')))]
        tr, te = train_test_split(valid, test_size=0.2, random_state=42)
        trs = set(tr)
        leak = defaultdict(lambda: [0, 0])
        for f in te:
            p = prefix_of(f)
            has = any((neighbor_name(f, d) in trs) for d in (-1, 1))
            for k in ('all', p):
                leak[k][0] += int(has)
                leak[k][1] += 1
        out['split_leak'] = {k: {'test': n, 'neighbor_in_train_pct': round(100.0 * h / max(n, 1), 1)}
                             for k, (h, n) in leak.items()}
        out['split_sizes'] = {'valid': len(valid), 'train': len(tr), 'test': len(te),
                              'test_by_prefix': dict(Counter(prefix_of(f) for f in te)),
                              'train_by_prefix': dict(Counter(prefix_of(f) for f in tr))}
        log(f"  train {len(tr)} / test {len(te)}   test 낮/밤: {out['split_sizes']['test_by_prefix']}   "
            f"train 낮/밤: {out['split_sizes']['train_by_prefix']}")
        for k, v in out['split_leak'].items():
            log(f"  {NAMES.get(k, k):15s}: test {v['test']}장 중 번호 ±1 이웃이 train 에 있는 비율 {v['neighbor_in_train_pct']}%")
    except Exception as e:
        log(f'  !! 분할 누수 계산 실패: {type(e).__name__}: {e}')

    log('\n[5] 번호 이웃 두 장 vs 같은 조건 무작위 두 장: RGB 평균 절대 차이 (0~255, 작을수록 비슷)')
    out['neighbor_similarity'] = {}
    fileset = set(files)
    for p in PREFIXES:
        fs = by_pre.get(p, [])
        cand = [f for f in fs if neighbor_name(f, 1) in fileset]
        if not cand:
            continue
        nb, rd = [], []
        for f in rng.sample(cand, min(args.pairs, len(cand))):
            a = load_hwc(os.path.join(img_dir, f))[..., :3].astype(np.float32)
            b = load_hwc(os.path.join(img_dir, neighbor_name(f, 1)))[..., :3].astype(np.float32)
            if a.shape == b.shape:
                nb.append(float(np.abs(a - b).mean()))
            g = rng.choice(fs)
            c = load_hwc(os.path.join(img_dir, g))[..., :3].astype(np.float32)
            if a.shape == c.shape:
                rd.append(float(np.abs(a - c).mean()))
        nums = sorted(int(re.search(r'(\d+)(?=\.npy$)', f).group(1)) for f in fs if re.search(r'(\d+)(?=\.npy$)', f))
        gaps = int(sum(1 for i in range(1, len(nums)) if nums[i] - nums[i - 1] > 1))
        info = {'neighbor_diff': pct(nb, (5, 50, 95)), 'random_diff': pct(rd, (5, 50, 95)),
                'number_range': [nums[0], nums[-1]] if nums else None, 'number_gaps': gaps}
        out['neighbor_similarity'][p] = info
        log(f"  {NAMES[p]:15s}: 이웃 {info['neighbor_diff']}   무작위 {info['random_diff']}   "
            f"번호 {info['number_range']} (끊긴 곳 {gaps}군데)")

    log('\nDATA_CHECK_JSON ' + json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
