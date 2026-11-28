#!/usr/bin/env python
"""
논문 재현 실험 실행 스크립트 (H200 제출용).

원본 코드(trimodaldet/)는 수정하지 않고 그대로 import 해서 사용한다.
두 가지 모드를 지원한다.

  --setting code   : 공개 코드 그대로 (scripts/train.py 와 동일)
                     - 입력값 0~255 그대로, mean/std = config.py 값
                     - torchvision 기본 리사이즈 (짧은 변 800px로 확대)
                     - AMP 없음
  --setting paper  : 논문 5.1절/3.2절에 적힌 설정
                     - 입력을 0~1로 스케일 (/255)
                     - RGB = ImageNet 통계, 열화상/이벤트 = 학습셋에서 계산한 통계
                     - 원본 해상도 301x391 그대로 (32 배수 패딩은 torchvision이 처리)
                     - AMP(mixed precision) 사용

공통 (논문 = 코드 동일):
  15 epoch, SGD(momentum 0.9, wd 1e-4), lr 0.02 @ batch 16,
  500 iter linear warmup + cosine, 80/20 무작위 분할(seed 42), MAGE+BiTE, FPN + Faster R-CNN

평가: COCO mAP / mAP50 을 전체(All), frame_(밝은 영상), nframe_(어두운 영상) 별로 출력.

--analyze (11/24 회의 후속 분석, 학습은 그대로이고 평가 뒤에 아래를 추가로 출력):
  1) 예측 통계 : 낮/밤 test 에서 모델이 낸 박스 수, 점수 분포 (못 찾는지 / 엉뚱한 걸 찾는지)
  2) 센서 끄고 채점 : 학습된 모델에서 시험 때 RGB / 열화상 / 이벤트를 하나씩 0으로 끄고 다시 채점
                     (원본 ablation 코드와 같은 방식: 정규화된 입력 채널을 0으로)
  3) MAGE 가중치 : stage 1~4 별로 MAGE 가 두 갈래(RGB / 열화상+이벤트) 사이에 주고받는
                   gate 값 평균과 각 갈래 feature 크기 (낮/밤 따로)
  4) [11/28 추가] Table 7 0점대 원인 진단 (재학습 없이 같은 모델로 다시 채점)
     a) 문턱값 낮추기 : torchvision 기본 score_thresh 0.05 -> 0.001 로 낮춰 다시 채점.
                        + 점수와 상관없이 정답 박스 근처(IoU>=0.5)에 예측이 있는 비율(recall)과
                          그 예측의 점수 분포 -> "못 찾는 것" vs "찾는데 확신이 낮아 버려지는 것"
     b) AdaBN       : 낮만(밤만) 학습 모델의 BatchNorm 통계(평균/분산)만 반대 조건 사진으로 다시 계산.
                        사진은 학습 split 의 반대 조건 이미지(라벨 사용 안 함, test 사진 사용 안 함).
                        가중치는 그대로. 점수가 크게 오르면 "BatchNorm 영점이 학습 조건에 고정된 것"이 원인.
                        (--train-subset day/night 일 때만 실행)
"""
import argparse
import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, box_iou

from trimodaldet.config import Config
from trimodaldet.data.dataset import NpyYoloDataset, collate_fn
from trimodaldet.models.encoder import get_encoder
from trimodaldet.models.backbone import InterModalBackbone
from trimodaldet.ablations.backbone_modality import ModalityConfigurableBackbone

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    sys.exit("scikit-learn 이 필요합니다.")

# 파일명 접두사 -> 조명 조건 (RGB 평균 밝기 측정 결과: frame_=123, nframe_=21)
SUBSETS = {'day': 'frame_', 'night': 'nframe_'}


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- data
class ScaledDataset(Dataset):
    """원본 NpyYoloDataset 을 감싸서, paper 모드일 때만 입력을 /255 한다."""

    def __init__(self, base, files, scale):
        self.ds = copy.copy(base)
        self.ds.image_files = list(files)
        self.scale = scale

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, tgt = self.ds[idx]
        if self.scale != 1.0:
            img = img * self.scale
        return img, tgt

    @property
    def files(self):
        return self.ds.image_files


def compute_aux_stats(image_dir, files, n_samples, seed):
    """열화상(ch3)/이벤트(ch4)의 mean/std 를 학습셋 일부에서 계산 (0~1 스케일)."""
    rng = random.Random(seed)
    pick = rng.sample(files, min(n_samples, len(files)))
    s = np.zeros(2)
    s2 = np.zeros(2)
    n = 0
    for f in pick:
        a = np.load(os.path.join(image_dir, f), mmap_mode='r')[:, :, 3:5].astype(np.float64) / 255.0
        a = a.reshape(-1, 2)
        s += a.sum(0)
        s2 += (a ** 2).sum(0)
        n += a.shape[0]
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    return mean.tolist(), std.tolist()


# --------------------------------------------------------------------------- model
def build_model(cfg, backbone_name, modalities, image_mean, image_std, min_size, max_size):
    encoder = get_encoder(backbone_name, in_chans_rgb=cfg.in_chans_rgb, in_chans_x=cfg.in_chans_x)
    if set(modalities) == {'rgb', 'thermal', 'event'}:
        backbone = InterModalBackbone(encoder, fpn_out_channels=cfg.fpn_out_channels)
    else:
        # 원본 ablation 코드와 같은 방식: 빠진 센서 채널을 0으로 마스킹
        backbone = ModalityConfigurableBackbone(encoder, fpn_out_channels=cfg.fpn_out_channels,
                                                active_modalities=modalities)
    kwargs = dict(
        num_classes=cfg.num_classes,
        rpn_anchor_generator=AnchorGenerator(sizes=cfg.anchor_sizes, aspect_ratios=cfg.anchor_aspect_ratios),
        box_roi_pool=MultiScaleRoIAlign(featmap_names=cfg.roi_featmap_names,
                                        output_size=cfg.roi_output_size,
                                        sampling_ratio=cfg.roi_sampling_ratio),
        image_mean=image_mean,
        image_std=image_std,
    )
    if min_size is not None:
        kwargs.update(min_size=min_size, max_size=max_size)
    return FasterRCNN(backbone, **kwargs)


# --------------------------------------------------------------------------- eval
MODALITY_CHANNELS = {'rgb': [0, 1, 2], 'thermal': [3], 'event': [4]}


def subsets_of(fname):
    return ['all'] + [sub for sub, prefix in SUBSETS.items() if fname.startswith(prefix)]


def add_drop_hook(model, drop):
    """시험 때 센서 끄기: backbone 에 들어가는 (정규화된) 입력의 해당 채널을 0으로."""
    ch = sum((MODALITY_CHANNELS[m] for m in drop), [])

    def pre(_mod, args):
        x = args[0].clone()
        x[:, ch] = 0
        return (x,) + tuple(args[1:])
    return model.backbone.register_forward_pre_hook(pre)


class MageProbe:
    """MAGE 가 주고받는 gate 값과 두 갈래 feature 크기를 stage 별, 낮/밤 별로 평균낸다.

    MAGE: out_rgb = rgb + 0.5*ch[1]*x + 0.5*sp[1]*x   (x -> rgb 로 넘어가는 양)
          out_x   = x   + 0.5*ch[0]*rgb + 0.5*sp[0]*rgb (rgb -> x 로 넘어가는 양)
    x = 열화상+이벤트 갈래
    """

    def __init__(self, encoder):
        self.cur = None          # 지금 배치의 이미지별 subset 목록
        self.sums = {}
        self.handles = []
        for i, mage in enumerate(encoder.MAGEs):
            self.handles.append(mage.channel_weights.register_forward_hook(self._gate_hook(i, 'ch')))
            self.handles.append(mage.spatial_weights.register_forward_hook(self._gate_hook(i, 'sp')))
            self.handles.append(mage.register_forward_hook(self._input_hook(i)))

    def _add(self, stage, key, per_sample):
        if self.cur is None:
            return
        for b, v in enumerate(per_sample.tolist()):
            if b >= len(self.cur):
                break
            for sub in self.cur[b]:
                s = self.sums.setdefault((sub, stage, key), [0.0, 0])
                s[0] += v
                s[1] += 1

    def _gate_hook(self, i, kind):
        def hook(_mod, _inp, out):
            w = out.detach().float()          # (2, B, ...)
            B = w.shape[1]
            self._add(i, f'{kind}_rgb2x', w[0].reshape(B, -1).mean(1))
            self._add(i, f'{kind}_x2rgb', w[1].reshape(B, -1).mean(1))
        return hook

    def _input_hook(self, i):
        def hook(_mod, inp, _out):
            x1, x2 = inp[0].detach().float(), inp[1].detach().float()
            B = x1.shape[0]
            self._add(i, 'feat_rgb', x1.abs().reshape(B, -1).mean(1))
            self._add(i, 'feat_x', x2.abs().reshape(B, -1).mean(1))
        return hook

    def result(self):
        out = {}
        for (sub, stage, key), (s, n) in sorted(self.sums.items()):
            out.setdefault(sub, {}).setdefault(f'stage{stage + 1}', {})[key] = round(s / max(n, 1), 4)
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


def new_pred_stats():
    return {'images': 0, 'gt_boxes': 0, 'pred_boxes': 0, 'pred_ge0.5': 0, 'images_no_pred_ge0.5': 0,
            'score_hist': [0] * 10, 'max_score_sum': 0.0}


def finish_pred_stats(s):
    n = max(s['images'], 1)
    return {'images': s['images'],
            'gt_per_img': round(s['gt_boxes'] / n, 2),
            'pred_per_img': round(s['pred_boxes'] / n, 2),
            'pred_ge0.5_per_img': round(s['pred_ge0.5'] / n, 2),
            'images_no_pred_ge0.5': s['images_no_pred_ge0.5'],
            'mean_max_score': round(s['max_score_sum'] / n, 3),
            'score_hist_0.0to1.0': s['score_hist']}


def new_match_stats():
    return {'gt': 0, 'found': 0, 'scores': []}


def finish_match_stats(s, low_thresh):
    sc = np.array(s['scores'], dtype=np.float64)
    out = {'gt_boxes': s['gt'],
           'recall50_any_score': round(100.0 * s['found'] / max(s['gt'], 1), 2)}
    if len(sc):
        out.update({'matched_score_median': round(float(np.median(sc)), 4),
                    'matched_score_mean': round(float(sc.mean()), 4),
                    'matched_below_0.05_pct': round(100.0 * float((sc < 0.05).mean()), 2),
                    'matched_ge_0.5_pct': round(100.0 * float((sc >= 0.5).mean()), 2)})
    return out


def update_match_stats(mstats, subs, p, t):
    """정답 박스마다 IoU>=0.5 인 예측이 하나라도 있는지(점수 무관)와, 있으면 그중 최고 점수를 모은다."""
    gt = t['boxes'].float()
    for sub in subs:
        mstats[sub]['gt'] += len(gt)
    if len(gt) == 0 or len(p['boxes']) == 0:
        return
    iou = box_iou(gt, p['boxes'])                       # (G, P)
    hit = iou >= 0.5
    found = hit.any(1)
    best = torch.where(hit, p['scores'].unsqueeze(0).expand_as(iou), torch.zeros_like(iou)).max(1).values
    sc = best[found].tolist()
    for sub in subs:
        mstats[sub]['found'] += int(found.sum())
        mstats[sub]['scores'].extend(sc)


def evaluate(model, loader, files, device, amp, max_images=None, drop=None, probe=None, pred_stats=False,
             match_stats=False):
    from torchmetrics.detection import MeanAveragePrecision
    metrics = {k: MeanAveragePrecision(box_format='xyxy') for k in ['all', 'day', 'night']}
    stats = {k: new_pred_stats() for k in ['all', 'day', 'night']}
    mstats = {k: new_match_stats() for k in ['all', 'day', 'night']}
    handle = add_drop_hook(model, drop) if drop else None
    model.eval()
    seen = 0
    try:
        with torch.no_grad():
            for images, targets in loader:
                images = [im.to(device) for im in images]
                if probe is not None:
                    probe.cur = [subsets_of(f) for f in files[seen:seen + len(images)]]
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    preds = model(images)
                for p, t in zip(preds, targets):
                    if max_images is not None and seen >= max_images:
                        break
                    p = {'boxes': p['boxes'].float().cpu(), 'scores': p['scores'].float().cpu(),
                         'labels': p['labels'].long().cpu()}
                    subs = subsets_of(files[seen])
                    for sub in subs:
                        metrics[sub].update([p], [t])
                    if pred_stats:
                        sc = p['scores']
                        for sub in subs:
                            s = stats[sub]
                            s['images'] += 1
                            s['gt_boxes'] += len(t['boxes'])
                            s['pred_boxes'] += len(sc)
                            n_hi = int((sc >= 0.5).sum())
                            s['pred_ge0.5'] += n_hi
                            s['images_no_pred_ge0.5'] += int(n_hi == 0)
                            s['max_score_sum'] += float(sc.max()) if len(sc) else 0.0
                            for v in sc.tolist():
                                s['score_hist'][min(int(v * 10), 9)] += 1
                    if match_stats:
                        update_match_stats(mstats, subs, p, t)
                    seen += 1
                if max_images is not None and seen >= max_images:
                    break
    finally:
        if handle is not None:
            handle.remove()
        if probe is not None:
            probe.cur = None
    out = {}
    for k, m in metrics.items():
        try:
            r = m.compute()
            out[k] = {'mAP': round(r['map'].item() * 100, 2), 'mAP50': round(r['map_50'].item() * 100, 2)}
        except Exception as e:  # 해당 subset 이미지가 없을 때 등
            log(f'!! METRIC ERROR [{k}]: {type(e).__name__}: {e}')
            out[k] = {'mAP': None, 'mAP50': None, 'note': str(e)[:200]}
    out['num_test_images'] = seen
    if pred_stats:
        out['pred_stats'] = {k: finish_pred_stats(v) for k, v in stats.items()}
    if match_stats:
        thr = float(getattr(model.roi_heads, 'score_thresh', float('nan')))
        out['score_thresh'] = thr
        out['match'] = {k: finish_match_stats(v, thr) for k, v in mstats.items()}
    return out


# --------------------------------------------------------------------------- 11/28 진단 (Table 7)
def evaluate_with_thresh(model, thresh, *eval_args, **eval_kwargs):
    """roi_heads.score_thresh 를 잠시 바꿔서 채점하고 원래 값으로 되돌린다."""
    old = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = thresh
    try:
        return evaluate(model, *eval_args, **eval_kwargs)
    finally:
        model.roi_heads.score_thresh = old


def adabn_copy(model, loader, device, amp):
    """AdaBN: 모델을 복사해서 BatchNorm 의 running mean/var 만 loader 사진들로 다시 계산한다.
    가중치(학습된 값)는 그대로, 라벨도 쓰지 않는다. 반환: (적응된 모델 복사본, BN 층 수, 사용한 사진 수)"""
    m = copy.deepcopy(model)
    bns = [x for x in m.modules() if isinstance(x, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return None, 0, 0
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None            # None = 들어온 배치들의 누적 평균
    m.eval()                          # 검출기는 평가 모드 (정답 필요 없음)
    for bn in bns:
        bn.train()                    # BatchNorm 만 통계 갱신 모드
    n = 0
    with torch.no_grad():
        for images, _ in loader:
            images = [im.to(device) for im in images]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                m(images)
            n += len(images)
    m.eval()
    return m, len(bns), n


def run_diagnostics(model, test_loader, files, device, amp, max_images, res, args, base, scale,
                    train_files_all):
    """Table 7 0점대 원인 진단: (a) 문턱값 낮추기, (b) AdaBN. 실패해도 본 결과에는 영향 없음."""
    diag = res.setdefault('diag', {})
    low = args.low_thresh
    eval_args = (test_loader, files, device, amp, max_images)

    # (a) 문턱값 낮추기
    log(f'=== DIAG (a): score_thresh {model.roi_heads.score_thresh} -> {low} 로 다시 채점 ===')
    try:
        r = evaluate_with_thresh(model, low, *eval_args, match_stats=True)
        diag['low_thresh'] = {k: r[k] for k in ['all', 'day', 'night']}
        diag['low_thresh']['match'] = r['match']
    except Exception as e:
        log(f'!! DIAG (a) FAILED: {type(e).__name__}: {e}')
        diag['low_thresh'] = {'error': f'{type(e).__name__}: {e}'[:300]}

    # (b) AdaBN (낮만/밤만 학습일 때만)
    if args.train_subset == 'all':
        diag['adabn'] = {'skipped': 'train_subset=all (전체 학습 모델은 대상 아님)'}
        return
    target = 'night' if args.train_subset == 'day' else 'day'
    adapt_pool = [f for f in train_files_all if f.startswith(SUBSETS[target])]
    rng = random.Random(42)
    adapt_files = rng.sample(adapt_pool, min(args.adabn_samples, len(adapt_pool)))
    log(f'=== DIAG (b): AdaBN -> BatchNorm 통계를 학습 split 의 {target} 사진 {len(adapt_files)}장으로 다시 계산 ===')
    try:
        adapt_ds = ScaledDataset(base, adapt_files, scale)
        adapt_loader = DataLoader(adapt_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                  collate_fn=collate_fn)
        m2, n_bn, n_img = adabn_copy(model, adapt_loader, device, amp)
        if m2 is None:
            diag['adabn'] = {'skipped': 'BatchNorm 층 없음'}
            return
        out = {'target': target, 'bn_layers': n_bn, 'adapt_images': n_img}
        r = evaluate(m2, *eval_args, match_stats=True)
        out['default_thresh'] = {k: r[k] for k in ['all', 'day', 'night']}
        out['default_thresh']['match'] = r['match']
        r = evaluate_with_thresh(m2, low, *eval_args, match_stats=True)
        out['low_thresh'] = {k: r[k] for k in ['all', 'day', 'night']}
        out['low_thresh']['match'] = r['match']
        diag['adabn'] = out
        del m2
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        log(f'!! DIAG (b) FAILED: {type(e).__name__}: {e}')
        log(traceback.format_exc()[-2000:])
        diag['adabn'] = {'error': f'{type(e).__name__}: {e}'[:300]}


def print_diag(res):
    d = res.get('diag')
    if not d:
        return
    log('=== DIAG (Table 7 원인 진단) ===')
    log('mAP / mAP50 (all | day | night),  recall50 = 정답 박스 근처(IoU>=0.5)에 예측이 있는 비율(점수 무관)')

    def row(label, r):
        if r is None:
            return
        if 'error' in r or 'skipped' in r:
            log(f"  {label:28s}: {r.get('error') or r.get('skipped')}")
            return
        mAP = ' | '.join(f"{r[k]['mAP']}/{r[k]['mAP50']}" for k in ['all', 'day', 'night'])
        log(f'  {label:28s}: {mAP}')
        for k in ['day', 'night']:
            m = r.get('match', {}).get(k)
            if m:
                extra = ''
                if 'matched_score_median' in m:
                    extra = (f"  찾은 박스 점수 중앙값 {m['matched_score_median']}"
                             f"  (0.05 미만 {m['matched_below_0.05_pct']}%, 0.5 이상 {m['matched_ge_0.5_pct']}%)")
                log(f"      {k:5s} recall50 {m['recall50_any_score']}%{extra}")

    base = {k: res[k] for k in ['all', 'day', 'night']}
    base['match'] = res.get('match_default', {})
    row('기본 (score_thresh 0.05)', base)
    row(f"문턱값 {res.get('low_thresh_value', '')}", d.get('low_thresh'))
    a = d.get('adabn', {})
    if 'default_thresh' in a:
        log(f"  AdaBN: {a['target']} 사진 {a['adapt_images']}장으로 BatchNorm {a['bn_layers']}개 층 통계 재계산")
        row('AdaBN (0.05)', a['default_thresh'])
        row('AdaBN + 문턱값 낮춤', a['low_thresh'])
    else:
        row('AdaBN', a)


def run_analysis(model, loader, files, device, amp, max_images, modalities, res):
    """--analyze: 센서 끄고 채점 (예측 통계와 MAGE 값은 본 평가에서 이미 수집)."""
    drops = [[m] for m in modalities] if len(modalities) > 1 else []
    if set(modalities) == {'rgb', 'thermal', 'event'}:
        drops.append(['thermal', 'event'])   # RGB 갈래만 남기기
    res['drop_eval'] = {}
    for d in drops:
        key = 'no_' + '+'.join(d)
        log(f'=== DROP EVAL: {key} ===')
        try:
            r = evaluate(model, loader, files, device, amp, max_images, drop=d)
            res['drop_eval'][key] = {k: r[k] for k in ['all', 'day', 'night']}
        except Exception as e:
            log(f'!! DROP EVAL FAILED [{key}]: {type(e).__name__}: {e}')
            res['drop_eval'][key] = {'error': f'{type(e).__name__}: {e}'[:300]}


def print_analysis(res):
    log('=== ANALYSIS ===')
    if 'pred_stats' in res:
        log('[예측 통계] 이미지당 정답 박스 / 예측 박스(전체, 점수>=0.5) / 점수>=0.5 예측이 없는 이미지 수 / 평균 최고점수')
        for k, s in res['pred_stats'].items():
            log(f"  {k:5s}: imgs {s['images']}  gt {s['gt_per_img']}  pred {s['pred_per_img']} "
                f"(>=0.5: {s['pred_ge0.5_per_img']})  no_pred {s['images_no_pred_ge0.5']}  "
                f"max_score {s['mean_max_score']}  hist {s['score_hist_0.0to1.0']}")
    if res.get('drop_eval'):
        log('[센서 끄고 채점] mAP all / day / night')
        log(f"  {'none (기본)':16s}: {res['all']['mAP']} / {res['day']['mAP']} / {res['night']['mAP']}")
        for key, r in res['drop_eval'].items():
            if 'error' in r:
                log(f'  {key:16s}: ERROR {r["error"]}')
            else:
                log(f"  {key:16s}: {r['all']['mAP']} / {r['day']['mAP']} / {r['night']['mAP']}")
    if 'mage' in res:
        log('[MAGE] stage별 평균: ch/sp_rgb2x = RGB->열화상+이벤트 gate, ch/sp_x2rgb = 열화상+이벤트->RGB gate, '
            'feat_rgb / feat_x = 각 갈래 feature 크기(|값| 평균)')
        for sub in ['day', 'night', 'all']:
            for st, v in res['mage'].get(sub, {}).items():
                log(f'  {sub:5s} {st}: ' + '  '.join(f'{k} {x}' for k, x in v.items()))


# --------------------------------------------------------------------------- main
def main():
    # 컨테이너에서 DataLoader worker 간 텐서 전달이 끊기는 문제("received 0 items of ancdata") 방지
    torch.multiprocessing.set_sharing_strategy('file_system')

    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='실험 이름 (출력 폴더명)')
    ap.add_argument('--data', required=True, help='images/ 와 labels/ 가 있는 폴더')
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--setting', choices=['code', 'paper'], default='paper')
    ap.add_argument('--backbone', default='mit_b1', choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4'])
    ap.add_argument('--modalities', default='rgb,thermal,event')
    ap.add_argument('--train-subset', default='all', choices=['all', 'day', 'night'])
    ap.add_argument('--stats-from', default='train', choices=['train', 'train_all'],
                    help="paper 모드의 열화상/이벤트 정규화 통계: train = 실제 학습에 쓰는 이미지(낮만이면 낮만), "
                         "train_all = 학습 split 전체 7,800장 (모든 실험이 같은 통계)")
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=16, help='한 번에 GPU에 올리는 이미지 수')
    ap.add_argument('--accum', type=int, default=1,
                    help='gradient accumulation 횟수. 실질 batch = batch-size x accum (메모리 부족 시 사용)')
    ap.add_argument('--lr', type=float, default=None, help='기본: 0.02 * 실질batch/16 (논문: 선형 스케일링)')
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log-every', type=int, default=100)
    ap.add_argument('--stats-samples', type=int, default=500)
    ap.add_argument('--max-steps', type=int, default=None, help='스모크 테스트용: 이 스텝 수만 학습')
    ap.add_argument('--eval-max', type=int, default=None, help='스모크 테스트용: 평가 이미지 수 제한')
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--analyze', action='store_true',
                    help='평가 뒤 예측 통계 / 센서 끄고 채점 / MAGE 가중치 / Table 7 진단(문턱값, AdaBN) 출력')
    ap.add_argument('--low-thresh', type=float, default=0.001, help='진단 (a): 낮춘 score_thresh')
    ap.add_argument('--adabn-samples', type=int, default=512, help='진단 (b): AdaBN 에 쓰는 반대 조건 사진 수')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paper = args.setting == 'paper'
    amp = paper and device.type == 'cuda'
    eff_batch = args.batch_size * args.accum
    lr = args.lr if args.lr is not None else 0.02 * eff_batch / 16
    modalities = [m.strip().lower() for m in args.modalities.split(',') if m.strip()]
    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)

    cfg = Config()
    cfg.image_dir = os.path.join(args.data, 'images')
    cfg.label_dir = os.path.join(args.data, 'labels')
    cfg.auto_detect_num_classes()

    # ---- split: 원본 코드와 완전히 같은 호출 (test_size=0.2, random_state=42)
    base = NpyYoloDataset(cfg.image_dir, cfg.label_dir, mode='all')
    train_files, test_files = train_test_split(base.image_files, test_size=cfg.test_size,
                                               random_state=cfg.random_state)
    train_files_all = list(train_files)
    if args.train_subset != 'all':
        prefix = SUBSETS[args.train_subset]
        train_files = [f for f in train_files if f.startswith(prefix)]

    # ---- normalization / resolution
    if paper:
        stats_files = train_files_all if args.stats_from == 'train_all' else train_files
        # 통계용 샘플 선택은 항상 42 로 고정 -> 학습 seed 를 바꿔도 정규화 값은 같음 (seed 반복 실험용)
        aux_mean, aux_std = compute_aux_stats(cfg.image_dir, stats_files, args.stats_samples, 42)
        image_mean = [0.485, 0.456, 0.406] + aux_mean
        image_std = [0.229, 0.224, 0.225] + aux_std
        scale, min_size, max_size = 1.0 / 255.0, 301, 391
    else:
        image_mean, image_std = cfg.image_mean, cfg.image_std
        scale, min_size, max_size = 1.0, None, None

    train_ds = ScaledDataset(base, train_files, scale)
    test_ds = ScaledDataset(base, test_files, scale)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn)
    # 평가는 worker 없이 (1,950장이라 충분히 빠르고, 서버에서 worker 통신 에러가 났던 부분)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)

    model = build_model(cfg, args.backbone, modalities, image_mean, image_std, min_size, max_size).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    info = {
        'name': args.name, 'setting': args.setting, 'backbone': args.backbone, 'modalities': modalities,
        'train_subset': args.train_subset, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'accum': args.accum, 'effective_batch': eff_batch, 'lr': lr,
        'amp': amp, 'min_size': min_size, 'max_size': max_size, 'stats_from': args.stats_from,
        'image_mean': [round(x, 4) for x in image_mean], 'image_std': [round(x, 4) for x in image_std],
        'num_train': len(train_files), 'num_test': len(test_files), 'params_M': round(n_params, 2),
        'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'torch': torch.__version__, 'seed': args.seed, 'analyze': args.analyze,
    }
    log('=== CONFIG === ' + json.dumps(info, ensure_ascii=False))

    # ---- optimizer / schedule: 원본 Trainer 와 동일
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    # step = optimizer 업데이트 횟수 (accum=1 이면 원본 Trainer 와 완전히 동일)
    steps_per_epoch = (len(train_loader) + args.accum - 1) // args.accum
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - args.warmup),
                                                           eta_min=0)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    step = 0
    t_start = time.time()
    stop = False
    for epoch in range(args.epochs):
        model.train()
        ep_loss, ep_n, t_ep = 0.0, 0, time.time()
        optimizer.zero_grad()
        n_batches = len(train_loader)
        for bi, (images, targets) in enumerate(train_loader):
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss / args.accum).backward()
            if not torch.isfinite(loss):
                log(f'!! loss is {loss.item()} at batch {bi + 1} (epoch {epoch + 1})')
            ep_loss += loss.item()
            ep_n += 1
            if (bi + 1) % args.accum != 0 and bi + 1 != n_batches:
                continue  # 아직 accum 횟수만큼 안 모음

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            step += 1
            if step <= args.warmup:
                for g in optimizer.param_groups:
                    g['lr'] = lr * min(1.0, step / args.warmup)
            else:
                scheduler.step()

            if step == 1 or step % args.log_every == 0:
                el = time.time() - t_start
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else 0
                log(f'step {step}/{total_steps} ep {epoch + 1} loss {loss.item():.4f} '
                    f'lr {optimizer.param_groups[0]["lr"]:.5f} {el / step:.2f}s/step mem {mem:.1f}GB '
                    f'eta {(total_steps - step) * el / step / 3600:.2f}h')
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        log(f'--- epoch {epoch + 1} done: avg loss {ep_loss / max(ep_n, 1):.4f}, '
            f'{(time.time() - t_ep) / 60:.1f} min')
        if not args.no_save:
            torch.save(model.state_dict(), os.path.join(out_dir, 'model.pth'))
        if stop:
            break

    train_hours = (time.time() - t_start) / 3600
    log('=== EVAL ===')
    probe = MageProbe(model.backbone.encoder) if args.analyze else None
    try:
        res = evaluate(model, test_loader, test_ds.files, device, amp, args.eval_max,
                       probe=probe, pred_stats=args.analyze, match_stats=args.analyze)
    except Exception as e:
        import traceback
        log('!! EVAL FAILED: ' + f'{type(e).__name__}: {e}')
        log(traceback.format_exc()[-3000:])
        res = {k: {'mAP': None, 'mAP50': None} for k in ['all', 'day', 'night']}
        res['eval_error'] = f'{type(e).__name__}: {e}'[:300]
    if 'match' in res:
        res['match_default'] = res.pop('match')
    if probe is not None:
        res['mage'] = probe.result()
        probe.remove()
    res.update(info)
    res['train_hours'] = round(train_hours, 3)
    res['steps'] = step

    def save_results():
        with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    # 본 채점 결과를 분석 전에 먼저 출력/저장 (분석 중 문제가 생겨도 본 결과는 남도록)
    log('=== EVAL RESULT (본 채점) === ' + '  '.join(
        f"{k} {res[k]['mAP']}/{res[k]['mAP50']}" for k in ['all', 'day', 'night']))
    save_results()
    if args.analyze and 'eval_error' not in res:
        try:
            run_analysis(model, test_loader, test_ds.files, device, amp, args.eval_max, modalities, res)
        except Exception as e:
            log(f'!! ANALYSIS FAILED: {type(e).__name__}: {e}')
        res['low_thresh_value'] = args.low_thresh
        try:
            run_diagnostics(model, test_loader, test_ds.files, device, amp, args.eval_max, res, args,
                            base, scale, train_files_all)
        except Exception as e:
            log(f'!! DIAG FAILED: {type(e).__name__}: {e}')
        save_results()

    log('=== RESULT ===')
    log(f"{args.name} | setting={args.setting} backbone={args.backbone} modal={'+'.join(modalities)} "
        f"train={args.train_subset} | params {n_params:.2f}M | {train_hours:.2f}h")
    for k in ['all', 'day', 'night']:
        log(f"  {k:5s}: mAP {res[k]['mAP']}  mAP50 {res[k]['mAP50']}")
    if args.analyze:
        print_analysis(res)
        print_diag(res)
    log('RESULT_JSON ' + json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    main()
