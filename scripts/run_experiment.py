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
from torchvision.ops import MultiScaleRoIAlign

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
def evaluate(model, loader, files, device, amp, max_images=None):
    from torchmetrics.detection import MeanAveragePrecision
    metrics = {k: MeanAveragePrecision(box_format='xyxy') for k in ['all', 'day', 'night']}
    model.eval()
    seen = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [im.to(device) for im in images]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                preds = model(images)
            for p, t in zip(preds, targets):
                if max_images is not None and seen >= max_images:
                    break
                p = {'boxes': p['boxes'].float().cpu(), 'scores': p['scores'].float().cpu(),
                     'labels': p['labels'].long().cpu()}
                fname = files[seen]
                metrics['all'].update([p], [t])
                for sub, prefix in SUBSETS.items():
                    if fname.startswith(prefix):
                        metrics[sub].update([p], [t])
                seen += 1
            if max_images is not None and seen >= max_images:
                break
    out = {}
    for k, m in metrics.items():
        try:
            r = m.compute()
            out[k] = {'mAP': round(r['map'].item() * 100, 2), 'mAP50': round(r['map_50'].item() * 100, 2)}
        except Exception as e:  # 해당 subset 이미지가 없을 때 등
            log(f'!! METRIC ERROR [{k}]: {type(e).__name__}: {e}')
            out[k] = {'mAP': None, 'mAP50': None, 'note': str(e)[:200]}
    out['num_test_images'] = seen
    return out


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
    if args.train_subset != 'all':
        prefix = SUBSETS[args.train_subset]
        train_files = [f for f in train_files if f.startswith(prefix)]

    # ---- normalization / resolution
    if paper:
        aux_mean, aux_std = compute_aux_stats(cfg.image_dir, train_files, args.stats_samples, args.seed)
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
        'amp': amp, 'min_size': min_size, 'max_size': max_size,
        'image_mean': [round(x, 4) for x in image_mean], 'image_std': [round(x, 4) for x in image_std],
        'num_train': len(train_files), 'num_test': len(test_files), 'params_M': round(n_params, 2),
        'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'torch': torch.__version__,
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
    try:
        res = evaluate(model, test_loader, test_ds.files, device, amp, args.eval_max)
    except Exception as e:
        import traceback
        log('!! EVAL FAILED: ' + f'{type(e).__name__}: {e}')
        log(traceback.format_exc()[-3000:])
        res = {k: {'mAP': None, 'mAP50': None} for k in ['all', 'day', 'night']}
        res['eval_error'] = f'{type(e).__name__}: {e}'[:300]
    res.update(info)
    res['train_hours'] = round(train_hours, 3)
    res['steps'] = step
    with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    log('=== RESULT ===')
    log(f"{args.name} | setting={args.setting} backbone={args.backbone} modal={'+'.join(modalities)} "
        f"train={args.train_subset} | params {n_params:.2f}M | {train_hours:.2f}h")
    for k in ['all', 'day', 'night']:
        log(f"  {k:5s}: mAP {res[k]['mAP']}  mAP50 {res[k]['mAP50']}")
    log('RESULT_JSON ' + json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    main()
