#!/usr/bin/env bash
# H200 (Aerodrone Jenkins) 실행용 진입점.
#   이슈 "실행 명령어" 예:  bash run_h200.sh t1_b1
#                          bash run_h200.sh t5_all      (여러 실험을 차례로)
#                          bash run_h200.sh t1_b1_code  (공개 코드 그대로 설정)
# 실험 목록은 아래 case 문 참고.
set -euo pipefail
cd "$(dirname "$0")"

EXP="${1:-smoke}"
DRIVE_FOLDER="https://drive.google.com/drive/folders/1U2NJQMRz8uP8e-z9lCguRWVPWpXs4MmB"
LOG_EVERY=200

# ---------------------------------------------------------------- 출력 폴더
if mkdir -p /app/output 2>/dev/null && [ -w /app/output ]; then OUT_ROOT=/app/output; else OUT_ROOT="$PWD/outputs"; fi
mkdir -p "$OUT_ROOT"
echo "[info] experiment=$EXP  out=$OUT_ROOT  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[info] nvidia-smi 없음"

# ---------------------------------------------------------------- 패키지
MISSING="$(python - <<'PY'
import importlib.util
pkgs = {"torchmetrics": "torchmetrics", "pycocotools": "pycocotools", "gdown": "gdown", "sklearn": "scikit-learn"}
print(" ".join(p for m, p in pkgs.items() if importlib.util.find_spec(m) is None))
PY
)"
if [ -n "$MISSING" ]; then
  echo "[pip] 설치: $MISSING"
  pip install -q --root-user-action=ignore $MISSING 2>&1 | grep -v -i "notice" || true
fi

# ---------------------------------------------------------------- 데이터 (data.zip 3.3GB)
# /app/output 이 실행 간에 유지되면 캐시를 재사용하고, 아니면 매번 내려받는다.
CACHE="$OUT_ROOT/data_cache"
find_data_dir() { find "$1" -type d -name images -not -path '*__MACOSX*' 2>/dev/null | head -n1 | xargs -r dirname; }

download_data() {
  local dst="$1"
  mkdir -p "$dst/dl"
  echo "[data] Google Drive 에서 다운로드 중..."
  gdown --folder "$DRIVE_FOLDER" -O "$dst/dl" --quiet
  local zip; zip="$(find "$dst/dl" -name '*.zip' | head -n1)"
  [ -n "$zip" ] || { echo "[error] zip 파일을 찾지 못함"; ls -R "$dst/dl" | head -n 50; exit 1; }
  echo "[data] 압축 해제: $zip"
  python -m zipfile -e "$zip" "$dst/unzipped"
  rm -f "$zip"
  touch "$dst/.complete"
}

if [ -f "$CACHE/.complete" ]; then
  echo "[data] 캐시 사용: $CACHE"
elif mkdir "$CACHE.lock" 2>/dev/null; then
  trap 'rmdir "$CACHE.lock" 2>/dev/null || true' EXIT
  rm -rf "$CACHE"; download_data "$CACHE"
  rmdir "$CACHE.lock"; trap - EXIT
else
  echo "[data] 다른 실행이 다운로드 중 -> 최대 60분 대기"
  for _ in $(seq 120); do [ -f "$CACHE/.complete" ] && break; sleep 30; done
  [ -f "$CACHE/.complete" ] || { echo "[data] 대기 시간 초과 -> 직접 다운로드"; CACHE="$PWD/data_local"; download_data "$CACHE"; }
fi
DATA="$(find_data_dir "$CACHE")"
[ -n "$DATA" ] || { echo "[error] images/ 폴더를 찾지 못함"; exit 1; }
echo "[data] $DATA : $(ls "$DATA/images" | wc -l) images, $(ls "$DATA/labels" | wc -l) labels"

# ---------------------------------------------------------------- 실험 1개 실행
# MIG 18GB 메모리 기준 batch x accum (실질 batch 는 항상 16, lr 0.02)
#   paper 설정: B0~B2 = 16x1, B3 = 8x2, B4 = 4x4
#   code 설정(800px): 1x16 (4x4도 18GB 초과로 실패했음)
DONE=()
#   이름 뒤 _code : 공개 코드 그대로 설정
#   이름 뒤 _an   : 분석 출력 추가 (예측 통계 / 센서 끄고 채점 / MAGE 가중치 / 문턱값 낮춰 재채점 / AdaBN)
#   이름 뒤 _sN   : 학습 seed = N (기본 42, train/test 분할은 항상 42 고정)
run_one() {   # run_one <실험이름>
  local name="$1" setting=paper exp="$1" seed=42 analyze=()
  if [[ "$exp" == *_code ]]; then setting=code; exp="${exp%_code}"; fi
  if [[ "$exp" == *_an ]]; then analyze=(--analyze); exp="${exp%_an}"; fi
  if [[ "$exp" =~ ^(.+)_s([0-9]+)$ ]]; then exp="${BASH_REMATCH[1]}"; seed="${BASH_REMATCH[2]}"; fi
  local bb=mit_b1 modal=rgb,thermal,event subset=all stats=train
  case "$exp" in
    smoke)            ;;
    smoke_t7)         subset=day ;;   # 진단 기능 시험용 (낮만 학습 20스텝)
    t1_b0|t1_b1|t1_b2|t1_b3|t1_b4) bb="mit_${exp#t1_}" ;;
    t5_rgb_thermal)   modal=rgb,thermal ;;
    t5_thermal_event) modal=thermal,event ;;
    t5_rgb_event)     modal=rgb,event ;;
    # 센서 1개
    t5_rgb)           modal=rgb ;;
    t5_thermal)       modal=thermal ;;
    t5_event)         modal=event ;;
    t7_day)           subset=day ;;
    t7_day_thermal)   subset=day; modal=thermal ;;
    t7_night)         subset=night ;;
    # 정규화 통계를 학습 split 전체(7,800장)로 고정한 버전
    t7_day_fs)        subset=day;   stats=train_all ;;
    t7_night_fs)      subset=night; stats=train_all ;;
    *) echo "[error] 알 수 없는 실험: $name"; return 1 ;;
  esac
  local bs=16 acc=1
  if [ "$setting" = code ]; then bs=1; acc=16
  elif [ "$bb" = mit_b3 ]; then bs=8; acc=2
  elif [ "$bb" = mit_b4 ]; then bs=4; acc=4; fi
  local extra=()
  if [[ "$exp" == smoke* ]]; then extra=(--epochs 1 --max-steps 20 --eval-max 64 --stats-samples 50 --adabn-samples 64 --no-save); fi

  echo ""
  echo "################ START $name  ($(date '+%m-%d %H:%M'))  backbone=$bb modal=$modal train=$subset setting=$setting batch=${bs}x${acc} stats=$stats seed=$seed ${analyze[*]}"
  if python -u scripts/run_experiment.py --data "$DATA" --out "$OUT_ROOT" --name "$name" \
       --setting "$setting" --backbone "$bb" --modalities "$modal" --train-subset "$subset" --stats-from "$stats" \
       --batch-size "$bs" --accum "$acc" --seed "$seed" --log-every "$LOG_EVERY" "${extra[@]}" "${analyze[@]}"; then
    echo "################ END $name OK"
  else
    echo "################ END $name FAILED (다음 실험으로 진행)"
  fi
  DONE+=("$name")
}

# 데이터 검사 (GPU 불필요, 몇 분): 파일 형식 / 라벨 / 밝기 / 차-주변 대비 / 분할 누수
run_check() {
  echo ""
  echo "################ START data_check  ($(date '+%m-%d %H:%M'))"
  if python -u scripts/check_data.py --data "$DATA"; then
    echo "################ END data_check OK"
  else
    echo "################ END data_check FAILED (다음 실험으로 진행)"
  fi
}

# ---------------------------------------------------------------- 실험 목록 (묶음)
case "$EXP" in
  smoke)    LIST=(smoke smoke_code t1_b4) ; LOG_EVERY=5 ;;
  t1_small) LIST=(t1_b0 t1_b2) ;;
  t1_large) LIST=(t1_b3 t1_b4) ;;
  t5_all)   LIST=(t5_rgb_thermal t5_thermal_event t5_rgb_event) ;;
  t7_all)   LIST=(t7_day t7_night) ;;
  t7_all_fs) LIST=(t7_day_fs t7_night_fs) ;;
  # ---- 11/24 회의 후속 (이슈 5개로 나눠 제출)
  wk_seed_tri)  LIST=(t1_b1_s1 t1_b1_s2) ;;                   # 3센서 seed 반복
  wk_seed_rgbt) LIST=(t5_rgb_thermal_s1 t5_rgb_thermal_s2) ;; # RGB+열화상 seed 반복
  wk_single)    LIST=(t5_rgb t5_thermal t5_event) ;;          # 센서 1개
  wk_analysis)  LIST=(t1_b1_an) ;;                            # 3센서 분석 (모델 저장)
  wk_t7)        LIST=(t7_day_an t7_day_thermal) ;;            # Table 7 원인 확인
  smoke_an)     LIST=(smoke_an) ; LOG_EVERY=5 ;;              # 분석 기능 시험 (20스텝)
  # ---- 11/28 Table 7 0점대 원인 진단 (이슈 1개로 전부)
  #   data_check        : 데이터 검사 (형식/라벨/밝기/차-주변 대비/분할 누수), 몇 분
  #   t7_day_an         : 낮만 학습 -> 본 채점 + 센서 끄기 + 문턱값 0.001 재채점 + AdaBN(밤 사진으로 BN 재계산)
  #   t7_night_an       : 밤만 학습 -> 같은 진단 (AdaBN 은 낮 사진으로)
  #   t7_day_thermal_an : 낮만 + 열화상만 학습 -> 밤에서 열화상만으로 통하는지
  wk_t7_diag)   LIST=(data_check t7_day_an t7_night_an t7_day_thermal_an) ;;
  smoke_diag)   LIST=(data_check smoke_t7_an) ; LOG_EVERY=5 ;;  # 위 진단 기능 시험 (20스텝)
  data_check)   LIST=(data_check) ;;
  *)        LIST=("$EXP") ;;
esac
if [ "$EXP" = smoke ]; then
  # smoke 에서 B4 는 20스텝만 (메모리 확인용)
  run_one smoke; run_one smoke_code
  echo ""; echo "################ START smoke_b4 (20 steps)"
  python -u scripts/run_experiment.py --data "$DATA" --out "$OUT_ROOT" --name smoke_b4 --backbone mit_b4 \
    --batch-size 4 --accum 4 --epochs 1 --max-steps 20 --eval-max 64 --stats-samples 50 --no-save --log-every 5 \
    && echo "################ END smoke_b4 OK" || echo "################ END smoke_b4 FAILED"
else
  for e in "${LIST[@]}"; do
    if [ "$e" = data_check ]; then run_check; else run_one "$e"; fi
  done
fi

# ---------------------------------------------------------------- 요약표
echo ""
echo "================ SUMMARY (COCO mAP / mAP50, x100) ================"
python - "$OUT_ROOT" "${DONE[@]}" <<'PY'
import json, os, sys
root, names = sys.argv[1], sys.argv[2:]
print(f"{'experiment':22s} {'all':>14s} {'day':>14s} {'night':>14s}  hours  batch")
for n in names:
    p = os.path.join(root, n, 'results.json')
    if not os.path.exists(p):
        print(f"{n:22s}  (결과 없음 - 실패)"); continue
    r = json.load(open(p, encoding='utf-8'))
    f = lambda k: f"{r[k]['mAP']}/{r[k]['mAP50']}"
    print(f"{n:22s} {f('all'):>14s} {f('day'):>14s} {f('night'):>14s}  {r.get('train_hours', '?'):>5}  "
          f"{r.get('batch_size')}x{r.get('accum', 1)}" + ("  EVAL_ERROR" if 'eval_error' in r else ""))

# ---- Table 7 진단 요약 (--analyze 로 돌린 실험만)
rows = []
for n in names:
    p = os.path.join(root, n, 'results.json')
    if not os.path.exists(p):
        continue
    r = json.load(open(p, encoding='utf-8'))
    d = r.get('diag')
    if not d:
        continue
    def m(x, k):
        try:
            return f"{x[k]['mAP']}/{x[k]['mAP50']}"
        except Exception:
            return '-'
    def rc(x, k):
        try:
            return f"{x['match'][k]['recall50_any_score']}%"
        except Exception:
            return '-'
    lt, ab = d.get('low_thresh', {}), d.get('adabn', {})
    base = {k: r[k] for k in ['all', 'day', 'night']}
    base['match'] = r.get('match_default', {})
    rows.append((n, '기본(0.05)', base))
    rows.append((n, f"문턱값 {r.get('low_thresh_value', '')}", lt))
    if 'default_thresh' in ab:
        rows.append((n, 'AdaBN', ab['default_thresh']))
        rows.append((n, 'AdaBN+문턱값', ab['low_thresh']))
    for k, v in r.get('drop_eval', {}).items():
        rows.append((n, f'센서끄기 {k}', v))
if rows:
    print("")
    print("================ DIAG SUMMARY (mAP/mAP50 x100, recall50 = 점수 무관 IoU>=0.5 검출 비율) ================")
    print(f"{'experiment':20s} {'조건':16s} {'all':>12s} {'day':>12s} {'night':>12s} {'rec_day':>8s} {'rec_night':>9s}")
    for n, label, x in rows:
        if not isinstance(x, dict) or 'error' in x or 'skipped' in x:
            print(f"{n:20s} {label:16s}  {x.get('error') or x.get('skipped') if isinstance(x, dict) else x}")
            continue
        print(f"{n:20s} {label:16s} {m(x,'all'):>12s} {m(x,'day'):>12s} {m(x,'night'):>12s} "
              f"{rc(x,'day'):>8s} {rc(x,'night'):>9s}")
PY
echo "[done] $EXP"
