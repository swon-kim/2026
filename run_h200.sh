#!/usr/bin/env bash
# H200 (Aerodrone Jenkins) 실행용 진입점.
#   이슈 "실행 명령어" 예:  bash run_h200.sh smoke
#                          bash run_h200.sh t1_b1
#                          bash run_h200.sh t1_b1_code   (공개 코드 그대로 설정)
# 실험 목록은 아래 case 문 참고.
set -euo pipefail
cd "$(dirname "$0")"

EXP="${1:-smoke}"
DRIVE_FOLDER="https://drive.google.com/drive/folders/1U2NJQMRz8uP8e-z9lCguRWVPWpXs4MmB"

# ---------------------------------------------------------------- 출력 폴더
if mkdir -p /app/output 2>/dev/null && [ -w /app/output ]; then OUT_ROOT=/app/output; else OUT_ROOT="$PWD/outputs"; fi
mkdir -p "$OUT_ROOT"
echo "[info] experiment=$EXP  out=$OUT_ROOT  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[info] nvidia-smi 없음"

# ---------------------------------------------------------------- 패키지
python - <<'PY' || pip install -q torchmetrics pycocotools gdown scikit-learn
import importlib, sys
missing = [m for m in ("torchmetrics", "pycocotools", "gdown", "sklearn") if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
PY

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

# ---------------------------------------------------------------- 실험 목록
run() { python -u scripts/run_experiment.py --data "$DATA" --out "$OUT_ROOT" "$@"; }

SETTING=paper
NAME="$EXP"
if [[ "$EXP" == *_code ]]; then SETTING=code; EXP="${EXP%_code}"; fi

case "$EXP" in
  smoke)
    # 환경/데이터/속도/메모리 확인용 (각 20 스텝 + 평가 32장)
    S="--epochs 1 --max-steps 20 --eval-max 32 --log-every 5 --no-save --stats-samples 50"
    run --name smoke_b1_paper --setting paper --backbone mit_b1 $S
    run --name smoke_b1_code  --setting code  --backbone mit_b1 $S || echo "[smoke] code 설정 실패 (메모리 부족 가능)"
    run --name smoke_b4_paper --setting paper --backbone mit_b4 $S || echo "[smoke] b4 실패 (메모리 부족 가능)"
    ;;
  # Table 1: 백본 크기 (3모달)
  t1_b0|t1_b1|t1_b2|t1_b3|t1_b4)
    run --name "$NAME" --setting "$SETTING" --backbone "mit_${EXP#t1_}" ;;
  # Table 5: 센서 조합 (MiT-B1)
  t5_rgb_thermal)   run --name "$NAME" --setting "$SETTING" --modalities rgb,thermal ;;
  t5_thermal_event) run --name "$NAME" --setting "$SETTING" --modalities thermal,event ;;
  t5_rgb_event)     run --name "$NAME" --setting "$SETTING" --modalities rgb,event ;;
  # Table 7: 낮만 / 밤만 학습 (전체 학습 = t1_b1). 평가는 항상 All/Day/Night
  t7_day)   run --name "$NAME" --setting "$SETTING" --train-subset day ;;
  t7_night) run --name "$NAME" --setting "$SETTING" --train-subset night ;;
  *)
    echo "[error] 알 수 없는 실험: $1"
    echo "가능: smoke, t1_b0~t1_b4, t5_rgb_thermal, t5_thermal_event, t5_rgb_event, t7_day, t7_night (+ 뒤에 _code)"
    exit 1 ;;
esac
echo "[done] $NAME"
