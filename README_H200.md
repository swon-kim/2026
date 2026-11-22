# TriModalDet 재현 실험 (H200 실행용)

원본 코드: <https://github.com/radlab-sketch/trimodal-uav-det> (MIT License)
논문: *Tri-Modal Fusion Transformers for UAV-based Object Detection* (arXiv:2604.16630)

이 레포는 원본 코드를 **수정하지 않고**, H200(Aerodrone Jenkins)에서 논문 Table 1 / 5 / 7을
재현하기 위한 파일만 추가했다.

| 추가 파일 | 역할 |
|---|---|
| `run_h200.sh` | 이슈 "실행 명령어"로 쓰는 진입점. 데이터 다운로드 → 학습 → 평가 → 결과 출력 |
| `scripts/run_experiment.py` | 실험 1회 실행 (원본 모듈을 import 해서 사용) |
| `README_H200.md` | 이 문서 |

## 이슈 작성 값

| 항목 | 값 |
|---|---|
| GitHub 링크 | 이 레포 주소 |
| 실행 명령어 | `bash run_h200.sh <실험이름>` |
| 사용 이미지 | `kau/pytorch-master` |
| 사용 언어 | Bash |
| GPU 할당량 | `1` |
| 추가 모듈 | `torchmetrics pycocotools gdown` |

## 실험 이름

| 이름 | 논문 | 내용 |
|---|---|---|
| `smoke` | - | 환경·데이터·속도·메모리 확인 (B1 paper/code, B4 paper 각 20스텝) |
| `t1_b0` ~ `t1_b4` | Table 1 | 백본 크기, 3모달 |
| `t5_rgb_thermal`, `t5_thermal_event`, `t5_rgb_event` | Table 5 | 센서 조합, MiT-B1 |
| `t7_day`, `t7_night` | Table 7 | 낮만/밤만 학습, MiT-B1 (전체 학습 = `t1_b1`) |

이름 뒤에 `_code`를 붙이면 공개 코드 그대로 설정으로 실행한다 (예: `t1_b1_code`).

## 두 가지 설정

| | `paper` (기본) | `code` (`_code`) |
|---|---|---|
| 입력 스케일 | /255 (0~1) | 0~255 그대로 |
| 정규화 | RGB ImageNet, 열화상/이벤트는 학습셋 통계 | config.py 값 (T/E = 0.5/0.5) |
| 해상도 | 원본 301×391 | torchvision 기본 (짧은 변 800) |
| AMP | 사용 | 미사용 |

공통: 15 epoch, SGD(0.9, 1e-4), lr 0.02 @ batch 16, warmup 500 + cosine,
원본과 동일한 80/20 무작위 분할(seed 42), MAGE+BiTE, FPN + Faster R-CNN.

## 결과

- 마지막에 `=== RESULT ===` 아래로 All / Day / Night 의 COCO mAP, mAP50 (×100)을 출력한다.
- `RESULT_JSON` 줄에 전체 설정과 결과가 JSON으로 들어 있다.
- `/app/output/<실험이름>/` 에 `results.json`, `model.pth` 저장.

## 참고 (논문과 데이터의 차이)

- Day/Night 구분은 파일명 접두사로 한다: `frame_` = 밝은 영상(RGB 평균 123), `nframe_` = 어두운 영상(평균 21).
  논문은 "day 6,412 / night 4,077"로 적었지만, 6,412장인 쪽(`nframe_`)이 실제로는 어둡다.
- 논문은 Table 5의 2모달 모델 구현 방식을 밝히지 않았다. 여기서는 원본 ablation 코드처럼 빠진 채널을 0으로 둔다.
