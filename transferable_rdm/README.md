# Transferable 1-RDM Surrogate

이 폴더는 기존 단일-system toy 스크립트를 다음 방향으로 확장한 버전이다.

- multi-system training
- held-out system validation
- KS-like synthetic system generator
- `sqrt(rho(r) rho(r')) K(r,r')` 구조
- Gaussian baseline kernel + low-rank residual kernel
- explicit near-diagonal derivative loss
- tau / trace / occupation penalty
- future DFT/KS 결과를 위한 NPZ loader

## 실행

기본 실행:

```bash
python train_transferable_1rdm.py
```

빠른 smoke test:

```bash
MPLBACKEND=Agg \
RDM_NUM_SYSTEMS=4 \
RDM_EPOCHS=2 \
RDM_STEPS_PER_EPOCH=2 \
RDM_BATCH_SIZE=128 \
RDM_AXIS_POINTS=5 \
python train_transferable_1rdm.py
```

1D, 2D, 3D toy는 각각 독립 실험으로 실행한다. 예를 들어 1D 400/50/50:

```bash
python train_transferable_1rdm.py \
  --dataset-mode toy \
  --toy-dimensions 1 \
  --num-systems 500 \
  --train-system-count 400 \
  --val-system-count 50 \
  --test-system-count 50 \
  --axis-points 9 \
  --pair-density-feature-mode rho-derivatives \
  --pair-density-hessian \
  --density-baseline-mode sad-multiplicative
```

2D와 3D는 각각 `--toy-dimensions 2`, `--toy-dimensions 3`으로 실행한다.
`--toy-dimensions 1,2,3`은 세 차원을 한 corpus에서 비교하는 추가 실험용이다.

1D와 2D toy는 현재 3D 모델과 동일한 입력 및 물리 loss를 사용하기 위해 3D
cubic grid에 임베딩된다. 활성 축에는 무작위 multi-well potential을 사용하고,
나머지 축은 고정 harmonic ground-state confinement를 사용한다. Toy split은
요청한 차원들이 train/validation/test에 가능한 한 균등하게 포함되도록 구성된다.

Toy generator와 feature adapter는 `transferable_rdm/toy/`에 분리되어 있고,
공통 `model.py`와 `training.py`는 QM9과 동일하게 사용한다. Toy feature schema는
QM9 patched NPZ와 같은 `local=32`, `global=11` 크기를 사용한다.

- 공통 물리 채널: 좌표, potential, potential gradient, potential Laplacian,
  radial distance, electron count
- QM9 원소별 Gaussian/vector 채널: toy potential-source Gaussian/vector로 대응
- 원자번호 채널: toy에는 원자가 없으므로 0
- SAD 대응 채널: 외부 potential의 정규화된 최저 product-orbital density
- global atom summary: source count, active dimension, source strength/radius summary

따라서 `rho-derivatives + Hessian` 설정에서는 QM9과 동일하게 point input 43,
pair input 43이 된다. 기존 QM9 NPZ 생성 및 로딩 경로는 toy adapter를 사용하지
않는다. Toy baseline은 target density 복사가 아니라 potential만으로 만든 초기
추정치이며, baseline 영향 자체를 제거하려면 `--density-baseline-mode learned`를
사용한다.

QMugs에서 변환한 NPZ subset을 쓰는 Phase 1 preset:

```bash
# Phase 1a: 300 train / 100 validation, axis_points=7
python train_transferable_1rdm.py \
  --phase phase1a \
  --npz-glob "qmugs_npz/phase1a/*.npz"

# Phase 1b: 800 train / 100 validation / 100 test, axis_points=7 by default
python train_transferable_1rdm.py \
  --phase phase1b \
  --npz-glob "qmugs_npz/phase1b/*.npz"

# Phase 1b를 9^3 grid로 만든 NPZ에 맞춰 실행하려면:
python train_transferable_1rdm.py \
  --phase phase1b \
  --axis-points 9 \
  --npz-glob "qmugs_npz/phase1b_axis9/*.npz"
```

Phase preset은 정확한 system split 개수를 강제한다. NPZ 파일 수가 부족하면 실행을 중단한다.

QM9/PySCF 변환기는 고정 `axis_points` 대신 목표 grid spacing을 받을 수 있다.
`--grid-spacing-bohr`를 지정하면 각 분자의 padded box 크기에 맞춰 `axis_points`가
자동으로 정해지고, 실제 spacing은 target 이하가 된다.

```bash
python scripts/build_qm9_pyscf_npz.py \
  --num-systems 114 \
  --max-atoms 8 \
  --selection smallest \
  --basis '6-31g(2df,p)' \
  --xc b3lyp \
  --grid-spacing-bohr 2.0 \
  --max-axis-points 15 \
  --output-dir qmugs_npz/qm9_pyscf_b631g2dfp_small_spacing2p0
```

## NPZ loader schema

`dataset_mode=npz` 또는 `dataset_mode=mixed`에서 NPZ 파일은 최소한 아래 key를 가져야 한다.

- `points`: `(n_points, 3)`
- `gamma_matrix`: `(n_points, n_points)`
- `local_features`: `(n_points, d_local)`
- `global_context`: `(d_global,)`

optional:

- `potential`: `(n_points, 1)`
- `grad_potential`: `(n_points, 3)`
- `occupancies`
- `orbital_energies`
- `electron_count`
- `axis_points`
- `grid_spacing_bohr`
- `box_length_bohr`

uniform Cartesian grid라고 가정하며, `points`로부터 axis를 복원한다.

## 파일 구조

Point density pretrain은 기본적으로 scaled MSE, integrated relative L1,
log-density MSE의 가중합을 사용한다. 부호가 있는 Fukui field에는 log 항을
적용하지 않는다. 각 항은 CLI 또는 대응하는 `RDM_POINT_DENSITY_*` 환경
변수로 조정할 수 있다.

```bash
python train_transferable_1rdm.py \
  --point-density-mse-weight 1.0 \
  --point-density-rel-l1-weight 0.25 \
  --point-density-log-weight 0.02 \
  --point-density-log-eps 1e-8
```

- `config.py`: 실험 설정
- `systems.py`: KS-like synthetic system 생성과 NPZ loader
- `data.py`: system split, pair feature, curriculum sampler
- `model.py`: baseline + residual kernel 모델
- `training.py`: 학습 / held-out system 평가
- `plotting.py`: 요약 figure
- `train_transferable_1rdm.py`: 실행 entry point
