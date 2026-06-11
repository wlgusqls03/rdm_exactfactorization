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

Reference density를 직접 넣어 density bottleneck을 제거하는 oracle 실험:

```bash
RDM_DENSITY_SOURCE=true \
python train_transferable_1rdm.py \
  --density-source true \
  --pair-density-feature-mode rho-derivatives \
  --pair-density-hessian
```

이 모드에서는 point-density pretrain을 건너뛰고 stored `rho_diag`와 동일한
finite-difference gradient/Laplacian/Hessian descriptor를 pair 및 stencil
예측에 사용한다. 따라서 density MAE는 0이어야 하며, 남는 gamma/tau/T 오차는
kernel 및 physics 학습 경로의 한계를 나타낸다. 기본값은 `predicted`다.

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

압축 NPZ corpus의 초기 로딩은 시스템 단위로 병렬화할 수 있다. 기본값은
기존과 같은 직렬 로딩이며, CPU와 저장장치 여유가 있으면 4 workers부터
사용한다.

```bash
RDM_NPZ_LOAD_WORKERS=4 \
python train_transferable_1rdm.py --dataset-mode npz ...
```

worker 수를 늘리면 압축 해제 시간은 줄 수 있지만 저장장치 종류와 파일 크기
분포에 따라 I/O 경합이 생길 수 있다. 500-system corpus에서는 `2`와 `4`의
최종 `rate`를 비교하고 더 빠른 값을 사용한다. worker 수만큼 일시적인 RAM
사용량도 증가한다.

반복 실험에서는 persistent mmap cache를 쓰는 편이 훨씬 빠르다. 첫 실행은
압축 NPZ를 `.npy` 파일로 추출하므로 기존 로딩과 비슷하거나 더 오래 걸릴 수
있지만, 이후 실행은 압축 해제 없이 cache를 memory-map한다.

```bash
RDM_NPZ_MMAP_CACHE=1 \
RDM_NPZ_MMAP_CACHE_DIR=qmugs_npz/.rdm_mmap_cache \
RDM_NPZ_LOAD_WORKERS=2 \
python train_transferable_1rdm.py --dataset-mode npz ...
```

cache에는 NPZ loader의 일반 필드와 각 시스템에서 계산한 정확한 axis/stencil
index가 저장된다. 서로 다른 시스템의 좌표를 같다고 가정하거나 공유하지
않는다. `gamma_matrix`와 `psi_occ`는 초기 cache 생성에서 제외되고 실제로
필요해지는 첫 시점에 개별적으로 추출된다. 원본 NPZ의 크기나 수정 시간이
바뀌면 해당 cache는 자동으로 다시 생성된다.

열린 파일 수가 시스템 수에 비례해 폭증하지 않도록 큰 `local_features`만
장기 memory-map하고, 나머지 cache 배열은 압축 없는 `.npy`에서 일반
로딩한다. 따라서 압축 해제 비용은 피하면서 per-array memmap의 file
descriptor 고갈도 방지한다.

학습 전에 cache만 구축할 수도 있다.

```bash
python build_npz_mmap_cache.py \
  --npz-glob 'qmugs_npz/qm9_gpaw_fd_h0p4_500_atoms10_axis56/*.npz' \
  --cache-dir qmugs_npz/.rdm_mmap_cache \
  --num-systems 500 \
  --workers 2
```

cache 구축이 끝난 뒤 학습에서는 `RDM_NPZ_LOAD_WORKERS=1`을 권장한다.

큰 3D grid에서는 validation과 종료 summary도 오래 걸린다. 아래 설정은
주기 validation을 고정된 10-system subset으로 제한하고, 종료 시 train은
20개만 다시 평가하면서 val/test는 전체 50개를 유지한다.

```bash
RDM_VAL_EVAL_SYSTEM_COUNT=10 \
RDM_FINAL_TRAIN_EVAL_SYSTEM_COUNT=20 \
RDM_FINAL_VAL_EVAL_SYSTEM_COUNT=0 \
RDM_FINAL_TEST_EVAL_SYSTEM_COUNT=0 \
RDM_EVAL_PAIR_COUNT=8192 \
RDM_EVAL_STENCIL_CENTERS=1024 \
RDM_EVAL_FULL_FINAL=0 \
python train_transferable_1rdm.py --dataset-mode npz ...
```

각 system-count 설정에서 `0`은 전체 split을 뜻한다. 빠른 sanity run에서는
final val/test도 `10` 또는 `20`으로 제한할 수 있지만, 최종 비교 실험에서는
val/test 전체 평가를 유지하는 편이 낫다.

kinetic loss가 활성화되고 stencil center 일부만 계산할 때는 기본적으로
정확한 target tau 적분에 sampling된 `tau_pred - tau_target` 적분을 더하는
control-variate estimator를 사용한다. raw `tau_pred` sample 합을 전체
grid로 확대하는 방식보다 분산이 작다.

```bash
RDM_KINETIC_CONTROL_VARIATE=1 \
python train_transferable_1rdm.py ...
```

`RDM_KINETIC_CONTROL_VARIATE=0`은 이전 raw sampled-sum estimator를
복원한다. Gradient diagnostics에는 kinetic gradient와 stored/physics-target/AO
kinetic integral이 함께 출력된다.

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
## GPAW NPZ storage profiles

New GPAW datasets default to `--storage-profile training-compact`. This keeps
the orbital factors, density and physics targets, model inputs, geometry,
scalar diagnostics, and reference metadata required for production training.
It omits duplicate legacy aliases and full gamma-derived diagnostic grids.

Use `--storage-profile full` only when the per-point central2/Richardson
diagnostic arrays are needed. Existing full datasets can be converted without
rerunning GPAW:

```bash
python scripts/compact_gpaw_training_npz.py \
  --input-dir qmugs_npz/qm9_gpaw_fd_orbital_v2_500_atoms10_h0p4 \
  --output-dir qmugs_npz/qm9_gpaw_fd_orbital_v2_500_atoms10_h0p4_compact
```
