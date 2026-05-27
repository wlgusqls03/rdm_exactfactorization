#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  else
    echo "[V2] ERROR: no python executable found. Set PYTHON=/path/to/python." >&2
    exit 1
  fi
fi

NPZ_GLOB="${NPZ_GLOB:-qmugs_npz/qm9_pyscf_ldavwn_b631gd_atoms10_400_50_50_spacing1p5_kp/*.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-v2_outputs}"
RUN_PREFIX="${RUN_PREFIX:-qm9_ldavwn_v2}"
EXPERIMENTS="${EXPERIMENTS:-baseline rho-only k-only gamma-only}"

TRAIN_SYSTEM_COUNT="${TRAIN_SYSTEM_COUNT:-400}"
VAL_SYSTEM_COUNT="${VAL_SYSTEM_COUNT:-50}"
TEST_SYSTEM_COUNT="${TEST_SYSTEM_COUNT:-50}"

EPOCHS="${EPOCHS:-120}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-40}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EVAL_PAIR_COUNT="${EVAL_PAIR_COUNT:-8192}"
VAL_EVERY="${VAL_EVERY:-10}"
LOG_EVERY="${LOG_EVERY:-1}"
LR="${LR:-3e-4}"

WIDTH="${WIDTH:-128}"
DEPTH="${DEPTH:-2}"
RANK="${RANK:-8}"
RFF_FEATURES="${RFF_FEATURES:-16}"
PAIR_SAMPLING_PROBS="${PAIR_SAMPLING_PROBS:-}"
PAIR_CATEGORY_WEIGHTS="${PAIR_CATEGORY_WEIGHTS:-20,8,4,1}"

echo "[V2] root        : ${ROOT_DIR}"
echo "[V2] python      : ${PYTHON}"
echo "[V2] npz glob    : ${NPZ_GLOB}"
echo "[V2] output root : ${OUTPUT_ROOT}"
echo "[V2] experiments : ${EXPERIMENTS}"
echo "[V2] pair probs  : ${PAIR_SAMPLING_PROBS:-curriculum}"
echo "[V2] pair weights: ${PAIR_CATEGORY_WEIGHTS}"

for EXPERIMENT in ${EXPERIMENTS}; do
  RUN_NAME="${RUN_PREFIX}_${EXPERIMENT}"
  OUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT}"
  EXTRA_ARGS=(--pair-category-weights "${PAIR_CATEGORY_WEIGHTS}")
  if [[ -n "${PAIR_SAMPLING_PROBS}" ]]; then
    EXTRA_ARGS+=(--pair-sampling-probs "${PAIR_SAMPLING_PROBS}")
  fi
  echo
  echo "[V2] Running ${EXPERIMENT} -> ${OUT_DIR}"
  "${PYTHON}" train_v2_ablation.py \
    --experiment "${EXPERIMENT}" \
    --dataset-mode npz \
    --npz-glob "${NPZ_GLOB}" \
    --train-system-count "${TRAIN_SYSTEM_COUNT}" \
    --val-system-count "${VAL_SYSTEM_COUNT}" \
    --test-system-count "${TEST_SYSTEM_COUNT}" \
    --output-dir "${OUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --epochs "${EPOCHS}" \
    --steps-per-epoch "${STEPS_PER_EPOCH}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-pair-count "${EVAL_PAIR_COUNT}" \
    --val-every "${VAL_EVERY}" \
    --log-every "${LOG_EVERY}" \
    --learning-rate "${LR}" \
    --width "${WIDTH}" \
    --depth "${DEPTH}" \
    --rank "${RANK}" \
    --rff-features "${RFF_FEATURES}" \
    "${EXTRA_ARGS[@]}"
done
