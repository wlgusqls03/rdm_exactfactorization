#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${PYTHON:=python}"
: "${BASE_RUN_DIR:=$ROOT_DIR/transferable_outputs/gpaw_h0p3_500_true_rho_central2_rank32_20260612_214602}"
: "${BASE_RUN_NAME:=gpaw_h0p3_500_true_rho_central2_rank32}"
: "${MMAP_CACHE_DIR:=$ROOT_DIR/qmugs_npz/.rdm_mmap_cache_h0p3_compact}"
: "${LAMBDA_KINETIC:=0.005}"
: "${FINETUNE_EPOCHS:=40}"
: "${FINETUNE_STEPS_PER_EPOCH:=200}"
: "${TRAIN_STENCIL_CENTERS:=4096}"
: "${FINETUNE_LR:=1e-5}"
: "${FINETUNE_MIN_LR:=1e-6}"
: "${KINETIC_RAMP_EPOCHS:=20}"

SOURCE_SUMMARY="$BASE_RUN_DIR/${BASE_RUN_NAME}_summary.json"
if [[ ! -f "$SOURCE_SUMMARY" ]]; then
  echo "ERROR: missing source summary: $SOURCE_SUMMARY" >&2
  exit 1
fi

for component in point mode pair context; do
  checkpoint="$BASE_RUN_DIR/${BASE_RUN_NAME}_${component}.weights.h5"
  if [[ ! -f "$checkpoint" ]]; then
    echo "ERROR: missing source checkpoint: $checkpoint" >&2
    exit 1
  fi
done

lambda_tag="${LAMBDA_KINETIC//./p}"
lambda_tag="${lambda_tag//-/m}"
RUN_NAME="${BASE_RUN_NAME}_kinetic_ft_lam${lambda_tag}"
LOG_PATH="$ROOT_DIR/logs/${RUN_NAME}.log"
mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/transferable_outputs"

echo "[Fine-tune] source summary  : $SOURCE_SUMMARY"
echo "[Fine-tune] run name        : $RUN_NAME"
echo "[Fine-tune] lambda kinetic  : $LAMBDA_KINETIC"
echo "[Fine-tune] epochs/steps    : $FINETUNE_EPOCHS / $FINETUNE_STEPS_PER_EPOCH"
echo "[Fine-tune] stencil centers : $TRAIN_STENCIL_CENTERS"
echo "[Fine-tune] learning rate   : $FINETUNE_LR"
echo "[Fine-tune] log             : $LOG_PATH"

MPLCONFIGDIR=/tmp/mplconfig \
MPLBACKEND=Agg \
TF_GPU_ALLOCATOR=cuda_malloc_async \
RDM_NPZ_MMAP_CACHE=1 \
RDM_NPZ_MMAP_CACHE_DIR="$MMAP_CACHE_DIR" \
RDM_NPZ_LOAD_WORKERS=1 \
RDM_LAZY_PSI_OCC=1 \
RDM_PSI_OCC_CACHE_GB=4 \
"$PYTHON" -u train_transferable_1rdm.py \
  --resume-summary-json "$SOURCE_SUMMARY" \
  --loss-preset staged-physics-kinetic \
  --use-kinetic-loss \
  --lambda-kinetic "$LAMBDA_KINETIC" \
  --deriv-start-epoch 0 \
  --deriv-ramp-epochs 0 \
  --tau-start-epoch 0 \
  --tau-ramp-epochs 0 \
  --kinetic-start-epoch 0 \
  --kinetic-ramp-epochs "$KINETIC_RAMP_EPOCHS" \
  --train-stencil-centers "$TRAIN_STENCIL_CENTERS" \
  --learning-rate "$FINETUNE_LR" \
  --min-learning-rate "$FINETUNE_MIN_LR" \
  --epochs "$FINETUNE_EPOCHS" \
  --steps-per-epoch "$FINETUNE_STEPS_PER_EPOCH" \
  --output-dir "$ROOT_DIR/transferable_outputs" \
  --run-name "$RUN_NAME" \
  --auto-run-dir \
  2>&1 | tee "$LOG_PATH"
