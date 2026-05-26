#!/usr/bin/env bash
set -euo pipefail

# End-to-end weekend run:
# 1. Build/resume a 500-molecule QM9 PySCF NPZ dataset with KP references.
# 2. Train the transferable 1-RDM model on a 400/50/50 split.
#
# Override any variable from the shell if needed, e.g.
#   STEPS_PER_EPOCH=80 EPOCHS=600 bash scripts/run_qm9_ldavwn_weekend_500.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${PYTHON:=/home/hbji/miniconda3/envs/polymer-gp/bin/python}"
: "${QM9_TAR:=data/qm9_raw/dsgdb9nsd.xyz.tar.bz2}"

: "${NUM_SYSTEMS:=500}"
: "${TRAIN_SYSTEM_COUNT:=400}"
: "${VAL_SYSTEM_COUNT:=50}"
: "${TEST_SYSTEM_COUNT:=50}"

: "${MAX_ATOMS:=10}"
: "${BASIS:=6-31g(d)}"
: "${XC:=lda,vwn}"
: "${GRID_SPACING_BOHR:=1.5}"
: "${MAX_AXIS_POINTS:=21}"
: "${PADDING_BOHR:=4.0}"
: "${SEED:=0}"
: "${SELECTION:=random}"

: "${NPZ_DIR:=qmugs_npz/qm9_pyscf_ldavwn_b631gd_atoms10_400_50_50_spacing1p5_kp}"
: "${OUTPUT_ROOT:=transferable_outputs}"
: "${RUN_NAME:=qm9_ldavwn_b631gd_atoms9_400_50_50_spacing1p5_gamma8_kpLate_Tlate_w192_r12}"

: "${EPOCHS:=520}"
: "${BATCH_SIZE:=1024}"
: "${STEPS_PER_EPOCH:=60}"
: "${VAL_EVERY:=20}"
: "${LOG_EVERY:=1}"
: "${LR_PATIENCE:=80}"
: "${PATIENCE:=240}"
: "${EVAL_PAIR_COUNT:=8192}"
: "${FULL_EVAL_MAX_POINTS:=2500}"
: "${GAMMA_CACHE_GB:=1.0}"

: "${MODEL_WIDTH:=192}"
: "${LEARNED_RANK:=12}"
: "${RFF_FEATURES:=32}"
: "${OCC_MAX:=2.0}"

: "${LAMBDA_GAMMA:=8.0}"
: "${LAMBDA_RHO:=2.0}"
: "${LAMBDA_KERNEL:=1.0}"
: "${LAMBDA_KP:=0.75}"
: "${LAMBDA_KINETIC:=0.20}"
: "${KP_START_EPOCH:=100}"
: "${KP_RAMP_EPOCHS:=60}"
: "${KINETIC_START_EPOCH:=240}"
: "${KINETIC_RAMP_EPOCHS:=80}"

STAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}_${STAMP}"
mkdir -p "$TRAIN_OUTPUT_DIR"
PIPELINE_LOG="${TRAIN_OUTPUT_DIR}/pipeline.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "[Pipeline] root               : $ROOT_DIR"
echo "[Pipeline] python             : $PYTHON"
echo "[Pipeline] qm9 tar            : $QM9_TAR"
echo "[Pipeline] npz dir            : $NPZ_DIR"
echo "[Pipeline] output dir         : $TRAIN_OUTPUT_DIR"
echo "[Pipeline] split              : ${TRAIN_SYSTEM_COUNT}/${VAL_SYSTEM_COUNT}/${TEST_SYSTEM_COUNT}"
echo "[Pipeline] model              : width=${MODEL_WIDTH}, rank=${LEARNED_RANK}, rff=${RFF_FEATURES}"
echo "[Pipeline] schedule           : KP ${KP_START_EPOCH}+${KP_RAMP_EPOCHS}, T ${KINETIC_START_EPOCH}+${KINETIC_RAMP_EPOCHS}"
echo "[Pipeline] loss weights       : gamma=${LAMBDA_GAMMA}, rho=${LAMBDA_RHO}, kernel=${LAMBDA_KERNEL}, KP=${LAMBDA_KP}, T=${LAMBDA_KINETIC}"
echo "[Pipeline] training           : epochs=${EPOCHS}, steps/epoch=${STEPS_PER_EPOCH}, batch=${BATCH_SIZE}"

if [[ ! -f "$QM9_TAR" ]]; then
  echo "[Pipeline] ERROR: QM9 tar file not found: $QM9_TAR" >&2
  exit 1
fi

mkdir -p "$NPZ_DIR"
existing_npz="$(find "$NPZ_DIR" -maxdepth 1 -name '*.npz' | wc -l)"
if (( existing_npz < NUM_SYSTEMS )); then
  echo "[Pipeline] Building NPZ dataset: ${existing_npz}/${NUM_SYSTEMS} files currently present."
  "$PYTHON" scripts/build_qm9_pyscf_npz.py \
    --qm9-tar "$QM9_TAR" \
    --output-dir "$NPZ_DIR" \
    --num-systems "$NUM_SYSTEMS" \
    --max-atoms "$MAX_ATOMS" \
    --basis "$BASIS" \
    --xc "$XC" \
    --grid-spacing-bohr "$GRID_SPACING_BOHR" \
    --max-axis-points "$MAX_AXIS_POINTS" \
    --padding-bohr "$PADDING_BOHR" \
    --selection "$SELECTION" \
    --seed "$SEED"
else
  echo "[Pipeline] NPZ dataset already has ${existing_npz} files. Skipping build."
fi

final_npz="$(find "$NPZ_DIR" -maxdepth 1 -name '*.npz' | wc -l)"
required_npz=$((TRAIN_SYSTEM_COUNT + VAL_SYSTEM_COUNT + TEST_SYSTEM_COUNT))
if (( final_npz < required_npz )); then
  echo "[Pipeline] ERROR: only ${final_npz} NPZ files available, but ${required_npz} are required." >&2
  exit 1
fi

echo "[Pipeline] Starting training."
MPLCONFIGDIR=/tmp/mplconfig \
MPLBACKEND=Agg \
RDM_OUTPUT_DIR="$TRAIN_OUTPUT_DIR" \
RDM_LOG_EVERY="$LOG_EVERY" \
RDM_NORMALIZE_RHO=1 \
RDM_TAU_STENCIL=richardson \
RDM_LOSS_PRESET=custom \
RDM_USE_GAMMA_LOSS=1 \
RDM_USE_RHO_LOSS=1 \
RDM_USE_KERNEL_LOSS=1 \
RDM_USE_KP_LOSS=1 \
RDM_USE_KINETIC_LOSS=1 \
RDM_USE_TRACE_LOSS=0 \
RDM_USE_MODE_LOSS=0 \
RDM_USE_DERIV_LOSS=0 \
RDM_USE_TAU_LOSS=0 \
RDM_USE_OCC_LOSS=0 \
RDM_LAMBDA_GAMMA="$LAMBDA_GAMMA" \
RDM_LAMBDA_RHO="$LAMBDA_RHO" \
RDM_LAMBDA_KERNEL="$LAMBDA_KERNEL" \
RDM_LAMBDA_KP="$LAMBDA_KP" \
RDM_LAMBDA_KINETIC="$LAMBDA_KINETIC" \
RDM_KP_START_EPOCH="$KP_START_EPOCH" \
RDM_KP_RAMP_EPOCHS="$KP_RAMP_EPOCHS" \
RDM_KINETIC_START_EPOCH="$KINETIC_START_EPOCH" \
RDM_KINETIC_RAMP_EPOCHS="$KINETIC_RAMP_EPOCHS" \
RDM_OCC_MAX="$OCC_MAX" \
RDM_MODEL_WIDTH="$MODEL_WIDTH" \
RDM_LEARNED_RANK="$LEARNED_RANK" \
RDM_RFF_FEATURES="$RFF_FEATURES" \
RDM_STEPS_PER_EPOCH="$STEPS_PER_EPOCH" \
RDM_VAL_EVERY="$VAL_EVERY" \
RDM_LR_PATIENCE="$LR_PATIENCE" \
RDM_PATIENCE="$PATIENCE" \
RDM_EVAL_PAIR_COUNT="$EVAL_PAIR_COUNT" \
RDM_GAMMA_CACHE_GB="$GAMMA_CACHE_GB" \
RDM_FULL_EVAL_MAX_POINTS="$FULL_EVAL_MAX_POINTS" \
"$PYTHON" train_transferable_1rdm.py \
  --dataset-mode npz \
  --npz-glob "${NPZ_DIR}/*.npz" \
  --train-system-count "$TRAIN_SYSTEM_COUNT" \
  --val-system-count "$VAL_SYSTEM_COUNT" \
  --test-system-count "$TEST_SYSTEM_COUNT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --run-name "$RUN_NAME"

echo "[Pipeline] Done."
echo "[Pipeline] output dir         : $TRAIN_OUTPUT_DIR"
echo "[Pipeline] log                : $PIPELINE_LOG"
