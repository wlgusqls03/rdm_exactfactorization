# Transferable 1-RDM Surrogate

This repository contains a research prototype for learning a transferable real-space
one-particle reduced density matrix (1-RDM) surrogate from DFT-derived molecular data.

The main target is the spin-summed closed-shell 1-RDM on a Cartesian grid:

```text
gamma(r_g, r_h) = sum_mu sum_nu chi_mu(r_g) P_mu,nu chi_nu(r_h)
```

where `P` is the AO density matrix from a PySCF Kohn-Sham DFT calculation and
`chi_mu(r)` are AO basis functions evaluated on the grid.

The learned model predicts

```text
gamma_theta(r, r') = sqrt(rho_theta(r) rho_theta(r')) K_theta(r, r')
```

with an optional density normalization constraint:

```text
int rho_theta(r) dr = N_e
```

It also supports auxiliary learning targets for the kinetic potential and kinetic
energy diagnostics.

## Current Scope

- Build compact QM9/PySCF DFT datasets as compressed NPZ files.
- Train a transferable 1-RDM model across many molecules.
- Split systems at the molecule level into train, validation, and test sets.
- Evaluate held-out molecules with gamma, density, kinetic-density, kinetic-energy,
  and kinetic-potential diagnostics.
- Save summary figures, JSON, CSV metrics, and model weights.

This is not a production-ready quantum chemistry package. It is a research codebase
for testing 1-RDM model structure, loss schedules, and data-generation choices.

## Model Summary

The surrogate is built from three neural networks.

### 1. Point Model

Input:

```text
[local point features, global molecular context]
```

Output:

```text
[density logit, kinetic-potential head, latent mode amplitudes]
```

The density is obtained with `softplus`; if density normalization is enabled, it is
rescaled so that the grid integral equals the electron count.

### 2. Pair Model

Input:

```text
[pair features for (r, r'), global molecular context]
```

Output:

```text
[baseline kernel width, residual-kernel gate]
```

It controls the baseline Gaussian-like kernel and the mixing between baseline and
learned residual kernel.

### 3. Context Model

Input:

```text
global molecular context
```

Output:

```text
system-level latent mode weights
```

These weights condition the low-rank residual kernel on the molecule.

All three models are RFF-enhanced MLPs using SiLU activations.

## Data Generation

The primary dataset builder is:

```text
scripts/build_qm9_pyscf_npz.py
```

It reads QM9 XYZ records, runs PySCF DFT, evaluates AO quantities on a real-space
grid, and writes one compressed NPZ file per molecule.

Stored targets include:

- `gamma_matrix`: real-space 1-RDM values on the grid.
- `tau_true_ao`: AO-gradient kinetic energy density.
- `derivative_true_ao`: directional gradient-product components.
- `kinetic_energy_hartree`: DFT Kohn-Sham kinetic energy from AO integrals.
- `kinetic_potential_centered`: centered kinetic potential reference.
- `local_features`: point descriptors used by the point model.
- `global_context`: molecule-level descriptors.
- molecule metadata such as formula, atom symbols, SMILES, grid spacing, and electron count.

For LDA-like local functionals, the kinetic-potential target is built from

```text
v_Ts(r) = mu - v_s(r)
```

where `v_s = v_ext + v_H + v_xc`. For nonlocal or hybrid functionals, the stored
scalar XC quantity should be treated as an approximation or diagnostic, not as a
rigorous local KS potential.

Example dataset build. Use the Python interpreter from the environment where
PySCF is installed.

```bash
python scripts/build_qm9_pyscf_npz.py \
  --qm9-tar data/qm9_raw/dsgdb9nsd.xyz.tar.bz2 \
  --output-dir qmugs_npz/qm9_pyscf_ldavwn_b631gd_atoms10_400_50_50_spacing1p5_kp \
  --num-systems 500 \
  --max-atoms 10 \
  --selection random \
  --basis '6-31g(d)' \
  --xc lda,vwn \
  --grid-spacing-bohr 1.5 \
  --max-axis-points 21 \
  --grid-level 1
```

## Training

The main training entry point is:

```text
train_transferable_1rdm.py
```

Recommended training command for an existing 500-molecule NPZ dataset:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
MPLBACKEND=Agg \
RDM_NORMALIZE_RHO=1 \
RDM_TAU_STENCIL=richardson \
RDM_LOSS_PRESET=core7 \
RDM_LEARNING_RATE=2.0e-4 \
RDM_MIN_LR=1.0e-5 \
RDM_LR_DECAY=0.5 \
RDM_LAMBDA_GAMMA=12.0 \
RDM_LAMBDA_RHO=1.0 \
RDM_LAMBDA_KERNEL=1.0 \
RDM_LAMBDA_KP=0.25 \
RDM_LAMBDA_KINETIC=0.02 \
RDM_KP_START_EPOCH=160 \
RDM_KP_RAMP_EPOCHS=160 \
RDM_KINETIC_START_EPOCH=460 \
RDM_KINETIC_RAMP_EPOCHS=180 \
RDM_OCC_MAX=2.0 \
RDM_MODEL_WIDTH=192 \
RDM_LEARNED_RANK=16 \
RDM_RFF_FEATURES=48 \
RDM_STEPS_PER_EPOCH=80 \
RDM_VAL_EVERY=20 \
RDM_LOG_EVERY=1 \
RDM_LR_PATIENCE=60 \
RDM_PATIENCE=260 \
RDM_EVAL_PAIR_COUNT=8192 \
RDM_GAMMA_CACHE_GB=1.0 \
RDM_FULL_EVAL_MAX_POINTS=2500 \
python train_transferable_1rdm.py \
  --dataset-mode npz \
  --npz-glob 'qmugs_npz/qm9_pyscf_ldavwn_b631gd_atoms10_400_50_50_spacing1p5_kp/*.npz' \
  --train-system-count 400 \
  --val-system-count 50 \
  --test-system-count 50 \
  --epochs 700 \
  --batch-size 1024 \
  --run-name qm9_ldavwn_b631gd_atoms10_400_50_50_spacing1p5_gamma12_kp025_Tlate_lr2e4_w192_r16 \
  --auto-run-dir
```

Learning rate can be set with any of these equivalent variables:

```text
RDM_LEARNING_RATE
RDM_INITIAL_LR
RDM_LR
```

Device selection is controlled by environment variables before TensorFlow is
imported:

```bash
# Default: let TensorFlow use any visible GPU, otherwise CPU.
RDM_DEVICE=auto

# Force CPU.
RDM_DEVICE=cpu

# Request GPU 0.
RDM_DEVICE=gpu RDM_GPU_IDS=0
```

`RDM_GPU_MEMORY_GROWTH=1` is the default and asks TensorFlow not to allocate all
GPU memory at startup. A CUDA-capable TensorFlow installation and working NVIDIA
driver are still required; if TensorFlow cannot see a GPU, the script prints a
warning and the run falls back to CPU.

## V2 Ablation Runner

The full prototype can become hard to diagnose because density, kernel, residual
modes, context conditioning, tau, KP, and kinetic energy are all coupled. The V2
runner keeps the same NPZ data and system-level splits, but resets the model and
loss structure into small ablation experiments.

Entry point:

```text
train_v2_ablation.py
```

Implemented experiments:

- `baseline`: no ML. Fits one scalar `alpha` for
  `gamma_base(r,r') = sqrt(rho_true(r) rho_true(r')) exp(-alpha |r-r'|^2)`.
- `rho-only`: point model only. Trains `rho_theta(r)` with density and trace loss.
- `k-only`: pair model only. Uses true density and trains `K_theta(r,r')`.
- `gamma-only`: point + pair model, trained only through sampled gamma loss.
- `gamma-simple`: point + pair model, trained with gamma + rho + trace losses.
- `gamma-residual`: adds a low-rank residual kernel.
- `gamma-context`: adds a context model that gates residual modes by molecule.

Run the four first diagnostic experiments on an existing NPZ dataset:

```bash
NPZ_GLOB='qmugs_npz/qm9_pyscf_ldavwn_b631gd_atoms10_400_50_50_spacing1p5_kp/*.npz' \
OUTPUT_ROOT=v2_outputs \
TRAIN_SYSTEM_COUNT=400 \
VAL_SYSTEM_COUNT=50 \
TEST_SYSTEM_COUNT=50 \
EPOCHS=120 \
STEPS_PER_EPOCH=40 \
BATCH_SIZE=1024 \
bash scripts/run_v2_ablation_suite.sh
```

Run one experiment directly:

```bash
python train_v2_ablation.py \
  --experiment k-only \
  --dataset-mode npz \
  --npz-glob 'qmugs_npz/qm9_pyscf_ldavwn_b631gd_atoms10_400_50_50_spacing1p5_kp/*.npz' \
  --train-system-count 400 \
  --val-system-count 50 \
  --test-system-count 50 \
  --epochs 120 \
  --steps-per-epoch 40 \
  --batch-size 1024 \
  --output-dir v2_outputs/k-only \
  --run-name qm9_v2_k_only
```

For capacity debugging, force train/validation/test to be the same molecule:

```bash
python train_v2_ablation.py \
  --experiment gamma-simple \
  --dataset-mode npz \
  --npz-glob 'qmugs_npz/qm9_pyscf_ldavwn_b631gd_atoms10_400_50_50_spacing1p5_kp/*.npz' \
  --overfit-one-system \
  --overfit-system-index 0 \
  --epochs 300 \
  --steps-per-epoch 80 \
  --batch-size 1024 \
  --output-dir v2_outputs/overfit_gamma_simple \
  --run-name qm9_v2_overfit_gamma_simple
```

If a one-molecule run cannot drive the training loss close to zero, the issue is
likely model capacity, feature scaling, kernel parameterization, or target scaling
rather than generalization.

Each V2 run writes `<run>_history.csv`, `<run>_per_system_metrics.csv`,
`<run>_summary.json`, and model weights when the experiment has trainable models.

## Losses

The main losses are:

- `gamma`: sampled pair loss for `gamma(r, r')`.
- `rho`: diagonal density loss.
- `kernel`: regularizes the kernel diagonal.
- `trace`: electron-count consistency through the grid trace.
- `kinetic`: scalar kinetic energy loss from predicted tau.
- `kp`: density-weighted centered kinetic-potential loss.
- `mode`: weak regularization of latent amplitudes.

For long runs, kinetic-potential and kinetic-energy losses can be staged in with
epoch schedules. This avoids forcing difficult auxiliary targets before the gamma
and density fit has stabilized.

## Outputs

Each run writes artifacts to `transferable_outputs/` or to a timestamped subdirectory
when `--auto-run-dir` is used.

Typical outputs:

- `<run>.png`: summary figure.
- `<run>_summary.json`: compact run summary.
- `<run>_history.csv`: epoch-level training and validation metrics.
- `<run>_split_metrics.csv`: averaged train/validation/test metrics.
- `<run>_per_system_metrics.csv`: molecule-by-molecule metrics.
- `<run>_point.weights.h5`
- `<run>_pair.weights.h5`
- `<run>_context.weights.h5`

For fixed output directories, previous results can be rotated automatically:

```bash
RDM_OUTPUT_DIR=result \
RDM_ROTATE_OUTPUT_DIR=1 \
RDM_OUTPUT_ROTATION_DEPTH=2 \
python train_transferable_1rdm.py ... --no-auto-run-dir
```

With depth `2`, an existing `old_old_result` is deleted, `old_result` is moved to
`old_old_result`, and `result` is moved to `old_result` before the new run writes
fresh outputs. Do not combine this with `--auto-run-dir` unless you intentionally
want to rotate timestamped directories.

## Code Layout

```text
train_transferable_1rdm.py
    Main training script. Builds the corpus, splits systems, constructs models,
    trains, evaluates, plots, and saves artifacts.

train_v2_ablation.py
    Separate ablation-first runner for baseline, rho-only, K-only, gamma-only,
    residual, and context experiments.

scripts/build_qm9_pyscf_npz.py
    QM9 to PySCF-DFT NPZ converter. Computes gamma, density, tau, kinetic energy,
    kinetic potential, features, and metadata.

scripts/run_qm9_ldavwn_weekend_500.sh
    Convenience pipeline for building a 500-molecule LDA/VWN QM9 subset and
    starting a long training run.

scripts/run_v2_ablation_suite.sh
    Convenience runner for the first V2 diagnostic experiments.

scripts/render_project_note_pdf.py
    Helper for rendering the project note PDF. Not needed for model training.

transferable_rdm/config.py
    Experiment configuration and environment-variable parsing.

transferable_rdm/systems.py
    SystemRecord definition, synthetic KS-like systems, NPZ loading, gamma cache,
    grid utilities, and reference target preparation.

transferable_rdm/data.py
    System-level train/validation/test splits, pair-feature construction, and
    category-balanced pair sampling.

transferable_rdm/model.py
    Point, pair, and context model definitions. Implements RFF-enhanced MLPs and
    the gamma_theta prediction formula.

transferable_rdm/training.py
    Training loop, scheduled loss weights, evaluation metrics, kinetic-potential
    loss, kinetic-energy diagnostics, and early stopping.

transferable_rdm/v2_ablation.py
    V2 ablation model classes, losses, baseline fitting, metrics, CSV/JSON output,
    and training loop.

transferable_rdm/plotting.py
    Summary plots for objectives, gamma parity, density slices, tau, KP, and
    kinetic-energy ratios.

transferable_rdm/utils.py
    Small utility functions for seeding, JSON writing, grid indexing, and logging.
```

Root-level `3D_1rdm_*.py` files are early exploratory prototypes and are not required
for the current transferable NPZ training pipeline.

## Dependencies

Core Python dependencies:

```text
numpy
tensorflow
matplotlib
pyscf
```

PySCF is required only for building DFT NPZ datasets. Training from existing NPZ
files does not run new quantum chemistry calculations.

## Files Usually Not Committed

The following are generated artifacts or large local datasets and should usually
be excluded from GitHub:

```text
data/
qmugs_npz/
transferable_outputs/
__pycache__/
*.npz
*.weights.h5
*.log
*.aux
*.out
*.pdf
*.png
```

Keep small source files, scripts, and documentation under version control. Store
large QM9 archives, NPZ datasets, trained weights, and run outputs outside the Git
repository or with a dedicated large-file workflow.

## Notes

- Validation and test splits are done at the molecule level.
- Pair batches are sampled stochastically; training objectives can fluctuate from
  epoch to epoch.
- Lower validation objective alone is not enough. Inspect component metrics such as
  pair loss, density MAE, KP loss, and kinetic-energy error.
- The code is intended for method development. Reported accuracy should be tied to
  a fixed dataset, split seed, basis, functional, grid spacing, and loss schedule.
