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

It also reports kinetic-density and kinetic-energy diagnostics derived from the
predicted 1-RDM. The V1 point model does not directly predict kinetic potential.

## Current Scope

- Build compact QM9/PySCF DFT datasets as compressed NPZ files.
- Train a transferable 1-RDM model across many molecules.
- Split systems at the molecule level into train, validation, and test sets.
- Evaluate held-out molecules with gamma, density, kinetic-density, kinetic-energy,
  and kinetic-energy diagnostics.
- Save summary figures, JSON, CSV metrics, and model weights.

This is not a production-ready quantum chemistry package. It is a research codebase
for testing 1-RDM model structure, loss schedules, and data-generation choices.

## Model Summary

The surrogate is built from four neural networks.

### 1. Point Model

Input:

```text
[local point features, global molecular context]
```

Output:

```text
[rho_N logit]
```

With `--pair-density-feature-mode fukui`, the point model instead emits three
density logits for `rho_N`, `rho_(N+1)`, and `rho_(N-1)`. In the default
`learned` mode, each density is obtained with `softplus` and rescaled to its
required electron count. The optional SAD residual mode described below instead
applies a positive multiplicative correction to an atomic-density baseline. The point model is
pretrained and frozen before 1-RDM training. Fukui-mode pretraining is staged:
`rho_N` starts first, charged densities enter at epoch 30, and direct Fukui
losses enter as a small auxiliary term at epoch 60. The defaults can be adjusted
with `--point-charged-weight`, `--point-fukui-weight`,
`--point-charged-start-epoch`, `--point-fukui-start-epoch`, and
`--point-fukui-ramp-epochs`. The ramp avoids abruptly perturbing the shared point
trunk when the Fukui auxiliary objective becomes active.

For NPZ archives patched with SAD features, the point model can predict a positive
multiplicative correction to the normalized atomic-density baseline:

```bash
python train_transferable_1rdm.py ... \
  --density-baseline-mode sad-multiplicative \
  --point-fukui-ramp-epochs 40
```

The original direct density-head behavior remains available as
`--density-baseline-mode learned`.

### 2. Mode Model

Input:

```text
[local point features, global molecular context]
```

Output:

```text
latent residual-mode amplitudes
```

This encoder remains trainable during 1-RDM training so that freezing the density
model does not freeze residual-kernel expressivity.

### 3. Pair Model

Input:

```text
[pair features for (r, r'), optional predicted-density descriptors, global molecular context]
```

Output:

```text
[baseline kernel width, residual-kernel gate]
```

It controls the baseline Gaussian-like kernel and the mixing between baseline and
learned residual kernel.

### 4. Context Model

Input:

```text
global molecular context
```

Output:

```text
system-level latent mode weights
```

These weights condition the low-rank residual kernel on the molecule.

All four models are RFF-enhanced MLPs using SiLU activations.

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

### FD/Pseudopotential Reference Builder

For a self-consistent real-space reference, use the GPAW finite-difference
builder:

```text
scripts/build_qm9_gpaw_fd_npz.py
```

It runs GPAW in real-space FD mode, extracts occupied pseudo-wavefunctions on
the same uniform grid used by the model, and writes a lazy/factorized reference
by default:

- `psi_occ`: occupied pseudo-wavefunctions on the model grid;
- `occupancies`;
- on-the-fly `gamma_ij = sum_n f_n psi_n(r_i) psi_n(r_j)`;
- `rho_diag = diag(gamma_matrix)`;
- `tau_orbital_gradient` and `derivative_orbital_gradient`;
- central2 and Richardson gamma-stencil targets stored under explicit
  `tau_gamma_*` and `derivative_gamma_*` names;
- full-grid orbital, central2-interior orbital, central2 gamma, and Richardson
  gamma kinetic integrals under separate keys;
- loader-compatible `tau_true_ao` and `derivative_true_ao` aliases only for
  legacy readers.

The v2 schema is named `gpaw_fd_orbital_v2`. The recommended primary training
configuration is `--tau-stencil central2 --kinetic-reference orbital-interior`
with `--physics-target orbital`. This uses the same central difference and
interior domain for local tau and integrated kinetic loss. Full-grid orbital
and Richardson quantities remain diagnostics.

The dense `gamma_matrix` is not stored unless `--store-full-gamma` is passed.
This is required for fine grids such as `0.4` or `0.3` bohr, where dense gamma
would scale as `N_grid^2`.

GPAW and ASE must be installed in the active environment. A small dry run can
select molecules and estimate grid sizes without running SCF:

```bash
python scripts/build_qm9_gpaw_fd_npz.py \
  --npz-glob 'qmugs_npz/qm9_pyscf_ldavwn_b631gd_atoms10_400_50_50_spacing1p5_kp/*.npz' \
  --output-dir qmugs_npz/qm9_gpaw_fd_smoke \
  --num-systems 5 \
  --grid-spacing-bohr 0.8 \
  --padding-bohr 4.0 \
  --max-axis-points 25 \
  --dry-run
```

The first actual smoke dataset should use only a few small molecules:

```bash
python scripts/build_qm9_gpaw_fd_npz.py \
  --npz-glob 'qmugs_npz/qm9_pyscf_ldavwn_b631gd_atoms10_400_50_50_spacing1p5_kp/*.npz' \
  --output-dir qmugs_npz/qm9_gpaw_fd_smoke \
  --num-systems 5 \
  --grid-spacing-bohr 0.8 \
  --padding-bohr 4.0 \
  --max-axis-points 25 \
  --xc LDA
```

After the smoke test passes, a more serious FD dataset should use a finer
spacing and the default lazy gamma format:

```bash
python scripts/build_qm9_gpaw_fd_npz.py \
  --npz-glob 'qmugs_npz/qm9_pyscf_ldavwn_b631gd_atoms10_400_50_50_spacing1p5_kp/*.npz' \
  --output-dir qmugs_npz/qm9_gpaw_fd_h0p4 \
  --num-systems 50 \
  --grid-spacing-bohr 0.4 \
  --padding-bohr 4.0 \
  --max-axis-points 55 \
  --tau-stencil central2 \
  --kinetic-reference orbital-interior \
  --xc LDA
```

## Training

The main training entry point is:

```text
train_transferable_1rdm.py
```

For staged derivative and kinetic-density training, use
`RDM_LOSS_PRESET=staged-physics`. This keeps the integrated kinetic-energy loss
off, trains the core gamma/rho terms first, then ramps target-RMS-normalized
Huber losses for the derivative and tau fields:

```bash
RDM_LOSS_PRESET=staged-physics \
RDM_LAMBDA_DERIV=1.0e-4 \
RDM_LAMBDA_TAU=1.0e-4 \
RDM_DERIV_START_EPOCH=30 \
RDM_DERIV_RAMP_EPOCHS=40 \
RDM_TAU_START_EPOCH=30 \
RDM_TAU_RAMP_EPOCHS=40 \
RDM_USE_KINETIC_LOSS=0 \
RDM_GRADIENT_DIAGNOSTICS=1 \
RDM_GRADIENT_DIAGNOSTICS_EVERY=5 \
python train_transferable_1rdm.py ...
```

Gradient diagnostics use one fixed train-system batch and report raw and
schedule-weighted gradient norms for gamma, derivative, and tau losses. They
also split each norm across the point, mode, pair, and context models and print
target/prediction RMS values for derivative and tau fields.

The default kernel also includes a near-diagonal, axis-resolved local curvature
correction:

```text
K = K_base * ((1 - gate) + gate * K_residual) + K_local
```

`K_local` is zero on the exact diagonal, so the density/kernel diagonal
constraint is preserved, and is windowed by `exp(-|r-r'|^2 / sigma^2)` so it
only acts on short-range off-diagonal stencil pairs. The pair model emits
axis-specific coefficients `c_x, c_y, c_z`, and the correction uses the
quadratic form

```text
K_local = scale * window * (c_x s_x^2 + c_y s_y^2 + c_z s_z^2)
```

where `s_a^2` is the pair-feature separation square rescaled from
`Delta_a^2 / domain^2` to approximately `Delta_a^2 / h^2`. The local heads are
zero-initialized; training starts as the previous kernel and learns local
curvature only if the derivative/tau losses supply useful signal. Useful
controls:

```bash
RDM_USE_LOCAL_CURVATURE_KERNEL=1 \
RDM_LOCAL_CURVATURE_SCALE=0.5 \
RDM_LOCAL_CURVATURE_SIGMA=1.0 \
RDM_LOCAL_CURVATURE_BASIS_SCALE=0
```

`RDM_LOCAL_CURVATURE_BASIS_SCALE=0` uses the automatic per-system
`(max(abs(axis)) / grid_step)^2` conversion. Set it to a positive value only to
force one global conversion factor for all systems.

Recommended Fukui-feature training command for an existing 500-molecule charged
oracle NPZ dataset:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
MPLBACKEND=Agg \
RDM_NORMALIZE_RHO=1 \
RDM_TAU_STENCIL=richardson \
RDM_LOSS_PRESET=core5 \
RDM_LEARNING_RATE=2.0e-4 \
RDM_MIN_LR=1.0e-5 \
RDM_LR_DECAY=0.5 \
RDM_LAMBDA_GAMMA=12.0 \
RDM_LAMBDA_RHO=1.0 \
RDM_LAMBDA_KERNEL=1.0 \
RDM_LAMBDA_KINETIC=0.02 \
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
  --npz-glob 'qmugs_npz/qm9_pyscf_ldavwn_b61plusgd_atoms10_500_charged_oracle/*.npz' \
  --train-system-count 400 \
  --val-system-count 50 \
  --test-system-count 50 \
  --pair-density-feature-mode fukui \
  --point-pretrain-epochs 200 \
  --point-pretrain-steps-per-epoch 80 \
  --epochs 700 \
  --batch-size 1024 \
  --run-name qm9_ldavwn_b61plusgd_fukui_v1 \
  --auto-run-dir
```

`RDM_GAMMA_CACHE_GB` limits the process-local CPU RAM LRU for expanded neutral
`gamma_matrix` arrays. It does not allocate GPU memory, and charged density
oracles do not increase this limit. The runtime prints both the expanded corpus
estimate and the approximate frozen density GPU-cache size.

Use `--pair-density-feature-mode off` for the original pair input, or
`--pair-density-feature-mode rho-derivatives` to add predicted `rho_N`, gradient
norm, and Laplacian descriptors without predicting charged densities. Fukui mode
adds the same descriptors for `rho_N`, `f+ = rho_(N+1) - rho_N`, and
`f- = rho_N - rho_(N-1)`.

The charged-oracle NPZ files are needed to pretrain Fukui mode. Once those NPZ
files exist, changing among the three training modes does not require rebuilding
the dataset.

Existing NPZ files can be upgraded with the MINAO atomic-density input feature
and directional atom descriptors without rerunning DFT:

```bash
python scripts/patch_npz_features.py \
  --npz-glob 'qmugs_npz/qm9_pyscf_ldavwn_b631plusgd_atoms10_500_charged_oracle/*.npz'
```

The patcher writes a temporary archive and atomically replaces the original only
after compression completes. It also recovers complete `*.npz.tmp.npz` archives
left by older versions of the script.

For a controlled comparison, keep all other options fixed and run:

```bash
# Predicted rho_N only; no density descriptor enters the pair model.
python train_transferable_1rdm.py ... --pair-density-feature-mode off --run-name v1_off

# Predicted rho_N plus its gradient norm and Laplacian.
python train_transferable_1rdm.py ... --pair-density-feature-mode rho-derivatives --run-name v1_rho_derivatives

# Predicted rho_N, f+, f- plus their gradient norms and Laplacians.
python train_transferable_1rdm.py ... --pair-density-feature-mode fukui --run-name v1_fukui
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
modes, context conditioning, tau, and kinetic energy are coupled. The V2
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

Pair sampling can be fixed instead of using the default curriculum. The values
are ordered as `diag,near,mid,far`, and category weights are normalized to mean
one inside each batch:

```bash
python train_v2_ablation.py \
  --experiment gamma-simple \
  --dataset-mode npz \
  --npz-glob 'qmugs_npz/qm9_pyscf_ldavwn_b631gd_atoms10_400_50_50_spacing1p5_kp/*.npz' \
  --overfit-one-system \
  --pair-sampling-probs 0.25,0.25,0.25,0.25 \
  --pair-category-weights 1,1,1,1 \
  --epochs 300 \
  --steps-per-epoch 80 \
  --batch-size 1024 \
  --output-dir v2_outputs/overfit_gamma_simple_balanced \
  --run-name qm9_v2_overfit_gamma_simple_balanced
```

For an off-diagonal-heavy diagnostic, use `--pair-sampling-probs 0.10,0.20,0.30,0.40`
with the same `--pair-category-weights 1,1,1,1`.

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
    Point, mode, pair, and context model definitions. Implements RFF-enhanced MLPs and
    the gamma_theta prediction formula.

transferable_rdm/training.py
    Point-density pretraining, frozen-density 1-RDM training, scheduled loss
    weights, evaluation metrics, kinetic-energy diagnostics, and early stopping.

transferable_rdm/v2_ablation.py
    V2 ablation model classes, losses, baseline fitting, metrics, CSV/JSON output,
    and training loop.

transferable_rdm/plotting.py
    Summary plots for objectives, gamma parity, density slices, tau, and
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
  pair loss, density MAE, tau MAE, and kinetic-energy error.
- The code is intended for method development. Reported accuracy should be tied to
  a fixed dataset, split seed, basis, functional, grid spacing, and loss schedule.
