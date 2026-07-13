# Nuclear 1-RDM toy experiments

This directory is a small experiment layer for testing transferable nuclear
1-RDM kinetic-energy prediction across particle masses.  It deliberately keeps
the existing `rdm_exactfactorization` implementation unchanged and calls its
validated toy generator and training engine.

The two central experiment modes are:

- `oracle`: use the exact density and test the off-diagonal 1-RDM/kinetic path.
- `predicted`: predict density from the potential and molecular context, then
  use that density in the 1-RDM model.  This is the test that removes the
  true-density assumption.

## Quick start

Print a smoke-test command without running it:

```bash
python nuclear_rdm_toy/run_experiment.py \
  --particle proton --density oracle --profile small --smoke --dry-run
```

Run it:

```bash
python nuclear_rdm_toy/run_experiment.py \
  --particle proton --density oracle --profile small --smoke
```

If the active environment is not the TensorFlow environment used for the
original mass tests, select it explicitly:

```bash
python nuclear_rdm_toy/run_experiment.py \
  --python /path/to/tensorflow-env/bin/python \
  --particle proton --density oracle --profile small --smoke
```

Run the same configuration without the true density:

```bash
python nuclear_rdm_toy/run_experiment.py \
  --particle proton --density predicted --profile small --smoke
```

Production-style proton run:

```bash
python nuclear_rdm_toy/run_experiment.py \
  --particle proton --density predicted --profile medium \
  --axis-points 31 --num-systems 500 --epochs 200
```

Outputs are written below `nuclear_rdm_toy/outputs/`; every run name records
particle, mass, density source, model profile, and grid size.

## Experiment matrix

Generate or execute matched oracle/predicted and model-size comparisons:

```bash
python nuclear_rdm_toy/run_matrix.py --suite core --dry-run
python nuclear_rdm_toy/run_matrix.py --suite core
```

Available suites:

- `core`: electron, muon, and proton; oracle and predicted; small model.
- `lightweight`: proton oracle/predicted with baseline, medium, small, and tiny models.
- `grid`: proton oracle/predicted at 15, 31, 51, and 71 points per axis.
- `all`: all unique runs from the three suites.

Summarize completed runs:

```bash
python nuclear_rdm_toy/summarize.py
```

## Model profiles

| profile | width | rank | point depth | pair depth | RFF |
|---|---:|---:|---:|---:|---:|
| baseline | 192 | 16 | 3 | 2 | 32 |
| medium | 96 | 4 | 2 | 2 | 16 |
| small | 64 | 2 | 2 | 1 | 16 |
| tiny | 32 | 1 | 1 | 1 | 8 |

The baseline reproduces the scale of the existing mass test.  The smaller
profiles test whether the nearly rank-one heavy-particle results need the full
model.

## Interpretation requirements

Do not judge a model from integrated kinetic energy alone.  Compare gamma MAE,
near-diagonal MAE, tau error, kinetic-energy MAE, and trace error together.
Heavy particles can collapse to one voxel on a coarse grid, so a proton result
must also pass the grid suite before it is treated as physically resolved.

The `predicted` experiment is intentionally harder.  A fair progression is:

1. reproduce the oracle baseline;
2. reduce width/rank in oracle mode;
3. establish grid convergence;
4. switch to predicted density on the converged grid;
5. compare predicted-density errors against the matched oracle run.
