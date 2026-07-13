#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from profiles import PARTICLE_MASSES, PROFILES


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
TRAIN_SCRIPT = REPO_ROOT / "train_transferable_1rdm.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a mass-conditioned nuclear 1-RDM toy experiment."
    )
    parser.add_argument("--particle", choices=sorted(PARTICLE_MASSES), default="proton")
    parser.add_argument("--mass", type=float, help="Override the named particle mass in electron-mass units.")
    parser.add_argument("--density", choices=["oracle", "predicted"], default="oracle")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="small")
    parser.add_argument("--axis-points", type=int, default=15)
    parser.add_argument("--num-systems", type=int, default=500)
    parser.add_argument("--train-systems", type=int, default=400)
    parser.add_argument("--val-systems", type=int, default=50)
    parser.add_argument("--test-systems", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable containing a TensorFlow/NumPy-compatible environment.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use a tiny end-to-end validation run.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command only.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Append one raw trainer argument; repeat as needed.")
    return parser.parse_args()


def resolved_counts(args: argparse.Namespace) -> tuple[int, int, int, int, int, int]:
    if args.smoke:
        return 8, 4, 2, 2, 2, 2
    return (
        args.num_systems,
        args.train_systems,
        args.val_systems,
        args.test_systems,
        args.epochs,
        args.steps_per_epoch,
    )


def build_command(args: argparse.Namespace) -> tuple[list[str], str]:
    profile = PROFILES[args.profile]
    mass = float(args.mass if args.mass is not None else PARTICLE_MASSES[args.particle])
    density_source = "true" if args.density == "oracle" else "predicted"
    total, train, val, test, epochs, steps = resolved_counts(args)
    mass_tag = f"{mass:.6g}".replace(".", "p")
    run_name = (
        f"{args.particle}_m{mass_tag}_{args.density}_{args.profile}"
        f"_n{args.axis_points}_seed{args.seed}"
    )
    output_dir = args.output_root.resolve() / run_name

    cmd = [
        str(args.python),
        str(TRAIN_SCRIPT),
        "--dataset-mode", "toy",
        "--toy-dimensions", "3",
        "--toy-particle-mass", str(mass),
        "--density-source", density_source,
        "--pair-density-feature-mode", "rho-derivatives",
        "--num-systems", str(total),
        "--train-system-count", str(train),
        "--val-system-count", str(val),
        "--test-system-count", str(test),
        "--axis-points", str(args.axis_points),
        "--epochs", str(epochs),
        "--steps-per-epoch", str(steps),
        "--seed", str(args.seed),
        "--width", str(profile.width),
        "--rank", str(profile.rank),
        "--point-model-depth", str(profile.point_depth),
        "--pair-model-depth", str(profile.pair_depth),
        "--loss-preset", "staged-physics-kinetic",
        "--use-kinetic-loss",
        "--use-tau-mse-loss",
        "--train-stencil-centers", "256" if args.smoke else "4096",
        "--eval-pair-count", "2048" if args.smoke else "32768",
        "--val-every", "1" if args.smoke else "5",
        "--run-name", run_name,
        "--output-dir", str(output_dir),
        "--no-auto-run-dir",
    ]

    # The trainer exposes RFF size through the environment rather than CLI.
    os.environ["RDM_RFF_FEATURES"] = str(profile.rff_features)
    if args.density == "predicted":
        cmd.extend([
            "--point-pretrain-epochs", "2" if args.smoke else "120",
            "--point-pretrain-steps-per-epoch", "2" if args.smoke else "80",
        ])
    for value in args.extra_arg:
        cmd.extend(shlex.split(value))
    return cmd, run_name


def main() -> None:
    args = parse_args()
    if not TRAIN_SCRIPT.exists():
        raise SystemExit(f"Training entry point not found: {TRAIN_SCRIPT}")
    cmd, run_name = build_command(args)
    print(f"Resolved run: {run_name}")
    print("RDM_RFF_FEATURES=" + os.environ["RDM_RFF_FEATURES"])
    print(shlex.join(cmd))
    if args.dry_run:
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=os.environ.copy())


if __name__ == "__main__":
    main()
