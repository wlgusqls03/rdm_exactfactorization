#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_experiment.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matched nuclear 1-RDM experiment suites.")
    parser.add_argument("--suite", choices=["core", "lightweight", "grid", "all"], default="core")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def suite_rows(name: str) -> list[tuple[str, str, str, int]]:
    core = [
        (particle, density, "small", 15)
        for particle in ("electron", "muon", "proton")
        for density in ("oracle", "predicted")
    ]
    lightweight = [
        ("proton", density, profile, 15)
        for density in ("oracle", "predicted")
        for profile in ("baseline", "medium", "small", "tiny")
    ]
    grid = [
        ("proton", density, "small", axis)
        for density in ("oracle", "predicted")
        for axis in (15, 31, 51, 71)
    ]
    selected = {"core": core, "lightweight": lightweight, "grid": grid}
    rows = core + lightweight + grid if name == "all" else selected[name]
    return list(dict.fromkeys(rows))


def main() -> None:
    args = parse_args()
    failures = []
    for particle, density, profile, axis in suite_rows(args.suite):
        cmd = [
            sys.executable, str(RUNNER),
            "--particle", particle,
            "--density", density,
            "--profile", profile,
            "--axis-points", str(axis),
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.smoke:
            cmd.append("--smoke")
        result = subprocess.run(cmd, cwd=HERE.parent, check=False)
        if result.returncode:
            failures.append((particle, density, profile, axis, result.returncode))
            if not args.continue_on_error:
                raise SystemExit(result.returncode)
    if failures:
        raise SystemExit(f"Failed runs: {failures}")


if __name__ == "__main__":
    main()
