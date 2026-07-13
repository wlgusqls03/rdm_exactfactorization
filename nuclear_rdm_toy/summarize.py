#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed nuclear 1-RDM runs.")
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    return parser.parse_args()


def metric_row(path: Path, split: str) -> dict[str, str] | None:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == split:
                return row
    return None


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def main() -> None:
    args = parse_args()
    paths = sorted(args.output_root.glob("**/*_split_metrics.csv"))
    if not paths:
        raise SystemExit(f"No split metrics found below {args.output_root}")
    print("run\tN\tgamma_MAE\ttau_rel_RMSE\tT_ref\tT_MAE\tT_MAE/T_ref\ttrace_abs_rel")
    for path in paths:
        row = metric_row(path, args.split)
        if row is None:
            continue
        t_ref = number(row, "kinetic_training_ref")
        t_mae = number(row, "kinetic_abs_error")
        relative = t_mae / abs(t_ref) if abs(t_ref) > 1e-30 else float("nan")
        print(
            f"{path.parent.name}\t{row.get('system_count', '')}"
            f"\t{number(row, 'pair_mae'):.6e}"
            f"\t{number(row, 'tau_rel_rmse'):.6e}"
            f"\t{t_ref:.6e}\t{t_mae:.6e}\t{relative:.6e}"
            f"\t{number(row, 'trace_abs_rel_error'):.6e}"
        )


if __name__ == "__main__":
    main()
