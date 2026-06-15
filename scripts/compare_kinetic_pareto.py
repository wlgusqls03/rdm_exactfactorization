#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare matched full-grid per-system metrics for kinetic fine-tuning runs."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="LABEL=path/to/per_system_metrics.csv; the first run is the baseline.",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    return parser.parse_args()


def load_run(spec: str, split: str) -> tuple[str, dict[str, dict[str, float]]]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=CSV_PATH, got: {spec}")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path).resolve()
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split:
                continue
            rows[row["system_id"]] = {
                key: float(row[key])
                for key in (
                    "pair_mae",
                    "tau_mae",
                    "kinetic_abs_error",
                    "kinetic_ref_error",
                    "kinetic_stencil_diag_error",
                    "kinetic_stencil_offdiag_error",
                    "kinetic_stencil_total_error",
                    "kinetic_stencil_reconstruction_residual",
                    "kinetic_stencil_reference_gap",
                )
            }
    if not rows:
        raise ValueError(f"No {split!r} rows found in {path}")
    return label, rows


def mean_metric(rows: dict[str, dict[str, float]], ids: list[str], key: str) -> float:
    return float(np.mean([rows[system_id][key] for system_id in ids]))


def main() -> None:
    args = parse_args()
    runs = [load_run(spec, args.split) for spec in args.runs]
    common_ids = sorted(set.intersection(*(set(rows) for _, rows in runs)))
    if not common_ids:
        raise ValueError("The supplied runs have no matched system IDs.")

    baseline_label, baseline = runs[0]
    baseline_abs = np.asarray(
        [baseline[system_id]["kinetic_abs_error"] for system_id in common_ids],
        dtype=np.float64,
    )
    print(f"split={args.split} matched_systems={len(common_ids)} baseline={baseline_label}")
    print(
        "label\tpair_mae\ttau_mae\tT_MAE_Ha\tT_median_Ha\t"
        "improved_vs_base\tdiag_signed_Ha\toffdiag_signed_Ha\t"
        "reconstruction_abs_max_Ha"
    )
    for label, rows in runs:
        abs_errors = np.asarray(
            [rows[system_id]["kinetic_abs_error"] for system_id in common_ids],
            dtype=np.float64,
        )
        reconstruction = np.asarray(
            [
                abs(rows[system_id]["kinetic_stencil_reconstruction_residual"])
                for system_id in common_ids
            ],
            dtype=np.float64,
        )
        improved = int(np.sum(abs_errors < baseline_abs)) if label != baseline_label else len(common_ids)
        print(
            f"{label}\t"
            f"{mean_metric(rows, common_ids, 'pair_mae'):.6e}\t"
            f"{mean_metric(rows, common_ids, 'tau_mae'):.6e}\t"
            f"{float(np.mean(abs_errors)):.6e}\t"
            f"{float(np.median(abs_errors)):.6e}\t"
            f"{improved}/{len(common_ids)}\t"
            f"{mean_metric(rows, common_ids, 'kinetic_stencil_diag_error'):.6e}\t"
            f"{mean_metric(rows, common_ids, 'kinetic_stencil_offdiag_error'):.6e}\t"
            f"{float(np.max(reconstruction)):.6e}"
        )


if __name__ == "__main__":
    main()
