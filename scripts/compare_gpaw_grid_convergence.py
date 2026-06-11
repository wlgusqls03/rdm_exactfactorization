from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ENERGY_METRICS = (
    "total_energy_hartree",
    "kinetic_energy_orbital_fd_hartree",
    "kinetic_energy_orbital_central2_interior_hartree",
    "kinetic_energy_gamma_central2_interior_hartree",
    "kinetic_energy_gamma_richardson_interior_hartree",
)

CONSISTENCY_METRICS = (
    "tau_orbital_vs_gamma_central2_mae",
    "tau_orbital_vs_gamma_richardson_mae",
)


def parse_dataset(value: str) -> tuple[str, Path]:
    try:
        tag, directory = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use TAG=DATASET_DIR, for example h0p4=qmugs_npz/run_h0p4.") from exc
    tag = tag.strip()
    if not tag:
        raise argparse.ArgumentTypeError("Dataset tag cannot be empty.")
    return tag, Path(directory).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GPAW grid-spacing convergence datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        required=True,
        help="Ordered TAG=DATASET_DIR entry. Supply at least two, coarse to fine.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qmugs_npz/gpaw_grid_convergence.csv"),
    )
    return parser.parse_args()


def scalar_from_npz(path: Path, key: str) -> float:
    with np.load(path, allow_pickle=True) as payload:
        if key not in payload:
            return float("nan")
        return float(np.asarray(payload[key]).item())


def load_dataset(tag: str, directory: Path) -> pd.DataFrame:
    index_path = directory / "molecule_index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing index for {tag}: {index_path}")

    frame = pd.read_csv(index_path)
    required = {"qm9_id", "formula", "npz_file"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{index_path} is missing columns: {sorted(missing)}")

    for metric in ENERGY_METRICS:
        if metric not in frame.columns:
            frame[metric] = [
                scalar_from_npz(directory / npz_file, metric)
                for npz_file in frame["npz_file"]
            ]

    selected = ["qm9_id", "formula", "axis_points", *ENERGY_METRICS, *CONSISTENCY_METRICS]
    selected = [column for column in selected if column in frame.columns]
    frame = frame[selected].copy()
    return frame.rename(
        columns={
            column: f"{column}_{tag}"
            for column in frame.columns
            if column not in {"qm9_id", "formula"}
        }
    )


def print_summary(frame: pd.DataFrame, tags: list[str]) -> None:
    for metric in ENERGY_METRICS:
        columns = []
        for coarse, fine in zip(tags, tags[1:]):
            coarse_column = f"{metric}_{coarse}"
            fine_column = f"{metric}_{fine}"
            if coarse_column not in frame or fine_column not in frame:
                continue
            delta_column = f"abs_delta_{metric}_{coarse}_{fine}"
            frame[delta_column] = (frame[coarse_column] - frame[fine_column]).abs()
            columns.append(delta_column)
        if columns:
            print(f"\n=== {metric}: absolute spacing difference [Ha] ===")
            print(frame[columns].describe().loc[["mean", "50%", "max"]].to_string())

    for metric in CONSISTENCY_METRICS:
        columns = [f"{metric}_{tag}" for tag in tags if f"{metric}_{tag}" in frame]
        if columns:
            print(f"\n=== {metric} ===")
            print(frame[columns].describe().loc[["mean", "50%", "max"]].to_string())


def main() -> None:
    args = parse_args()
    if len(args.dataset) < 2:
        raise ValueError("Supply at least two --dataset entries.")

    tags = [tag for tag, _ in args.dataset]
    if len(tags) != len(set(tags)):
        raise ValueError(f"Dataset tags must be unique: {tags}")

    frames = [load_dataset(tag, directory) for tag, directory in args.dataset]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["qm9_id", "formula"], how="inner", validate="one_to_one")

    print(f"Matched molecules: {len(merged)}")
    if len(merged) == 0:
        raise RuntimeError("No common qm9_id/formula records were found.")

    print_summary(merged, tags)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
