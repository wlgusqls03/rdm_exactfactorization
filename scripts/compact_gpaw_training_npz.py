#!/usr/bin/env python3
"""Create compact GPAW training archives without rerunning DFT."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "points",
    "psi_occ",
    "occupancies",
    "rho_diag",
    "electron_count",
    "derivative_orbital_gradient",
    "tau_orbital_gradient",
    "local_features",
    "global_context",
    "potential",
    "grad_potential",
    "atom_symbols",
    "atom_coords_bohr",
    "axis_points",
    "grid_spacing_bohr",
    "grid_radius_bohr",
    "box_length_bohr",
    "total_energy_hartree",
    "kinetic_energy_hartree",
    "kinetic_energy_orbital_fd_hartree",
    "kinetic_energy_orbital_central2_interior_hartree",
    "kinetic_energy_gamma_central2_interior_hartree",
    "kinetic_energy_gamma_richardson_interior_hartree",
    "tau_orbital_vs_gamma_central2_mae",
    "tau_orbital_vs_gamma_richardson_mae",
    "tau_gamma_central2_over_orbital_rms",
    "tau_gamma_richardson_over_orbital_rms",
    "tau_fd_orbital_vs_gamma_mae",
    "tau_fd_gamma_over_orbital_rms",
    "reference_schema",
    "tau_reference_primary",
    "tau_reference",
    "gamma_reference",
    "reference_backend",
    "gpaw_mode",
    "gpaw_xc",
    "gpaw_setups",
    "gpaw_fd_order",
    "local_feature_schema",
}

OPTIONAL_KEYS = {
    "kinetic_reference",
    "orbital_energies",
    "formula",
    "smiles_gdb",
    "smiles_relaxed",
    "inchi_gdb",
    "inchi_relaxed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def compact_archive(source: Path, destination: Path, overwrite: bool) -> tuple[int, int]:
    if destination.exists() and not overwrite:
        return source.stat().st_size, destination.stat().st_size

    with np.load(source, allow_pickle=True) as archive:
        missing = sorted(REQUIRED_KEYS.difference(archive.files))
        if missing:
            raise KeyError(f"{source} is missing required compact keys: {missing}")
        payload = {
            key: np.asarray(archive[key])
            for key in sorted(REQUIRED_KEYS | OPTIONAL_KEYS)
            if key in archive
        }
    payload["storage_profile"] = np.asarray("training-compact")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(destination)
    return source.stat().st_size, destination.stat().st_size


def copy_sidecars(input_dir: Path, output_dir: Path) -> None:
    for name in (
        "manifest.json",
        "molecule_index.csv",
        "molecule_index.md",
        "skipped_records.json",
    ):
        source = input_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    xyz_source = input_dir / "xyz"
    xyz_destination = output_dir / "xyz"
    if xyz_source.is_dir() and not xyz_destination.exists():
        shutil.copytree(xyz_source, xyz_destination)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("--output-dir must differ from --input-dir.")

    sources = sorted(input_dir.glob("*.npz"))
    if not sources:
        raise FileNotFoundError(f"No NPZ files found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_bytes = 0
    compact_bytes = 0
    for index, source in enumerate(sources, start=1):
        before, after = compact_archive(source, output_dir / source.name, args.overwrite)
        source_bytes += before
        compact_bytes += after
        if index == 1 or index % max(args.progress_every, 1) == 0 or index == len(sources):
            ratio = compact_bytes / max(source_bytes, 1)
            print(
                f"[compact] {index}/{len(sources)} | "
                f"source={source_bytes / 2**30:.2f} GiB | "
                f"compact={compact_bytes / 2**30:.2f} GiB | ratio={ratio:.3f}",
                flush=True,
            )

    copy_sidecars(input_dir, output_dir)
    print(f"Saved compact dataset: {output_dir}")


if __name__ == "__main__":
    main()
