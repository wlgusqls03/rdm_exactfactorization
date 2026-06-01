from __future__ import annotations

import argparse
import glob
import os
import shutil
from pathlib import Path

import numpy as np
from pyscf import dft, gto

from build_qm9_pyscf_npz import (
    ANGSTROM_TO_BOHR,
    ELEMENT_Z,
    build_local_features,
    normalized_density_on_grid,
)


def get_sad_density(symbols: list[str], coords_bohr: np.ndarray, basis: str, absolute_grid_bohr: np.ndarray, cell_volume: float) -> np.ndarray:
    coords_ang = coords_bohr / ANGSTROM_TO_BOHR
    atom_spec = [(sym, tuple(c)) for sym, c in zip(symbols, coords_ang)]
    
    mol = gto.Mole()
    mol.atom = atom_spec
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.verbose = 0
    mol.build()
    
    mf = dft.RKS(mol)
    try:
        sad_dm = mf.get_init_guess(key="atom")
    except Exception:
        print(f"  Fallback to minao for {symbols}")
        sad_dm = mf.get_init_guess(key="minao")
        
    rho_sad, _ = normalized_density_on_grid(mol, sad_dm, absolute_grid_bohr, cell_volume)
    return rho_sad


def patch_npz_file(path_str: str) -> None:
    path = Path(path_str)
    try:
        with np.load(path, allow_pickle=True) as payload:
            local_features = payload["local_features"]
            if local_features.shape[1] >= 31:
                # Already patched
                return
            
            print(f"Patching {path.name} ...")
            
            # Extract basic info
            points_bohr = payload["points"]
            coords_bohr_centered = payload["atom_coords_bohr"]
            symbols = [str(s) for s in payload["atom_symbols"]]
            atomic_numbers = np.asarray([ELEMENT_Z[symbol] for symbol in symbols], dtype=np.float64)
            potential = payload["potential"]
            grad = payload["grad_potential"]
            electron_count = int(payload["electron_count"])
            
            # Use fallback for basis and step if they are missing from older NPZ files
            basis = str(payload["basis"]) if "basis" in payload else "6-31+g(d)"
            step = float(payload["grid_spacing_bohr"]) if "grid_spacing_bohr" in payload else 1.5
            
            # Reconstruct absolute grid
            # In build_qm9_pyscf_npz: coords_bohr_centered = coords - mean
            # absolute_grid = points + mean
            # Since atom_coords_bohr in NPZ IS centered, we can't easily get the original mean.
            # But wait, does PySCF care about the absolute center as long as atoms and grid are consistent?
            # Yes, if we just use coords_bohr_centered for atoms, and points_bohr for grid, it's perfectly consistent!
            absolute_grid_bohr = points_bohr
            
            rho_sad = get_sad_density(
                symbols, 
                coords_bohr_centered, 
                basis, 
                absolute_grid_bohr, 
                step**3
            )
            
            new_local_features = build_local_features(
                points_bohr,
                coords_bohr_centered,
                atomic_numbers,
                potential,
                grad,
                electron_count=electron_count,
                rho_sad=rho_sad,
                include_vectors=True,
            )
            
            # Copy all data to a new dictionary
            new_data = {k: payload[k] for k in payload.files}
            new_data["local_features"] = new_local_features
            
        # Save to a temporary file first, then replace to avoid corruption
        temp_path = path.with_suffix(".npz.tmp")
        np.savez_compressed(temp_path, **new_data)
        shutil.move(temp_path, path)
        
    except Exception as e:
        print(f"Error processing {path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Patch existing NPZ files with SAD and directional features.")
    parser.add_argument("--npz-glob", required=True, help="Glob pattern for NPZ files to patch.")
    args = parser.parse_args()
    
    paths = sorted(glob.glob(args.npz_glob))
    if not paths:
        print(f"No files found matching {args.npz_glob}")
        return
        
    print(f"Found {len(paths)} NPZ files. Checking and patching...")
    for i, path in enumerate(paths):
        patch_npz_file(path)
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(paths)} files...")
    print("Done!")

if __name__ == "__main__":
    main()
