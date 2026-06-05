from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
from pyscf import dft, gto

from build_qm9_pyscf_npz import (
    ANGSTROM_TO_BOHR,
    ALLOWED_ELEMENTS,
    ELEMENT_Z,
    normalized_density_on_grid,
)

LOCAL_FEATURE_SCHEMA = "sad_vectors_lapv_v1"
POTENTIAL_LAPLACIAN_CLIP = 8.0


def signed_log_scaled_np(values: np.ndarray, clip: float) -> np.ndarray:
    clip = max(float(clip), 1.0)
    clipped = np.clip(values, -clip, clip)
    return np.sign(clipped) * np.log1p(np.abs(clipped)) / np.log1p(clip)


def richardson_laplacian_np(grid_values: np.ndarray, n_axis: int, h: float) -> np.ndarray:
    vol = np.reshape(grid_values, (n_axis, n_axis, n_axis))
    padded = np.pad(vol, ((2, 2), (2, 2), (2, 2)), mode="symmetric")
    center = padded[2 : n_axis + 2, 2 : n_axis + 2, 2 : n_axis + 2]

    def second_derivative(axis: int) -> np.ndarray:
        slices = [slice(2, n_axis + 2), slice(2, n_axis + 2), slice(2, n_axis + 2)]
        values = []
        for offset in (0, 1, 3, 4):
            shifted = list(slices)
            shifted[axis] = slice(offset, offset + n_axis)
            values.append(padded[tuple(shifted)])
        return (-values[0] + 16.0 * values[1] - 30.0 * center + 16.0 * values[2] - values[3]) / (
            12.0 * h * h
        )

    return np.reshape(second_derivative(0) + second_derivative(1) + second_derivative(2), (-1, 1))


def potential_laplacian_feature(potential: np.ndarray, n_axis: int, step: float, clip: float) -> np.ndarray:
    lap = richardson_laplacian_np(potential, n_axis, step)
    pot_scale = max(float(np.std(potential)), 1.0)
    return signed_log_scaled_np(lap * (step**2) / pot_scale, clip).astype(np.float32)


def build_local_features_with_lapv(
    points_bohr: np.ndarray,
    coords_bohr_centered: np.ndarray,
    atomic_numbers: np.ndarray,
    potential: np.ndarray,
    grad: np.ndarray,
    electron_count: int,
    rho_sad: np.ndarray,
    step: float,
    laplacian_clip: float = POTENTIAL_LAPLACIAN_CLIP,
) -> np.ndarray:
    radius = max(float(np.max(np.abs(points_bohr))), 1e-6)
    coords_norm = points_bohr / radius
    pot_scale = max(float(np.std(potential)), 1.0)
    pot_feat = potential / pot_scale
    grad_feat = grad / pot_scale
    n_axis = round(len(points_bohr) ** (1.0 / 3.0))
    if n_axis**3 != len(points_bohr):
        raise ValueError(f"Expected cubic grid, got {len(points_bohr)} points.")
    lap_feat = potential_laplacian_feature(potential, n_axis, step, laplacian_clip)
    radial = np.linalg.norm(points_bohr, axis=1, keepdims=True) / radius

    gaussian_by_element = []
    vector_by_element = []
    for symbol in ALLOWED_ELEMENTS:
        z = ELEMENT_Z[symbol]
        centers = coords_bohr_centered[atomic_numbers == z]
        if len(centers) == 0:
            gaussian_by_element.append(np.zeros((len(points_bohr), 1), dtype=np.float32))
            vector_by_element.append(np.zeros((len(points_bohr), 3), dtype=np.float32))
            continue
        diff = points_bohr[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff**2, axis=2)
        weights = np.exp(-0.45 * dist2)
        gaussian_by_element.append(np.sum(weights, axis=1, keepdims=True).astype(np.float32))
        vector_by_element.append((np.sum(diff * weights[:, :, None], axis=1) / radius).astype(np.float32))

    nearest_z = np.zeros((len(points_bohr), 1), dtype=np.float32)
    if len(coords_bohr_centered):
        dist2_all = np.sum((points_bohr[:, None, :] - coords_bohr_centered[None, :, :]) ** 2, axis=2)
        nearest_z[:, 0] = atomic_numbers[np.argmin(dist2_all, axis=1)] / 10.0
    electron_col = np.full((len(points_bohr), 1), electron_count / 30.0, dtype=np.float32)
    return np.concatenate(
        [
            coords_norm,
            pot_feat,
            grad_feat,
            lap_feat,
            radial,
        ]
        + gaussian_by_element
        + [
            nearest_z,
            electron_col,
            np.log1p(np.maximum(rho_sad, 0.0)).astype(np.float32),
        ]
        + vector_by_element,
        axis=1,
    ).astype(np.float32)


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
        # Use minao for speed. 'atom' runs actual HF for each atom, which is slow.
        sad_dm = mf.get_init_guess(key="minao")
    except Exception:
        print(f"  Fallback to 1e for {symbols}")
        sad_dm = mf.get_init_guess(key="1e")
        
    rho_sad, _ = normalized_density_on_grid(mol, sad_dm, absolute_grid_bohr, cell_volume)
    return rho_sad


def is_patched_archive(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=True) as payload:
            local_features = payload["local_features"]
            if "local_feature_schema" in payload.files:
                return str(payload["local_feature_schema"]) == LOCAL_FEATURE_SCHEMA
            return local_features.shape[1] == 32
    except Exception:
        return False


def recover_legacy_temp_file(path: Path) -> bool:
    """Promote a complete temp archive left by the old NumPy suffix bug."""
    legacy_temp_path = Path(str(path) + ".tmp.npz")
    if not legacy_temp_path.exists():
        return False
    if not is_patched_archive(legacy_temp_path):
        legacy_temp_path.unlink()
        return False
    print(f"Recovering completed temp archive for {path.name} ...")
    os.replace(legacy_temp_path, path)
    return True


def patch_npz_file(path_str: str) -> bool:
    path = Path(path_str)
    try:
        if recover_legacy_temp_file(path):
            return True
        with np.load(path, allow_pickle=True) as payload:
            local_features = payload["local_features"]
            if is_patched_archive(path):
                # Already patched
                return True
            
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
            basis = str(payload["basis"]) if "basis" in payload.files else "6-31+g(d)"
            step = float(payload["grid_spacing_bohr"]) if "grid_spacing_bohr" in payload.files else 1.5
            
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
            
            new_local_features = build_local_features_with_lapv(
                points_bohr,
                coords_bohr_centered,
                atomic_numbers,
                potential,
                grad,
                electron_count=electron_count,
                rho_sad=rho_sad,
                step=step,
            )
            
            # Copy all data to a new dictionary
            new_data = {k: payload[k] for k in payload.files}
            new_data["local_features"] = new_local_features
            new_data["rho_sad"] = np.asarray(rho_sad, dtype=np.float32)
            new_data["local_feature_schema"] = np.asarray(LOCAL_FEATURE_SCHEMA)
            new_data["potential_laplacian_clip"] = np.asarray(POTENTIAL_LAPLACIAN_CLIP, dtype=np.float32)
            
        # Save to a temporary file first, then replace to avoid corruption
        temp_path = Path(str(path) + ".tmp")
        with temp_path.open("wb") as handle:
            np.savez_compressed(handle, **new_data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        return True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing {path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Patch existing NPZ files with SAD and directional features.")
    parser.add_argument("--npz-glob", required=True, help="Glob pattern for NPZ files to patch.")
    args = parser.parse_args()
    
    paths = [
        path
        for path in sorted(glob.glob(args.npz_glob))
        if not path.endswith(".tmp.npz")
    ]
    if not paths:
        print(f"No files found matching {args.npz_glob}")
        return
        
    print(f"Found {len(paths)} NPZ files. Checking and patching...")
    failed = []
    for i, path in enumerate(paths):
        if not patch_npz_file(path):
            failed.append(path)
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(paths)} files...")
    print(f"Done: succeeded={len(paths) - len(failed)} failed={len(failed)}")
    if failed:
        print("Failed files:")
        for path in failed:
            print(f"  {path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
