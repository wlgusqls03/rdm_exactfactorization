from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ANGSTROM_TO_BOHR = 1.889726124565062
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR
ALLOWED_ELEMENTS = ("H", "C", "N", "O", "F")
ELEMENT_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


@dataclass
class Qm9Record:
    qm9_id: str
    symbols: list[str]
    coords_angstrom: np.ndarray
    smiles_gdb: str = ""
    smiles_relaxed: str = ""
    inchi_gdb: str = ""
    inchi_relaxed: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small QM9 real-space FD/pseudopotential 1-RDM NPZ dataset "
            "with GPAW finite-difference orbitals."
        )
    )
    parser.add_argument("--qm9-tar", type=Path, default=Path("data/qm9_raw/dsgdb9nsd.xyz.tar.bz2"))
    parser.add_argument("--npz-glob", type=str, default=None, help="Optional existing NPZ source for molecule records.")
    parser.add_argument("--output-dir", type=Path, default=Path("qmugs_npz/qm9_gpaw_fd_demo"))
    parser.add_argument("--num-systems", type=int, default=5)
    parser.add_argument("--max-atoms", type=int, default=7)
    parser.add_argument("--selection", choices=["random", "smallest"], default="smallest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-spacing-bohr", type=float, default=0.55)
    parser.add_argument("--padding-bohr", type=float, default=5.0)
    parser.add_argument("--max-axis-points", type=int, default=35)
    parser.add_argument("--xc", type=str, default="LDA")
    parser.add_argument(
        "--setups",
        type=str,
        default="",
        help="Optional GPAW setup selector, e.g. sg15 if installed. Empty uses GPAW's default PAW setups.",
    )
    parser.add_argument("--fd-order", type=int, default=4)
    parser.add_argument("--nbands-extra", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=180)
    parser.add_argument("--energy-convergence-ev", type=float, default=5e-4)
    parser.add_argument("--density-convergence", type=float, default=1e-4)
    parser.add_argument("--tau-stencil", choices=["central2", "richardson"], default="richardson")
    parser.add_argument("--dry-run", action="store_true", help="Select molecules and print grid sizes without running GPAW.")
    return parser.parse_args()


def require_gpaw() -> tuple[object, object, object]:
    try:
        from ase import Atoms
        from gpaw import FD, GPAW
        from gpaw import Mixer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "GPAW/ASE is required for this builder. Install them in the active environment, "
            "then rerun this command. Missing module: "
            f"{exc.name}"
        ) from exc
    return Atoms, GPAW, (FD, Mixer)


def flat_index(i: int, j: int, k: int, n: int) -> int:
    return (i * n + j) * n + k


def stencil_offsets(axis_points: int, stencil: str) -> tuple[int, ...]:
    method = stencil.strip().lower()
    if method in {"richardson", "richardson4", "extrapolated"} and axis_points >= 5:
        return (1, 2)
    if method in {"central2", "second", "second_order", "legacy"} or axis_points < 5:
        return (1,)
    raise ValueError(f"Unknown tau stencil: {stencil}")


def mixed_derivative_from_stencil(values: np.ndarray, step: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    d_h = (values[0] - values[1] - values[2] + values[3]) / (4.0 * step * step)
    if len(values) < 8:
        return float(d_h)
    d_2h = (values[4] - values[5] - values[6] + values[7]) / (16.0 * step * step)
    return float((4.0 * d_h - d_2h) / 3.0)


def prepare_stencil_targets(
    axis_points: int,
    gamma_matrix: np.ndarray,
    step: float,
    tau_stencil: str,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    offsets = stencil_offsets(axis_points, tau_stencil)
    margin = max(offsets)
    interior = []
    left_idx = []
    right_idx = []
    derivative_true = []
    for i in range(margin, axis_points - margin):
        for j in range(margin, axis_points - margin):
            for k in range(margin, axis_points - margin):
                interior.append(flat_index(i, j, k, axis_points))
                per_dim_left = []
                per_dim_right = []
                per_dim_deriv = []
                for dim in range(3):
                    dim_left = []
                    dim_right = []
                    for offset in offsets:
                        plus = [i, j, k]
                        minus = [i, j, k]
                        plus[dim] += offset
                        minus[dim] -= offset
                        idx_plus = flat_index(plus[0], plus[1], plus[2], axis_points)
                        idx_minus = flat_index(minus[0], minus[1], minus[2], axis_points)
                        dim_left.extend([idx_plus, idx_plus, idx_minus, idx_minus])
                        dim_right.extend([idx_plus, idx_minus, idx_plus, idx_minus])
                    values = gamma_matrix[dim_left, dim_right]
                    per_dim_left.append(dim_left)
                    per_dim_right.append(dim_right)
                    per_dim_deriv.append(mixed_derivative_from_stencil(values, step))
                left_idx.append(per_dim_left)
                right_idx.append(per_dim_right)
                derivative_true.append(per_dim_deriv)
    derivative_arr = np.asarray(derivative_true, dtype=np.float32)
    tau = 0.5 * np.sum(derivative_arr, axis=1, keepdims=True)
    return (
        np.asarray(interior, dtype=np.int64),
        np.asarray(left_idx, dtype=np.int64),
        np.asarray(right_idx, dtype=np.int64),
        derivative_arr,
    ), tau.astype(np.float32)


def parse_qm9_float(value: str) -> float:
    return float(value.replace("*^", "e"))


def read_qm9_records(path: Path, max_atoms: int) -> list[Qm9Record]:
    records: list[Qm9Record] = []
    with tarfile.open(path, mode="r:bz2") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".xyz"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            lines = handle.read().decode("utf-8").splitlines()
            if not lines:
                continue
            n_atoms = int(lines[0].strip())
            if n_atoms > max_atoms:
                continue
            symbols: list[str] = []
            coords: list[list[float]] = []
            ok = True
            for line in lines[2 : 2 + n_atoms]:
                parts = line.split()
                symbol = parts[0]
                if symbol not in ALLOWED_ELEMENTS:
                    ok = False
                    break
                symbols.append(symbol)
                coords.append([parse_qm9_float(parts[1]), parse_qm9_float(parts[2]), parse_qm9_float(parts[3])])
            if not ok:
                continue
            if sum(ELEMENT_Z[symbol] for symbol in symbols) % 2:
                continue
            smiles_gdb = smiles_relaxed = inchi_gdb = inchi_relaxed = ""
            if len(lines) > 2 + n_atoms + 1:
                smiles = lines[2 + n_atoms + 1].split()
                if smiles:
                    smiles_gdb = smiles[0]
                    smiles_relaxed = smiles[-1]
            if len(lines) > 2 + n_atoms + 2:
                inchi = lines[2 + n_atoms + 2].split()
                if inchi:
                    inchi_gdb = inchi[0]
                    inchi_relaxed = inchi[-1]
            records.append(
                Qm9Record(
                    qm9_id=Path(member.name).stem,
                    symbols=symbols,
                    coords_angstrom=np.asarray(coords, dtype=np.float64),
                    smiles_gdb=smiles_gdb,
                    smiles_relaxed=smiles_relaxed,
                    inchi_gdb=inchi_gdb,
                    inchi_relaxed=inchi_relaxed,
                )
            )
    return records


def read_records_from_npz(pattern: str, max_atoms: int) -> list[Qm9Record]:
    records: list[Qm9Record] = []
    for path_str in sorted(glob.glob(pattern)):
        if path_str.endswith(".tmp.npz"):
            continue
        path = Path(path_str)
        with np.load(path, allow_pickle=True) as payload:
            symbols = [str(symbol) for symbol in payload["atom_symbols"]]
            if len(symbols) > max_atoms:
                continue
            coords_angstrom = np.asarray(payload["atom_coords_bohr"], dtype=np.float64) * BOHR_TO_ANGSTROM
            stem = path.stem
            qm9_id = stem.split("_", 1)[1] if "_" in stem else stem
            records.append(
                Qm9Record(
                    qm9_id=qm9_id,
                    symbols=symbols,
                    coords_angstrom=coords_angstrom,
                    smiles_gdb=str(np.asarray(payload.get("smiles_gdb", "")).item()),
                    smiles_relaxed=str(np.asarray(payload.get("smiles_relaxed", "")).item()),
                    inchi_gdb=str(np.asarray(payload.get("inchi_gdb", "")).item()),
                    inchi_relaxed=str(np.asarray(payload.get("inchi_relaxed", "")).item()),
                )
            )
    return records


def molecular_formula(symbols: list[str]) -> str:
    parts = []
    for symbol in ALLOWED_ELEMENTS:
        count = symbols.count(symbol)
        if count:
            parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def choose_axis_points(coords_bohr_centered: np.ndarray, spacing_bohr: float, padding_bohr: float, cap: int) -> int:
    radius = float(np.max(np.abs(coords_bohr_centered)) + padding_bohr)
    n = int(np.ceil(2.0 * radius / spacing_bohr))
    n = max(8, n)
    if n % 2 == 0:
        n += 1
    if cap > 0 and n > cap:
        raise RuntimeError(
            f"axis_points={n} exceeds --max-axis-points={cap}. "
            "Use a coarser grid, smaller padding, or a larger cap."
        )
    return n


def centered_grid(axis_points: int, spacing_bohr: float) -> tuple[np.ndarray, np.ndarray]:
    axis = (np.arange(axis_points, dtype=np.float64) - 0.5 * (axis_points - 1)) * spacing_bohr
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return axis.astype(np.float32), points.astype(np.float32)


def nuclear_potential_and_grad(
    points_bohr: np.ndarray,
    coords_bohr_centered: np.ndarray,
    atomic_numbers: np.ndarray,
    softening: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    potential = np.zeros((len(points_bohr),), dtype=np.float64)
    grad = np.zeros((len(points_bohr), 3), dtype=np.float64)
    for z, center in zip(atomic_numbers, coords_bohr_centered):
        delta = points_bohr - center[None, :]
        r2 = np.sum(delta * delta, axis=1) + softening * softening
        r = np.sqrt(r2)
        potential -= float(z) / r
        grad += float(z) * delta / (r2[:, None] * r[:, None])
    return potential.reshape(-1, 1).astype(np.float32), grad.astype(np.float32)


def build_local_features(
    points_bohr: np.ndarray,
    coords_bohr_centered: np.ndarray,
    atomic_numbers: np.ndarray,
    potential: np.ndarray,
    grad: np.ndarray,
    electron_count: float,
) -> np.ndarray:
    radius = max(float(np.max(np.abs(points_bohr))), 1e-6)
    pot_scale = max(float(np.std(potential)), 1.0)
    coords_norm = points_bohr / radius
    pot_feat = potential / pot_scale
    grad_feat = grad / pot_scale
    radial = np.linalg.norm(points_bohr, axis=1, keepdims=True) / radius
    gaussian_by_element = []
    for symbol in ALLOWED_ELEMENTS:
        z = ELEMENT_Z[symbol]
        centers = coords_bohr_centered[atomic_numbers == z]
        if len(centers) == 0:
            gaussian_by_element.append(np.zeros((len(points_bohr), 1), dtype=np.float32))
            continue
        dist2 = np.sum((points_bohr[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        gaussian_by_element.append(np.sum(np.exp(-0.45 * dist2), axis=1, keepdims=True).astype(np.float32))
    nearest_z = np.zeros((len(points_bohr), 1), dtype=np.float32)
    if len(coords_bohr_centered):
        dist2_all = np.sum((points_bohr[:, None, :] - coords_bohr_centered[None, :, :]) ** 2, axis=2)
        nearest_z[:, 0] = atomic_numbers[np.argmin(dist2_all, axis=1)] / 10.0
    electron_col = np.full((len(points_bohr), 1), electron_count / 30.0, dtype=np.float32)
    return np.concatenate(
        [coords_norm, pot_feat, grad_feat, radial] + gaussian_by_element + [nearest_z, electron_col],
        axis=1,
    ).astype(np.float32)


def build_global_context(symbols: list[str], coords_bohr_centered: np.ndarray, electron_count: float) -> np.ndarray:
    atomic_numbers = np.asarray([ELEMENT_Z[symbol] for symbol in symbols], dtype=np.float32)
    counts = np.asarray([symbols.count(symbol) for symbol in ALLOWED_ELEMENTS], dtype=np.float32)
    heavy_count = sum(symbol != "H" for symbol in symbols)
    radius = float(np.max(np.linalg.norm(coords_bohr_centered, axis=1))) if len(coords_bohr_centered) else 0.0
    return np.concatenate(
        [
            np.asarray(
                [
                    electron_count / 30.0,
                    len(symbols) / 30.0,
                    heavy_count / 10.0,
                    float(np.mean(atomic_numbers)) / 10.0,
                    float(np.std(atomic_numbers)) / 10.0,
                    radius / 10.0,
                ],
                dtype=np.float32,
            ),
            counts / 10.0,
        ]
    ).astype(np.float32)


def normalize_orbitals(psi_matrix: np.ndarray, cell_volume: float) -> np.ndarray:
    norms = np.sqrt(np.sum(psi_matrix * psi_matrix, axis=0, keepdims=True) * cell_volume)
    norms = np.maximum(norms, 1e-14)
    return psi_matrix / norms


def kinetic_density_from_orbitals(
    orbitals_4d: np.ndarray,
    occupancies: np.ndarray,
    spacing_bohr: float,
) -> tuple[np.ndarray, np.ndarray]:
    derivative_components = np.zeros(orbitals_4d.shape[:3] + (3,), dtype=np.float64)
    for band in range(orbitals_4d.shape[3]):
        occ = float(occupancies[band])
        if abs(occ) < 1e-12:
            continue
        grads = np.gradient(orbitals_4d[:, :, :, band], spacing_bohr, edge_order=2)
        for dim, grad in enumerate(grads):
            derivative_components[:, :, :, dim] += occ * grad * grad
    flat_deriv = derivative_components.reshape(-1, 3)
    tau = 0.5 * np.sum(flat_deriv, axis=1, keepdims=True)
    return flat_deriv.astype(np.float32), tau.astype(np.float32)


def build_atoms(record: Qm9Record, axis_points: int, spacing_bohr: float, Atoms: object) -> tuple[object, np.ndarray]:
    coords_bohr = record.coords_angstrom * ANGSTROM_TO_BOHR
    coords_bohr_centered = coords_bohr - np.mean(coords_bohr, axis=0, keepdims=True)
    box_angstrom = axis_points * spacing_bohr * BOHR_TO_ANGSTROM
    positions_angstrom = coords_bohr_centered * BOHR_TO_ANGSTROM + 0.5 * box_angstrom
    atoms = Atoms(record.symbols, positions=positions_angstrom, cell=[box_angstrom] * 3, pbc=False)
    return atoms, coords_bohr_centered


def run_gpaw(record: Qm9Record, args: argparse.Namespace) -> dict[str, object]:
    Atoms, GPAW, gpaw_helpers = require_gpaw()
    FD, Mixer = gpaw_helpers

    coords_bohr = record.coords_angstrom * ANGSTROM_TO_BOHR
    coords_bohr_centered = coords_bohr - np.mean(coords_bohr, axis=0, keepdims=True)
    axis_points = choose_axis_points(coords_bohr_centered, args.grid_spacing_bohr, args.padding_bohr, args.max_axis_points)
    axis, points_bohr = centered_grid(axis_points, args.grid_spacing_bohr)
    atoms, coords_bohr_centered = build_atoms(record, axis_points, args.grid_spacing_bohr, Atoms)

    electron_guess = sum(ELEMENT_Z[symbol] for symbol in record.symbols)
    nbands = max(1, electron_guess // 2 + int(args.nbands_extra))
    txt = str(args.output_dir / "gpaw_logs" / f"{record.qm9_id}.txt")
    Path(txt).parent.mkdir(parents=True, exist_ok=True)
    calc = GPAW(
        mode=FD(nn=args.fd_order),
        xc=args.xc,
        gpts=(axis_points, axis_points, axis_points),
        nbands=nbands,
        setups=(args.setups if args.setups else None),
        convergence={"energy": args.energy_convergence_ev, "density": args.density_convergence},
        maxiter=args.maxiter,
        mixer=Mixer(0.05, 5, 50.0),
        txt=txt,
    )
    atoms.calc = calc
    energy_ev = float(atoms.get_potential_energy())

    occupancies_all = np.asarray(calc.get_occupation_numbers(spin=0, kpt=0), dtype=np.float64)
    eigenvalues_ev_all = np.asarray(calc.get_eigenvalues(spin=0, kpt=0), dtype=np.float64)
    occupied = np.where(occupancies_all > 1e-8)[0]
    if occupied.size == 0:
        raise RuntimeError(f"No occupied bands found for {record.qm9_id}.")

    orbitals = []
    for band in occupied:
        psi = np.asarray(calc.get_pseudo_wave_function(band=int(band), spin=0, kpt=0), dtype=np.float64)
        if psi.shape != (axis_points, axis_points, axis_points):
            raise RuntimeError(f"Unexpected GPAW wavefunction shape {psi.shape}; expected {(axis_points,) * 3}.")
        orbitals.append(psi.reshape(-1))
    psi_matrix = normalize_orbitals(np.stack(orbitals, axis=1), args.grid_spacing_bohr**3)
    occupancies = occupancies_all[occupied]
    eigenvalues_ev = eigenvalues_ev_all[occupied]
    electron_count = float(np.sum(occupancies))

    gamma_matrix = (psi_matrix * occupancies[None, :]) @ psi_matrix.T
    gamma_matrix = 0.5 * (gamma_matrix + gamma_matrix.T)
    rho_diag = np.diag(gamma_matrix).reshape(-1, 1)
    trace = float(np.sum(rho_diag) * args.grid_spacing_bohr**3)
    if abs(trace - electron_count) > max(2e-4, 2e-4 * electron_count):
        raise RuntimeError(f"Trace check failed for {record.qm9_id}: grid trace={trace}, occ sum={electron_count}")

    orbitals_4d = psi_matrix.reshape(axis_points, axis_points, axis_points, -1)
    derivative_orbital_fd, tau_orbital_fd = kinetic_density_from_orbitals(
        orbitals_4d,
        occupancies,
        args.grid_spacing_bohr,
    )
    (stencil_info, tau_gamma_fd) = prepare_stencil_targets(
        axis_points=axis_points,
        gamma_matrix=gamma_matrix,
        step=args.grid_spacing_bohr,
        tau_stencil=args.tau_stencil,
    )
    interior_idx, _, _, derivative_gamma_fd = stencil_info
    interior_tau_orbital = tau_orbital_fd[interior_idx]
    tau_consistency_mae = float(np.mean(np.abs(interior_tau_orbital - tau_gamma_fd)))
    tau_consistency_rms_ratio = float(
        np.sqrt(np.mean(tau_gamma_fd**2)) / max(np.sqrt(np.mean(interior_tau_orbital**2)), 1e-30)
    )

    atomic_numbers = np.asarray([ELEMENT_Z[symbol] for symbol in record.symbols], dtype=np.float64)
    potential, grad = nuclear_potential_and_grad(points_bohr, coords_bohr_centered, atomic_numbers)
    local_features = build_local_features(points_bohr, coords_bohr_centered, atomic_numbers, potential, grad, electron_count)
    global_context = build_global_context(record.symbols, coords_bohr_centered, electron_count)
    return {
        "axis_points": axis_points,
        "axis": axis,
        "points_bohr": points_bohr,
        "coords_bohr_centered": coords_bohr_centered,
        "potential": potential,
        "grad_potential": grad,
        "local_features": local_features,
        "global_context": global_context,
        "gamma_matrix": gamma_matrix.astype(np.float32),
        "rho_diag": rho_diag.astype(np.float32),
        "derivative_orbital_fd": derivative_orbital_fd,
        "tau_orbital_fd": tau_orbital_fd,
        "derivative_gamma_fd_interior": derivative_gamma_fd.astype(np.float32),
        "tau_gamma_fd_interior": tau_gamma_fd.astype(np.float32),
        "occupancies": occupancies.astype(np.float32),
        "orbital_energies": (eigenvalues_ev / 27.211386245988).astype(np.float32),
        "electron_count": electron_count,
        "total_energy_hartree": energy_ev / 27.211386245988,
        "kinetic_energy_hartree": float(np.sum(tau_orbital_fd) * args.grid_spacing_bohr**3),
        "tau_consistency_mae": tau_consistency_mae,
        "tau_consistency_rms_ratio": tau_consistency_rms_ratio,
    }


def write_npz(record: Qm9Record, args: argparse.Namespace, output_path: Path) -> dict[str, object]:
    result = run_gpaw(record, args)
    axis_points = int(result["axis_points"])
    points_bohr = np.asarray(result["points_bohr"], dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points=points_bohr,
        gamma_matrix=np.asarray(result["gamma_matrix"], dtype=np.float32),
        rho_diag=np.asarray(result["rho_diag"], dtype=np.float32),
        # Legacy training-loader names. In this dataset these are FD orbital-gradient references.
        derivative_true_ao=np.asarray(result["derivative_orbital_fd"], dtype=np.float32),
        tau_true_ao=np.asarray(result["tau_orbital_fd"], dtype=np.float32),
        derivative_true_fd_orbital=np.asarray(result["derivative_orbital_fd"], dtype=np.float32),
        tau_true_fd_orbital=np.asarray(result["tau_orbital_fd"], dtype=np.float32),
        derivative_true_fd_gamma=np.asarray(result["derivative_gamma_fd_interior"], dtype=np.float32),
        tau_true_fd_gamma=np.asarray(result["tau_gamma_fd_interior"], dtype=np.float32),
        local_features=np.asarray(result["local_features"], dtype=np.float32),
        global_context=np.asarray(result["global_context"], dtype=np.float32),
        potential=np.asarray(result["potential"], dtype=np.float32),
        grad_potential=np.asarray(result["grad_potential"], dtype=np.float32),
        rho_sad=np.empty((0, 1), dtype=np.float32),
        electron_count=np.asarray(result["electron_count"], dtype=np.float32),
        occupancies=np.asarray(result["occupancies"], dtype=np.float32),
        orbital_energies=np.asarray(result["orbital_energies"], dtype=np.float32),
        atom_symbols=np.asarray(record.symbols),
        atom_coords_bohr=np.asarray(result["coords_bohr_centered"], dtype=np.float32),
        formula=np.asarray(molecular_formula(record.symbols)),
        smiles_gdb=np.asarray(record.smiles_gdb),
        smiles_relaxed=np.asarray(record.smiles_relaxed),
        inchi_gdb=np.asarray(record.inchi_gdb),
        inchi_relaxed=np.asarray(record.inchi_relaxed),
        axis_points=np.asarray(axis_points, dtype=np.int32),
        grid_spacing_bohr=np.asarray(args.grid_spacing_bohr, dtype=np.float32),
        grid_radius_bohr=np.asarray(float(np.max(np.abs(result["axis"]))), dtype=np.float32),
        box_length_bohr=np.asarray(axis_points * args.grid_spacing_bohr, dtype=np.float32),
        kinetic_energy_hartree=np.asarray(result["kinetic_energy_hartree"], dtype=np.float32),
        total_energy_hartree=np.asarray(result["total_energy_hartree"], dtype=np.float32),
        tau_reference=np.asarray("gpaw_fd_orbital_gradient"),
        gamma_reference=np.asarray("gpaw_fd_pseudo_wavefunctions"),
        reference_backend=np.asarray("gpaw_fd_pseudopotential"),
        gpaw_xc=np.asarray(args.xc),
        gpaw_setups=np.asarray(args.setups if args.setups else "default"),
        gpaw_fd_order=np.asarray(args.fd_order, dtype=np.int32),
        tau_fd_orbital_vs_gamma_mae=np.asarray(result["tau_consistency_mae"], dtype=np.float32),
        tau_fd_gamma_over_orbital_rms=np.asarray(result["tau_consistency_rms_ratio"], dtype=np.float32),
        local_feature_schema=np.asarray("gpaw_fd_legacy_v1"),
    )
    return {
        "system_id": output_path.stem,
        "qm9_id": record.qm9_id,
        "formula": molecular_formula(record.symbols),
        "n_atoms": len(record.symbols),
        "electron_count": float(result["electron_count"]),
        "total_energy_hartree": float(result["total_energy_hartree"]),
        "kinetic_energy_hartree": float(result["kinetic_energy_hartree"]),
        "tau_fd_orbital_vs_gamma_mae": float(result["tau_consistency_mae"]),
        "tau_fd_gamma_over_orbital_rms": float(result["tau_consistency_rms_ratio"]),
        "axis_points": axis_points,
        "grid_spacing_bohr": float(args.grid_spacing_bohr),
        "box_length_bohr": axis_points * float(args.grid_spacing_bohr),
        "xc": args.xc,
        "setups": args.setups,
    }


def write_xyz(record: Qm9Record, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(record.symbols)), f"{record.qm9_id} formula={molecular_formula(record.symbols)}"]
    for symbol, coord in zip(record.symbols, record.coords_angstrom):
        lines.append(f"{symbol:2s} {coord[0]: .10f} {coord[1]: .10f} {coord[2]: .10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_indices(output_dir: Path, manifest: list[dict[str, object]]) -> None:
    fields = [
        "index",
        "system_id",
        "qm9_id",
        "formula",
        "n_atoms",
        "electron_count",
        "axis_points",
        "grid_spacing_bohr",
        "kinetic_energy_hartree",
        "tau_fd_orbital_vs_gamma_mae",
        "tau_fd_gamma_over_orbital_rms",
        "npz_file",
        "xyz_file",
    ]
    with (output_dir / "molecule_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in manifest:
            writer.writerow({field: row.get(field, "") for field in fields})
    lines = ["# GPAW FD Molecule Index", "", "| index | system_id | formula | atoms | electrons | axis | tau FD MAE |", "|---:|---|---|---:|---:|---:|---:|"]
    for row in manifest:
        lines.append(
            "| {index} | {system_id} | {formula} | {n_atoms} | {electron_count:.3f} | {axis_points} | {tau_fd_orbital_vs_gamma_mae:.3e} |".format(
                **row
            )
        )
    (output_dir / "molecule_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_records(args: argparse.Namespace) -> list[Qm9Record]:
    if args.npz_glob:
        records = read_records_from_npz(args.npz_glob, args.max_atoms)
    else:
        records = read_qm9_records(args.qm9_tar, args.max_atoms)
    if args.selection == "smallest":
        records = sorted(records, key=lambda r: (len(r.symbols), sum(ELEMENT_Z[s] for s in r.symbols), r.qm9_id))
    else:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(records)
    return records


def main() -> None:
    args = parse_args()
    records = select_records(args)
    if len(records) < args.num_systems:
        raise RuntimeError(f"Only found {len(records)} suitable records.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for record in records:
        if len(manifest) >= args.num_systems:
            break
        idx = len(manifest)
        coords_bohr = record.coords_angstrom * ANGSTROM_TO_BOHR
        centered = coords_bohr - np.mean(coords_bohr, axis=0, keepdims=True)
        try:
            axis_points = choose_axis_points(centered, args.grid_spacing_bohr, args.padding_bohr, args.max_axis_points)
        except RuntimeError as exc:
            skipped.append({"qm9_id": record.qm9_id, "reason": str(exc)})
            continue
        output_path = args.output_dir / f"{idx:04d}_{record.qm9_id}.npz"
        if output_path.exists():
            print(f"[{idx + 1}/{args.num_systems}] exists: {output_path}")
            with np.load(output_path, allow_pickle=True) as payload:
                row = {
                    "system_id": output_path.stem,
                    "qm9_id": record.qm9_id,
                    "formula": molecular_formula(record.symbols),
                    "n_atoms": len(record.symbols),
                    "electron_count": float(payload["electron_count"]),
                    "axis_points": int(payload["axis_points"]),
                    "grid_spacing_bohr": float(payload["grid_spacing_bohr"]),
                    "kinetic_energy_hartree": float(payload["kinetic_energy_hartree"]),
                    "tau_fd_orbital_vs_gamma_mae": float(payload["tau_fd_orbital_vs_gamma_mae"]),
                    "tau_fd_gamma_over_orbital_rms": float(payload["tau_fd_gamma_over_orbital_rms"]),
                }
        elif args.dry_run:
            print(f"[dry-run] {record.qm9_id}: atoms={len(record.symbols)} axis={axis_points} points={axis_points ** 3}")
            row = {
                "system_id": output_path.stem,
                "qm9_id": record.qm9_id,
                "formula": molecular_formula(record.symbols),
                "n_atoms": len(record.symbols),
                "electron_count": float(sum(ELEMENT_Z[s] for s in record.symbols)),
                "axis_points": axis_points,
                "grid_spacing_bohr": float(args.grid_spacing_bohr),
                "kinetic_energy_hartree": float("nan"),
                "tau_fd_orbital_vs_gamma_mae": float("nan"),
                "tau_fd_gamma_over_orbital_rms": float("nan"),
            }
        else:
            print(f"[{idx + 1}/{args.num_systems}] GPAW FD -> NPZ: {record.qm9_id} axis={axis_points}")
            try:
                row = write_npz(record, args, output_path)
            except Exception as exc:
                print(f"[skip] {record.qm9_id}: {exc}")
                skipped.append({"qm9_id": record.qm9_id, "reason": str(exc)})
                continue
        row.update({"index": idx, "npz_file": output_path.name, "xyz_file": f"xyz/{output_path.stem}.xyz"})
        manifest.append(row)
        write_xyz(record, args.output_dir / "xyz" / f"{output_path.stem}.xyz")

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "skipped_records.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    write_indices(args.output_dir, manifest)
    print(f"Wrote {len(manifest)} records to {args.output_dir}")
    print(f"Skipped {len(skipped)} records")


if __name__ == "__main__":
    main()
