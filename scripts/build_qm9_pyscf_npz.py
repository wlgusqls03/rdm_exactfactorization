from __future__ import annotations

import argparse
import csv
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyscf import df, dft, gto


ANGSTROM_TO_BOHR = 1.889726124565062
ALLOWED_ELEMENTS = ("H", "C", "N", "O", "F")
ELEMENT_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


@dataclass
class Qm9Record:
    qm9_id: str
    symbols: list[str]
    coords_angstrom: np.ndarray
    smiles_gdb: str
    smiles_relaxed: str
    inchi_gdb: str
    inchi_relaxed: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiny PySCF-DFT 1-RDM NPZ subset from QM9 XYZ files.")
    parser.add_argument("--qm9-tar", type=Path, default=Path("data/qm9_raw/dsgdb9nsd.xyz.tar.bz2"))
    parser.add_argument("--output-dir", type=Path, default=Path("qmugs_npz/qm9_pyscf_demo_axis7"))
    parser.add_argument("--num-systems", type=int, default=12)
    parser.add_argument("--axis-points", type=int, default=7)
    parser.add_argument(
        "--grid-spacing-bohr",
        type=float,
        default=None,
        help=(
            "Target Cartesian grid spacing in bohr. If set, axis_points is computed "
            "per molecule from the padded box size instead of using --axis-points."
        ),
    )
    parser.add_argument(
        "--max-axis-points",
        type=int,
        default=0,
        help="Optional safety cap for spacing-derived axis points. 0 disables the cap.",
    )
    parser.add_argument("--padding-bohr", type=float, default=4.0)
    parser.add_argument("--max-atoms", type=int, default=7)
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--xc", type=str, default="b3lyp")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument(
        "--skip-kinetic-potential",
        action="store_true",
        help="Do not store kinetic-potential reference arrays in the NPZ files.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection", choices=["random", "smallest"], default="random")
    return parser.parse_args()


def parse_qm9_float(value: str) -> float:
    """QM9 occasionally uses Mathematica-style scientific notation."""
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
            atom_lines = lines[2 : 2 + n_atoms]
            symbols: list[str] = []
            coords: list[list[float]] = []
            ok = True
            for line in atom_lines:
                parts = line.split()
                symbol = parts[0]
                if symbol not in ALLOWED_ELEMENTS:
                    ok = False
                    break
                symbols.append(symbol)
                coords.append([parse_qm9_float(parts[1]), parse_qm9_float(parts[2]), parse_qm9_float(parts[3])])
            if not ok:
                continue
            nelec = sum(ELEMENT_Z[symbol] for symbol in symbols)
            if nelec % 2 != 0:
                continue
            smiles_gdb = ""
            smiles_relaxed = ""
            inchi_gdb = ""
            inchi_relaxed = ""
            if len(lines) > 2 + n_atoms + 1:
                smiles_parts = lines[2 + n_atoms + 1].split()
                if smiles_parts:
                    smiles_gdb = smiles_parts[0]
                    smiles_relaxed = smiles_parts[-1]
            if len(lines) > 2 + n_atoms + 2:
                inchi_parts = lines[2 + n_atoms + 2].split()
                if inchi_parts:
                    inchi_gdb = inchi_parts[0]
                    inchi_relaxed = inchi_parts[-1]
            qm9_id = Path(member.name).stem
            records.append(
                Qm9Record(
                    qm9_id=qm9_id,
                    symbols=symbols,
                    coords_angstrom=np.asarray(coords, dtype=np.float64),
                    smiles_gdb=smiles_gdb,
                    smiles_relaxed=smiles_relaxed,
                    inchi_gdb=inchi_gdb,
                    inchi_relaxed=inchi_relaxed,
                )
            )
    return records


def molecular_formula(symbols: list[str]) -> str:
    counts = {symbol: symbols.count(symbol) for symbol in ALLOWED_ELEMENTS}
    parts = []
    for symbol in ALLOWED_ELEMENTS:
        count = counts[symbol]
        if count == 0:
            continue
        parts.append(symbol if count == 1 else f"{symbol}{count}")
    return "".join(parts)


def choose_axis_points(box_length: float, target_spacing: float) -> int:
    """Choose an odd number of points so the origin is included and h <= target."""
    intervals = int(np.ceil(box_length / target_spacing))
    axis_points = max(3, intervals + 1)
    if axis_points % 2 == 0:
        axis_points += 1
    return axis_points


def make_grid(
    coords_bohr: np.ndarray,
    axis_points: int,
    padding_bohr: float,
    grid_spacing_bohr: float | None = None,
    max_axis_points: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    centered = coords_bohr - np.mean(coords_bohr, axis=0, keepdims=True)
    radius = float(np.max(np.abs(centered)) + padding_bohr)
    if grid_spacing_bohr is not None:
        if grid_spacing_bohr <= 0.0:
            raise ValueError("--grid-spacing-bohr must be positive.")
        axis_points = choose_axis_points(2.0 * radius, grid_spacing_bohr)
        if max_axis_points > 0 and axis_points > max_axis_points:
            raise RuntimeError(
                f"Spacing-derived axis_points={axis_points} exceeds --max-axis-points={max_axis_points}. "
                "Increase the cap, use a coarser spacing, or reduce molecule size/padding."
            )
    axis = np.linspace(-radius, radius, axis_points, dtype=np.float32)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
    return axis, points


def nuclear_potential_and_grad(
    points_bohr: np.ndarray,
    coords_bohr_centered: np.ndarray,
    atomic_numbers: np.ndarray,
    softening: float = 0.25,
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
    electron_count: int,
) -> np.ndarray:
    radius = max(float(np.max(np.abs(points_bohr))), 1e-6)
    coords_norm = points_bohr / radius
    pot_scale = max(float(np.std(potential)), 1.0)
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
        nearest = np.argmin(dist2_all, axis=1)
        nearest_z[:, 0] = atomic_numbers[nearest] / 10.0

    electron_col = np.full((len(points_bohr), 1), electron_count / 30.0, dtype=np.float32)
    return np.concatenate(
        [coords_norm, pot_feat, grad_feat, radial] + gaussian_by_element + [nearest_z, electron_col],
        axis=1,
    ).astype(np.float32)


def build_global_context(symbols: list[str], coords_bohr_centered: np.ndarray, electron_count: int) -> np.ndarray:
    atomic_numbers = np.asarray([ELEMENT_Z[symbol] for symbol in symbols], dtype=np.float32)
    counts = np.array([symbols.count(symbol) for symbol in ALLOWED_ELEMENTS], dtype=np.float32)
    heavy_count = sum(symbol != "H" for symbol in symbols)
    radius = float(np.max(np.linalg.norm(coords_bohr_centered, axis=1))) if len(coords_bohr_centered) else 0.0
    context = np.concatenate(
        [
            np.array(
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
        ],
    )
    return context.astype(np.float32)


def run_dft(record: Qm9Record, basis: str, xc: str, grid_level: int) -> tuple[gto.Mole, dft.rks.RKS, np.ndarray]:
    atom_spec = [
        (symbol, tuple(coord.tolist()))
        for symbol, coord in zip(record.symbols, record.coords_angstrom)
    ]
    mol = gto.Mole()
    mol.atom = atom_spec
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = grid_level
    mf.conv_tol = 1e-8
    mf.max_cycle = 80
    energy = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for {record.qm9_id}; last energy={energy}")
    return mol, mf, mf.make_rdm1().astype(np.float64)


def kinetic_energy_from_dm(mol: gto.Mole, dm: np.ndarray) -> float:
    """Kohn-Sham non-interacting kinetic energy T_s = Tr[P T_AO].

    PySCF's int1e_kin integral already includes the -1/2 Laplacian operator.
    The RKS density matrix is spin-summed, so no extra factor of 2 is applied.
    """
    kinetic_ao = mol.intor("int1e_kin")
    return float(np.einsum("ij,ji->", dm, kinetic_ao))


def kinetic_density_from_dm(
    mol: gto.Mole,
    dm: np.ndarray,
    coords_bohr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """AO-gradient kinetic density components on real-space grid.

    derivative_components[:, a] is
        sum_mu,nu P_mu,nu d_a chi_mu(r) d_a chi_nu(r)
    and tau = 1/2 * sum_a derivative_components[:, a].
    """
    ao_deriv = np.asarray(dft.numint.eval_ao(mol, coords_bohr, deriv=1), dtype=np.float64)
    grad_ao = ao_deriv[1:4]  # (3, n_grid, n_ao)
    derivative_components = np.einsum("xgi,ij,xgj->gx", grad_ao, dm, grad_ao, optimize=True)
    tau = 0.5 * np.sum(derivative_components, axis=1, keepdims=True)
    return derivative_components.astype(np.float64), tau.astype(np.float64)


def hartree_potential_from_dm(
    mol: gto.Mole,
    dm: np.ndarray,
    coords_bohr: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    """Electron Hartree potential v_H(r) from AO density matrix."""
    values = np.empty((len(coords_bohr), 1), dtype=np.float64)
    for start in range(0, len(coords_bohr), chunk_size):
        stop = min(start + chunk_size, len(coords_bohr))
        fake_mol = gto.fakemol_for_charges(coords_bohr[start:stop])
        ints = df.incore.aux_e2(mol, fake_mol, intor="int3c2e", aosym="s1")
        values[start:stop, 0] = np.einsum("ijp,ij->p", ints, dm, optimize=True)
    return values


def local_xc_potential_from_dm(
    mol: gto.Mole,
    mf: dft.rks.RKS,
    dm: np.ndarray,
    coords_bohr: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Local/semilocal XC potential ingredient on grid.

    For LDA this is the actual local v_xc(r). For GGA/hybrid functionals this
    stores the vrho ingredient only, so it is an approximate scalar diagnostic
    rather than a rigorous multiplicative KS potential.
    """
    ni = dft.numint.NumInt()
    xctype = ni._xc_type(mf.xc)
    deriv = 1 if xctype in {"GGA", "MGGA"} else 0
    ao_eval = dft.numint.eval_ao(mol, coords_bohr, deriv=deriv)
    rho_eval = ni.eval_rho(mol, ao_eval, dm, xctype=xctype)
    _, vxc, _, _ = ni.eval_xc(mf.xc, rho_eval, spin=0, deriv=1)
    vrho = np.asarray(vxc[0], dtype=np.float64).reshape(-1, 1)
    if xctype == "LDA":
        reference = "lda_local_vxc"
    elif xctype == "GGA":
        reference = "gga_vrho_only_approx"
    else:
        reference = f"{xctype.lower()}_vrho_only_approx"
    return vrho, reference


def chemical_potential_from_mo(mf: dft.rks.RKS) -> float:
    occupied = np.asarray(mf.mo_occ) > 1e-8
    if not np.any(occupied):
        return float("nan")
    return float(np.max(np.asarray(mf.mo_energy)[occupied]))


def kinetic_potential_reference(
    mol: gto.Mole,
    mf: dft.rks.RKS,
    dm: np.ndarray,
    coords_bohr: np.ndarray,
    v_ext: np.ndarray,
    rho_diag: np.ndarray,
    cell_volume: float,
) -> dict[str, object]:
    """KEDF kinetic-potential reference v_Ts = mu - v_s on the grid.

    This is rigorous for LDA-like local KS potentials. For GGA and hybrid
    functionals the stored XC scalar is a vrho-only diagnostic and is marked as
    approximate in the metadata.
    """
    v_h = hartree_potential_from_dm(mol, dm, coords_bohr)
    v_xc, xc_reference = local_xc_potential_from_dm(mol, mf, dm, coords_bohr)
    mu = chemical_potential_from_mo(mf)
    v_s = v_ext + v_h + v_xc
    v_ts = mu - v_s

    rho_weight = np.maximum(np.asarray(rho_diag, dtype=np.float64), 0.0)
    norm = float(np.sum(rho_weight) * cell_volume)
    if norm > 1e-14:
        center = float(np.sum(v_ts * rho_weight) * cell_volume / norm)
    else:
        center = float(np.mean(v_ts))
    v_ts_centered = v_ts - center
    return {
        "hartree_potential": v_h,
        "xc_potential_local": v_xc,
        "ks_potential": v_s,
        "kinetic_potential": v_ts,
        "kinetic_potential_centered": v_ts_centered,
        "kinetic_potential_center": center,
        "chemical_potential_hartree": mu,
        "kinetic_potential_reference": f"mu_minus_vs_{xc_reference}",
    }


def write_npz(record: Qm9Record, args: argparse.Namespace, output_path: Path) -> dict[str, object]:
    mol, mf, dm = run_dft(record, basis=args.basis, xc=args.xc, grid_level=args.grid_level)
    kinetic_energy = kinetic_energy_from_dm(mol, dm)
    coords_bohr = record.coords_angstrom * ANGSTROM_TO_BOHR
    coords_bohr_centered = coords_bohr - np.mean(coords_bohr, axis=0, keepdims=True)
    atomic_numbers = np.asarray([ELEMENT_Z[symbol] for symbol in record.symbols], dtype=np.float64)
    axis, points_bohr = make_grid(
        coords_bohr,
        args.axis_points,
        args.padding_bohr,
        grid_spacing_bohr=args.grid_spacing_bohr,
        max_axis_points=args.max_axis_points,
    )
    axis_points = len(axis)

    absolute_grid_bohr = points_bohr + np.mean(coords_bohr, axis=0, keepdims=True)
    ao = mol.eval_gto("GTOval_sph", absolute_grid_bohr)
    gamma_matrix = (ao @ dm @ ao.T).astype(np.float64)
    gamma_matrix = 0.5 * (gamma_matrix + gamma_matrix.T)
    derivative_true_ao, tau_true_ao = kinetic_density_from_dm(mol, dm, absolute_grid_bohr)

    step = float(axis[1] - axis[0])
    trace_grid = float(np.trace(gamma_matrix) * step**3)
    gamma_trace_scale = 1.0
    if abs(trace_grid) > 1e-12:
        gamma_trace_scale = mol.nelectron / trace_grid
        gamma_matrix *= gamma_trace_scale
        derivative_true_ao *= gamma_trace_scale
        tau_true_ao *= gamma_trace_scale

    potential, grad = nuclear_potential_and_grad(points_bohr, coords_bohr_centered, atomic_numbers)
    rho_diag = np.diag(gamma_matrix).reshape(-1, 1).astype(np.float64)
    kp_payload: dict[str, object] = {}
    if not args.skip_kinetic_potential:
        kp_payload = kinetic_potential_reference(
            mol,
            mf,
            dm,
            absolute_grid_bohr,
            potential.astype(np.float64),
            rho_diag,
            cell_volume=step**3,
        )
    local_features = build_local_features(
        points_bohr,
        coords_bohr_centered,
        atomic_numbers,
        potential,
        grad,
        electron_count=mol.nelectron,
    )
    global_context = build_global_context(record.symbols, coords_bohr_centered, electron_count=mol.nelectron)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points=points_bohr.astype(np.float32),
        gamma_matrix=gamma_matrix.astype(np.float32),
        derivative_true_ao=derivative_true_ao.astype(np.float32),
        tau_true_ao=tau_true_ao.astype(np.float32),
        local_features=local_features,
        global_context=global_context,
        potential=potential,
        grad_potential=grad,
        hartree_potential=np.asarray(
            kp_payload.get("hartree_potential", np.full((len(points_bohr), 1), np.nan)),
            dtype=np.float32,
        ),
        xc_potential_local=np.asarray(
            kp_payload.get("xc_potential_local", np.full((len(points_bohr), 1), np.nan)),
            dtype=np.float32,
        ),
        ks_potential=np.asarray(
            kp_payload.get("ks_potential", np.full((len(points_bohr), 1), np.nan)),
            dtype=np.float32,
        ),
        kinetic_potential=np.asarray(
            kp_payload.get("kinetic_potential", np.full((len(points_bohr), 1), np.nan)),
            dtype=np.float32,
        ),
        kinetic_potential_centered=np.asarray(
            kp_payload.get("kinetic_potential_centered", np.full((len(points_bohr), 1), np.nan)),
            dtype=np.float32,
        ),
        electron_count=np.array(mol.nelectron, dtype=np.float32),
        occupancies=mf.mo_occ.astype(np.float32),
        orbital_energies=mf.mo_energy.astype(np.float32),
        atom_symbols=np.asarray(record.symbols),
        atom_coords_bohr=coords_bohr_centered.astype(np.float32),
        smiles_gdb=np.asarray(record.smiles_gdb),
        smiles_relaxed=np.asarray(record.smiles_relaxed),
        inchi_gdb=np.asarray(record.inchi_gdb),
        inchi_relaxed=np.asarray(record.inchi_relaxed),
        formula=np.asarray(molecular_formula(record.symbols)),
        axis_points=np.array(axis_points, dtype=np.int32),
        grid_spacing_bohr=np.array(step, dtype=np.float32),
        target_grid_spacing_bohr=np.array(
            args.grid_spacing_bohr if args.grid_spacing_bohr is not None else step,
            dtype=np.float32,
        ),
        grid_radius_bohr=np.array(float(np.max(np.abs(axis))), dtype=np.float32),
        box_length_bohr=np.array(float(axis[-1] - axis[0]), dtype=np.float32),
        kinetic_energy_hartree=np.array(kinetic_energy, dtype=np.float32),
        gamma_trace_scale=np.array(gamma_trace_scale, dtype=np.float32),
        chemical_potential_hartree=np.array(
            float(kp_payload.get("chemical_potential_hartree", np.nan)),
            dtype=np.float32,
        ),
        kinetic_potential_center=np.array(
            float(kp_payload.get("kinetic_potential_center", np.nan)),
            dtype=np.float32,
        ),
        kinetic_potential_reference=np.asarray(
            str(kp_payload.get("kinetic_potential_reference", "not_computed"))
        ),
    )
    return {
        "system_id": output_path.stem,
        "qm9_id": record.qm9_id,
        "formula": molecular_formula(record.symbols),
        "smiles_gdb": record.smiles_gdb,
        "smiles_relaxed": record.smiles_relaxed,
        "inchi_gdb": record.inchi_gdb,
        "inchi_relaxed": record.inchi_relaxed,
        "n_atoms": len(record.symbols),
        "electron_count": int(mol.nelectron),
        "dft_energy_hartree": float(mf.e_tot),
        "kinetic_energy_hartree": kinetic_energy,
        "gamma_trace_scale": gamma_trace_scale,
        "chemical_potential_hartree": float(kp_payload.get("chemical_potential_hartree", np.nan)),
        "kinetic_potential_reference": str(kp_payload.get("kinetic_potential_reference", "not_computed")),
        "grid_trace": float(np.trace(gamma_matrix) * step**3),
        "axis_points": int(axis_points),
        "grid_spacing_bohr": float(step),
        "target_grid_spacing_bohr": (
            float(args.grid_spacing_bohr) if args.grid_spacing_bohr is not None else float(step)
        ),
        "grid_radius_bohr": float(np.max(np.abs(axis))),
        "box_length_bohr": float(axis[-1] - axis[0]),
        "basis": args.basis,
        "xc": args.xc,
    }


REQUIRED_EXISTING_NPZ_KEYS = (
    "points",
    "gamma_matrix",
    "local_features",
    "global_context",
    "electron_count",
)


def manifest_row_from_existing_npz(
    *,
    output_path: Path,
    record: Qm9Record,
    args: argparse.Namespace,
    index: int,
) -> dict[str, object] | None:
    """Return manifest row for a reusable NPZ, or remove it if it is invalid."""
    try:
        with np.load(output_path, allow_pickle=True) as payload:
            missing = [key for key in REQUIRED_EXISTING_NPZ_KEYS if key not in payload]
            if missing:
                raise KeyError(f"missing required keys: {', '.join(missing)}")
            return {
                "index": index,
                "split": "train" if index < 500 else "val",
                "system_id": output_path.stem,
                "qm9_id": record.qm9_id,
                "formula": molecular_formula(record.symbols),
                "smiles_gdb": record.smiles_gdb,
                "smiles_relaxed": record.smiles_relaxed,
                "inchi_gdb": record.inchi_gdb,
                "inchi_relaxed": record.inchi_relaxed,
                "n_atoms": len(record.symbols),
                "electron_count": float(payload["electron_count"]),
                "basis": args.basis,
                "xc": args.xc,
                "axis_points": int(payload["axis_points"]) if "axis_points" in payload else "",
                "grid_spacing_bohr": float(payload["grid_spacing_bohr"]) if "grid_spacing_bohr" in payload else "",
                "box_length_bohr": float(payload["box_length_bohr"]) if "box_length_bohr" in payload else "",
                "npz_file": output_path.name,
                "xyz_file": f"xyz/{output_path.stem}.xyz",
            }
    except Exception as exc:
        print(f"[repair] invalid existing NPZ, rebuilding: {output_path} ({exc})")
        if output_path.exists():
            output_path.unlink()
        return None


def write_xyz(record: Qm9Record, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        str(len(record.symbols)),
        f"{record.qm9_id} formula={molecular_formula(record.symbols)} smiles={record.smiles_gdb}",
    ]
    for symbol, coord in zip(record.symbols, record.coords_angstrom):
        lines.append(f"{symbol:2s} {coord[0]: .10f} {coord[1]: .10f} {coord[2]: .10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readable_indices(output_dir: Path, manifest: list[dict[str, object]]) -> None:
    fields = [
        "index",
        "split",
        "system_id",
        "qm9_id",
        "formula",
        "n_atoms",
        "electron_count",
        "smiles_gdb",
        "inchi_gdb",
        "basis",
        "xc",
        "axis_points",
        "grid_spacing_bohr",
        "box_length_bohr",
        "npz_file",
        "xyz_file",
    ]
    csv_path = output_dir / "molecule_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in manifest:
            writer.writerow({field: row.get(field, "") for field in fields})

    md_path = output_dir / "molecule_index.md"
    lines = [
        "# Molecule Index",
        "",
        "| index | split | system_id | formula | atoms | electrons | SMILES |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    for row in manifest:
        lines.append(
            "| {index} | {split} | {system_id} | {formula} | {n_atoms} | {electron_count} | `{smiles_gdb}` |".format(
                **row
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = read_qm9_records(args.qm9_tar, args.max_atoms)
    if args.selection == "smallest":
        records = sorted(
            records,
            key=lambda record: (
                len(record.symbols),
                sum(ELEMENT_Z[symbol] for symbol in record.symbols),
                record.qm9_id,
            ),
        )
    else:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(records)
    selected = records[: args.num_systems]
    if len(selected) < args.num_systems:
        raise RuntimeError(f"Only found {len(selected)} suitable QM9 records.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, record in enumerate(selected):
        output_path = args.output_dir / f"{idx:04d}_{record.qm9_id}.npz"
        if output_path.exists():
            print(f"[{idx + 1}/{len(selected)}] exists: {output_path}")
            row = manifest_row_from_existing_npz(
                output_path=output_path,
                record=record,
                args=args,
                index=idx,
            )
            if row is not None:
                manifest.append(row)
                write_xyz(record, args.output_dir / "xyz" / f"{output_path.stem}.xyz")
                continue
        print(f"[{idx + 1}/{len(selected)}] DFT -> NPZ: {record.qm9_id} ({len(record.symbols)} atoms)")
        row = write_npz(record, args, output_path)
        row.update(
            {
                "index": idx,
                "split": "train" if idx < 500 else "val",
                "npz_file": output_path.name,
                "xyz_file": f"xyz/{output_path.stem}.xyz",
            }
        )
        manifest.append(row)
        write_xyz(record, args.output_dir / "xyz" / f"{output_path.stem}.xyz")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readable_indices(args.output_dir, manifest)
    print(f"Wrote {len(manifest)} NPZ files to {args.output_dir}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote readable index to {args.output_dir / 'molecule_index.csv'}")


if __name__ == "__main__":
    main()
