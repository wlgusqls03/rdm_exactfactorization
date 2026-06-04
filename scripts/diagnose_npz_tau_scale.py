from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


ANGSTROM_TO_BOHR = 1.889726124565062


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def scalar(payload: np.lib.npyio.NpzFile, key: str, default: float = float("nan")) -> float:
    if key not in payload:
        return default
    value = np.asarray(payload[key])
    if value.size == 0:
        return default
    return float(value.reshape(-1)[0])


def infer_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = []
    for dim in range(3):
        axis = np.unique(np.round(points[:, dim], decimals=8)).astype(np.float64)
        axes.append(axis)
    return axes[0], axes[1], axes[2]


def spacing_stats(axis: np.ndarray) -> tuple[float, float, float]:
    diffs = np.diff(axis)
    return float(np.min(diffs)), float(np.mean(diffs)), float(np.max(diffs))


def print_row(label: str, value: object) -> None:
    print(f"{label:<42}: {value}")


def percentile_summary(values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    qs = np.percentile(flat, [0, 50, 95, 99, 100])
    return " / ".join(f"{value:.8e}" for value in qs)


def tau_fd_from_gamma(gamma: np.ndarray, axis_points: int, step: float, stencil: str) -> tuple[np.ndarray, np.ndarray]:
    (_, _, _, derivative), tau = prepare_stencil_targets(
        axis_points=axis_points,
        gamma_matrix=np.asarray(gamma, dtype=np.float64),
        step=float(step),
        tau_stencil=stencil,
    )
    return derivative.astype(np.float64), tau.astype(np.float64)


def summarize_tau_candidate(
    name: str,
    gamma: np.ndarray,
    axis_points: int,
    step: float,
    stencil: str,
    tau_ao: np.ndarray | None,
) -> tuple[str, float, float, float]:
    _, tau_fd = tau_fd_from_gamma(gamma, axis_points, step, stencil)
    fd_rms = rms(tau_fd)
    if tau_ao is None:
        return name, fd_rms, float("nan"), float("nan")
    tau_ao_interior = interior_values(tau_ao, axis_points, stencil)
    ao_rms = rms(tau_ao_interior)
    ratio = fd_rms / max(ao_rms, 1e-30)
    return name, fd_rms, ratio, mae(tau_fd, tau_ao_interior)


def interior_indices(axis_points: int, stencil: str) -> np.ndarray:
    offsets = (1, 2) if stencil.strip().lower() in {"richardson", "richardson4", "extrapolated"} else (1,)
    margin = max(offsets)
    idx = []
    for i in range(margin, axis_points - margin):
        for j in range(margin, axis_points - margin):
            for k in range(margin, axis_points - margin):
                idx.append((i * axis_points + j) * axis_points + k)
    return np.asarray(idx, dtype=np.int64)


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

    derivative_arr = np.asarray(derivative_true, dtype=np.float64)
    tau = 0.5 * np.sum(derivative_arr, axis=1, keepdims=True)
    return (
        np.asarray(interior, dtype=np.int64),
        np.asarray(left_idx, dtype=np.int64),
        np.asarray(right_idx, dtype=np.int64),
        derivative_arr,
    ), tau


def interior_values(values: np.ndarray, axis_points: int, stencil: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr[interior_indices(axis_points, stencil)]


def diagnose(path: Path, stencil: str) -> None:
    with np.load(path, allow_pickle=True) as payload:
        print()
        print("=" * 88)
        print(path)
        print("=" * 88)
        print_row("keys", ", ".join(sorted(payload.files)))

        points = np.asarray(payload["points"], dtype=np.float64)
        gamma = np.asarray(payload["gamma_matrix"], dtype=np.float64)
        rho_payload = np.asarray(payload["rho_diag"], dtype=np.float64) if "rho_diag" in payload else None
        tau_ao = np.asarray(payload["tau_true_ao"], dtype=np.float64) if "tau_true_ao" in payload else None
        deriv_ao = np.asarray(payload["derivative_true_ao"], dtype=np.float64) if "derivative_true_ao" in payload else None

        axis_x, axis_y, axis_z = infer_axis(points)
        if not (len(axis_x) == len(axis_y) == len(axis_z)):
            raise ValueError("Grid is not cubic.")
        axis_points = len(axis_x)
        hx = spacing_stats(axis_x)
        hy = spacing_stats(axis_y)
        hz = spacing_stats(axis_z)
        h_coord = float(np.mean([hx[1], hy[1], hz[1]]))
        h_stored = scalar(payload, "grid_spacing_bohr", h_coord)
        dvol = h_coord**3
        electron_count = scalar(payload, "electron_count", float("nan"))
        gamma_trace_scale = scalar(payload, "gamma_trace_scale", float("nan"))

        print()
        print("[Grid]")
        print_row("axis points", axis_points)
        print_row("h coord x min/mean/max", f"{hx[0]:.8e} / {hx[1]:.8e} / {hx[2]:.8e}")
        print_row("h coord y min/mean/max", f"{hy[0]:.8e} / {hy[1]:.8e} / {hy[2]:.8e}")
        print_row("h coord z min/mean/max", f"{hz[0]:.8e} / {hz[1]:.8e} / {hz[2]:.8e}")
        print_row("h stored grid_spacing_bohr", f"{h_stored:.8e}")
        print_row("h stored / h coord", f"{h_stored / max(h_coord, 1e-30):.8e}")
        print_row("cell volume h_coord^3", f"{dvol:.8e}")
        print_row("electron_count", f"{electron_count:.8e}")
        print_row("gamma_trace_scale", f"{gamma_trace_scale:.8e}")

        gamma_diag = np.diag(gamma).reshape(-1, 1)
        rho_ref = rho_payload.reshape(-1, 1) if rho_payload is not None else gamma_diag
        print()
        print("[Gamma Diagonal / Density]")
        print_row("gamma shape", gamma.shape)
        print_row("mean gamma_ii", f"{float(np.mean(gamma_diag)):.8e}")
        print_row("mean rho_ref", f"{float(np.mean(rho_ref)):.8e}")
        print_row("MAE(gamma_ii, rho_ref)", f"{mae(gamma_diag, rho_ref):.8e}")
        print_row("MAE(gamma_ii / dV, rho_ref)", f"{mae(gamma_diag / dvol, rho_ref):.8e}")
        print_row("MAE(gamma_ii * dV, rho_ref)", f"{mae(gamma_diag * dvol, rho_ref):.8e}")
        print_row("sum gamma_ii * dV", f"{float(np.sum(gamma_diag) * dvol):.8e}")
        print_row("sum gamma_ii", f"{float(np.sum(gamma_diag)):.8e}")
        print_row("sum rho_ref * dV", f"{float(np.sum(rho_ref) * dvol):.8e}")
        print_row("mean gamma_ii/rho_ref", f"{float(np.mean(gamma_diag / np.maximum(rho_ref, 1e-30))):.8e}")
        print_row("mean gamma_ii/(rho_ref*dV)", f"{float(np.mean(gamma_diag / np.maximum(rho_ref * dvol, 1e-30))):.8e}")

        if tau_ao is not None:
            tau_ao_interior = interior_values(tau_ao, axis_points, stencil)
            print()
            print("[AO Tau Reference]")
            print_row("tau AO shape", tau_ao.shape)
            print_row("tau AO RMS interior", f"{rms(tau_ao_interior):.8e}")
            print_row("tau AO RMS full", f"{rms(tau_ao):.8e}")
            print_row("tau AO integral interior", f"{float(np.sum(tau_ao_interior) * dvol):.8e}")
            print_row("tau AO integral full", f"{float(np.sum(tau_ao) * dvol):.8e}")
            print_row("tau AO min/p50/p95/p99/max", percentile_summary(tau_ao))
            print_row("stored kinetic_energy_hartree", f"{scalar(payload, 'kinetic_energy_hartree'):.8e}")
        else:
            tau_ao_interior = None

        if deriv_ao is not None:
            deriv_ao_interior = interior_values(deriv_ao, axis_points, stencil)
            print_row("deriv AO RMS interior", f"{rms(deriv_ao_interior):.8e}")

        print()
        print("[FD Tau Candidates]")
        candidates: list[tuple[str, np.ndarray, float]] = [
            ("gamma, h_coord", gamma, h_coord),
            ("gamma, h_stored", gamma, h_stored),
            ("gamma, h_coord/10", gamma, h_coord / 10.0),
            ("gamma, h_coord*AngstromToBohr", gamma, h_coord * ANGSTROM_TO_BOHR),
            ("gamma, h_coord/AngstromToBohr", gamma, h_coord / ANGSTROM_TO_BOHR),
            ("gamma/dV, h_coord", gamma / dvol, h_coord),
            ("gamma*dV, h_coord", gamma * dvol, h_coord),
        ]
        rows = [summarize_tau_candidate(name, cand_gamma, axis_points, step, stencil, tau_ao) for name, cand_gamma, step in candidates]
        print(f"{'candidate':<32} {'RMS(FD)':>14} {'RMS(FD)/RMS(AO)':>18} {'MAE(FD,AO)':>14}")
        for name, fd_rms, ratio, err in rows:
            print(f"{name:<32} {fd_rms:14.6e} {ratio:18.6e} {err:14.6e}")

        if tau_ao is not None:
            _, tau_fd = tau_fd_from_gamma(gamma, axis_points, h_coord, stencil)
            fd_rms = rms(tau_fd)
            ao_rms = rms(tau_ao_interior)
            best_h_factor = np.sqrt(fd_rms / max(ao_rms, 1e-30))
            print()
            print("[Scale Inference]")
            print_row("sqrt(RMS(FD)/RMS(AO))", f"{best_h_factor:.8e}")
            print_row("h needed if only h^2 scale wrong", f"{h_coord * best_h_factor:.8e}")
            print_row("h_needed / h_coord", f"{best_h_factor:.8e}")
            print_row("AO/FD RMS scale", f"{ao_rms / max(fd_rms, 1e-30):.8e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose gamma/rho/grid/tau scale consistency in transferable 1-RDM NPZ files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--npz", type=Path, help="Single NPZ file to inspect.")
    group.add_argument("--npz-glob", type=str, help="Glob pattern for NPZ files.")
    parser.add_argument("--index", type=int, default=0, help="Index into --npz-glob results.")
    parser.add_argument("--max-files", type=int, default=1, help="Number of globbed files to inspect.")
    parser.add_argument("--tau-stencil", default="richardson", help="Stencil used for FD tau.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.npz is not None:
        paths = [args.npz]
    else:
        matches = [Path(path) for path in sorted(glob.glob(args.npz_glob)) if not path.endswith(".tmp.npz")]
        if not matches:
            raise FileNotFoundError(f"No NPZ files matched: {args.npz_glob}")
        start = max(int(args.index), 0)
        stop = min(start + max(int(args.max_files), 1), len(matches))
        paths = matches[start:stop]
    for path in paths:
        diagnose(path, args.tau_stencil)


if __name__ == "__main__":
    main()
