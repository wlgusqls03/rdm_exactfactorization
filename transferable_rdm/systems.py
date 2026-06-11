from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .toy import build_toy_raw_system, parse_toy_dimensions
from .utils import flat_index, make_uniform_grid, print_block


_GAMMA_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()
_GAMMA_CACHE_BYTES = 0
_PSI_OCC_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()
_PSI_OCC_CACHE_BYTES = 0
_LIGHT_NPZ_CACHE_VERSION = 1
_MMAP_NPZ_CACHE_VERSION = 1
_MMAP_LAZY_KEYS = {"gamma_matrix", "psi_occ"}
_MMAP_RESIDENT_KEYS = {"local_features"}


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no", "n"}


def gamma_cache_limit_bytes() -> int:
    return int(float(os.environ.get("RDM_GAMMA_CACHE_GB", "1.0")) * (1024**3))


def psi_occ_cache_limit_bytes() -> int:
    return int(float(os.environ.get("RDM_PSI_OCC_CACHE_GB", "2.0")) * (1024**3))


def npz_mmap_cache_enabled() -> bool:
    return env_flag("RDM_NPZ_MMAP_CACHE", False)


def npz_mmap_cache_path(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    root_value = os.environ.get("RDM_NPZ_MMAP_CACHE_DIR", "").strip()
    root = Path(root_value).expanduser() if root_value else source.parent / ".rdm_mmap_cache"
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
    return root / f"{source.stem}-{digest}"


def npz_mmap_cache_valid(path: str | Path, cache_path: Path) -> bool:
    manifest_path = cache_path / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        source = Path(path).expanduser().resolve()
        stat = source.stat()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (
            int(manifest["cache_version"]) == _MMAP_NPZ_CACHE_VERSION
            and int(manifest["source_size"]) == stat.st_size
            and int(manifest["source_mtime_ns"]) == stat.st_mtime_ns
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def build_npz_mmap_cache(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    cache_path = npz_mmap_cache_path(source)
    if npz_mmap_cache_valid(source, cache_path):
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_name(
        f".{cache_path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    temp_path.mkdir(parents=True, exist_ok=False)
    try:
        with np.load(source, allow_pickle=True) as payload:
            keys = list(payload.files)
            for key in keys:
                if key in _MMAP_LAZY_KEYS:
                    continue
                np.save(temp_path / f"{key}.npy", np.asarray(payload[key]), allow_pickle=True)
        stat = source.stat()
        manifest = {
            "cache_version": _MMAP_NPZ_CACHE_VERSION,
            "source_path": str(source),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "keys": keys,
        }
        (temp_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if cache_path.exists():
            shutil.rmtree(cache_path)
        temp_path.replace(cache_path)
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise
    return cache_path


def load_cached_npz_key(path: str | Path, key: str) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    cache_path = build_npz_mmap_cache(source)
    array_path = cache_path / f"{key}.npy"
    if not array_path.exists():
        temp_path = array_path.with_name(f".{array_path.name}.tmp-{os.getpid()}-{time.time_ns()}")
        with np.load(source, allow_pickle=True) as payload:
            if key not in payload:
                raise KeyError(f"{source} is missing {key}.")
            with temp_path.open("wb") as handle:
                np.save(handle, np.asarray(payload[key]), allow_pickle=True)
        temp_path.replace(array_path)
    if key in _MMAP_RESIDENT_KEYS:
        try:
            return np.load(array_path, mmap_mode="r", allow_pickle=True)
        except ValueError:
            pass
    return np.load(array_path, allow_pickle=True)


def write_cached_array(array_path: Path, value: np.ndarray) -> None:
    temp_path = array_path.with_name(f".{array_path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with temp_path.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
    temp_path.replace(array_path)


def load_or_build_grid_cache(
    path: str | Path,
    points: np.ndarray,
    tau_stencil: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = build_npz_mmap_cache(path)
    suffix = tau_stencil.strip().lower()
    array_paths = {
        "axis": cache_path / "__rdm_axis.npy",
        "interior": cache_path / f"__rdm_interior_{suffix}.npy",
        "left": cache_path / f"__rdm_stencil_left_{suffix}.npy",
        "right": cache_path / f"__rdm_stencil_right_{suffix}.npy",
    }
    if not all(array_path.exists() for array_path in array_paths.values()):
        axis = infer_uniform_axis(points)
        interior, left, right = prepare_stencil_indices(len(axis), tau_stencil)
        write_cached_array(array_paths["axis"], axis)
        write_cached_array(array_paths["interior"], interior.astype(np.int32))
        write_cached_array(array_paths["left"], left.astype(np.int32))
        write_cached_array(array_paths["right"], right.astype(np.int32))
    return tuple(
        np.load(array_paths[name], allow_pickle=False)
        for name in ("axis", "interior", "left", "right")
    )


class MmapNpzPayload:
    def __init__(self, source: str | Path):
        self.source = Path(source).expanduser().resolve()
        self.cache_path = build_npz_mmap_cache(self.source)
        manifest = json.loads((self.cache_path / "manifest.json").read_text(encoding="utf-8"))
        self.files = list(manifest["keys"])

    def __contains__(self, key: str) -> bool:
        return key in self.files

    def __getitem__(self, key: str) -> np.ndarray:
        array_path = self.cache_path / f"{key}.npy"
        if not array_path.exists():
            return load_cached_npz_key(self.source, key)
        if key in _MMAP_RESIDENT_KEYS:
            try:
                return np.load(array_path, mmap_mode="r", allow_pickle=True)
            except ValueError:
                pass
        return np.load(array_path, allow_pickle=True)

    def __enter__(self) -> "MmapNpzPayload":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


def open_npz_payload(path: str | Path):
    if npz_mmap_cache_enabled():
        return MmapNpzPayload(path)
    return np.load(path, allow_pickle=True)


def load_gamma_matrix_cached(path: str | Path) -> np.ndarray:
    """Load a large NPZ gamma matrix with a small process-local LRU cache."""
    global _GAMMA_CACHE_BYTES
    key = str(path)
    cached = _GAMMA_CACHE.get(key)
    if cached is not None:
        _GAMMA_CACHE.move_to_end(key)
        return cached

    with open_npz_payload(key) as payload:
        gamma = np.asarray(payload["gamma_matrix"], dtype=np.float32)

    limit = gamma_cache_limit_bytes()
    if limit <= 0:
        return gamma

    while _GAMMA_CACHE and _GAMMA_CACHE_BYTES + gamma.nbytes > limit:
        _, old = _GAMMA_CACHE.popitem(last=False)
        _GAMMA_CACHE_BYTES -= old.nbytes
    if gamma.nbytes <= limit:
        _GAMMA_CACHE[key] = gamma
        _GAMMA_CACHE_BYTES += gamma.nbytes
    return gamma


def load_psi_occ_cached(path: str | Path) -> np.ndarray:
    """Load occupied pseudo-orbitals from NPZ with a process-local LRU cache."""
    global _PSI_OCC_CACHE_BYTES
    key = str(path)
    cached = _PSI_OCC_CACHE.get(key)
    if cached is not None:
        _PSI_OCC_CACHE.move_to_end(key)
        return cached

    with open_npz_payload(key) as payload:
        psi_occ = np.asarray(payload["psi_occ"], dtype=np.float32)

    limit = psi_occ_cache_limit_bytes()
    if limit <= 0:
        return psi_occ

    while _PSI_OCC_CACHE and _PSI_OCC_CACHE_BYTES + psi_occ.nbytes > limit:
        _, old = _PSI_OCC_CACHE.popitem(last=False)
        _PSI_OCC_CACHE_BYTES -= old.nbytes
    if psi_occ.nbytes <= limit:
        _PSI_OCC_CACHE[key] = psi_occ
        _PSI_OCC_CACHE_BYTES += psi_occ.nbytes
    return psi_occ


def npz_light_cache_dir() -> Path | None:
    """Directory for small per-NPZ metadata caches.

    The cache stores arrays such as rho_diag that are needed during corpus
    construction. This lets later runs avoid inflating the full gamma matrix
    before training starts.
    """
    value = os.environ.get("RDM_NPZ_LIGHT_CACHE_DIR", "~/.cache/rdm_exactfactorization/npz_light")
    if value.strip().lower() in {"", "0", "false", "off", "no", "none"}:
        return None
    return Path(value).expanduser()


def npz_light_cache_path(path: str | Path) -> Path | None:
    cache_dir = npz_light_cache_dir()
    if cache_dir is None:
        return None
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npz"


def read_npz_light_cache(path: str | Path) -> dict[str, np.ndarray | float] | None:
    cache_path = npz_light_cache_path(path)
    if cache_path is None or not cache_path.exists():
        return None

    source = Path(path).expanduser().resolve()
    try:
        stat = source.stat()
        with np.load(cache_path, allow_pickle=False) as payload:
            version = int(np.asarray(payload["cache_version"]).reshape(-1)[0])
            source_size = int(np.asarray(payload["source_size"]).reshape(-1)[0])
            source_mtime_ns = int(np.asarray(payload["source_mtime_ns"]).reshape(-1)[0])
            if (
                version != _LIGHT_NPZ_CACHE_VERSION
                or source_size != stat.st_size
                or source_mtime_ns != stat.st_mtime_ns
            ):
                return None
            return {
                "rho_diag": np.asarray(payload["rho_diag"], dtype=np.float32),
                "electron_count": float(np.asarray(payload["electron_count"]).reshape(-1)[0]),
            }
    except (OSError, KeyError, ValueError):
        return None


def write_npz_light_cache(path: str | Path, rho_diag: np.ndarray, electron_count: float) -> None:
    cache_path = npz_light_cache_path(path)
    if cache_path is None:
        return
    source = Path(path).expanduser().resolve()
    try:
        stat = source.stat()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            cache_version=np.asarray([_LIGHT_NPZ_CACHE_VERSION], dtype=np.int32),
            source_size=np.asarray([stat.st_size], dtype=np.int64),
            source_mtime_ns=np.asarray([stat.st_mtime_ns], dtype=np.int64),
            rho_diag=np.asarray(rho_diag, dtype=np.float32),
            electron_count=np.asarray([electron_count], dtype=np.float32),
        )
    except OSError:
        # Cache failures should never stop training.
        return


def system_resident_nbytes(system: "SystemRecord") -> int:
    total = 0
    for value in (
        system.points,
        system.local_features,
        system.potential,
        system.grad_potential,
        system.hartree_potential,
        system.xc_potential_local,
        system.ks_potential,
        system.kinetic_potential,
        system.kinetic_potential_centered,
        system.global_context,
        system.gamma_matrix,
        system.gamma_pairs,
        system.rho_diag,
        system.psi_occ,
        system.rho_sad,
        system.rho_cation,
        system.rho_anion,
        system.pair_left,
        system.pair_right,
        system.pair_distance,
        system.pair_weights,
        system.diagonal_pair_indices,
        system.interior_point_indices,
        system.stencil_left,
        system.stencil_right,
        system.derivative_true,
        system.tau_true,
        system.derivative_true_fd,
        system.tau_true_fd,
        system.occupancies,
        system.orbital_energies,
        system.spectral_subset,
    ):
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
    return total


@dataclass
class SystemRecord:
    """한 시스템에 대한 학습 / 검증용 데이터를 한 묶음으로 저장.

    Shapes
    ------
    axis              : (n_axis,)
    points            : (n_points, 3)
    local_features    : (n_points, d_local)
    potential         : (n_points, 1)
    grad_potential    : (n_points, 3)
    global_context    : (d_global,)
    gamma_matrix      : (n_points, n_points)
    gamma_pairs       : (n_points^2, 1)
    rho_diag          : (n_points, 1)
    rho_sad           : (n_points, 1) or None
    pair_left/right   : (n_points^2,)
    pair_distance     : (n_points^2,)
    pair_weights      : (n_points^2, 1)
    stencil_left/right: (n_interior, 3, 4)
    derivative_true   : (n_interior, 3)
    tau_true          : (n_interior, 1)
    """

    system_id: str
    family: str
    axis: np.ndarray
    points: np.ndarray
    step: float
    cell_volume: float

    local_features: np.ndarray
    potential: np.ndarray
    grad_potential: np.ndarray
    hartree_potential: np.ndarray
    xc_potential_local: np.ndarray
    ks_potential: np.ndarray
    kinetic_potential: np.ndarray
    kinetic_potential_centered: np.ndarray
    global_context: np.ndarray

    gamma_matrix: np.ndarray
    gamma_pairs: np.ndarray
    rho_diag: np.ndarray
    psi_occ: np.ndarray | None
    rho_sad: np.ndarray | None
    rho_cation: np.ndarray | None
    rho_anion: np.ndarray | None

    pair_left: np.ndarray
    pair_right: np.ndarray
    pair_distance: np.ndarray
    pair_weights: np.ndarray
    diagonal_pair_indices: np.ndarray
    category_indices: dict[str, np.ndarray]

    interior_point_indices: np.ndarray
    stencil_left: np.ndarray
    stencil_right: np.ndarray
    derivative_true: np.ndarray
    tau_true: np.ndarray
    derivative_true_fd: np.ndarray | None
    tau_true_fd: np.ndarray | None

    electron_count: float
    occupancies: np.ndarray
    orbital_energies: np.ndarray
    spectral_subset: np.ndarray
    metadata: dict[str, object]

    def load_gamma_matrix(self) -> np.ndarray:
        if self.gamma_matrix is not None and self.gamma_matrix.size:
            return self.gamma_matrix
        source_path = self.metadata.get("source_path")
        if not source_path:
            raise RuntimeError(f"System {self.system_id} has no gamma matrix or source_path.")
        return load_gamma_matrix_cached(str(source_path))

    def gamma_values(self, left_idx: np.ndarray, right_idx: np.ndarray) -> np.ndarray:
        if (self.psi_occ is not None and self.psi_occ.size) or self.metadata.get("has_psi_occ", False):
            psi = (
                np.asarray(self.psi_occ, dtype=np.float32)
                if self.psi_occ is not None and self.psi_occ.size
                else load_psi_occ_cached(str(self.metadata["source_path"]))
            )
            occ = np.asarray(self.occupancies, dtype=np.float32)
            if psi.shape[1] != occ.shape[0]:
                raise RuntimeError(
                    f"System {self.system_id} psi_occ/occupancies mismatch: "
                    f"{psi.shape[1]} orbitals vs {occ.shape[0]} occupations."
                )
            values = np.sum(psi[left_idx] * psi[right_idx] * occ[None, :], axis=1)
            return values.reshape(-1, 1).astype(np.float32)
        gamma = self.load_gamma_matrix()
        return gamma[left_idx, right_idx].reshape(-1, 1).astype(np.float32)

    def gamma_submatrix(self, indices: np.ndarray) -> np.ndarray:
        if (self.psi_occ is not None and self.psi_occ.size) or self.metadata.get("has_psi_occ", False):
            psi_all = (
                np.asarray(self.psi_occ, dtype=np.float32)
                if self.psi_occ is not None and self.psi_occ.size
                else load_psi_occ_cached(str(self.metadata["source_path"]))
            )
            psi = np.asarray(psi_all[indices], dtype=np.float32)
            occ = np.asarray(self.occupancies, dtype=np.float32)
            weighted = psi * occ[None, :]
            gamma = weighted @ psi.T
            return (0.5 * (gamma + gamma.T)).astype(np.float32)
        gamma = self.load_gamma_matrix()
        return gamma[np.ix_(indices, indices)].astype(np.float32)


def second_derivative_matrix(n: int, step: float) -> np.ndarray:
    """1D second-order central finite-difference Laplacian."""
    main = -2.0 * np.ones(n, dtype=np.float64)
    off = np.ones(n - 1, dtype=np.float64)
    return (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / (step * step)


def solve_1d_schrodinger(axis: np.ndarray, potential: np.ndarray, n_keep: int) -> tuple[np.ndarray, np.ndarray]:
    """1D noninteracting Hamiltonian eigenstate.

    H = -1/2 d^2/dx^2 + V(x)
    """
    step = float(axis[1] - axis[0])
    lap = second_derivative_matrix(len(axis), step)
    hamiltonian = -0.5 * lap + np.diag(potential.astype(np.float64))
    eigvals, eigvecs = np.linalg.eigh(hamiltonian)
    eigvals = eigvals[:n_keep]
    eigvecs = eigvecs[:, :n_keep]

    # discrete normalization: sum_i |psi_i|^2 * dx = 1
    norms = np.sqrt(np.sum(eigvecs**2, axis=0) * step)
    eigvecs = eigvecs / norms[None, :]
    return eigvals.astype(np.float32), eigvecs.astype(np.float32)


def fermi_occupations(energies: np.ndarray, electron_count: float, temperature: float) -> np.ndarray:
    """작은 finite-temperature smearing으로 occupation 생성."""
    temperature = max(float(temperature), 1e-3)
    energies = np.asarray(energies, dtype=np.float64)

    lo = float(np.min(energies) - 20.0 * temperature)
    hi = float(np.max(energies) + 20.0 * temperature)
    for _ in range(120):
        mu = 0.5 * (lo + hi)
        occ = 1.0 / (1.0 + np.exp((energies - mu) / temperature))
        if np.sum(occ) > electron_count:
            hi = mu
        else:
            lo = mu
    mu = 0.5 * (lo + hi)
    occ = 1.0 / (1.0 + np.exp((energies - mu) / temperature))
    return occ.astype(np.float32)


def evaluate_x_potential_and_grad(x: np.ndarray, params: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """x 방향 potential과 미분."""
    omega_x = float(params["omega_x"])
    quartic_x = float(params["quartic_x"])
    centers_x = np.asarray(params["centers_x"], dtype=np.float32)
    depths = np.asarray(params["depths"], dtype=np.float32)
    widths = np.asarray(params["widths"], dtype=np.float32)

    potential = 0.5 * (omega_x**2) * x**2 + quartic_x * x**4
    grad = (omega_x**2) * x + 4.0 * quartic_x * x**3

    for center, depth, width in zip(centers_x, depths, widths):
        dx = x - center
        gauss = np.exp(-(dx / width) ** 2)
        potential -= depth * gauss
        grad += depth * (2.0 * dx / (width * width)) * gauss
    return potential.astype(np.float32), grad.astype(np.float32)


def evaluate_separable_potential(points: np.ndarray, params: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """3D separable KS-like potential V(x,y,z)=Vx(x)+Vy(y)+Vz(z)."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    vx, dvx = evaluate_x_potential_and_grad(x, params)
    omega_y = float(params["omega_y"])
    omega_z = float(params["omega_z"])

    vy = 0.5 * (omega_y**2) * y**2
    vz = 0.5 * (omega_z**2) * z**2
    dvy = (omega_y**2) * y
    dvz = (omega_z**2) * z

    potential = (vx + vy + vz).reshape(-1, 1).astype(np.float32)
    grad = np.stack([dvx, dvy, dvz], axis=1).astype(np.float32)
    return potential, grad


def sample_ks_like_parameters(config: ExperimentConfig, rng: np.random.Generator) -> dict[str, object]:
    """학습용 separable KS-like 계 파라미터를 랜덤 생성."""
    num_wells = int(rng.integers(1, config.max_wells + 1))

    if num_wells == 1:
        centers_x = np.array([rng.uniform(-1.2, 1.2)], dtype=np.float32)
    else:
        bond = rng.uniform(1.0, 2.4)
        shift = rng.uniform(-0.3, 0.3)
        centers_x = np.array([shift - 0.5 * bond, shift + 0.5 * bond], dtype=np.float32)
        centers_x = np.sort(centers_x).astype(np.float32)

    depths = rng.uniform(0.9, 2.2, size=num_wells).astype(np.float32)
    widths = rng.uniform(0.45, 1.05, size=num_wells).astype(np.float32)

    electron_count = float(rng.uniform(1.0, min(4.8, config.max_orbitals - 0.2)))
    temperature = float(rng.uniform(0.03, 0.12))

    return {
        "num_wells": num_wells,
        "centers_x": centers_x,
        "depths": depths,
        "widths": widths,
        "omega_x": float(rng.uniform(0.06, 0.16)),
        "omega_y": float(rng.uniform(0.10, 0.32)),
        "omega_z": float(rng.uniform(0.10, 0.30)),
        "quartic_x": float(rng.uniform(0.0, 0.01)),
        "electron_count": electron_count,
        "temperature": temperature,
    }


def enumerate_3d_states(
    eig_x: np.ndarray,
    eig_y: np.ndarray,
    eig_z: np.ndarray,
    max_orbitals: int,
) -> list[tuple[float, tuple[int, int, int]]]:
    """낮은 에너지의 3D product state 목록."""
    states: list[tuple[float, tuple[int, int, int]]] = []
    nx_keep = min(len(eig_x), max_orbitals)
    ny_keep = min(len(eig_y), max_orbitals)
    nz_keep = min(len(eig_z), max_orbitals)
    for ix in range(nx_keep):
        for iy in range(ny_keep):
            for iz in range(nz_keep):
                energy = float(eig_x[ix] + eig_y[iy] + eig_z[iz])
                states.append((energy, (ix, iy, iz)))
    states.sort(key=lambda item: item[0])
    return states[:max_orbitals]


def build_global_context(params: dict[str, object], config: ExperimentConfig) -> np.ndarray:
    """시스템 전체를 요약하는 고정 길이 global descriptor."""
    centers = np.zeros(config.max_wells, dtype=np.float32)
    depths = np.zeros(config.max_wells, dtype=np.float32)
    widths = np.zeros(config.max_wells, dtype=np.float32)

    centers[: len(params["centers_x"])] = np.asarray(params["centers_x"], dtype=np.float32)
    depths[: len(params["depths"])] = np.asarray(params["depths"], dtype=np.float32)
    widths[: len(params["widths"])] = np.asarray(params["widths"], dtype=np.float32)

    volume = (2.0 * config.domain_radius) ** 3
    avg_density = float(params["electron_count"]) / volume
    context = np.concatenate(
        [
            np.array(
                [
                    params["electron_count"] / max(config.max_orbitals, 1),
                    params["num_wells"] / max(config.max_wells, 1),
                    params["omega_x"],
                    params["omega_y"],
                    params["omega_z"],
                    params["quartic_x"],
                    params["temperature"],
                    avg_density,
                ],
                dtype=np.float32,
            ),
            centers / max(config.domain_radius, 1e-6),
            depths / 3.0,
            widths / max(config.domain_radius, 1e-6),
        ]
    )
    return context.astype(np.float32)


def build_local_features(
    points: np.ndarray,
    potential: np.ndarray,
    grad: np.ndarray,
    params: dict[str, object],
    config: ExperimentConfig,
) -> np.ndarray:
    """각 point에서 사용할 local descriptor.

    미래에 실제 DFT 데이터를 넣을 때도 이 자리만 바꾸면 된다.
    synthetic system에서는 analytic potential 정보와 기하 정보를 함께 넣는다.
    """
    coords_norm = points / max(config.domain_radius, 1e-6)  # (n_points, 3)
    pot_scale = max(float(np.std(potential)), 1.0)
    pot_feat = potential / pot_scale  # (n_points, 1)
    grad_feat = grad / pot_scale      # (n_points, 3)

    radial_columns: list[np.ndarray] = []
    centers_x = np.asarray(params["centers_x"], dtype=np.float32)
    depths = np.asarray(params["depths"], dtype=np.float32)
    widths = np.asarray(params["widths"], dtype=np.float32)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    for slot in range(config.max_wells):
        if slot < len(centers_x):
            dx = x - centers_x[slot]
            radial = np.sqrt(dx * dx + y * y + z * z)
            width = float(widths[slot])
            depth = float(depths[slot])
            radial_columns.extend(
                [
                    (radial / max(config.domain_radius, 1e-6)).reshape(-1, 1),
                    np.exp(-((radial / max(width, 1e-6)) ** 2)).reshape(-1, 1),
                    (depth / np.sqrt(radial * radial + width * width)).reshape(-1, 1),
                ]
            )
        else:
            radial_columns.extend(
                [
                    np.zeros((len(points), 1), dtype=np.float32),
                    np.zeros((len(points), 1), dtype=np.float32),
                    np.zeros((len(points), 1), dtype=np.float32),
                ]
            )

    density_scale = np.full((len(points), 1), params["electron_count"] / max(config.max_orbitals, 1), dtype=np.float32)
    temperature_col = np.full((len(points), 1), params["temperature"], dtype=np.float32)

    features = np.concatenate(
        [coords_norm, pot_feat, grad_feat, density_scale, temperature_col] + radial_columns,
        axis=1,
    )
    return features.astype(np.float32)


def make_pair_weights(points: np.ndarray, left: np.ndarray, right: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """diagonal / near-diagonal pair를 더 강조하는 static base weight."""
    dist = np.linalg.norm(points[left] - points[right], axis=1).astype(np.float32)
    weights = np.ones((len(dist), 1), dtype=np.float32)

    diag_mask = dist < 1e-7
    near_mask = (dist >= 1e-7) & (dist <= step + 1e-7)
    mid_mask = (dist > step + 1e-7) & (dist <= 2.0 * step + 1e-7)
    far_mask = ~(diag_mask | near_mask | mid_mask)

    weights[diag_mask] = 20.0
    weights[near_mask] = 8.0
    weights[mid_mask] = 4.0
    weights[far_mask] = 1.0
    weights /= np.mean(weights)

    categories = {
        "diag": np.where(diag_mask)[0],
        "near": np.where(near_mask)[0],
        "mid": np.where(mid_mask)[0],
        "far": np.where(far_mask)[0],
    }
    return dist, weights.astype(np.float32), categories


def choose_spectral_subset(axis_points: int, target_count: int) -> np.ndarray:
    """occupation penalty에 쓸 coarse point subset.

    full grid에서 너무 큰 eigendecomposition을 피하기 위해, 균일한 coarse subset을 고른다.
    """
    k = int(round(target_count ** (1.0 / 3.0)))
    k = max(2, min(axis_points, k))
    coords = np.linspace(0, axis_points - 1, k).round().astype(int)
    indices = []
    for i in coords:
        for j in coords:
            for l in coords:
                indices.append(flat_index(int(i), int(j), int(l), axis_points))
    return np.array(sorted(set(indices)), dtype=np.int64)


def mixed_derivative_from_stencil(values: np.ndarray, step: float, stencil_order: int) -> np.ndarray:
    """Mixed derivative from stored gamma stencil values.

    `values[..., :4]` contains the standard +/-h central mixed stencil.
    If `stencil_order == 8`, `values[..., 4:8]` contains the same stencil at
    +/-2h and Richardson extrapolation is used.
    """
    d_h = (values[..., 0] - values[..., 1] - values[..., 2] + values[..., 3]) / (4.0 * step * step)
    if stencil_order < 8:
        return d_h
    d_2h = (values[..., 4] - values[..., 5] - values[..., 6] + values[..., 7]) / (16.0 * step * step)
    return (4.0 * d_h - d_2h) / 3.0


def stencil_offsets(axis_points: int, tau_stencil: str) -> tuple[int, ...]:
    method = tau_stencil.strip().lower()
    if method in {"richardson", "richardson4", "extrapolated"} and axis_points >= 5:
        return (1, 2)
    elif method in {"central2", "second", "second_order", "legacy"}:
        return (1,)
    elif axis_points < 5:
        return (1,)
    raise ValueError(f"Unknown RDM_TAU_STENCIL: {tau_stencil}")


def prepare_stencil_indices(
    axis_points: int,
    tau_stencil: str = "central2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """near-diagonal mixed derivative target을 위한 stencil index만 준비."""
    interior = []
    left_idx = []
    right_idx = []

    offsets = stencil_offsets(axis_points, tau_stencil)
    margin = max(offsets)
    for i in range(margin, axis_points - margin):
        for j in range(margin, axis_points - margin):
            for k in range(margin, axis_points - margin):
                center_idx = flat_index(i, j, k, axis_points)
                interior.append(center_idx)

                per_dim_left = []
                per_dim_right = []

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

                    per_dim_left.append(dim_left)
                    per_dim_right.append(dim_right)

                left_idx.append(per_dim_left)
                right_idx.append(per_dim_right)

    return (
        np.asarray(interior, dtype=np.int64),
        np.asarray(left_idx, dtype=np.int64),
        np.asarray(right_idx, dtype=np.int64),
    )


def prepare_stencil_targets(
    axis_points: int,
    gamma_matrix: np.ndarray,
    step: float,
    tau_stencil: str = "central2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """near-diagonal mixed derivative target을 위한 stencil index와 true 값을 준비."""
    interior_idx, left_idx, right_idx = prepare_stencil_indices(axis_points, tau_stencil)
    derivative_true = []

    for per_point_left, per_point_right in zip(left_idx, right_idx):
        per_dim_true = []
        for dim_left, dim_right in zip(per_point_left, per_point_right):
            values = gamma_matrix[dim_left, dim_right]
            deriv = mixed_derivative_from_stencil(np.asarray(values, dtype=np.float64), step, len(values))
            per_dim_true.append(deriv)
        derivative_true.append(per_dim_true)

    derivative_true_arr = np.asarray(derivative_true, dtype=np.float32)  # (n_interior, 3)
    tau_true = 0.5 * np.sum(derivative_true_arr, axis=1, keepdims=True)  # (n_interior, 1)
    return (
        interior_idx,
        left_idx,
        right_idx,
        derivative_true_arr,
    ), tau_true.astype(np.float32)


def finalize_system_record(
    *,
    system_id: str,
    family: str,
    axis: np.ndarray,
    points: np.ndarray,
    local_features: np.ndarray,
    potential: np.ndarray,
    grad_potential: np.ndarray,
    global_context: np.ndarray,
    gamma_matrix: np.ndarray,
    electron_count: float,
    occupancies: np.ndarray,
    orbital_energies: np.ndarray,
    metadata: dict[str, object],
    config: ExperimentConfig,
    keep_gamma_matrix: bool = True,
    derivative_true_grid: np.ndarray | None = None,
    tau_true_grid: np.ndarray | None = None,
    derivative_true_fd: np.ndarray | None = None,
    tau_true_fd: np.ndarray | None = None,
    hartree_potential: np.ndarray | None = None,
    xc_potential_local: np.ndarray | None = None,
    ks_potential: np.ndarray | None = None,
    kinetic_potential: np.ndarray | None = None,
    kinetic_potential_centered: np.ndarray | None = None,
    rho_diag_override: np.ndarray | None = None,
    psi_occ: np.ndarray | None = None,
    rho_sad: np.ndarray | None = None,
    rho_cation: np.ndarray | None = None,
    rho_anion: np.ndarray | None = None,
    stencil_indices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> SystemRecord:
    """raw system data를 학습용 record로 정리."""
    n_points = len(points)
    step = float(axis[1] - axis[0])
    if rho_diag_override is not None:
        rho_diag = np.asarray(rho_diag_override, dtype=np.float32)
        if rho_diag.shape == (n_points,):
            rho_diag = rho_diag.reshape(-1, 1)
        if rho_diag.shape != (n_points, 1):
            raise ValueError(f"rho_diag must have shape ({n_points}, 1), got {rho_diag.shape}")
    elif gamma_matrix.size:
        rho_diag = np.diag(gamma_matrix).reshape(-1, 1).astype(np.float32)
    else:
        raise ValueError("gamma_matrix or rho_diag_override is required to finalize a system record.")

    if stencil_indices is not None and not gamma_matrix.size:
        interior_idx, stencil_left, stencil_right = stencil_indices
        derivative_true = np.zeros((len(interior_idx), 3), dtype=np.float32)
        tau_true = np.zeros((len(interior_idx), 1), dtype=np.float32)
    elif gamma_matrix.size:
        (interior_idx, stencil_left, stencil_right, derivative_true), tau_true = prepare_stencil_targets(
            axis_points=len(axis),
            gamma_matrix=gamma_matrix,
            step=step,
            tau_stencil=config.tau_stencil,
        )
    elif derivative_true_grid is not None and tau_true_grid is not None:
        interior_idx, stencil_left, stencil_right = prepare_stencil_indices(
            axis_points=len(axis),
            tau_stencil=config.tau_stencil,
        )
        derivative_true = np.zeros((len(interior_idx), 3), dtype=np.float32)
        tau_true = np.zeros((len(interior_idx), 1), dtype=np.float32)
    else:
        raise ValueError(
            "Lazy gamma loading requires stored orbital-gradient derivative and tau grids "
            "(legacy keys derivative_true_ao/tau_true_ao are also accepted)."
        )
    if derivative_true_grid is not None:
        derivative_grid = np.asarray(derivative_true_grid, dtype=np.float32)
        if derivative_grid.shape != (n_points, 3):
            raise ValueError(
                f"derivative_true_grid must have shape ({n_points}, 3), got {derivative_grid.shape}"
            )
        derivative_true = derivative_grid[interior_idx]
    if tau_true_grid is not None:
        tau_grid = np.asarray(tau_true_grid, dtype=np.float32)
        if tau_grid.shape == (n_points,):
            tau_grid = tau_grid.reshape(-1, 1)
        if tau_grid.shape != (n_points, 1):
            raise ValueError(f"tau_true_grid must have shape ({n_points}, 1), got {tau_grid.shape}")
        tau_true = tau_grid[interior_idx]

    return SystemRecord(
        system_id=system_id,
        family=family,
        axis=np.asarray(axis, dtype=np.float32),
        points=np.asarray(points, dtype=np.float32),
        step=step,
        cell_volume=step**3,
        local_features=np.asarray(local_features, dtype=np.float32),
        potential=np.asarray(potential, dtype=np.float32),
        grad_potential=np.asarray(grad_potential, dtype=np.float32),
        hartree_potential=(
            np.asarray(hartree_potential, dtype=np.float32)
            if hartree_potential is not None
            else np.empty((0, 1), dtype=np.float32)
        ),
        xc_potential_local=(
            np.asarray(xc_potential_local, dtype=np.float32)
            if xc_potential_local is not None
            else np.empty((0, 1), dtype=np.float32)
        ),
        ks_potential=(
            np.asarray(ks_potential, dtype=np.float32)
            if ks_potential is not None
            else np.empty((0, 1), dtype=np.float32)
        ),
        kinetic_potential=(
            np.asarray(kinetic_potential, dtype=np.float32)
            if kinetic_potential is not None
            else np.empty((0, 1), dtype=np.float32)
        ),
        kinetic_potential_centered=(
            np.asarray(kinetic_potential_centered, dtype=np.float32)
            if kinetic_potential_centered is not None
            else np.empty((0, 1), dtype=np.float32)
        ),
        global_context=np.asarray(global_context, dtype=np.float32),
        gamma_matrix=np.asarray(gamma_matrix, dtype=np.float32) if keep_gamma_matrix else np.empty((0, 0), dtype=np.float32),
        gamma_pairs=np.empty((0, 1), dtype=np.float32),
        rho_diag=rho_diag,
        psi_occ=(
            np.asarray(psi_occ, dtype=np.float32)
            if psi_occ is not None
            else None
        ),
        rho_sad=(
            np.asarray(rho_sad, dtype=np.float32).reshape(n_points, 1)
            if rho_sad is not None
            else None
        ),
        rho_cation=(
            np.asarray(rho_cation, dtype=np.float32).reshape(n_points, 1)
            if rho_cation is not None
            else None
        ),
        rho_anion=(
            np.asarray(rho_anion, dtype=np.float32).reshape(n_points, 1)
            if rho_anion is not None
            else None
        ),
        pair_left=np.empty((0,), dtype=np.int64),
        pair_right=np.empty((0,), dtype=np.int64),
        pair_distance=np.empty((0,), dtype=np.float32),
        pair_weights=np.empty((0, 1), dtype=np.float32),
        diagonal_pair_indices=np.arange(n_points, dtype=np.int64) * (n_points + 1),
        category_indices={},
        interior_point_indices=np.asarray(interior_idx, dtype=np.int32),
        stencil_left=np.asarray(stencil_left, dtype=np.int32),
        stencil_right=np.asarray(stencil_right, dtype=np.int32),
        derivative_true=derivative_true.astype(np.float32),
        tau_true=tau_true.astype(np.float32),
        derivative_true_fd=(
            np.asarray(derivative_true_fd, dtype=np.float32)
            if derivative_true_fd is not None
            else None
        ),
        tau_true_fd=(
            np.asarray(tau_true_fd, dtype=np.float32)
            if tau_true_fd is not None
            else None
        ),
        electron_count=float(electron_count),
        occupancies=np.asarray(occupancies, dtype=np.float32),
        orbital_energies=np.asarray(orbital_energies, dtype=np.float32),
        spectral_subset=choose_spectral_subset(len(axis), config.spectral_subset_points),
        metadata=metadata,
    )


def build_ks_like_system(config: ExperimentConfig, system_index: int, rng: np.random.Generator) -> SystemRecord:
    """separable 3D KS-like synthetic system 생성."""
    axis = np.linspace(-config.domain_radius, config.domain_radius, config.axis_points, dtype=np.float32)
    points = make_uniform_grid(axis)  # (n_points, 3)
    params = sample_ks_like_parameters(config, rng)

    x_potential, _ = evaluate_x_potential_and_grad(axis, params)
    y_potential = 0.5 * (float(params["omega_y"]) ** 2) * axis**2
    z_potential = 0.5 * (float(params["omega_z"]) ** 2) * axis**2

    keep_1d = max(4, min(config.axis_points, config.max_orbitals))
    eig_x, vec_x = solve_1d_schrodinger(axis, x_potential, keep_1d)
    eig_y, vec_y = solve_1d_schrodinger(axis, y_potential, keep_1d)
    eig_z, vec_z = solve_1d_schrodinger(axis, z_potential, keep_1d)

    states = enumerate_3d_states(eig_x, eig_y, eig_z, config.max_orbitals)
    energies = np.array([state[0] for state in states], dtype=np.float32)
    occupancies = fermi_occupations(energies, float(params["electron_count"]), float(params["temperature"]))

    orbitals = []
    for _, (ix, iy, iz) in states:
        psi = np.einsum("i,j,k->ijk", vec_x[:, ix], vec_y[:, iy], vec_z[:, iz]).reshape(-1)
        orbitals.append(psi.astype(np.float32))
    orbital_matrix = np.stack(orbitals, axis=1).astype(np.float32)  # (n_points, n_orb)

    gamma_matrix = np.zeros((len(points), len(points)), dtype=np.float32)
    for orb, occ in zip(orbitals, occupancies):
        gamma_matrix += float(occ) * np.outer(orb, orb).astype(np.float32)

    potential, grad = evaluate_separable_potential(points, params)
    local_features = build_local_features(points, potential, grad, params, config)
    global_context = build_global_context(params, config)

    metadata = {
        "num_wells": int(params["num_wells"]),
        "centers_x": np.asarray(params["centers_x"], dtype=np.float32).tolist(),
        "depths": np.asarray(params["depths"], dtype=np.float32).tolist(),
        "widths": np.asarray(params["widths"], dtype=np.float32).tolist(),
        "omega_x": float(params["omega_x"]),
        "omega_y": float(params["omega_y"]),
        "omega_z": float(params["omega_z"]),
        "quartic_x": float(params["quartic_x"]),
        "temperature": float(params["temperature"]),
        "electron_count": float(params["electron_count"]),
    }
    return finalize_system_record(
        system_id=f"ks_like_{system_index:03d}",
        family="ks_like",
        axis=axis,
        points=points,
        local_features=local_features,
        potential=potential,
        grad_potential=grad,
        global_context=global_context,
        gamma_matrix=gamma_matrix,
        electron_count=float(params["electron_count"]),
        occupancies=occupancies,
        orbital_energies=energies,
        metadata=metadata,
        config=config,
    )


def build_dimensional_toy_system(
    config: ExperimentConfig,
    system_index: int,
    active_dimension: int,
    rng: np.random.Generator,
) -> SystemRecord:
    """Finalize an isolated toy generator result into the shared model schema."""
    raw = build_toy_raw_system(config, active_dimension, rng)
    return finalize_system_record(
        system_id=f"toy_{active_dimension}d_{system_index:04d}",
        family="toy_dimensional",
        axis=raw.axis,
        points=raw.points,
        local_features=raw.local_features,
        potential=raw.potential,
        grad_potential=raw.gradient,
        global_context=raw.global_context,
        gamma_matrix=raw.gamma_matrix,
        electron_count=raw.electron_count,
        occupancies=raw.occupancies,
        orbital_energies=raw.orbital_energies,
        metadata=raw.metadata,
        config=config,
        psi_occ=raw.orbital_matrix,
        rho_sad=raw.rho_baseline,
    )


def infer_uniform_axis(points: np.ndarray) -> np.ndarray:
    """uniform Cartesian grid axis 복원."""
    unique_x = np.unique(np.round(points[:, 0], decimals=8))
    unique_y = np.unique(np.round(points[:, 1], decimals=8))
    unique_z = np.unique(np.round(points[:, 2], decimals=8))
    if len(unique_x) != len(unique_y) or len(unique_x) != len(unique_z):
        raise ValueError("Loaded NPZ points are not a cubic Cartesian grid.")
    return unique_x.astype(np.float32)


def load_npz_system(path: str | Path, config: ExperimentConfig) -> SystemRecord:
    """미래의 실제 DFT / KS 결과를 같은 파이프라인으로 읽기 위한 loader.

    기대하는 최소 key
    ------------------
    points         : (n_points, 3)
    gamma_matrix   : (n_points, n_points)
    local_features : (n_points, d_local)
    global_context : (d_global,)

    optional key
    ------------
    potential      : (n_points, 1)
    grad_potential : (n_points, 3)
    occupancies    : (n_modes,)
    orbital_energies
    electron_count
    """
    def scalar_payload(payload, key: str, default: object = "") -> object:
        if key not in payload:
            return default
        value = np.asarray(payload[key])
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()

    with open_npz_payload(path) as payload:
        required = ["points", "local_features", "global_context"]
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"{path} is missing required keys: {missing}")
        if "gamma_matrix" not in payload and "rho_diag" not in payload:
            raise KeyError(f"{path} is missing gamma_matrix or rho_diag.")

        points = np.asarray(payload["points"], dtype=np.float32)
        local_features = np.asarray(payload["local_features"], dtype=np.float32)
        global_context = np.asarray(payload["global_context"], dtype=np.float32)
        if npz_mmap_cache_enabled():
            axis, interior_idx, stencil_left, stencil_right = load_or_build_grid_cache(
                path,
                points,
                config.tau_stencil,
            )
            cached_stencil_indices = (interior_idx, stencil_left, stencil_right)
        else:
            axis = infer_uniform_axis(points)
            cached_stencil_indices = None

        potential = np.asarray(payload["potential"], dtype=np.float32) if "potential" in payload else np.zeros((len(points), 1), dtype=np.float32)
        grad_potential = np.asarray(payload["grad_potential"], dtype=np.float32) if "grad_potential" in payload else np.zeros((len(points), 3), dtype=np.float32)
        hartree_potential = (
            np.asarray(payload["hartree_potential"], dtype=np.float32)
            if "hartree_potential" in payload
            else None
        )
        xc_potential_local = (
            np.asarray(payload["xc_potential_local"], dtype=np.float32)
            if "xc_potential_local" in payload
            else None
        )
        ks_potential = np.asarray(payload["ks_potential"], dtype=np.float32) if "ks_potential" in payload else None
        kinetic_potential = (
            np.asarray(payload["kinetic_potential"], dtype=np.float32)
            if "kinetic_potential" in payload
            else None
        )
        kinetic_potential_centered = (
            np.asarray(payload["kinetic_potential_centered"], dtype=np.float32)
            if "kinetic_potential_centered" in payload
            else None
        )
        occupancies = np.asarray(payload["occupancies"], dtype=np.float32) if "occupancies" in payload else np.array([], dtype=np.float32)
        has_psi_occ = "psi_occ" in payload
        lazy_psi_occ = os.environ.get("RDM_LAZY_PSI_OCC", "1").strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
            "n",
        }
        psi_occ = (
            np.asarray(payload["psi_occ"], dtype=np.float32)
            if has_psi_occ and not lazy_psi_occ
            else None
        )
        orbital_energies = np.asarray(payload["orbital_energies"], dtype=np.float32) if "orbital_energies" in payload else np.array([], dtype=np.float32)
        atom_symbols = np.asarray(payload["atom_symbols"]).astype(str).tolist() if "atom_symbols" in payload else []
        atom_coords_bohr = (
            np.asarray(payload["atom_coords_bohr"], dtype=np.float32)
            if "atom_coords_bohr" in payload
            else np.empty((0, 3), dtype=np.float32)
        )
        orbital_derivative_key = (
            "derivative_orbital_gradient"
            if "derivative_orbital_gradient" in payload
            else "derivative_true_ao"
        )
        orbital_tau_key = "tau_orbital_gradient" if "tau_orbital_gradient" in payload else "tau_true_ao"
        derivative_true_grid = (
            np.asarray(payload[orbital_derivative_key], dtype=np.float32)
            if orbital_derivative_key in payload
            else None
        )
        tau_true_grid = (
            np.asarray(payload[orbital_tau_key], dtype=np.float32)
            if orbital_tau_key in payload
            else None
        )
        gamma_stencil = str(config.tau_stencil).strip().lower()
        gamma_derivative_key = (
            "derivative_gamma_central2_interior"
            if gamma_stencil in {"central2", "second", "second_order", "legacy"}
            else "derivative_gamma_richardson_interior"
        )
        gamma_tau_key = (
            "tau_gamma_central2_interior"
            if gamma_stencil in {"central2", "second", "second_order", "legacy"}
            else "tau_gamma_richardson_interior"
        )
        if gamma_derivative_key not in payload:
            gamma_derivative_key = "derivative_true_fd_gamma"
        if gamma_tau_key not in payload:
            gamma_tau_key = "tau_true_fd_gamma"
        derivative_true_fd = (
            np.asarray(payload[gamma_derivative_key], dtype=np.float32)
            if gamma_derivative_key in payload
            else None
        )
        tau_true_fd = (
            np.asarray(payload[gamma_tau_key], dtype=np.float32)
            if gamma_tau_key in payload
            else None
        )
        electron_count_from_payload = float(payload["electron_count"]) if "electron_count" in payload else None
        rho_diag_override = np.asarray(payload["rho_diag"], dtype=np.float32) if "rho_diag" in payload else None
        rho_sad = np.asarray(payload["rho_sad"], dtype=np.float32) if "rho_sad" in payload else None
        if rho_sad is not None and rho_sad.size == 0:
            rho_sad = None
        local_feature_schema = str(scalar_payload(payload, "local_feature_schema", ""))
        if rho_sad is None and local_features.shape[1] >= 16 and local_feature_schema in {"", "sad_vectors_v1"}:
            # Patched legacy archives stored log1p(rho_sad) as feature column 15
            # before rho_sad became an explicit NPZ channel.
            rho_sad = np.expm1(local_features[:, 15:16]).astype(np.float32)
        rho_cation = np.asarray(payload["rho_cation"], dtype=np.float32) if "rho_cation" in payload else None
        rho_anion = np.asarray(payload["rho_anion"], dtype=np.float32) if "rho_anion" in payload else None
        light_cache = None if rho_diag_override is not None else read_npz_light_cache(path)
        if rho_diag_override is None and light_cache is not None:
            rho_diag_override = np.asarray(light_cache["rho_diag"], dtype=np.float32)

        can_skip_initial_gamma = (
            rho_diag_override is not None
            and derivative_true_grid is not None
            and tau_true_grid is not None
            and (electron_count_from_payload is not None or light_cache is not None)
        )
        if can_skip_initial_gamma:
            gamma_matrix = np.empty((0, 0), dtype=np.float32)
            electron_count = (
                electron_count_from_payload
                if electron_count_from_payload is not None
                else float(light_cache["electron_count"])  # type: ignore[index]
            )
        else:
            gamma_matrix = np.asarray(payload["gamma_matrix"], dtype=np.float32)
            if rho_diag_override is None:
                rho_diag_override = np.diag(gamma_matrix).reshape(-1, 1).astype(np.float32)
            electron_count = (
                electron_count_from_payload
                if electron_count_from_payload is not None
                else float(np.trace(gamma_matrix) * config.cell_volume)
            )
            write_npz_light_cache(path, rho_diag_override, electron_count)
        metadata = {
            "source_path": str(path),
            "formula": scalar_payload(payload, "formula", ""),
            "smiles_gdb": scalar_payload(payload, "smiles_gdb", ""),
            "smiles_relaxed": scalar_payload(payload, "smiles_relaxed", ""),
            "inchi_gdb": scalar_payload(payload, "inchi_gdb", ""),
            "inchi_relaxed": scalar_payload(payload, "inchi_relaxed", ""),
            "atom_symbols": atom_symbols,
            "atom_coords_bohr": atom_coords_bohr,
            "axis_points": int(scalar_payload(payload, "axis_points", len(axis))),
            "grid_spacing_bohr": float(scalar_payload(payload, "grid_spacing_bohr", axis[1] - axis[0])),
            "grid_radius_bohr": float(scalar_payload(payload, "grid_radius_bohr", np.max(np.abs(axis)))),
            "box_length_bohr": float(scalar_payload(payload, "box_length_bohr", axis[-1] - axis[0])),
            "total_energy_hartree": float(scalar_payload(payload, "total_energy_hartree", np.nan)),
            "kinetic_energy_hartree": float(scalar_payload(payload, "kinetic_energy_hartree", np.nan)),
            "kinetic_energy_orbital_fd_hartree": float(
                scalar_payload(payload, "kinetic_energy_orbital_fd_hartree", np.nan)
            ),
            "kinetic_energy_orbital_central2_interior_hartree": float(
                scalar_payload(payload, "kinetic_energy_orbital_central2_interior_hartree", np.nan)
            ),
            "kinetic_energy_gamma_stencil_hartree": float(
                scalar_payload(payload, "kinetic_energy_gamma_stencil_hartree", np.nan)
            ),
            "kinetic_energy_gamma_central2_interior_hartree": float(
                scalar_payload(payload, "kinetic_energy_gamma_central2_interior_hartree", np.nan)
            ),
            "kinetic_energy_gamma_richardson_interior_hartree": float(
                scalar_payload(payload, "kinetic_energy_gamma_richardson_interior_hartree", np.nan)
            ),
            "kinetic_reference": scalar_payload(payload, "kinetic_reference", "legacy"),
            "reference_schema": scalar_payload(payload, "reference_schema", "legacy"),
            "chemical_potential_hartree": float(scalar_payload(payload, "chemical_potential_hartree", np.nan)),
            "kinetic_potential_center": float(scalar_payload(payload, "kinetic_potential_center", np.nan)),
            "kinetic_potential_reference": scalar_payload(payload, "kinetic_potential_reference", "not_computed"),
            "tau_reference": scalar_payload(
                payload,
                "tau_reference",
                "orbital_gradient" if tau_true_grid is not None else f"finite_difference_{config.tau_stencil}",
            ),
            "tau_reference_primary": scalar_payload(
                payload,
                "tau_reference_primary",
                "orbital_gradient" if tau_true_grid is not None else f"finite_difference_{config.tau_stencil}",
            ),
            "gamma_reference": scalar_payload(payload, "gamma_reference", ""),
            "reference_backend": scalar_payload(payload, "reference_backend", ""),
            "has_psi_occ": has_psi_occ,
            "lazy_psi_occ": bool(has_psi_occ and lazy_psi_occ),
            "charged_density_oracles": rho_cation is not None and rho_anion is not None,
            "local_feature_schema": local_feature_schema,
        }

    return finalize_system_record(
        system_id=Path(path).stem,
        family="npz",
        axis=axis,
        points=points,
        local_features=local_features,
        potential=potential,
        grad_potential=grad_potential,
        global_context=global_context,
        gamma_matrix=gamma_matrix,
        electron_count=electron_count,
        occupancies=occupancies,
        orbital_energies=orbital_energies,
        metadata=metadata,
        config=config,
        keep_gamma_matrix=False,
        derivative_true_grid=derivative_true_grid,
        tau_true_grid=tau_true_grid,
        derivative_true_fd=derivative_true_fd,
        tau_true_fd=tau_true_fd,
        hartree_potential=hartree_potential,
        xc_potential_local=xc_potential_local,
        ks_potential=ks_potential,
        kinetic_potential=kinetic_potential,
        kinetic_potential_centered=kinetic_potential_centered,
        rho_diag_override=rho_diag_override,
        psi_occ=psi_occ,
        rho_sad=rho_sad,
        rho_cation=rho_cation,
        rho_anion=rho_anion,
        stencil_indices=cached_stencil_indices,
    )


def build_system_corpus(config: ExperimentConfig) -> list[SystemRecord]:
    """학습에 사용할 전체 시스템 목록 생성."""
    rng = np.random.default_rng(config.seed)
    systems: list[SystemRecord] = []

    if config.dataset_mode in {"ks_like", "mixed"}:
        for idx in range(config.num_systems):
            systems.append(build_ks_like_system(config, idx, rng))

    if config.dataset_mode == "toy":
        dimensions = parse_toy_dimensions(config.toy_dimensions)
        for idx in range(config.num_systems):
            dimension = dimensions[idx % len(dimensions)]
            systems.append(build_dimensional_toy_system(config, idx, dimension, rng))

    if config.dataset_mode in {"npz", "mixed"}:
        paths = sorted(glob.glob(config.npz_glob)) if config.npz_glob else []
        if config.dataset_mode == "npz" and config.num_systems > 0:
            paths = paths[: config.num_systems]
        if config.dataset_mode == "npz" and not paths:
            raise FileNotFoundError("dataset_mode='npz' but RDM_NPZ_GLOB did not match any files.")
        progress_every = int(os.environ.get("RDM_LOAD_PROGRESS_EVERY", "25"))
        progress_enabled = env_flag("RDM_LOAD_PROGRESS", True) and progress_every > 0
        load_workers = max(int(os.environ.get("RDM_NPZ_LOAD_WORKERS", "1")), 1)
        memory_label = "mapped/logical arrays" if npz_mmap_cache_enabled() else "resident arrays"
        if npz_mmap_cache_enabled():
            valid_cache_entries = sum(
                npz_mmap_cache_valid(path, npz_mmap_cache_path(path))
                for path in paths
            )
            print(
                "[NPZ mmap cache] "
                f"valid entries={valid_cache_entries}/{len(paths)} | "
                f"dir={npz_mmap_cache_path(paths[0]).parent}"
            )
        start_time = time.perf_counter()
        resident_bytes = 0
        if progress_enabled and load_workers > 1:
            print(f"[NPZ load] using {load_workers} parallel workers")

        if load_workers == 1:
            for idx, path in enumerate(paths, start=1):
                system = load_npz_system(path, config)
                systems.append(system)
                resident_bytes += system_resident_nbytes(system)
                if progress_enabled and (idx == 1 or idx % progress_every == 0 or idx == len(paths)):
                    elapsed = time.perf_counter() - start_time
                    print(
                        "[NPZ load] "
                        f"{idx}/{len(paths)} systems | "
                        f"{memory_label} ~{resident_bytes / (1024**3):.2f} GiB | "
                        f"elapsed {elapsed:.1f}s | "
                        f"rate {idx / max(elapsed, 1e-9):.2f} systems/s"
                    )
        else:
            loaded_by_index: list[SystemRecord | None] = [None] * len(paths)
            with ThreadPoolExecutor(max_workers=load_workers) as executor:
                future_to_index = {
                    executor.submit(load_npz_system, path, config): index
                    for index, path in enumerate(paths)
                }
                for completed, future in enumerate(as_completed(future_to_index), start=1):
                    index = future_to_index[future]
                    system = future.result()
                    loaded_by_index[index] = system
                    resident_bytes += system_resident_nbytes(system)
                    if progress_enabled and (
                        completed == 1
                        or completed % progress_every == 0
                        or completed == len(paths)
                    ):
                        elapsed = time.perf_counter() - start_time
                        print(
                            "[NPZ load] "
                            f"{completed}/{len(paths)} systems | "
                            f"{memory_label} ~{resident_bytes / (1024**3):.2f} GiB | "
                            f"elapsed {elapsed:.1f}s | "
                            f"rate {completed / max(elapsed, 1e-9):.2f} systems/s"
                        )
            systems.extend(system for system in loaded_by_index if system is not None)

    if not systems:
        raise RuntimeError("No systems were generated or loaded.")

    axis_counts = [len(system.axis) for system in systems]
    point_counts = [len(system.points) for system in systems]
    local_feature_dims = sorted({system.local_features.shape[1] for system in systems})
    global_context_dims = sorted({len(system.global_context) for system in systems})
    if len(local_feature_dims) != 1:
        raise ValueError(
            f"Inconsistent local_feature dimensions across the corpus: {local_feature_dims}. "
            "Finish patching every NPZ file before training."
        )
    if len(global_context_dims) != 1:
        raise ValueError(f"Inconsistent global_context dimensions across the corpus: {global_context_dims}.")
    tau_references = sorted({str(system.metadata.get("tau_reference", f"finite_difference_{config.tau_stencil}")) for system in systems})
    print_block(
        "System corpus",
        [
            ("dataset_mode", config.dataset_mode),
            ("num_systems", len(systems)),
            ("axis_points", f"{min(axis_counts)}..{max(axis_counts)}"),
            ("n_points/system", f"{min(point_counts)}..{max(point_counts)}"),
            ("local_feature_dim", local_feature_dims[0]),
            ("global_context_dim", global_context_dims[0]),
            ("tau_stencil", config.tau_stencil),
            ("tau_reference", tau_references),
        ],
    )
    return systems
