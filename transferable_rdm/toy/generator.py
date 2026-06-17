from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ExperimentConfig
from ..utils import make_uniform_grid


LOCAL_FEATURE_SCHEMA = "toy_qm9_analog_32x11_v1"
TOY_TEMPERATURE = 0.05
SOURCE_CHANNELS = 5


@dataclass(frozen=True)
class ToyRawSystem:
    axis: np.ndarray
    points: np.ndarray
    potential: np.ndarray
    gradient: np.ndarray
    local_features: np.ndarray
    global_context: np.ndarray
    gamma_matrix: np.ndarray
    orbital_matrix: np.ndarray
    occupancies: np.ndarray
    orbital_energies: np.ndarray
    electron_count: float
    rho_baseline: np.ndarray
    metadata: dict[str, object]


def parse_toy_dimensions(value: str) -> tuple[int, ...]:
    try:
        dimensions = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    except ValueError as exc:
        raise ValueError("RDM_TOY_DIMENSIONS must be a comma-separated subset of 1,2,3.") from exc
    if not dimensions or any(dimension not in {1, 2, 3} for dimension in dimensions):
        raise ValueError("RDM_TOY_DIMENSIONS must be a comma-separated subset of 1,2,3.")
    return dimensions


def second_derivative_matrix(n: int, step: float) -> np.ndarray:
    main = -2.0 * np.ones(n, dtype=np.float64)
    off = np.ones(n - 1, dtype=np.float64)
    return (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / (step * step)


def solve_1d_schrodinger(
    axis: np.ndarray,
    potential: np.ndarray,
    n_keep: int,
    particle_mass: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    step = float(axis[1] - axis[0])
    mass = max(float(particle_mass), 1e-12)
    hamiltonian = -(0.5 / mass) * second_derivative_matrix(len(axis), step) + np.diag(
        potential.astype(np.float64)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    eigenvalues = eigenvalues[:n_keep]
    eigenvectors = eigenvectors[:, :n_keep]
    norms = np.sqrt(np.sum(eigenvectors**2, axis=0) * step)
    return eigenvalues.astype(np.float32), (eigenvectors / norms[None, :]).astype(np.float32)


def fermi_occupations(
    energies: np.ndarray,
    electron_count: float,
    temperature: float,
) -> np.ndarray:
    temperature = max(float(temperature), 1e-3)
    energies64 = np.asarray(energies, dtype=np.float64)
    low = float(np.min(energies64) - 20.0 * temperature)
    high = float(np.max(energies64) + 20.0 * temperature)
    for _ in range(120):
        chemical_potential = 0.5 * (low + high)
        occupations = 1.0 / (1.0 + np.exp((energies64 - chemical_potential) / temperature))
        if np.sum(occupations) > electron_count:
            high = chemical_potential
        else:
            low = chemical_potential
    chemical_potential = 0.5 * (low + high)
    return (
        1.0 / (1.0 + np.exp((energies64 - chemical_potential) / temperature))
    ).astype(np.float32)


def sample_axis_parameters(
    config: ExperimentConfig,
    rng: np.random.Generator,
    *,
    active: bool,
) -> dict[str, object]:
    if not active:
        return {
            "num_wells": 0,
            "centers": np.empty((0,), dtype=np.float32),
            "depths": np.empty((0,), dtype=np.float32),
            "widths": np.empty((0,), dtype=np.float32),
            "omega": float(config.toy_inactive_omega),
            "quartic": 0.0,
        }

    num_wells = int(rng.integers(1, config.max_wells + 1))
    if num_wells == 1:
        centers = np.array([rng.uniform(-1.2, 1.2)], dtype=np.float32)
    else:
        separation = rng.uniform(1.0, 2.4)
        shift = rng.uniform(-0.3, 0.3)
        centers = np.sort(
            np.array([shift - 0.5 * separation, shift + 0.5 * separation], dtype=np.float32)
        )
    return {
        "num_wells": num_wells,
        "centers": centers,
        "depths": rng.uniform(0.9, 2.2, size=num_wells).astype(np.float32),
        "widths": rng.uniform(0.45, 1.05, size=num_wells).astype(np.float32),
        "omega": float(rng.uniform(0.06, 0.20)),
        "quartic": float(rng.uniform(0.0, 0.012)),
    }


def evaluate_axis_potential(
    coordinate: np.ndarray,
    params: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    omega = float(params["omega"])
    quartic = float(params["quartic"])
    potential = 0.5 * omega**2 * coordinate**2 + quartic * coordinate**4
    gradient = omega**2 * coordinate + 4.0 * quartic * coordinate**3
    for center, depth, width in zip(params["centers"], params["depths"], params["widths"]):
        displacement = coordinate - float(center)
        gaussian = np.exp(-((displacement / float(width)) ** 2))
        potential -= float(depth) * gaussian
        gradient += float(depth) * 2.0 * displacement / float(width) ** 2 * gaussian
    return potential.astype(np.float32), gradient.astype(np.float32)


def enumerate_states(
    eigenvalues: tuple[np.ndarray, np.ndarray, np.ndarray],
    active_dimension: int,
    max_orbitals: int,
) -> list[tuple[float, tuple[int, int, int]]]:
    index_ranges = [
        range(min(len(eigenvalues[axis_index]), max_orbitals))
        if axis_index < active_dimension
        else range(1)
        for axis_index in range(3)
    ]
    states = [
        (
            float(
                eigenvalues[0][ix]
                + eigenvalues[1][iy]
                + eigenvalues[2][iz]
            ),
            (ix, iy, iz),
        )
        for ix in index_ranges[0]
        for iy in index_ranges[1]
        for iz in index_ranges[2]
    ]
    states.sort(key=lambda item: item[0])
    return states[:max_orbitals]


def signed_log_scaled(values: np.ndarray, clip: float) -> np.ndarray:
    clip = max(float(clip), 1.0)
    clipped = np.clip(values, -clip, clip)
    return np.sign(clipped) * np.log1p(np.abs(clipped)) / np.log1p(clip)


def richardson_laplacian(values: np.ndarray, n_axis: int, step: float) -> np.ndarray:
    volume = np.asarray(values, dtype=np.float64).reshape(n_axis, n_axis, n_axis)
    padded = np.pad(volume, ((2, 2), (2, 2), (2, 2)), mode="symmetric")
    center = padded[2 : n_axis + 2, 2 : n_axis + 2, 2 : n_axis + 2]

    def second_derivative(axis_index: int) -> np.ndarray:
        base = [slice(2, n_axis + 2), slice(2, n_axis + 2), slice(2, n_axis + 2)]
        shifted_values = []
        for offset in (0, 1, 3, 4):
            shifted = list(base)
            shifted[axis_index] = slice(offset, offset + n_axis)
            shifted_values.append(padded[tuple(shifted)])
        return (
            -shifted_values[0]
            + 16.0 * shifted_values[1]
            - 30.0 * center
            + 16.0 * shifted_values[2]
            - shifted_values[3]
        ) / (12.0 * step * step)

    laplacian = sum(second_derivative(axis_index) for axis_index in range(3))
    return laplacian.reshape(-1, 1)


def source_records(axis_params: list[dict[str, object]]) -> list[tuple[np.ndarray, float, float]]:
    records: list[tuple[np.ndarray, float, float]] = []
    for axis_index, params in enumerate(axis_params):
        for center, depth, width in zip(params["centers"], params["depths"], params["widths"]):
            position = np.zeros(3, dtype=np.float32)
            position[axis_index] = float(center)
            records.append((position, float(depth), float(width)))
    return records


def source_channel_groups(
    records: list[tuple[np.ndarray, float, float]],
) -> list[list[tuple[np.ndarray, float, float]]]:
    groups: list[list[tuple[np.ndarray, float, float]]] = [[] for _ in range(SOURCE_CHANNELS)]
    for index, record in enumerate(records):
        groups[min(index, SOURCE_CHANNELS - 1)].append(record)
    return groups


def ground_state_density_baseline(
    ground_orbital: np.ndarray,
    electron_count: float,
    cell_volume: float,
) -> np.ndarray:
    """Potential-derived density baseline analogous to an atomic initial guess."""
    baseline = np.square(np.asarray(ground_orbital, dtype=np.float64)).reshape(-1, 1)
    baseline = np.maximum(baseline, 1e-12)
    normalizer = float(np.sum(baseline) * cell_volume)
    return (baseline * (electron_count / max(normalizer, 1e-12))).astype(np.float32)


def build_qm9_analog_features(
    points: np.ndarray,
    potential: np.ndarray,
    gradient: np.ndarray,
    axis_params: list[dict[str, object]],
    active_dimension: int,
    electron_count: float,
    rho_baseline: np.ndarray,
    step: float,
    laplacian_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius = max(float(np.max(np.abs(points))), 1e-6)
    potential_scale = max(float(np.std(potential)), 1.0)
    laplacian = signed_log_scaled(
        richardson_laplacian(potential, round(len(points) ** (1.0 / 3.0)), step)
        * step**2
        / potential_scale,
        laplacian_clip,
    ).astype(np.float32)
    radial = np.linalg.norm(points, axis=1, keepdims=True) / radius

    records = source_records(axis_params)
    groups = source_channel_groups(records)
    gaussian_channels = []
    vector_channels = []
    for group in groups:
        if not group:
            gaussian_channels.append(np.zeros((len(points), 1), dtype=np.float32))
            vector_channels.append(np.zeros((len(points), 3), dtype=np.float32))
            continue
        gaussian_sum = np.zeros((len(points), 1), dtype=np.float32)
        vector_sum = np.zeros((len(points), 3), dtype=np.float32)
        for position, depth, width in group:
            displacement = points - position[None, :]
            distance_sq = np.sum(displacement**2, axis=1, keepdims=True)
            weight = depth * np.exp(-distance_sq / max(width * width, 1e-8))
            gaussian_sum += weight.astype(np.float32)
            vector_sum += (displacement * weight / radius).astype(np.float32)
        gaussian_channels.append(gaussian_sum)
        vector_channels.append(vector_sum)

    unavailable_atomic_number = np.zeros((len(points), 1), dtype=np.float32)
    electron_column = np.full((len(points), 1), electron_count / 30.0, dtype=np.float32)
    local_features = np.concatenate(
        [
            points / radius,
            potential / potential_scale,
            gradient / potential_scale,
            laplacian,
            radial,
        ]
        + gaussian_channels
        + [
            unavailable_atomic_number,
            electron_column,
            np.log1p(np.maximum(rho_baseline, 0.0)).astype(np.float32),
        ]
        + vector_channels,
        axis=1,
    ).astype(np.float32)

    depths = np.asarray([record[1] for record in records], dtype=np.float32)
    positions = np.asarray([record[0] for record in records], dtype=np.float32)
    source_radius = float(np.max(np.linalg.norm(positions, axis=1))) if len(positions) else 0.0
    group_counts = np.asarray([len(group) for group in groups], dtype=np.float32)
    global_context = np.concatenate(
        [
            np.asarray(
                [
                    electron_count / 30.0,
                    len(records) / 30.0,
                    active_dimension / 3.0,
                    float(np.mean(depths)) / 3.0 if len(depths) else 0.0,
                    float(np.std(depths)) / 3.0 if len(depths) else 0.0,
                    source_radius / 10.0,
                ],
                dtype=np.float32,
            ),
            group_counts / 10.0,
        ]
    ).astype(np.float32)
    if local_features.shape[1] != 32 or global_context.shape != (11,):
        raise RuntimeError(
            f"Toy feature schema mismatch: local={local_features.shape}, global={global_context.shape}."
        )
    return local_features, global_context


def build_toy_raw_system(
    config: ExperimentConfig,
    active_dimension: int,
    rng: np.random.Generator,
) -> ToyRawSystem:
    axis = np.linspace(-config.domain_radius, config.domain_radius, config.axis_points, dtype=np.float32)
    points = make_uniform_grid(axis)
    step = float(axis[1] - axis[0])
    cell_volume = step**3
    axis_params = [
        sample_axis_parameters(config, rng, active=axis_index < active_dimension)
        for axis_index in range(3)
    ]
    solutions = []
    keep_1d = max(4, min(config.axis_points, config.max_orbitals))
    particle_mass = max(float(config.toy_particle_mass), 1e-12)
    for params in axis_params:
        axis_potential, _ = evaluate_axis_potential(axis, params)
        solutions.append(solve_1d_schrodinger(axis, axis_potential, keep_1d, particle_mass))

    eigenvalues = tuple(solution[0] for solution in solutions)
    eigenvectors = tuple(solution[1] for solution in solutions)
    states = enumerate_states(eigenvalues, active_dimension, config.max_orbitals)
    orbital_energies = np.asarray([state[0] for state in states], dtype=np.float32)
    max_electron_count = max(0.6, min(4.8, len(states) - 0.2))
    min_electron_count = min(1.0, 0.5 * max_electron_count)
    electron_count = float(rng.uniform(min_electron_count, max_electron_count))
    occupancies = fermi_occupations(orbital_energies, electron_count, TOY_TEMPERATURE)
    orbitals = [
        np.einsum(
            "i,j,k->ijk",
            eigenvectors[0][:, ix],
            eigenvectors[1][:, iy],
            eigenvectors[2][:, iz],
        )
        .reshape(-1)
        .astype(np.float32)
        for _, (ix, iy, iz) in states
    ]
    orbital_matrix = np.stack(orbitals, axis=1).astype(np.float32)
    gamma_matrix = ((orbital_matrix * occupancies[None, :]) @ orbital_matrix.T).astype(np.float32)

    potential_parts = []
    gradient_parts = []
    for axis_index, params in enumerate(axis_params):
        axis_potential, axis_gradient = evaluate_axis_potential(points[:, axis_index], params)
        potential_parts.append(axis_potential)
        gradient_parts.append(axis_gradient)
    potential = np.sum(np.stack(potential_parts, axis=1), axis=1, keepdims=True).astype(np.float32)
    gradient = np.stack(gradient_parts, axis=1).astype(np.float32)
    rho_baseline = ground_state_density_baseline(
        orbital_matrix[:, 0],
        electron_count,
        cell_volume,
    )
    local_features, global_context = build_qm9_analog_features(
        points,
        potential,
        gradient,
        axis_params,
        active_dimension,
        electron_count,
        rho_baseline,
        step,
        config.potential_laplacian_clip,
    )
    metadata = {
        "toy_dimension": active_dimension,
        "toy_embedding": "active_axes_in_3d_grid",
        "toy_temperature": TOY_TEMPERATURE,
        "particle_mass": particle_mass,
        "kinetic_prefactor": 0.5 / particle_mass,
        "toy_source_count": sum(int(params["num_wells"]) for params in axis_params),
        "toy_density_baseline": "normalized_ground_state_orbital",
        "local_feature_schema": LOCAL_FEATURE_SCHEMA,
        "electron_count": electron_count,
        "tau_reference": f"finite_difference_{config.tau_stencil}",
    }
    return ToyRawSystem(
        axis=axis,
        points=points,
        potential=potential,
        gradient=gradient,
        local_features=local_features,
        global_context=global_context,
        gamma_matrix=gamma_matrix,
        orbital_matrix=orbital_matrix,
        occupancies=occupancies,
        orbital_energies=orbital_energies,
        electron_count=electron_count,
        rho_baseline=rho_baseline,
        metadata=metadata,
    )
