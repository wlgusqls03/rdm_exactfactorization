from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from .config import ExperimentConfig
from .systems import SystemRecord


PAIR_DENSITY_FEATURE_MODES = ("off", "rho-derivatives", "fukui")
DENSITY_BASELINE_MODES = ("learned", "sad-multiplicative")
DENSITY_SOURCES = ("predicted", "true")


@dataclass
class DensityFeatureState:
    """Predicted densities and reusable grid derivatives for one system."""

    rho_neutral: tf.Tensor
    rho_cation: tf.Tensor | None
    rho_anion: tf.Tensor | None
    descriptor_fields: tuple[tuple[str, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], ...]


_FROZEN_DENSITY_STATE_CACHE: dict[int, DensityFeatureState] = {}


def cached_frozen_density_state(system: SystemRecord) -> DensityFeatureState | None:
    return _FROZEN_DENSITY_STATE_CACHE.get(id(system))


def cache_frozen_density_state(system: SystemRecord, state: DensityFeatureState) -> None:
    _FROZEN_DENSITY_STATE_CACHE[id(system)] = state


def clear_frozen_density_state_cache() -> None:
    _FROZEN_DENSITY_STATE_CACHE.clear()


def pair_density_feature_mode(config: ExperimentConfig) -> str:
    mode = config.pair_density_feature_mode.strip().lower()
    if mode not in PAIR_DENSITY_FEATURE_MODES:
        raise ValueError(
            f"Unknown pair density feature mode: {config.pair_density_feature_mode!r}. "
            f"Choose one of: {', '.join(PAIR_DENSITY_FEATURE_MODES)}."
        )
    return mode


def density_head_count(config: ExperimentConfig) -> int:
    return 3 if pair_density_feature_mode(config) == "fukui" else 1


def density_source_mode(config: ExperimentConfig) -> str:
    mode = config.density_source.strip().lower()
    if mode not in DENSITY_SOURCES:
        raise ValueError(
            f"Unknown density source: {config.density_source!r}. "
            f"Choose one of: {', '.join(DENSITY_SOURCES)}."
        )
    return mode


def pair_density_feature_dim(config: ExperimentConfig) -> int:
    mode = pair_density_feature_mode(config)
    if mode == "off":
        return 0
    per_field_dim = 12 if config.pair_density_hessian else 6
    return 3 * per_field_dim if mode == "fukui" else per_field_dim


def density_baseline_mode(config: ExperimentConfig) -> str:
    mode = config.density_baseline_mode.strip().lower()
    if mode not in DENSITY_BASELINE_MODES:
        raise ValueError(
            f"Unknown density baseline mode: {config.density_baseline_mode!r}. "
            f"Choose one of: {', '.join(DENSITY_BASELINE_MODES)}."
        )
    return mode


def normalized_density_head(
    system: SystemRecord,
    raw_head: tf.Tensor,
    electron_count: float,
    *,
    config: ExperimentConfig,
    normalize: bool = True,
) -> tf.Tensor:
    mode = density_baseline_mode(config)
    if mode == "sad-multiplicative":
        if system.rho_sad is None:
            raise ValueError(
                f"System {system.system_id} has no SAD density baseline. "
                "Patch NPZ files with scripts/patch_npz_features.py or use --density-baseline-mode learned."
            )
        sad = tf.convert_to_tensor(system.rho_sad, dtype=tf.float32)
        floor = max(float(config.sad_density_floor), 1e-30)
        clip = max(float(config.sad_residual_clip), 0.0)
        rho = tf.maximum(sad, floor) * tf.exp(tf.clip_by_value(raw_head, -clip, clip))
    else:
        rho = tf.nn.softplus(raw_head) + 1e-6
    if not normalize:
        return rho
    normalizer = tf.reduce_sum(rho) * system.cell_volume
    return rho * (float(electron_count) / tf.maximum(normalizer, 1e-12))


def richardson_gradient_3d(grid_values: tf.Tensor, n_axis: int, h: float) -> tf.Tensor:
    """Compute an O(h^4) 3D gradient with symmetric boundary extension."""
    vol = tf.reshape(grid_values, (n_axis, n_axis, n_axis))
    padded = tf.pad(vol, [[2, 2], [2, 2], [2, 2]], mode="SYMMETRIC")
    n = n_axis
    grad_x = (
        padded[0:n, 2 : n + 2, 2 : n + 2]
        - 8.0 * padded[1 : n + 1, 2 : n + 2, 2 : n + 2]
        + 8.0 * padded[3 : n + 3, 2 : n + 2, 2 : n + 2]
        - padded[4 : n + 4, 2 : n + 2, 2 : n + 2]
    ) / (12.0 * h)
    grad_y = (
        padded[2 : n + 2, 0:n, 2 : n + 2]
        - 8.0 * padded[2 : n + 2, 1 : n + 1, 2 : n + 2]
        + 8.0 * padded[2 : n + 2, 3 : n + 3, 2 : n + 2]
        - padded[2 : n + 2, 4 : n + 4, 2 : n + 2]
    ) / (12.0 * h)
    grad_z = (
        padded[2 : n + 2, 2 : n + 2, 0:n]
        - 8.0 * padded[2 : n + 2, 2 : n + 2, 1 : n + 1]
        + 8.0 * padded[2 : n + 2, 2 : n + 2, 3 : n + 3]
        - padded[2 : n + 2, 2 : n + 2, 4 : n + 4]
    ) / (12.0 * h)
    return tf.reshape(tf.stack([grad_x, grad_y, grad_z], axis=-1), (-1, 3))


def richardson_laplacian_3d(grid_values: tf.Tensor, n_axis: int, h: float) -> tf.Tensor:
    """Compute an O(h^4) 3D Laplacian with symmetric boundary extension."""
    vol = tf.reshape(grid_values, (n_axis, n_axis, n_axis))
    padded = tf.pad(vol, [[2, 2], [2, 2], [2, 2]], mode="SYMMETRIC")
    n = n_axis
    center = padded[2 : n + 2, 2 : n + 2, 2 : n + 2]

    def second_derivative(axis: int) -> tf.Tensor:
        slices = [slice(2, n + 2), slice(2, n + 2), slice(2, n + 2)]
        values = []
        for offset in (0, 1, 3, 4):
            shifted = list(slices)
            shifted[axis] = slice(offset, offset + n)
            values.append(padded[tuple(shifted)])
        return (-values[0] + 16.0 * values[1] - 30.0 * center + 16.0 * values[2] - values[3]) / (
            12.0 * h * h
        )

    return tf.reshape(second_derivative(0) + second_derivative(1) + second_derivative(2), (-1, 1))


def richardson_hessian_diag_3d(grid_values: tf.Tensor, n_axis: int, h: float) -> tf.Tensor:
    """Compute O(h^4) diagonal Hessian components dxx, dyy, dzz."""
    vol = tf.reshape(grid_values, (n_axis, n_axis, n_axis))
    padded = tf.pad(vol, [[2, 2], [2, 2], [2, 2]], mode="SYMMETRIC")
    n = n_axis
    center = padded[2 : n + 2, 2 : n + 2, 2 : n + 2]

    def second_derivative(axis: int) -> tf.Tensor:
        slices = [slice(2, n + 2), slice(2, n + 2), slice(2, n + 2)]
        values = []
        for offset in (0, 1, 3, 4):
            shifted = list(slices)
            shifted[axis] = slice(offset, offset + n)
            values.append(padded[tuple(shifted)])
        return (-values[0] + 16.0 * values[1] - 30.0 * center + 16.0 * values[2] - values[3]) / (
            12.0 * h * h
        )

    return tf.reshape(
        tf.stack([second_derivative(axis) for axis in range(3)], axis=-1),
        (-1, 3),
    )


def descriptor_field(
    system: SystemRecord,
    name: str,
    values: tf.Tensor,
) -> tuple[str, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    n_axis = len(system.axis)
    grad = richardson_gradient_3d(values, n_axis, system.step)
    lap = richardson_laplacian_3d(values, n_axis, system.step)
    hess_diag = richardson_hessian_diag_3d(values, n_axis, system.step)
    return name, values, grad, lap, hess_diag


def build_density_feature_state(
    system: SystemRecord,
    point_out: tf.Tensor,
    config: ExperimentConfig,
) -> DensityFeatureState:
    neutral = normalized_density_head(
        system, point_out[:, 0:1], system.electron_count, config=config, normalize=config.normalize_rho
    )
    mode = pair_density_feature_mode(config)
    cation = None
    anion = None
    fields: list[tuple[str, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]] = []
    if mode != "off":
        fields.append(normalized_descriptor_field(system, "rho_neutral", neutral, config))
    if mode == "fukui":
        cation = normalized_density_head(
            system, point_out[:, 1:2], max(system.electron_count - 1.0, 1e-6),
            config=config, normalize=config.normalize_rho
        )
        anion = normalized_density_head(
            system, point_out[:, 2:3], system.electron_count + 1.0,
            config=config, normalize=config.normalize_rho
        )
        fields.append(normalized_descriptor_field(system, "fukui_plus", anion - neutral, config))
        fields.append(normalized_descriptor_field(system, "fukui_minus", neutral - cation, config))
    if config.freeze_point_after_pretrain:
        neutral = tf.stop_gradient(neutral)
        cation = tf.stop_gradient(cation) if cation is not None else None
        anion = tf.stop_gradient(anion) if anion is not None else None
        fields = [
            (
                name,
                tf.stop_gradient(values),
                tf.stop_gradient(grad),
                tf.stop_gradient(lap),
                tf.stop_gradient(hess_diag),
            )
            for name, values, grad, lap, hess_diag in fields
        ]
    return DensityFeatureState(neutral, cation, anion, tuple(fields))


def build_true_density_feature_state(
    system: SystemRecord,
    config: ExperimentConfig,
) -> DensityFeatureState:
    """Build oracle density inputs from the stored reference density."""
    neutral = tf.convert_to_tensor(system.rho_diag, dtype=tf.float32)
    mode = pair_density_feature_mode(config)
    cation = None
    anion = None
    fields: list[tuple[str, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]] = []
    if mode != "off":
        fields.append(normalized_descriptor_field(system, "rho_neutral", neutral, config))
    if mode == "fukui":
        if system.rho_cation is None or system.rho_anion is None:
            raise ValueError(
                f"True-density Fukui mode requires cation and anion densities for {system.system_id}."
            )
        cation = tf.convert_to_tensor(system.rho_cation, dtype=tf.float32)
        anion = tf.convert_to_tensor(system.rho_anion, dtype=tf.float32)
        fields.append(normalized_descriptor_field(system, "fukui_plus", anion - neutral, config))
        fields.append(normalized_descriptor_field(system, "fukui_minus", neutral - cation, config))
    return DensityFeatureState(
        tf.stop_gradient(neutral),
        tf.stop_gradient(cation) if cation is not None else None,
        tf.stop_gradient(anion) if anion is not None else None,
        tuple(
            (
                name,
                tf.stop_gradient(values),
                tf.stop_gradient(grad),
                tf.stop_gradient(lap),
                tf.stop_gradient(hess_diag),
            )
            for name, values, grad, lap, hess_diag in fields
        ),
    )


def signed_log_scaled(values: tf.Tensor, clip: float) -> tf.Tensor:
    clip = max(float(clip), 1.0)
    clipped = tf.clip_by_value(values, -clip, clip)
    return tf.sign(clipped) * tf.math.log1p(tf.abs(clipped)) / tf.math.log1p(tf.constant(clip, tf.float32))


def normalized_descriptor_field(
    system: SystemRecord,
    name: str,
    values: tf.Tensor,
    config: ExperimentConfig,
) -> tuple[str, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Precompute normalized endpoint descriptors once per frozen density state."""
    _, values, grad, lap, hess_diag = descriptor_field(system, name, values)
    scale = tf.maximum(tf.reduce_mean(tf.abs(values)), max(config.pair_density_eps, 1e-30))
    scaled_values = signed_log_scaled(values / scale, config.pair_density_value_clip)
    scaled_grad = tf.math.log1p(
        tf.clip_by_value(
            tf.norm(grad, axis=1, keepdims=True) * system.step / scale,
            0.0,
            config.pair_density_value_clip,
        )
    ) / tf.math.log1p(tf.constant(max(config.pair_density_value_clip, 1.0), tf.float32))
    scaled_lap = signed_log_scaled(lap * (system.step**2) / scale, config.pair_density_laplacian_clip)
    scaled_hess_diag = signed_log_scaled(
        hess_diag * (system.step**2) / scale,
        config.pair_density_hessian_clip,
    )
    return name, scaled_values, scaled_grad, scaled_lap, scaled_hess_diag


def pair_density_features(
    system: SystemRecord,
    state: DensityFeatureState,
    left_idx,
    right_idx,
    config: ExperimentConfig,
) -> tf.Tensor:
    """Build endpoint value, gradient-norm, and Laplacian descriptors."""
    if not state.descriptor_fields:
        return tf.zeros((tf.shape(left_idx)[0], 0), dtype=tf.float32)
    left_idx = tf.convert_to_tensor(left_idx, dtype=tf.int64)
    right_idx = tf.convert_to_tensor(right_idx, dtype=tf.int64)
    features = []
    for _, scaled_values, scaled_grad, scaled_lap, scaled_hess_diag in state.descriptor_fields:
        values_left = tf.gather(scaled_values, left_idx)
        values_right = tf.gather(scaled_values, right_idx)
        grad_left = tf.gather(scaled_grad, left_idx)
        grad_right = tf.gather(scaled_grad, right_idx)
        lap_left = tf.gather(scaled_lap, left_idx)
        lap_right = tf.gather(scaled_lap, right_idx)
        hess_left = tf.gather(scaled_hess_diag, left_idx)
        hess_right = tf.gather(scaled_hess_diag, right_idx)
        if config.pair_density_symmetric:
            features.extend(
                [
                    0.5 * (values_left + values_right),
                    tf.abs(values_left - values_right),
                    0.5 * (grad_left + grad_right),
                    tf.abs(grad_left - grad_right),
                    0.5 * (lap_left + lap_right),
                    tf.abs(lap_left - lap_right),
                ]
            )
            if config.pair_density_hessian:
                features.extend([0.5 * (hess_left + hess_right), tf.abs(hess_left - hess_right)])
            continue
        features.extend(
            [
                values_left,
                values_right,
                grad_left,
                grad_right,
                lap_left,
                lap_right,
            ]
        )
        if config.pair_density_hessian:
            features.extend([hess_left, hess_right])
    return tf.concat(features, axis=1)
