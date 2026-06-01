from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from .config import ExperimentConfig
from .systems import SystemRecord


PAIR_DENSITY_FEATURE_MODES = ("off", "rho-derivatives", "fukui")
_FROZEN_DENSITY_STATE_CACHE: dict[int, DensityFeatureState] = {}


@dataclass
class DensityFeatureState:
    """Predicted densities and reusable grid derivatives for one system."""

    rho_neutral: tf.Tensor
    rho_cation: tf.Tensor | None
    rho_anion: tf.Tensor | None
    descriptor_fields: tuple[tuple[str, tf.Tensor, tf.Tensor, tf.Tensor], ...]


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


def pair_density_feature_dim(config: ExperimentConfig) -> int:
    mode = pair_density_feature_mode(config)
    if mode == "off":
        return 0
    return 18 if mode == "fukui" else 6


def normalized_density_head(
    system: SystemRecord,
    raw_head: tf.Tensor,
    electron_count: float,
    *,
    normalize: bool = True,
) -> tf.Tensor:
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


def descriptor_field(system: SystemRecord, name: str, values: tf.Tensor) -> tuple[str, tf.Tensor, tf.Tensor, tf.Tensor]:
    n_axis = len(system.axis)
    grad = richardson_gradient_3d(values, n_axis, system.step)
    lap = richardson_laplacian_3d(values, n_axis, system.step)
    return name, values, grad, lap


def build_density_feature_state(
    system: SystemRecord,
    point_out: tf.Tensor,
    config: ExperimentConfig,
) -> DensityFeatureState:
    neutral = normalized_density_head(system, point_out[:, 0:1], system.electron_count, normalize=config.normalize_rho)
    mode = pair_density_feature_mode(config)
    cation = None
    anion = None
    fields: list[tuple[str, tf.Tensor, tf.Tensor, tf.Tensor]] = []
    if mode != "off":
        fields.append(descriptor_field(system, "rho_neutral", neutral))
    if mode == "fukui":
        cation = normalized_density_head(
            system, point_out[:, 1:2], max(system.electron_count - 1.0, 1e-6), normalize=config.normalize_rho
        )
        anion = normalized_density_head(
            system, point_out[:, 2:3], system.electron_count + 1.0, normalize=config.normalize_rho
        )
        fields.append(descriptor_field(system, "fukui_plus", anion - neutral))
        fields.append(descriptor_field(system, "fukui_minus", neutral - cation))
    if config.freeze_point_after_pretrain:
        neutral = tf.stop_gradient(neutral)
        cation = tf.stop_gradient(cation) if cation is not None else None
        anion = tf.stop_gradient(anion) if anion is not None else None
        fields = [(name, tf.stop_gradient(values), tf.stop_gradient(grad), tf.stop_gradient(lap)) for name, values, grad, lap in fields]
    return DensityFeatureState(neutral, cation, anion, tuple(fields))


def signed_log_scaled(values: tf.Tensor, clip: float) -> tf.Tensor:
    clip = max(float(clip), 1.0)
    clipped = tf.clip_by_value(values, -clip, clip)
    return tf.sign(clipped) * tf.math.log1p(tf.abs(clipped)) / tf.math.log1p(tf.constant(clip, tf.float32))


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
    for _, values, grad, lap in state.descriptor_fields:
        scale = tf.maximum(tf.reduce_mean(tf.abs(values)), max(config.pair_density_eps, 1e-30))
        scaled_values = signed_log_scaled(values / scale, config.pair_density_value_clip)
        scaled_grad = tf.math.log1p(
            tf.clip_by_value(tf.norm(grad, axis=1, keepdims=True) * system.step / scale, 0.0, config.pair_density_value_clip)
        ) / tf.math.log1p(tf.constant(max(config.pair_density_value_clip, 1.0), tf.float32))
        scaled_lap = signed_log_scaled(lap * (system.step**2) / scale, config.pair_density_laplacian_clip)
        features.extend(
            [
                tf.gather(scaled_values, left_idx),
                tf.gather(scaled_values, right_idx),
                tf.gather(scaled_grad, left_idx),
                tf.gather(scaled_grad, right_idx),
                tf.gather(scaled_lap, left_idx),
                tf.gather(scaled_lap, right_idx),
            ]
        )
    return tf.concat(features, axis=1)
