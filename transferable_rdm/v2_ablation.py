from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

from .data import (
    DatasetSplit,
    build_pair_features,
    choose_system,
    pair_weights_from_categories,
    sample_pair_indices,
)
from .model import RandomFourierFeatures
from .systems import SystemRecord
from .utils import print_block


EXPERIMENTS = (
    "baseline",
    "rho-only",
    "k-only",
    "gamma-only",
    "gamma-simple",
    "gamma-residual",
    "gamma-context",
)


@dataclass(frozen=True)
class V2Config:
    experiment: str = "baseline"
    output_dir: str = "v2_outputs"
    run_name: str = "v2_ablation"
    seed: int = 0

    width: int = 128
    depth: int = 2
    rank: int = 8
    rff_features: int = 16
    rff_scale: float = 2.0
    context_rff: bool = False
    residual_scale: float = 0.25

    batch_size: int = 1024
    steps_per_epoch: int = 40
    epochs: int = 120
    val_every: int = 10
    log_every: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    lr_decay_factor: float = 1.0
    lr_decay_patience: int = 0
    lr_decay_min: float = 1e-6
    lr_decay_min_delta: float = 0.0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    restore_best_weights: bool = False

    eval_pair_count: int = 8192
    cache_eval_batches: bool = True
    steps_per_system: int = 1
    pair_features_on_device: bool = True
    baseline_fit_batches: int = 24
    baseline_alpha_min: float = 1e-3
    baseline_alpha_max: float = 3.0
    baseline_alpha_count: int = 36
    baseline_density_power: str = "sqrt"

    normalize_rho: bool = True
    kernel_rho_floor: float = 1e-8
    kernel_target_clip: float = 20.0
    kernel_base_alpha: float = 0.0
    sep_factor_scale: float = 0.05
    pair_sampling_probs: tuple[float, float, float, float] | None = None
    pair_category_weights: tuple[float, float, float, float] = (20.0, 8.0, 4.0, 1.0)

    lambda_gamma: float = 1.0
    lambda_rho: float = 1.0
    lambda_trace: float = 1.0
    lambda_kernel: float = 1.0
    lambda_k_highrho: float = 0.0
    k_highrho_cut: float = 1e-6
    k_highrho_eps: float = 1e-6
    pair_rho_log_mean: bool = False
    pair_rho_log_diff: bool = False
    pair_rho_scaled_product: bool = False
    pair_rho_grad_norm: bool = False
    pair_rho_laplacian: bool = False
    pair_rho_directional_grad: bool = False
    pair_rho_source: str = "auto"
    pair_rho_stop_gradient: bool = True
    pair_rho_eps: float = 1e-14
    pair_rho_log_scale: float = 8.0
    pair_rho_log_clip: float = 4.0
    pair_rho_scaled_clip: float = 20.0
    pair_rho_product_transform: str = "log1p"
    pair_rho_laplacian_clip: float = 50.0
    pair_rho_oracle_mode: str = "off"

    save_weights: bool = True
    pretrained_point_weights: str | None = None


_SYSTEM_TENSOR_CACHE: dict[tuple[int, str], tf.Tensor] = {}
_EVAL_BATCH_CACHE: dict[tuple[object, ...], dict[str, object]] = {}
_TRUE_FIELD_DERIVATIVE_CACHE: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}


@dataclass
class V2Models:
    point: tf.keras.Model | None = None
    pair: tf.keras.Model | None = None
    context: tf.keras.Model | None = None

    def trainable_variables(self) -> list[tf.Variable]:
        variables: list[tf.Variable] = []
        for model in (self.point, self.pair, self.context):
            if model is not None:
                variables.extend(model.trainable_variables)
        return variables

    def get_weights(self) -> list[list[np.ndarray] | None]:
        return [model.get_weights() if model is not None else None for model in (self.point, self.pair, self.context)]

    def set_weights(self, weights: list[list[np.ndarray] | None]) -> None:
        for model, model_weights in zip((self.point, self.pair, self.context), weights):
            if model is not None and model_weights is not None:
                model.set_weights(model_weights)


def to_tensor(array: np.ndarray) -> tf.Tensor:
    return tf.convert_to_tensor(array, dtype=tf.float32)


def system_tensor(system: SystemRecord, name: str, array: np.ndarray) -> tf.Tensor:
    key = (id(system), name)
    tensor = _SYSTEM_TENSOR_CACHE.get(key)
    if tensor is None:
        tensor = to_tensor(array)
        _SYSTEM_TENSOR_CACHE[key] = tensor
    return tensor


def system_scalar(system: SystemRecord, name: str, value_factory) -> tf.Tensor:
    key = (id(system), name)
    tensor = _SYSTEM_TENSOR_CACHE.get(key)
    if tensor is None:
        tensor = tf.constant(float(value_factory()), dtype=tf.float32)
        _SYSTEM_TENSOR_CACHE[key] = tensor
    return tensor


def batch_tensor(batch: dict[str, object], key: str, dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
    tensor_key = f"_tf_{key}"
    cached = batch.get(tensor_key)
    if isinstance(cached, tf.Tensor):
        return cached
    tensor = tf.convert_to_tensor(batch[key], dtype=dtype)
    batch[tensor_key] = tensor
    return tensor


def index_tensor(indices: np.ndarray | tf.Tensor) -> tf.Tensor:
    if isinstance(indices, tf.Tensor):
        return tf.cast(indices, tf.int32)
    return tf.convert_to_tensor(indices, dtype=tf.int32)


def batch_indices(batch: dict[str, object], key: str) -> tf.Tensor:
    return batch_tensor(batch, key, dtype=tf.int32)


def build_mlp(
    *,
    input_dim: int,
    output_dim: int,
    width: int,
    depth: int,
    seed: int,
    rff_features: int,
    rff_scale: float,
    name: str,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name=f"{name}_input")
    if rff_features > 0:
        x = RandomFourierFeatures(
            n_features=rff_features,
            scale=rff_scale,
            seed=seed,
            include_input=True,
            name=f"{name}_rff",
        )(inputs)
    else:
        x = inputs
    for layer_idx in range(depth):
        x = tf.keras.layers.Dense(
            width,
            activation=tf.nn.silu,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed + layer_idx),
            bias_initializer="zeros",
            name=f"{name}_dense_{layer_idx}",
        )(x)
    outputs = tf.keras.layers.Dense(
        output_dim,
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed + depth + 100),
        bias_initializer="zeros",
        name=f"{name}_output",
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def uses_point_model(experiment: str) -> bool:
    return experiment in {"rho-only", "gamma-only", "gamma-simple", "gamma-residual", "gamma-context"}


def uses_pair_model(experiment: str) -> bool:
    return experiment in {"k-only", "gamma-only", "gamma-simple", "gamma-residual", "gamma-context"}


def uses_residual(experiment: str) -> bool:
    return experiment in {"gamma-residual", "gamma-context"}


def uses_context(experiment: str) -> bool:
    return experiment == "gamma-context"


def active_pair_rho_feature_names(config: V2Config) -> list[str]:
    names: list[str] = []
    if config.pair_rho_log_mean:
        names.append("rho_log_mean")
    if config.pair_rho_log_diff:
        names.append("rho_log_diff")
    if config.pair_rho_scaled_product:
        names.append("rho_sqrt_product_scaled")
    if config.pair_rho_grad_norm:
        names.append("rho_grad_norm")
    if config.pair_rho_laplacian:
        names.append("rho_laplacian")
    if config.pair_rho_directional_grad:
        names.append("rho_directional_grad")
    if config.pair_rho_oracle_mode != "off":
        names.append(f"oracle_{config.pair_rho_oracle_mode}")
    return names


def needs_rho_derivatives(config: V2Config) -> bool:
    return config.pair_rho_grad_norm or config.pair_rho_laplacian or config.pair_rho_directional_grad


def pair_rho_source(config: V2Config) -> str:
    if not active_pair_rho_feature_names(config):
        return "off"
    source = config.pair_rho_source.strip().lower()
    if source == "auto":
        # Derivatives always require the point model (pred), but if only scalars
        # are used and we are in k-only, we can use true.
        if needs_rho_derivatives(config):
            return "pred"
        return "true" if config.experiment == "k-only" else "pred"
    if source not in {"true", "pred"}:
        raise ValueError("pair_rho_source must be one of: auto, true, pred.")
    return source


def validate_pair_rho_config(config: V2Config) -> None:
    oracle_mode = config.pair_rho_oracle_mode.strip().lower()
    if oracle_mode not in {"off", "neutral-derivatives", "three-density", "fukui"}:
        raise ValueError(
            "pair_rho_oracle_mode must be one of: off, neutral-derivatives, three-density, fukui."
        )
    source = pair_rho_source(config)
    if source == "off":
        return
    if source == "true" and config.experiment != "k-only":
        raise ValueError("True-rho pair features are allowed only for k-only oracle ablations.")
    if source == "true" and needs_rho_derivatives(config):
        raise ValueError("Derivative features are currently only implemented for predicted-rho source.")
    if oracle_mode != "off" and source != "true":
        raise ValueError("Oracle rho descriptor modes require --pair-rho-source true.")
    if oracle_mode != "off" and config.experiment != "k-only":
        raise ValueError("Oracle rho descriptor modes are k-only ablations. Use --experiment k-only.")
    if source == "pred":
        if not uses_point_model(config.experiment) or not uses_pair_model(config.experiment):
            raise ValueError("Predicted-rho pair features require an experiment with both point and pair models.")


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

    lap = second_derivative(0) + second_derivative(1) + second_derivative(2)
    return tf.reshape(lap, (-1, 1))


def true_field_derivatives(
    system: SystemRecord,
    field_name: str,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite-difference gradient and Laplacian for a stored oracle field."""
    key = (id(system), field_name)
    cached = _TRUE_FIELD_DERIVATIVE_CACHE.get(key)
    if cached is not None:
        return cached
    n_axis = len(system.axis)
    volume = np.asarray(values, dtype=np.float32).reshape(n_axis, n_axis, n_axis)
    edge_order = 2 if n_axis >= 3 else 1
    grad_components = np.gradient(volume, system.step, edge_order=edge_order)
    grad = np.stack(grad_components, axis=-1).reshape(-1, 3).astype(np.float32)
    lap = sum(
        np.gradient(component, system.step, axis=axis, edge_order=edge_order)
        for axis, component in enumerate(grad_components)
    ).reshape(-1, 1).astype(np.float32)
    result = (grad, lap)
    _TRUE_FIELD_DERIVATIVE_CACHE[key] = result
    return result


def signed_log_scaled(values: np.ndarray, clip: float) -> np.ndarray:
    clip = max(float(clip), 1.0)
    clipped = np.clip(values, -clip, clip)
    return (np.sign(clipped) * np.log1p(np.abs(clipped)) / math.log1p(clip)).astype(np.float32)


def oracle_field_pair_features(
    system: SystemRecord,
    field_name: str,
    values: np.ndarray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    config: V2Config,
) -> np.ndarray:
    """Build endpoint value, gradient-norm, and Laplacian descriptors for one field."""
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    grad, lap = true_field_derivatives(system, field_name, values)
    scale = max(float(np.mean(np.abs(values))), float(config.pair_rho_eps), 1e-30)
    value_clip = max(float(config.pair_rho_scaled_clip), 1.0)
    lap_clip = max(float(config.pair_rho_laplacian_clip), 1.0)

    scaled_values = signed_log_scaled(values / scale, value_clip)
    scaled_grad_norm = np.log1p(
        np.clip(np.linalg.norm(grad, axis=1, keepdims=True) * system.step / scale, 0.0, value_clip)
    ) / math.log1p(value_clip)
    scaled_lap = signed_log_scaled(lap * (system.step**2) / scale, lap_clip)
    return np.concatenate(
        [
            scaled_values[left_idx],
            scaled_values[right_idx],
            scaled_grad_norm[left_idx],
            scaled_grad_norm[right_idx],
            scaled_lap[left_idx],
            scaled_lap[right_idx],
        ],
        axis=1,
    ).astype(np.float32)


def oracle_rho_fields(system: SystemRecord, config: V2Config) -> list[tuple[str, np.ndarray]]:
    """Select stored true-density or Fukui fields for a pair-model oracle ablation."""
    mode = config.pair_rho_oracle_mode.strip().lower()
    if mode == "off":
        return []
    neutral = system.rho_diag
    if mode == "neutral-derivatives":
        return [("rho_neutral", neutral)]
    if system.rho_cation is None or system.rho_anion is None:
        raise ValueError(
            f"System {system.system_id} has no charged density oracle channels. "
            "Rebuild NPZ files with --include-charged-density-oracles."
        )
    if mode == "three-density":
        return [
            ("rho_neutral", neutral),
            ("rho_cation", system.rho_cation),
            ("rho_anion", system.rho_anion),
        ]
    if mode == "fukui":
        return [
            ("rho_neutral", neutral),
            ("fukui_plus", system.rho_anion - neutral),
            ("fukui_minus", neutral - system.rho_cation),
        ]
    raise AssertionError(f"Unhandled oracle mode: {mode}")


def true_rho_pair_features(
    system: SystemRecord,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    config: V2Config,
) -> np.ndarray:
    rho_left = np.maximum(system.rho_diag[left_idx].astype(np.float32), 0.0)
    rho_right = np.maximum(system.rho_diag[right_idx].astype(np.float32), 0.0)
    eps = max(float(config.pair_rho_eps), 1e-30)
    rho_scale = max(float(np.mean(np.maximum(system.rho_diag, 0.0))), eps)

    log_left = np.log(rho_left + eps)
    log_right = np.log(rho_right + eps)
    log_scale = math.log(rho_scale + eps)
    log_feature_scale = max(float(config.pair_rho_log_scale), 1e-6)
    log_feature_clip = float(config.pair_rho_log_clip)

    features: list[np.ndarray] = []
    if config.pair_rho_log_mean:
        log_mean = (0.5 * (log_left + log_right) - log_scale) / log_feature_scale
        if log_feature_clip > 0.0:
            log_mean = np.clip(log_mean, -log_feature_clip, log_feature_clip)
        features.append(log_mean.reshape(-1, 1))
    if config.pair_rho_log_diff:
        log_diff = np.abs(log_left - log_right) / log_feature_scale
        if log_feature_clip > 0.0:
            log_diff = np.clip(log_diff, 0.0, log_feature_clip)
        features.append(log_diff.reshape(-1, 1))
    if config.pair_rho_scaled_product:
        scaled = np.sqrt(rho_left * rho_right) / rho_scale
        if config.pair_rho_scaled_clip > 0.0:
            scaled = np.clip(scaled, 0.0, config.pair_rho_scaled_clip)
        if config.pair_rho_product_transform == "log1p":
            product_scale = max(float(config.pair_rho_scaled_clip), 1.0)
            scaled = np.log1p(scaled) / math.log1p(product_scale)
        elif config.pair_rho_product_transform == "linear":
            product_scale = max(float(config.pair_rho_scaled_clip), 1.0)
            scaled = scaled / product_scale
        else:
            raise ValueError("pair_rho_product_transform must be one of: log1p, linear.")
        features.append(scaled.reshape(-1, 1))

    for field_name, values in oracle_rho_fields(system, config):
        features.append(oracle_field_pair_features(system, field_name, values, left_idx, right_idx, config))

    if not features:
        return np.empty((len(left_idx), 0), dtype=np.float32)
    return np.concatenate(features, axis=1).astype(np.float32)


def predicted_rho_pair_features(
    system: SystemRecord,
    rho_values: tf.Tensor,
    left_idx: np.ndarray | tf.Tensor,
    right_idx: np.ndarray | tf.Tensor,
    config: V2Config,
) -> tf.Tensor:
    eps = tf.constant(max(float(config.pair_rho_eps), 1e-30), dtype=tf.float32)
    rho_clean = tf.maximum(rho_values, 0.0)
    rho_scale = tf.maximum(tf.reduce_mean(rho_clean), eps)

    rho_left = tf.maximum(gather(rho_values, left_idx), 0.0)
    rho_right = tf.maximum(gather(rho_values, right_idx), 0.0)
    log_left = tf.math.log(rho_left + eps)
    log_right = tf.math.log(rho_right + eps)
    log_scale = tf.math.log(rho_scale + eps)
    log_feature_scale = tf.constant(max(float(config.pair_rho_log_scale), 1e-6), dtype=tf.float32)
    log_feature_clip = float(config.pair_rho_log_clip)

    features: list[tf.Tensor] = []
    if config.pair_rho_log_mean:
        log_mean = (0.5 * (log_left + log_right) - log_scale) / log_feature_scale
        if log_feature_clip > 0.0:
            log_mean = tf.clip_by_value(log_mean, -log_feature_clip, log_feature_clip)
        features.append(log_mean)
    if config.pair_rho_log_diff:
        log_diff = tf.abs(log_left - log_right) / log_feature_scale
        if log_feature_clip > 0.0:
            log_diff = tf.clip_by_value(log_diff, 0.0, log_feature_clip)
        features.append(log_diff)
    if config.pair_rho_scaled_product:
        scaled = tf.sqrt(tf.maximum(rho_left * rho_right, 0.0)) / rho_scale
        if config.pair_rho_scaled_clip > 0.0:
            scaled = tf.clip_by_value(scaled, 0.0, float(config.pair_rho_scaled_clip))
        product_scale = max(float(config.pair_rho_scaled_clip), 1.0)
        if config.pair_rho_product_transform == "log1p":
            scaled = tf.math.log1p(scaled) / math.log1p(product_scale)
        elif config.pair_rho_product_transform == "linear":
            scaled = scaled / product_scale
        else:
            raise ValueError("pair_rho_product_transform must be one of: log1p, linear.")
        features.append(scaled)

    if needs_rho_derivatives(config):
        grad_all = richardson_gradient_3d(rho_values, len(system.axis), system.step)
        lap_all = richardson_laplacian_3d(rho_values, len(system.axis), system.step)

        if config.pair_rho_grad_norm:
            # reduced gradient |grad| / (rho + eps)
            grad_left = gather(grad_all, left_idx)
            grad_right = gather(grad_all, right_idx)
            norm_left = tf.norm(grad_left, axis=1, keepdims=True) / (rho_left + eps)
            norm_right = tf.norm(grad_right, axis=1, keepdims=True) / (rho_right + eps)
            # Clip and log to keep it friendly for MLPs
            norm_left = tf.math.log1p(tf.clip_by_value(norm_left, 0.0, 100.0))
            norm_right = tf.math.log1p(tf.clip_by_value(norm_right, 0.0, 100.0))
            features.extend([norm_left, norm_right])

        if config.pair_rho_laplacian:
            lap_left = gather(lap_all, left_idx) / (rho_left + eps)
            lap_right = gather(lap_all, right_idx) / (rho_right + eps)
            clip = float(config.pair_rho_laplacian_clip)
            lap_left = tf.clip_by_value(lap_left, -clip, clip) / clip
            lap_right = tf.clip_by_value(lap_right, -clip, clip) / clip
            features.extend([lap_left, lap_right])

        if config.pair_rho_directional_grad:
            # grad_i . (r_j - r_i) / |r_j - r_i|
            points = system_tensor(system, "points", system.points)
            r_left = gather(points, left_idx)
            r_right = gather(points, right_idx)
            diff = r_right - r_left
            dist = tf.norm(diff, axis=1, keepdims=True)
            unit_diff = diff / (dist + 1e-8)

            grad_left = gather(grad_all, left_idx)
            grad_right = gather(grad_all, right_idx)

            # dot products
            dg_left = tf.reduce_sum(grad_left * unit_diff, axis=1, keepdims=True) / (rho_left + eps)
            dg_right = tf.reduce_sum(grad_right * (-unit_diff), axis=1, keepdims=True) / (rho_right + eps)

            # clip
            dg_left = tf.clip_by_value(dg_left, -10.0, 10.0)
            dg_right = tf.clip_by_value(dg_right, -10.0, 10.0)
            features.extend([dg_left, dg_right])

    if not features:
        return tf.zeros((tf.shape(index_tensor(left_idx))[0], 0), dtype=tf.float32)
    return tf.concat(features, axis=1)


def build_v2_pair_features(
    system: SystemRecord,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    config: V2Config,
) -> np.ndarray:
    base = build_pair_features(system, left_idx, right_idx)
    if pair_rho_source(config) != "true":
        return base.astype(np.float32)
    rho_features = true_rho_pair_features(system, left_idx, right_idx, config)
    return np.concatenate([base, rho_features], axis=1).astype(np.float32)


def v2_pair_feature_dim(system: SystemRecord, config: V2Config) -> int:
    sample_idx = np.array([0], dtype=np.int64)
    base_dim = build_pair_features(system, sample_idx, sample_idx).shape[1]
    rho_dim = 0
    if pair_rho_source(config) == "true":
        rho_dim = true_rho_pair_features(system, sample_idx, sample_idx, config).shape[1]
    elif pair_rho_source(config) == "pred":
        rho_dim += int(config.pair_rho_log_mean)
        rho_dim += int(config.pair_rho_log_diff)
        rho_dim += int(config.pair_rho_scaled_product)
        rho_dim += 2 * int(config.pair_rho_grad_norm)
        rho_dim += 2 * int(config.pair_rho_laplacian)
        rho_dim += 2 * int(config.pair_rho_directional_grad)
    return base_dim + rho_dim


def build_v2_models(point_dim: int, pair_dim: int, global_dim: int, config: V2Config) -> V2Models:
    if config.experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown V2 experiment: {config.experiment}")
    validate_pair_rho_config(config)
    pair_rho_names = active_pair_rho_feature_names(config)
    rho_source = pair_rho_source(config)

    point_model = None
    pair_model = None
    context_model = None

    if uses_point_model(config.experiment):
        point_output_dim = 1 + (config.rank if uses_residual(config.experiment) else 0)
        point_model = build_mlp(
            input_dim=point_dim + global_dim,
            output_dim=point_output_dim,
            width=config.width,
            depth=config.depth,
            seed=config.seed + 101,
            rff_features=config.rff_features,
            rff_scale=config.rff_scale,
            name=f"{config.experiment}_point",
        )

    if uses_pair_model(config.experiment):
        pair_model = build_mlp(
            input_dim=pair_dim + global_dim,
            output_dim=1,
            width=config.width,
            depth=config.depth,
            seed=config.seed + 211,
            rff_features=config.rff_features,
            rff_scale=config.rff_scale,
            name=f"{config.experiment}_pair",
        )

    if uses_context(config.experiment):
        context_model = build_mlp(
            input_dim=global_dim,
            output_dim=config.rank,
            width=max(32, config.width // 2),
            depth=max(1, config.depth),
            seed=config.seed + 307,
            rff_features=max(4, config.rff_features // 2) if config.context_rff else 0,
            rff_scale=config.rff_scale,
            name=f"{config.experiment}_context",
        )

    models = V2Models(point=point_model, pair=pair_model, context=context_model)

    if config.pretrained_point_weights and models.point is not None:
        models.point.load_weights(config.pretrained_point_weights)
        models.point.trainable = False
        print(f"Loaded pretrained point weights and frozen point model: {config.pretrained_point_weights}")

    print_block(
        "V2 model",
        [
            ("experiment", config.experiment),
            ("point input dim", point_dim + global_dim if point_model is not None else "off"),
            ("pair input dim", pair_dim + global_dim if pair_model is not None else "off"),
            ("global dim", global_dim),
            ("rank", config.rank if uses_residual(config.experiment) else 0),
            ("kernel base alpha", config.kernel_base_alpha if pair_model is not None else "off"),
            ("pretrained point weights", config.pretrained_point_weights or "off"),
            ("pair rho features", ", ".join(pair_rho_names) if pair_rho_names else "off"),
            ("pair rho source", rho_source if pair_rho_names else "off"),
            ("pair rho stop-gradient", config.pair_rho_stop_gradient if rho_source == "pred" else "n/a"),
            (
                "rho derivatives",
                "predicted rho, O(h^4) symmetric-boundary stencil"
                if needs_rho_derivatives(config)
                else (
                    f"true oracle: {config.pair_rho_oracle_mode}"
                    if config.pair_rho_oracle_mode != "off"
                    else "off"
                ),
            ),
            ("context RFF", config.context_rff if context_model is not None else "off"),
            ("point params", point_model.count_params() if point_model is not None else 0),
            ("point trainable", point_model.trainable if point_model is not None else "n/a"),
            ("pair params", pair_model.count_params() if pair_model is not None else 0),
            ("context params", context_model.count_params() if context_model is not None else 0),
        ],
    )
    return models


def tile_global(global_context: tf.Tensor, count: tf.Tensor) -> tf.Tensor:
    global_context = tf.reshape(global_context, (1, -1))
    return tf.repeat(global_context, repeats=count, axis=0)


def point_inputs(system: SystemRecord) -> tf.Tensor:
    key = (id(system), "point_inputs")
    cached = _SYSTEM_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached
    local = system_tensor(system, "local_features", system.local_features)
    tiled = tile_global(system_tensor(system, "global_context", system.global_context), tf.shape(local)[0])
    inputs = tf.concat([local, tiled], axis=1)
    _SYSTEM_TENSOR_CACHE[key] = inputs
    return inputs


def rho_diag_tensor(system: SystemRecord) -> tf.Tensor:
    return system_tensor(system, "rho_diag", system.rho_diag)


def global_context_tensor(system: SystemRecord) -> tf.Tensor:
    return system_tensor(system, "global_context", system.global_context)


def base_pair_features_tf(system: SystemRecord, left_idx: np.ndarray | tf.Tensor, right_idx: np.ndarray | tf.Tensor) -> tf.Tensor:
    points = system_tensor(system, "points", system.points)
    potential = system_tensor(system, "potential", system.potential)
    grad_potential = system_tensor(system, "grad_potential", system.grad_potential)
    left = index_tensor(left_idx)
    right = index_tensor(right_idx)

    points_r = tf.gather(points, left)
    points_rp = tf.gather(points, right)
    midpoint = 0.5 * (points_r + points_rp)
    separation = points_r - points_rp
    abs_separation = tf.abs(separation)
    sep_sq_components = separation**2
    sep_norm = tf.norm(separation, axis=1, keepdims=True)
    sep_sq = tf.reduce_sum(separation**2, axis=1, keepdims=True)

    pot_r = tf.gather(potential, left)
    pot_rp = tf.gather(potential, right)
    grad_r = tf.gather(grad_potential, left)
    grad_rp = tf.gather(grad_potential, right)

    domain_scale = system_scalar(system, "pair_domain_scale", lambda: max(float(np.max(np.abs(system.axis))), 1e-6))
    step_scale = system_scalar(system, "pair_step_scale", lambda: max(float(system.step), 1e-6))
    pot_scale = system_scalar(system, "pair_potential_scale", lambda: max(float(np.std(system.potential)), 1.0))

    return tf.concat(
        [
            midpoint / domain_scale,
            abs_separation / domain_scale,
            sep_sq_components / (domain_scale * domain_scale),
            sep_norm / domain_scale,
            sep_sq / (domain_scale * domain_scale),
            0.5 * (pot_r + pot_rp) / pot_scale,
            0.5 * (grad_r + grad_rp) * step_scale / pot_scale,
            tf.abs(grad_r - grad_rp) * step_scale / pot_scale,
        ],
        axis=1,
    )


def pair_feature_tensor(system: SystemRecord, batch: dict[str, object], config: V2Config) -> tf.Tensor:
    cached = batch.get("_tf_pair_feat")
    if isinstance(cached, tf.Tensor):
        return cached
    if "pair_feat" in batch:
        return batch_tensor(batch, "pair_feat")
    pair_feat = base_pair_features_tf(system, batch_indices(batch, "left"), batch_indices(batch, "right"))
    batch["_tf_pair_feat"] = pair_feat
    return pair_feat


def predict_rho_and_modes(system: SystemRecord, models: V2Models, config: V2Config) -> tuple[tf.Tensor, tf.Tensor | None]:
    if models.point is None:
        rho_true = rho_diag_tensor(system)
        return rho_true, None

    point_out = models.point(point_inputs(system))
    rho_raw = tf.nn.softplus(point_out[:, :1]) + 1e-8
    if config.normalize_rho:
        normalizer = tf.reduce_sum(rho_raw) * system.cell_volume
        rho = rho_raw * (system.electron_count / tf.maximum(normalizer, 1e-12))
    else:
        rho = rho_raw
    modes = point_out[:, 1:] if point_out.shape[-1] and int(point_out.shape[-1]) > 1 else None
    return rho, modes


def gather(values: tf.Tensor, indices: np.ndarray | tf.Tensor) -> tf.Tensor:
    return tf.gather(values, index_tensor(indices))


def separation_factor(pair_feat: tf.Tensor, config: V2Config) -> tf.Tensor:
    sep_sq = tf.maximum(pair_feat[:, 10:11], 0.0)
    return 1.0 - tf.exp(-sep_sq / max(config.sep_factor_scale, 1e-8))


def kernel_base(pair_feat: tf.Tensor, config: V2Config) -> tf.Tensor:
    sep_sq = tf.maximum(pair_feat[:, 10:11], 0.0)
    alpha = tf.constant(max(config.kernel_base_alpha, 0.0), dtype=tf.float32)
    return tf.exp(-alpha * sep_sq)


def predict_kernel(
    system: SystemRecord,
    batch: dict[str, object],
    models: V2Models,
    config: V2Config,
    modes_all: tf.Tensor | None,
    rho_pred: tf.Tensor | None = None,
) -> tf.Tensor:
    if models.pair is None:
        raise RuntimeError("This experiment has no pair model.")

    pair_feat = pair_feature_tensor(system, batch, config)
    left_idx = batch_indices(batch, "left")
    right_idx = batch_indices(batch, "right")
    if pair_rho_source(config) == "pred":
        if rho_pred is None:
            raise RuntimeError("Predicted-rho pair features require rho_pred.")
        rho_feature_values = tf.stop_gradient(rho_pred) if config.pair_rho_stop_gradient else rho_pred
        pair_feat = tf.concat(
            [pair_feat, predicted_rho_pair_features(system, rho_feature_values, batch["left"], batch["right"], config)],
            axis=1,
        )

    global_tiled = tile_global(global_context_tensor(system), tf.shape(pair_feat)[0])
    pair_input = tf.concat([pair_feat, global_tiled], axis=1)
    delta_pair = models.pair(pair_input)
    sep_factor = separation_factor(pair_feat, config)
    kernel = kernel_base(pair_feat, config) + sep_factor * delta_pair

    if uses_residual(config.experiment):
        if modes_all is None:
            raise RuntimeError("Residual experiment requires point modes.")
        left_modes = gather(modes_all, left_idx)
        right_modes = gather(modes_all, right_idx)

        if models.context is not None:
            context_weights = tf.nn.softplus(models.context(tf.reshape(global_context_tensor(system), (1, -1)))) + 1e-6
            left_modes = left_modes * tf.sqrt(context_weights)
            right_modes = right_modes * tf.sqrt(context_weights)

        left_norm = left_modes / tf.sqrt(tf.reduce_sum(left_modes**2, axis=1, keepdims=True) + 1e-8)
        right_norm = right_modes / tf.sqrt(tf.reduce_sum(right_modes**2, axis=1, keepdims=True) + 1e-8)
        residual = tf.reduce_sum(left_norm * right_norm, axis=1, keepdims=True)
        kernel = kernel + config.residual_scale * sep_factor * residual

    return kernel


def gamma_from_rho_kernel(rho_left: tf.Tensor, rho_right: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    return tf.sqrt(tf.maximum(rho_left * rho_right, 0.0) + 1e-12) * kernel


def make_batch(
    system: SystemRecord,
    batch_size: int,
    epoch: int,
    total_epochs: int,
    rng: np.random.Generator,
    config: V2Config,
) -> dict[str, object]:
    left, right, categories = sample_pair_indices(
        system,
        batch_size,
        epoch,
        total_epochs,
        rng,
        category_probs=config.pair_sampling_probs,
    )
    gamma_true = system.gamma_values(left, right)
    rho_left = system.rho_diag[left].astype(np.float32)
    rho_right = system.rho_diag[right].astype(np.float32)
    batch: dict[str, object] = {
        "left": left.astype(np.int64),
        "right": right.astype(np.int64),
        "gamma_true": gamma_true.astype(np.float32),
        "rho_left_true": rho_left,
        "rho_right_true": rho_right,
        "weights": pair_weights_from_categories(categories, config.pair_category_weights),
    }
    if not config.pair_features_on_device or pair_rho_source(config) == "true":
        pair_feat = build_v2_pair_features(system, left, right, config)
        batch["pair_feat"] = pair_feat.astype(np.float32)
    return batch


def prepare_batch_tensors(batch: dict[str, object]) -> dict[str, object]:
    """Warm TensorFlow tensors for arrays that are reused inside a step/eval."""
    for key in ("left", "right"):
        batch_indices(batch, key)
    for key in ("gamma_true", "rho_left_true", "rho_right_true", "weights"):
        batch_tensor(batch, key)
    if "pair_feat" in batch:
        batch_tensor(batch, "pair_feat")
    return batch


def eval_batch_cache_key(system: SystemRecord, config: V2Config) -> tuple[object, ...]:
    return (
        id(system),
        config.eval_pair_count,
        config.pair_sampling_probs,
        config.pair_category_weights,
        tuple(active_pair_rho_feature_names(config)),
        pair_rho_source(config),
        config.pair_rho_eps,
        config.pair_rho_log_scale,
        config.pair_rho_log_clip,
        config.pair_rho_scaled_clip,
        config.pair_rho_product_transform,
        config.pair_rho_laplacian_clip,
        config.pair_rho_oracle_mode,
        config.pair_features_on_device,
    )


def make_eval_batch(
    system: SystemRecord,
    config: V2Config,
    rng: np.random.Generator,
) -> dict[str, object]:
    if not config.cache_eval_batches:
        return prepare_batch_tensors(make_batch(system, config.eval_pair_count, 0, 1, rng, config))

    key = eval_batch_cache_key(system, config)
    cached = _EVAL_BATCH_CACHE.get(key)
    if cached is not None:
        return cached
    batch = prepare_batch_tensors(make_batch(system, config.eval_pair_count, 0, 1, rng, config))
    _EVAL_BATCH_CACHE[key] = batch
    return batch


def true_kernel_target(batch: dict[str, np.ndarray], config: V2Config) -> tuple[np.ndarray, np.ndarray]:
    rho_product = np.maximum(batch["rho_left_true"] * batch["rho_right_true"], config.kernel_rho_floor)
    denom = np.sqrt(rho_product).astype(np.float32)
    target = batch["gamma_true"] / denom
    if config.kernel_target_clip > 0:
        target = np.clip(target, -config.kernel_target_clip, config.kernel_target_clip)

    rho_scale = max(float(np.mean(np.concatenate([batch["rho_left_true"], batch["rho_right_true"]]))), config.kernel_rho_floor)
    density_weight = np.clip(denom / rho_scale, 0.05, 1.0).astype(np.float32)
    weights = batch["weights"] * density_weight
    return target.astype(np.float32), weights.astype(np.float32)


def highrho_kernel_loss(
    batch: dict[str, object],
    gamma_pred: tf.Tensor,
    config: V2Config,
) -> tuple[tf.Tensor, tf.Tensor]:
    rho_left = batch_tensor(batch, "rho_left_true")
    rho_right = batch_tensor(batch, "rho_right_true")
    denom = tf.sqrt(tf.maximum(rho_left * rho_right, 0.0))
    mask = tf.cast(denom > config.k_highrho_cut, tf.float32)
    weights = batch_tensor(batch, "weights") * mask
    scaled_err = (gamma_pred - batch_tensor(batch, "gamma_true")) / tf.maximum(denom, config.k_highrho_eps)
    weight_sum = tf.reduce_sum(weights)
    loss = tf.reduce_sum(weights * tf.square(scaled_err)) / tf.maximum(weight_sum, 1e-12)
    active_frac = tf.reduce_mean(mask)
    return loss, active_frac


def highrho_kernel_metrics_np(
    batch: dict[str, np.ndarray],
    gamma_pred: np.ndarray,
    config: V2Config,
) -> dict[str, float]:
    denom = np.sqrt(np.maximum(batch["rho_left_true"] * batch["rho_right_true"], 0.0)).astype(np.float32)
    mask = denom > config.k_highrho_cut
    active_frac = float(np.mean(mask))
    if not np.any(mask):
        return {
            "kernel_highrho_loss": float("nan"),
            "kernel_highrho_mae": float("nan"),
            "kernel_highrho_frac": active_frac,
        }
    weights = batch["weights"] * mask.astype(np.float32)
    scaled_err = (gamma_pred - batch["gamma_true"]) / np.maximum(denom, config.k_highrho_eps)
    weight_sum = max(float(np.sum(weights)), 1e-12)
    return {
        "kernel_highrho_loss": float(np.sum(weights * scaled_err**2) / weight_sum),
        "kernel_highrho_mae": float(np.mean(np.abs(scaled_err[mask]))),
        "kernel_highrho_frac": active_frac,
    }


def weighted_mse(true: tf.Tensor, pred: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(weights * tf.square(true - pred)) / tf.maximum(tf.reduce_sum(weights), 1e-12)


def rho_losses(system: SystemRecord, rho_pred: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    rho_loss = tf.reduce_mean(tf.square(rho_pred - rho_diag_tensor(system)))
    trace_pred = tf.reduce_sum(rho_pred) * system.cell_volume
    trace_loss = tf.square((trace_pred - system.electron_count) / max(system.electron_count, 1.0))
    return rho_loss, trace_loss


def compute_step_loss(
    system: SystemRecord,
    batch: dict[str, object] | None,
    models: V2Models,
    config: V2Config,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    rho_pred, modes_all = predict_rho_and_modes(system, models, config)
    rho_loss, trace_loss = rho_losses(system, rho_pred)

    zero = tf.constant(0.0, dtype=tf.float32)
    gamma_loss = zero
    kernel_loss = zero
    kernel_highrho_loss = zero
    kernel_highrho_frac = zero

    if config.experiment == "rho-only":
        total = config.lambda_rho * rho_loss + config.lambda_trace * trace_loss
        return total, {
            "rho_loss": rho_loss,
            "trace_loss": trace_loss,
            "gamma_loss": zero,
            "kernel_loss": zero,
            "kernel_highrho_loss": zero,
            "kernel_highrho_frac": zero,
        }

    if batch is None:
        raise RuntimeError("Pair batch is required for non-rho-only experiments.")

    left_idx = batch_indices(batch, "left")
    right_idx = batch_indices(batch, "right")
    if config.experiment == "k-only":
        rho_left = batch_tensor(batch, "rho_left_true")
        rho_right = batch_tensor(batch, "rho_right_true")
    else:
        rho_left = gather(rho_pred, left_idx)
        rho_right = gather(rho_pred, right_idx)

    kernel = predict_kernel(system, batch, models, config, modes_all, rho_pred)
    gamma_pred = gamma_from_rho_kernel(rho_left, rho_right, kernel)
    gamma_loss = weighted_mse(batch_tensor(batch, "gamma_true"), gamma_pred, batch_tensor(batch, "weights"))

    if config.experiment == "k-only":
        kernel_target, kernel_weights = true_kernel_target(batch, config)
        kernel_loss = weighted_mse(to_tensor(kernel_target), kernel, to_tensor(kernel_weights))
        kernel_highrho_loss, kernel_highrho_frac = highrho_kernel_loss(batch, gamma_pred, config)
        total = config.lambda_kernel * gamma_loss + config.lambda_k_highrho * kernel_highrho_loss
    elif config.experiment == "gamma-only":
        total = config.lambda_gamma * gamma_loss
    else:
        total = (
            config.lambda_gamma * gamma_loss
            + config.lambda_rho * rho_loss
            + config.lambda_trace * trace_loss
        )

    return total, {
        "rho_loss": rho_loss,
        "trace_loss": trace_loss,
        "gamma_loss": gamma_loss,
        "kernel_loss": kernel_loss,
        "kernel_highrho_loss": kernel_highrho_loss,
        "kernel_highrho_frac": kernel_highrho_frac,
    }


def baseline_gamma(
    system: SystemRecord,
    left: np.ndarray,
    right: np.ndarray,
    alpha: float,
    density_power: str = "sqrt",
) -> np.ndarray:
    rho_left = system.rho_diag[left]
    rho_right = system.rho_diag[right]
    dist_sq = np.sum((system.points[left] - system.points[right]) ** 2, axis=1, keepdims=True)
    if density_power == "product":
        prefactor = rho_left * rho_right
    elif density_power == "sqrt":
        prefactor = np.sqrt(np.maximum(rho_left * rho_right, 0.0))
    else:
        raise ValueError("baseline_density_power must be 'sqrt' or 'product'.")
    return (prefactor * np.exp(-float(alpha) * dist_sq)).astype(np.float32)


def fit_baseline_alpha(split: DatasetSplit, config: V2Config) -> float:
    candidates = [0.0]
    if config.baseline_alpha_count > 1:
        candidates.extend(
            np.geomspace(config.baseline_alpha_min, config.baseline_alpha_max, config.baseline_alpha_count).tolist()
        )
    rng = np.random.default_rng(config.seed + 401)
    losses = np.zeros(len(candidates), dtype=np.float64)

    n_batches = max(1, config.baseline_fit_batches)
    for batch_idx in range(n_batches):
        system = choose_system(split.train_systems, rng)
        batch = make_batch(system, config.batch_size, batch_idx, n_batches, rng, config)
        gamma_true = batch["gamma_true"]
        weights = batch["weights"]
        denom = max(float(np.sum(weights)), 1e-12)
        for idx, alpha in enumerate(candidates):
            pred = baseline_gamma(system, batch["left"], batch["right"], alpha, config.baseline_density_power)
            losses[idx] += float(np.sum(weights * (pred - gamma_true) ** 2) / denom)

    best_idx = int(np.argmin(losses))
    best_alpha = float(candidates[best_idx])
    print_block(
        "V2 baseline fit",
        [
            ("density power", config.baseline_density_power),
            ("alpha", f"{best_alpha:.6g}"),
            ("fit loss", f"{losses[best_idx] / n_batches:.6e}"),
        ],
    )
    return best_alpha


def evaluate_system(
    system: SystemRecord,
    models: V2Models,
    config: V2Config,
    rng: np.random.Generator,
    alpha: float | None = None,
) -> dict[str, float]:
    batch = make_eval_batch(system, config, rng)

    rho_pred_np: np.ndarray
    gamma_pred_np: np.ndarray | None = None
    kernel_pred_np: np.ndarray | None = None

    if config.experiment == "baseline":
        rho_pred_np = system.rho_diag.astype(np.float32)
        gamma_pred_np = baseline_gamma(
            system,
            batch["left"],  # type: ignore[arg-type]
            batch["right"],  # type: ignore[arg-type]
            alpha=0.0 if alpha is None else alpha,
            density_power=config.baseline_density_power,
        )
    else:
        rho_pred, modes_all = predict_rho_and_modes(system, models, config)
        rho_pred_np = rho_pred.numpy().astype(np.float32)
        if config.experiment != "rho-only":
            if config.experiment == "k-only":
                rho_left = batch_tensor(batch, "rho_left_true")
                rho_right = batch_tensor(batch, "rho_right_true")
            else:
                rho_left = gather(rho_pred, batch_indices(batch, "left"))
                rho_right = gather(rho_pred, batch_indices(batch, "right"))
            kernel_pred = predict_kernel(system, batch, models, config, modes_all, rho_pred=rho_pred)
            gamma_pred = gamma_from_rho_kernel(rho_left, rho_right, kernel_pred)
            gamma_pred_np = gamma_pred.numpy().astype(np.float32)
            kernel_pred_np = kernel_pred.numpy().astype(np.float32)

    rho_loss = float(np.mean((rho_pred_np - system.rho_diag) ** 2))
    rho_mae = float(np.mean(np.abs(rho_pred_np - system.rho_diag)))
    trace_pred = float(np.sum(rho_pred_np) * system.cell_volume)
    trace_rel_error = float((trace_pred - system.electron_count) / max(system.electron_count, 1.0))

    metrics = {
        "rho_loss": rho_loss,
        "rho_mae": rho_mae,
        "trace_pred": trace_pred,
        "trace_rel_error": trace_rel_error,
        "pair_loss": float("nan"),
        "pair_mae": float("nan"),
        "kernel_loss": float("nan"),
        "kernel_mae": float("nan"),
        "kernel_highrho_loss": float("nan"),
        "kernel_highrho_mae": float("nan"),
        "kernel_highrho_frac": float("nan"),
    }

    if gamma_pred_np is not None:
        weights = batch["weights"]
        gamma_err = gamma_pred_np - batch["gamma_true"]
        metrics["pair_loss"] = float(np.sum(weights * gamma_err**2) / max(float(np.sum(weights)), 1e-12))
        metrics["pair_mae"] = float(np.mean(np.abs(gamma_err)))

    if kernel_pred_np is not None:
        kernel_target, kernel_weights = true_kernel_target(batch, config)
        kernel_err = kernel_pred_np - kernel_target
        metrics["kernel_loss"] = float(
            np.sum(kernel_weights * kernel_err**2) / max(float(np.sum(kernel_weights)), 1e-12)
        )
        metrics["kernel_mae"] = float(np.mean(np.abs(kernel_err)))
        if gamma_pred_np is not None:
            metrics.update(highrho_kernel_metrics_np(batch, gamma_pred_np, config))

    return metrics


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    avg: dict[str, float] = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows if key in row and np.isfinite(row[key])], dtype=np.float64)
        avg[key] = float(np.mean(values)) if values.size else float("nan")
    return avg


def evaluate_split(
    systems: Iterable[SystemRecord],
    split_name: str,
    models: V2Models,
    config: V2Config,
    alpha: float | None,
    seed_offset: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    rng = np.random.default_rng(config.seed + seed_offset)
    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, float]] = []
    for system in systems:
        metrics = evaluate_system(system, models, config, rng, alpha=alpha)
        metric_rows.append(metrics)
        rows.append(
            {
                "split": split_name,
                "system_id": system.system_id,
                "formula": system.metadata.get("formula", ""),
                "axis_points": len(system.axis),
                "n_points": len(system.points),
                "electron_count": system.electron_count,
                **metrics,
            }
        )
    return average_metrics(metric_rows), rows


def history_header() -> list[str]:
    return [
        "epoch",
        "train_objective",
        "train_gamma_loss",
        "train_rho_loss",
        "train_trace_loss",
        "train_kernel_loss",
        "train_kernel_highrho_loss",
        "train_kernel_highrho_frac",
        "learning_rate",
        "val_pair_loss",
        "val_pair_mae",
        "val_rho_loss",
        "val_rho_mae",
        "val_trace_rel_error",
        "val_kernel_loss",
        "val_kernel_mae",
        "val_kernel_highrho_loss",
        "val_kernel_highrho_mae",
        "val_kernel_highrho_frac",
    ]


def write_csv(path: Path, rows: list[dict[str, object]], header: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if header is None:
        header = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def optimizer_learning_rate(optimizer: tf.keras.optimizers.Optimizer) -> float:
    lr = optimizer.learning_rate
    if callable(lr):
        lr = lr(optimizer.iterations)
    return float(tf.keras.backend.get_value(lr))


def set_optimizer_learning_rate(optimizer: tf.keras.optimizers.Optimizer, value: float) -> None:
    try:
        tf.keras.backend.set_value(optimizer.learning_rate, float(value))
    except (AttributeError, TypeError):
        optimizer.learning_rate = float(value)


def train_v2(config: V2Config, split: DatasetSplit, point_dim: int, pair_dim: int, global_dim: int) -> dict[str, object]:
    validate_pair_rho_config(config)
    out_dir = Path(config.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.experiment == "baseline":
        alpha = fit_baseline_alpha(split, config)
        models = V2Models()
        val_avg, val_rows = evaluate_split(split.val_systems, "val", models, config, alpha, seed_offset=701)
        test_avg, test_rows = evaluate_split(split.test_systems, "test", models, config, alpha, seed_offset=801)
        train_avg, train_rows = evaluate_split(split.train_systems, "train", models, config, alpha, seed_offset=601)
        summary = {
            "config": asdict(config),
            "baseline_alpha": alpha,
            "train": train_avg,
            "val": val_avg,
            "test": test_avg,
        }
        write_csv(out_dir / f"{config.run_name}_per_system_metrics.csv", train_rows + val_rows + test_rows)
        save_json(out_dir / f"{config.run_name}_summary.json", summary)
        return summary

    models = build_v2_models(point_dim, pair_dim, global_dim, config)

    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    rng = np.random.default_rng(config.seed + 501)

    history: list[dict[str, object]] = []
    best_val = math.inf
    best_summary: dict[str, object] = {}
    best_weights: list[list[np.ndarray] | None] | None = None
    epochs_since_best = 0
    lr_plateau_epochs = 0
    stopped_epoch: int | None = None
    pair_rho_names = active_pair_rho_feature_names(config)
    rho_source = pair_rho_source(config)

    print_block(
        "V2 training",
        [
            ("experiment", config.experiment),
            ("epochs", config.epochs),
            ("steps/epoch", config.steps_per_epoch),
            ("steps/system", max(int(config.steps_per_system), 1)),
            ("batch", config.batch_size),
            ("cache eval batches", config.cache_eval_batches),
            ("pair feature placement", "tensorflow" if config.pair_features_on_device else "numpy"),
            ("learning rate", config.learning_rate),
            (
                "lr decay",
                (
                    f"factor={config.lr_decay_factor:g}, patience={config.lr_decay_patience}, "
                    f"min={config.lr_decay_min:g}, min_delta={config.lr_decay_min_delta:g}"
                )
                if config.lr_decay_factor < 1.0 and config.lr_decay_patience > 0
                else "off",
            ),
            (
                "early stopping",
                (
                    f"patience={config.early_stopping_patience}, "
                    f"min_delta={config.early_stopping_min_delta:g}, "
                    f"restore_best={config.restore_best_weights}"
                )
                if config.early_stopping_patience > 0
                else "off",
            ),
            ("k-only objective", "true-rho gamma loss" if config.experiment == "k-only" else "n/a"),
            ("kernel form", "exp(-alpha d^2) + sep * deltaK_pair"),
            ("kernel base alpha", config.kernel_base_alpha),
            ("k high-rho lambda", config.lambda_k_highrho if config.experiment == "k-only" else "n/a"),
            ("k high-rho cut", config.k_highrho_cut if config.experiment == "k-only" else "n/a"),
            ("k high-rho eps", config.k_highrho_eps if config.experiment == "k-only" else "n/a"),
            ("pair rho features", ", ".join(pair_rho_names) or "off"),
            ("pair rho source", rho_source if pair_rho_names else "off"),
            ("pair rho stop-gradient", config.pair_rho_stop_gradient if rho_source == "pred" else "n/a"),
            (
                "pair rho normalization",
                (
                    f"eps={config.pair_rho_eps:g}, log/scale={config.pair_rho_log_scale:g}, "
                    f"log clip={config.pair_rho_log_clip:g}, product={config.pair_rho_product_transform}"
                )
                if pair_rho_names
                else "off",
            ),
            ("pair sampling probs", config.pair_sampling_probs or "curriculum"),
            ("pair weights", config.pair_category_weights),
        ],
    )

    for epoch in range(config.epochs):
        accum = {
            "objective": [],
            "gamma_loss": [],
            "rho_loss": [],
            "trace_loss": [],
            "kernel_loss": [],
            "kernel_highrho_loss": [],
            "kernel_highrho_frac": [],
        }
        steps_per_system = max(int(config.steps_per_system), 1)
        current_system: SystemRecord | None = None
        for step_idx in range(config.steps_per_epoch):
            if current_system is None or step_idx % steps_per_system == 0:
                current_system = choose_system(split.train_systems, rng)
            system = current_system
            batch = None
            if config.experiment != "rho-only":
                batch = prepare_batch_tensors(make_batch(system, config.batch_size, epoch, config.epochs, rng, config))
            with tf.GradientTape() as tape:
                total, losses = compute_step_loss(system, batch, models, config)
            variables = models.trainable_variables()
            gradients = tape.gradient(total, variables)
            optimizer.apply_gradients((grad, var) for grad, var in zip(gradients, variables) if grad is not None)

            accum["objective"].append(float(total.numpy()))
            for name in (
                "gamma_loss",
                "rho_loss",
                "trace_loss",
                "kernel_loss",
                "kernel_highrho_loss",
                "kernel_highrho_frac",
            ):
                accum[name].append(float(losses[name].numpy()))

        row: dict[str, object] = {
            "epoch": epoch,
            "train_objective": float(np.mean(accum["objective"])),
            "train_gamma_loss": float(np.mean(accum["gamma_loss"])),
            "train_rho_loss": float(np.mean(accum["rho_loss"])),
            "train_trace_loss": float(np.mean(accum["trace_loss"])),
            "train_kernel_loss": float(np.mean(accum["kernel_loss"])),
            "train_kernel_highrho_loss": float(np.mean(accum["kernel_highrho_loss"])),
            "train_kernel_highrho_frac": float(np.mean(accum["kernel_highrho_frac"])),
            "learning_rate": optimizer_learning_rate(optimizer),
        }

        should_validate = epoch == 0 or (epoch + 1) % max(config.val_every, 1) == 0 or epoch == config.epochs - 1
        if should_validate:
            val_avg, _ = evaluate_split(split.val_systems, "val", models, config, None, seed_offset=900 + epoch)
            for key in (
                "pair_loss",
                "pair_mae",
                "rho_loss",
                "rho_mae",
                "trace_rel_error",
                "kernel_loss",
                "kernel_mae",
                "kernel_highrho_loss",
                "kernel_highrho_mae",
                "kernel_highrho_frac",
            ):
                row[f"val_{key}"] = val_avg.get(key, float("nan"))
            objective_key = "rho_loss" if config.experiment == "rho-only" else "pair_loss"
            val_objective = float(val_avg.get(objective_key, float("nan")))
            min_delta = max(config.early_stopping_min_delta, config.lr_decay_min_delta, 0.0)
            improved = np.isfinite(val_objective) and val_objective < best_val - min_delta
            if improved:
                best_val = val_objective
                best_summary = {"epoch": epoch, "val": val_avg}
                if config.restore_best_weights:
                    best_weights = models.get_weights()
                epochs_since_best = 0
                lr_plateau_epochs = 0
            elif np.isfinite(val_objective):
                epochs_since_best += max(config.val_every, 1)
                lr_plateau_epochs += max(config.val_every, 1)

                if config.lr_decay_factor < 1.0 and config.lr_decay_patience > 0:
                    if lr_plateau_epochs >= config.lr_decay_patience:
                        current_lr = optimizer_learning_rate(optimizer)
                        new_lr = max(config.lr_decay_min, current_lr * config.lr_decay_factor)
                        if new_lr < current_lr:
                            set_optimizer_learning_rate(optimizer, new_lr)
                            row["learning_rate"] = new_lr
                            print(
                                f"Epoch {epoch:4d} | reduce lr {current_lr:.3e} -> {new_lr:.3e}",
                                flush=True,
                            )
                        lr_plateau_epochs = 0

                if config.early_stopping_patience > 0 and epochs_since_best >= config.early_stopping_patience:
                    stopped_epoch = epoch
        history.append(row)

        if epoch == 0 or (epoch + 1) % max(config.log_every, 1) == 0 or epoch == config.epochs - 1:
            val_text = ""
            if should_validate:
                val_text = (
                    f" | val pair={row.get('val_pair_loss', float('nan')):.3e}"
                    f" rho={row.get('val_rho_loss', float('nan')):.3e}"
                    f" K={row.get('val_kernel_loss', float('nan')):.3e}"
                )
                if config.experiment == "k-only" and config.lambda_k_highrho > 0.0:
                    val_text += f" Khi={row.get('val_kernel_highrho_loss', float('nan')):.3e}"
            train_highrho_text = ""
            if config.experiment == "k-only" and config.lambda_k_highrho > 0.0:
                train_highrho_text = (
                    f" Khi={row['train_kernel_highrho_loss']:.3e}"
                    f" highrho={row['train_kernel_highrho_frac']:.2f}"
                )
            print(
                f"Epoch {epoch:4d} | train obj={row['train_objective']:.6e}"
                f" gamma={row['train_gamma_loss']:.3e}"
                f" rho={row['train_rho_loss']:.3e}"
                f" K={row['train_kernel_loss']:.3e}"
                f" lr={row['learning_rate']:.3e}"
                f"{train_highrho_text}"
                f"{val_text}",
                flush=True,
            )

        if stopped_epoch is not None:
            print(
                f"Early stopping at epoch {stopped_epoch}; best validation epoch={best_summary.get('epoch', 'n/a')}.",
                flush=True,
            )
            break

    if config.restore_best_weights and best_weights is not None:
        models.set_weights(best_weights)
        print(f"Restored best weights from epoch {best_summary.get('epoch', 'n/a')}.", flush=True)

    train_avg, train_rows = evaluate_split(split.train_systems, "train", models, config, None, seed_offset=1001)
    val_avg, val_rows = evaluate_split(split.val_systems, "val", models, config, None, seed_offset=1101)
    test_avg, test_rows = evaluate_split(split.test_systems, "test", models, config, None, seed_offset=1201)
    summary = {
        "config": asdict(config),
        "best": best_summary,
        "stopped_epoch": stopped_epoch,
        "train": train_avg,
        "val": val_avg,
        "test": test_avg,
    }

    write_csv(out_dir / f"{config.run_name}_history.csv", history, history_header())
    write_csv(out_dir / f"{config.run_name}_per_system_metrics.csv", train_rows + val_rows + test_rows)
    save_json(out_dir / f"{config.run_name}_summary.json", summary)

    if config.save_weights:
        if models.point is not None:
            models.point.save_weights(out_dir / f"{config.run_name}_point.weights.h5")
        if models.pair is not None:
            models.pair.save_weights(out_dir / f"{config.run_name}_pair.weights.h5")
        if models.context is not None:
            models.context.save_weights(out_dir / f"{config.run_name}_context.weights.h5")

    return summary
