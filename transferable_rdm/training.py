from __future__ import annotations

import gc
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from .config import ExperimentConfig
from .density_features import (
    DensityFeatureState,
    build_density_feature_state,
    build_true_density_feature_state,
    cache_frozen_density_state,
    cached_frozen_density_state,
    clear_frozen_density_state_cache,
    density_baseline_mode,
    density_source_mode,
    normalized_density_head,
    pair_density_features,
    pair_density_feature_mode,
    richardson_gradient_3d,
)
from .data import (
    DatasetSplit,
    PairBatch,
    build_pair_features,
    choose_system,
    full_pair_chunks,
    pair_weights_from_categories,
    sample_pair_batch,
    sample_pair_indices,
)
from .model import ModelBundle, predict_from_features
from .systems import SystemRecord
from .utils import print_block, topk_descending


LOSS_NAMES = (
    "gamma",
    "rho",
    "kernel",
    "deriv",
    "tau",
    "tau_mse",
    "stencil_gamma",
    "trace",
    "occ",
    "mode",
    "kinetic",
    "kinetic_mse",
)
TRAIN_HISTORY_KEYS = (
    "pair_loss",
    "rho_loss",
    "kernel_loss",
    "deriv_loss",
    "tau_loss",
    "tau_mse_loss",
    "stencil_gamma_loss",
    "trace_loss",
    "occ_penalty",
    "mode_reg",
    "kinetic_loss",
    "kinetic_mse_loss",
)
VAL_HISTORY_KEYS = (
    "pair_loss",
    "pair_mae",
    "pair_rmse",
    "diag_pair_mae",
    "diag_pair_rmse",
    "near_diag_mae",
    "near_diag_rmse",
    "mid_pair_mae",
    "mid_pair_rmse",
    "far_offdiag_mae",
    "far_offdiag_rmse",
    "rho_loss",
    "density_mae",
    "kernel_loss",
    "kernel_diag_error",
    "deriv_loss",
    "deriv_raw_mse",
    "deriv_mae",
    "deriv_pred_ao_mae",
    "deriv_fd_ao_mae",
    "deriv_fd_ao_rms_ratio",
    "deriv_pred_fd_mae",
    "tau_loss",
    "tau_raw_mse",
    "tau_rmse",
    "tau_rel_mse_loss",
    "tau_rel_rmse",
    "tau_mae",
    "tau_pred_ao_mae",
    "tau_fd_ao_mae",
    "tau_fd_ao_rms_ratio",
    "tau_pred_fd_mae",
    "stencil_gamma_huber",
    "stencil_gamma_mae",
    "stencil_gamma_rmse",
    "stencil_gamma_rel_mae",
    "stencil_gamma_rel_rmse",
    "kinetic_loss",
    "kinetic_mse_loss",
    "kinetic_abs_error",
    "kinetic_abs_error_p90",
    "kinetic_sq_error",
    "kinetic_rmse",
    "kinetic_rel_abs_error",
    "kinetic_rel_abs_error_p90",
    "kinetic_rel_sq_error",
    "kinetic_rel_rmse",
    "energy_total_abs_error",
    "energy_total_rmse",
    "energy_grid_total_abs_error",
    "energy_grid_total_rmse",
    "trace_loss",
    "trace_abs_rel_error",
    "occ_penalty",
    "symmetry_mae",
)
TRAIN_OBJECTIVE_TERMS = (
    ("gamma", "pair_loss"),
    ("rho", "rho_loss"),
    ("kernel", "kernel_loss"),
    ("deriv", "deriv_loss"),
    ("tau", "tau_loss"),
    ("tau_mse", "tau_mse_loss"),
    ("stencil_gamma", "stencil_gamma_loss"),
    ("trace", "trace_loss"),
    ("occ", "occ_penalty"),
    ("mode", "mode_reg"),
    ("kinetic", "kinetic_loss"),
    ("kinetic_mse", "kinetic_mse_loss"),
)
EVAL_OBJECTIVE_TERMS = (
    ("gamma", "pair_loss"),
    ("rho", "rho_loss"),
    ("kernel", "kernel_loss"),
    ("deriv", "deriv_loss"),
    ("tau", "tau_loss"),
    ("tau_mse", "tau_mse_loss"),
    ("stencil_gamma", "stencil_gamma_huber"),
    ("trace", "trace_loss"),
    ("occ", "occ_penalty"),
    ("kinetic", "kinetic_loss"),
    ("kinetic_mse", "kinetic_mse_loss"),
)

PHYSICS_TARGET_MODES = ("orbital", "fd")
_TRUE_GAMMA_STENCIL_TARGET_CACHE: dict[tuple[int, int, float], tuple[np.ndarray, np.ndarray]] = {}
_PHYSICS_TAU_INTEGRAL_CACHE: dict[tuple[int, str, int, float], float] = {}
_STENCIL_PAIR_FEATURE_CACHE: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
_SYSTEM_TENSOR_CACHE: OrderedDict[int, "SystemTensorState"] = OrderedDict()
_COULOMB_KERNEL_FFT_CACHE: OrderedDict[tuple[int, float], np.ndarray] = OrderedDict()
_ELEMENT_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}


@dataclass
class TrainingHistory:
    train_objective: list[float] = field(default_factory=list)
    val_objective: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    kinetic_weight: list[float] = field(default_factory=list)
    validation_ran: list[int] = field(default_factory=list)
    loss_weights: dict[str, list[float]] = field(default_factory=lambda: {key: [] for key in LOSS_NAMES})
    train_components: dict[str, list[float]] = field(
        default_factory=lambda: {key: [] for key in TRAIN_HISTORY_KEYS}
    )
    val_components: dict[str, list[float]] = field(
        default_factory=lambda: {key: [] for key in VAL_HISTORY_KEYS}
    )


@dataclass
class PointPretrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)


@dataclass
class SystemTensorState:
    local_features: tf.Tensor
    global_context: tf.Tensor
    rho_diag: tf.Tensor


def active_system_tensors(system: SystemRecord, config: ExperimentConfig) -> SystemTensorState:
    """Small active-system tensor cache; never scales with full dataset size."""
    max_entries = int(config.active_system_tensor_cache_size)
    if max_entries <= 0:
        return SystemTensorState(
            to_tensor(system.local_features),
            to_tensor(system.global_context),
            to_tensor(system.rho_diag),
        )
    key = id(system)
    cached = _SYSTEM_TENSOR_CACHE.get(key)
    if cached is not None:
        _SYSTEM_TENSOR_CACHE.move_to_end(key)
        return cached
    state = SystemTensorState(
        to_tensor(system.local_features),
        to_tensor(system.global_context),
        to_tensor(system.rho_diag),
    )
    _SYSTEM_TENSOR_CACHE[key] = state
    while len(_SYSTEM_TENSOR_CACHE) > max_entries:
        _SYSTEM_TENSOR_CACHE.popitem(last=False)
    return state


def weighted_mse(y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(weights * tf.square(y_true - y_pred)) / tf.reduce_sum(weights)


def rms_normalized_huber(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    scale_floor: float,
    delta: float,
) -> tf.Tensor:
    """Huber loss after scaling errors by the target RMS."""
    if scale_floor <= 0.0:
        raise ValueError("RMS-normalized Huber scale_floor must be positive.")
    if delta <= 0.0:
        raise ValueError("RMS-normalized Huber delta must be positive.")
    scale = tf.maximum(tf.sqrt(tf.reduce_mean(tf.square(y_true))), float(scale_floor))
    scaled_abs_error = tf.abs((y_pred - y_true) / scale)
    quadratic = tf.minimum(scaled_abs_error, float(delta))
    linear = scaled_abs_error - quadratic
    return tf.reduce_mean(0.5 * tf.square(quadratic) + float(delta) * linear)


def density_normalized_huber(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    rho_left: tf.Tensor,
    rho_right: tf.Tensor,
    *,
    scale_floor: float,
    delta: float,
) -> tf.Tensor:
    """Huber loss after normalizing gamma errors by sqrt(rho_l rho_r)."""
    if scale_floor <= 0.0:
        raise ValueError("Density-normalized Huber scale_floor must be positive.")
    if delta <= 0.0:
        raise ValueError("Density-normalized Huber delta must be positive.")
    scale = tf.sqrt(tf.maximum(rho_left * rho_right, 0.0)) + float(scale_floor)
    scaled_abs_error = tf.abs((y_pred - y_true) / scale)
    quadratic = tf.minimum(scaled_abs_error, float(delta))
    linear = scaled_abs_error - quadratic
    return tf.reduce_mean(0.5 * tf.square(quadratic) + float(delta) * linear)


def rms_normalized_mse(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    *,
    scale_floor: float,
) -> tf.Tensor:
    """MSE after scaling errors by the target RMS."""
    if scale_floor <= 0.0:
        raise ValueError("RMS-normalized MSE scale_floor must be positive.")
    scale_sq = tf.maximum(tf.reduce_mean(tf.square(y_true)), float(scale_floor) ** 2)
    return tf.reduce_mean(tf.square(y_pred - y_true)) / scale_sq


def to_tensor(array: np.ndarray) -> tf.Tensor:
    return tf.convert_to_tensor(array, dtype=tf.float32)


def point_model_outputs(system: SystemRecord, models: ModelBundle) -> tf.Tensor:
    point_features = to_tensor(system.local_features)
    global_context_t = tf.reshape(to_tensor(system.global_context), (1, -1))
    tiled_global = tf.repeat(global_context_t, repeats=tf.shape(point_features)[0], axis=0)
    point_input = tf.concat([point_features, tiled_global], axis=1)
    return models.point_model(point_input)


def point_output_and_state(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
) -> tuple[tf.Tensor, DensityFeatureState]:
    if density_source_mode(config) == "true":
        cached = cached_frozen_density_state(system)
        if cached is None:
            cached = build_true_density_feature_state(system, config)
            cache_frozen_density_state(system, cached)
        return tf.zeros((0, 0), dtype=tf.float32), cached
    if config.freeze_point_after_pretrain:
        cached = cached_frozen_density_state(system)
        if cached is not None:
            return tf.zeros((0, 0), dtype=tf.float32), cached
    point_out = point_model_outputs(system, models)
    state = build_density_feature_state(system, point_out, config)
    if config.freeze_point_after_pretrain:
        cache_frozen_density_state(system, state)
    return point_out, state
def point_density_predictions(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
) -> dict[str, tf.Tensor]:
    if density_source_mode(config) == "true":
        state = build_true_density_feature_state(system, config)
        predictions = {
            "delta_raw": tf.zeros(
                (len(system.points), 3 if pair_density_feature_mode(config) == "fukui" else 1),
                dtype=tf.float32,
            ),
            "rho_neutral": state.rho_neutral,
        }
        if pair_density_feature_mode(config) == "fukui":
            predictions.update(
                {
                    "rho_cation": state.rho_cation,
                    "rho_anion": state.rho_anion,
                    "fukui_plus": state.rho_anion - state.rho_neutral,
                    "fukui_minus": state.rho_neutral - state.rho_cation,
                }
            )
        return predictions
    point_out = point_model_outputs(system, models)
    predictions = {
        "delta_raw": point_out,
        "rho_neutral": normalized_density_head(
            system, point_out[:, 0:1], system.electron_count, config=config, normalize=config.normalize_rho
        ),
    }
    if pair_density_feature_mode(config) == "fukui":
        if system.rho_cation is None or system.rho_anion is None:
            raise ValueError(
                f"System {system.system_id} has no charged density oracle channels. "
                "Rebuild NPZ files with --include-charged-density-oracles."
            )
        predictions["rho_cation"] = normalized_density_head(
            system, point_out[:, 1:2], max(system.electron_count - 1.0, 1e-6),
            config=config, normalize=config.normalize_rho
        )
        predictions["rho_anion"] = normalized_density_head(
            system, point_out[:, 2:3], system.electron_count + 1.0,
            config=config, normalize=config.normalize_rho
        )
        predictions["fukui_plus"] = predictions["rho_anion"] - predictions["rho_neutral"]
        predictions["fukui_minus"] = predictions["rho_neutral"] - predictions["rho_cation"]
    return predictions


def empty_delta_diagnostics(n_heads: int) -> dict[str, np.ndarray | int]:
    return {
        "min": np.full(n_heads, np.inf, dtype=np.float64),
        "max": np.full(n_heads, -np.inf, dtype=np.float64),
        "clip_low_count": np.zeros(n_heads, dtype=np.int64),
        "clip_high_count": np.zeros(n_heads, dtype=np.int64),
        "count": 0,
    }


def update_delta_diagnostics(
    diagnostics: dict[str, np.ndarray | int],
    raw_heads: tf.Tensor,
    clip: float,
) -> None:
    clip = max(float(clip), 0.0)
    diagnostics["min"] = np.minimum(diagnostics["min"], tf.reduce_min(raw_heads, axis=0).numpy())
    diagnostics["max"] = np.maximum(diagnostics["max"], tf.reduce_max(raw_heads, axis=0).numpy())
    diagnostics["clip_low_count"] += tf.reduce_sum(tf.cast(raw_heads <= -clip, tf.int64), axis=0).numpy()
    diagnostics["clip_high_count"] += tf.reduce_sum(tf.cast(raw_heads >= clip, tf.int64), axis=0).numpy()
    diagnostics["count"] += int(tf.shape(raw_heads)[0].numpy())


def delta_diagnostic_scalars(diagnostics: dict[str, np.ndarray | int]) -> dict[str, float]:
    labels = ("N", "N_minus_1", "N_plus_1")
    count = max(int(diagnostics["count"]), 1)
    scalars = {}
    for index in range(len(diagnostics["min"])):
        prefix = f"delta_{labels[index]}"
        scalars[f"{prefix}_min"] = float(diagnostics["min"][index])
        scalars[f"{prefix}_max"] = float(diagnostics["max"][index])
        scalars[f"{prefix}_clip_low_fraction"] = float(diagnostics["clip_low_count"][index] / count)
        scalars[f"{prefix}_clip_high_fraction"] = float(diagnostics["clip_high_count"][index] / count)
    return scalars


def format_delta_diagnostics(diagnostics: dict[str, np.ndarray | int]) -> str:
    labels = ("N", "N-1", "N+1")
    scalars = delta_diagnostic_scalars(diagnostics)
    chunks = []
    for index, label in enumerate(labels[: len(diagnostics["min"])]):
        key = ("N", "N_minus_1", "N_plus_1")[index]
        chunks.append(
            f"{label}=[{scalars[f'delta_{key}_min']:.2f},{scalars[f'delta_{key}_max']:.2f}] "
            f"clip={100.0 * scalars[f'delta_{key}_clip_low_fraction']:.2f}/"
            f"{100.0 * scalars[f'delta_{key}_clip_high_fraction']:.2f}%"
        )
    return "; ".join(chunks)


def format_delta_scalar_summary(summary: dict[str, object]) -> str:
    labels = (("N", "N"), ("N-1", "N_minus_1"), ("N+1", "N_plus_1"))
    chunks = []
    for display_label, key in labels:
        if f"delta_{key}_min" not in summary:
            continue
        chunks.append(
            f"{display_label}=[{float(summary[f'delta_{key}_min']):.2f},"
            f"{float(summary[f'delta_{key}_max']):.2f}] "
            f"clip={100.0 * float(summary[f'delta_{key}_clip_low_fraction']):.2f}/"
            f"{100.0 * float(summary[f'delta_{key}_clip_high_fraction']):.2f}%"
        )
    return "; ".join(chunks)


def scaled_field_mse(y_true: tf.Tensor, y_pred: tf.Tensor, scale_floor: float) -> tf.Tensor:
    scale = tf.maximum(tf.sqrt(tf.reduce_mean(tf.square(y_true))), scale_floor)
    return tf.reduce_mean(tf.square((y_pred - y_true) / scale))


def relative_field_l1(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.abs(y_pred - y_true)) / tf.maximum(tf.reduce_sum(tf.abs(y_true)), 1e-12)


def log_density_mse(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float) -> tf.Tensor:
    floor = max(float(eps), 1e-30)
    log_true = tf.math.log(tf.maximum(y_true, floor))
    log_pred = tf.math.log(tf.maximum(y_pred, floor))
    return tf.reduce_mean(tf.square(log_pred - log_true))


def density_field_loss_components(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    config: ExperimentConfig,
    *,
    include_log: bool = True,
) -> dict[str, tf.Tensor]:
    scaled_mse = scaled_field_mse(y_true, y_pred, config.point_density_scale_floor)
    relative_l1 = relative_field_l1(y_true, y_pred)
    log_mse = (
        log_density_mse(y_true, y_pred, config.point_density_log_eps)
        if include_log
        else tf.constant(0.0, dtype=tf.float32)
    )
    total = (
        config.point_density_mse_weight
        * scaled_mse
        + config.point_density_rel_l1_weight * relative_l1
        + config.point_density_log_weight * log_mse
    )
    return {
        "total": total,
        "scaled_mse": scaled_mse,
        "relative_l1": relative_l1,
        "log_mse": log_mse,
    }


def density_field_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    config: ExperimentConfig,
    *,
    include_log: bool = True,
) -> tf.Tensor:
    return density_field_loss_components(y_true, y_pred, config, include_log=include_log)["total"]


def normalized_sad_density(system: SystemRecord, config: ExperimentConfig) -> tf.Tensor:
    if system.rho_sad is None:
        raise ValueError(f"System {system.system_id} has no SAD density baseline.")
    sad = tf.maximum(to_tensor(system.rho_sad), max(float(config.sad_density_floor), 1e-30))
    normalizer = tf.reduce_sum(sad) * system.cell_volume
    return sad * (float(system.electron_count) / tf.maximum(normalizer, 1e-12))


def sad_density_relative_l1(system: SystemRecord, config: ExperimentConfig) -> float:
    sad = normalized_sad_density(system, config).numpy()
    return float(
        np.sum(np.abs(sad - system.rho_diag))
        * system.cell_volume
        / max(system.electron_count, 1e-12)
    )


def sad_delta_residual_loss(
    system: SystemRecord,
    raw_delta: tf.Tensor,
    config: ExperimentConfig,
) -> tf.Tensor:
    """Centered log-residual target for sad-multiplicative density heads."""
    if density_baseline_mode(config) != "sad-multiplicative" or config.point_delta_weight <= 0.0:
        return tf.constant(0.0, dtype=tf.float32)
    if system.rho_sad is None:
        raise ValueError(f"System {system.system_id} has no SAD density baseline.")

    rho_true = tf.maximum(to_tensor(system.rho_diag), max(float(config.point_delta_eps), 1e-30))
    rho_sad = tf.maximum(to_tensor(system.rho_sad), max(float(config.point_delta_eps), 1e-30))
    delta_target = tf.math.log(rho_true / rho_sad)
    weights = rho_true / tf.maximum(tf.reduce_sum(rho_true), 1e-12)

    target_center = tf.reduce_sum(weights * delta_target)
    pred_center = tf.reduce_sum(weights * raw_delta[:, 0:1])
    residual = (raw_delta[:, 0:1] - pred_center) - (delta_target - target_center)
    abs_residual = tf.abs(residual)
    delta = max(float(config.point_delta_huber), 1e-12)
    quadratic = tf.minimum(abs_residual, delta)
    linear = abs_residual - quadratic
    return tf.reduce_mean(0.5 * tf.square(quadratic) + delta * linear)


def point_fukui_weight_at_epoch(config: ExperimentConfig, epoch: int) -> float:
    if pair_density_feature_mode(config) != "fukui" or epoch < config.point_fukui_start_epoch:
        return 0.0
    ramp_epochs = max(int(config.point_fukui_ramp_epochs), 0)
    if ramp_epochs == 0:
        return float(config.point_fukui_weight)
    progress = min((epoch - config.point_fukui_start_epoch + 1) / ramp_epochs, 1.0)
    return float(config.point_fukui_weight) * progress


def point_fukui_ramp_active(config: ExperimentConfig, epoch: int) -> bool:
    ramp_epochs = max(int(config.point_fukui_ramp_epochs), 0)
    return (
        pair_density_feature_mode(config) == "fukui"
        and ramp_epochs > 0
        and epoch >= config.point_fukui_start_epoch
        and epoch < config.point_fukui_start_epoch + ramp_epochs - 1
    )


def point_pretrain_losses(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
    epoch: int,
) -> dict[str, tf.Tensor]:
    pred = point_density_predictions(system, models, config)
    zero = tf.constant(0.0, dtype=tf.float32)
    neutral_components = density_field_loss_components(to_tensor(system.rho_diag), pred["rho_neutral"], config)
    neutral_loss = neutral_components["total"]
    delta_loss = sad_delta_residual_loss(system, pred["delta_raw"], config)
    charged_loss = zero
    fukui_loss = zero
    if pair_density_feature_mode(config) == "fukui":
        true_cation = to_tensor(system.rho_cation)
        true_anion = to_tensor(system.rho_anion)
        if epoch >= config.point_charged_start_epoch:
            charged_loss = 0.5 * (
                density_field_loss(true_cation, pred["rho_cation"], config)
                + density_field_loss(true_anion, pred["rho_anion"], config)
            )
        if epoch >= config.point_fukui_start_epoch:
            rho_neutral = to_tensor(system.rho_diag)
            fukui_loss = 0.5 * (
                density_field_loss(true_anion - rho_neutral, pred["fukui_plus"], config, include_log=False)
                + density_field_loss(rho_neutral - true_cation, pred["fukui_minus"], config, include_log=False)
            )
    fukui_weight = point_fukui_weight_at_epoch(config, epoch)
    total = (
        neutral_loss
        + config.point_delta_weight * delta_loss
        + config.point_charged_weight * charged_loss
        + fukui_weight * fukui_loss
    )
    return {
        "total": total,
        "neutral": neutral_loss,
        "delta": delta_loss,
        "charged": charged_loss,
        "fukui": fukui_loss,
        "neutral_scaled_mse": neutral_components["scaled_mse"],
        "neutral_relative_l1": neutral_components["relative_l1"],
        "neutral_log_mse": neutral_components["log_mse"],
        "fukui_weight": tf.constant(fukui_weight, dtype=tf.float32),
        "delta_raw": pred["delta_raw"],
    }


def evaluate_point_model(
    systems: list[SystemRecord],
    models: ModelBundle,
    config: ExperimentConfig,
    *,
    keep_arrays: bool = False,
) -> dict[str, object]:
    per_system = []
    delta_diagnostics = None
    for idx, system in enumerate(systems):
        pred = point_density_predictions(system, models, config)
        if density_baseline_mode(config) == "sad-multiplicative" and system.rho_sad is not None:
            if delta_diagnostics is None:
                delta_diagnostics = empty_delta_diagnostics(int(pred["delta_raw"].shape[1]))
            update_delta_diagnostics(delta_diagnostics, pred["delta_raw"], config.sad_residual_clip)
            sad_rel_l1 = sad_density_relative_l1(system, config)
        else:
            sad_rel_l1 = float("nan")
        entry: dict[str, object] = {
            "system_id": system.system_id,
            "rho_neutral_mae": float(np.mean(np.abs(pred["rho_neutral"].numpy() - system.rho_diag))),
            "rho_neutral_rel_l1": float(
                np.sum(np.abs(pred["rho_neutral"].numpy() - system.rho_diag))
                * system.cell_volume
                / max(system.electron_count, 1e-12)
            ),
            "rho_sad_rel_l1": sad_rel_l1,
        }
        if pair_density_feature_mode(config) == "fukui":
            fukui_plus_true = system.rho_anion - system.rho_diag
            fukui_minus_true = system.rho_diag - system.rho_cation
            entry.update(
                {
                    "rho_cation_mae": float(np.mean(np.abs(pred["rho_cation"].numpy() - system.rho_cation))),
                    "rho_anion_mae": float(np.mean(np.abs(pred["rho_anion"].numpy() - system.rho_anion))),
                    "fukui_plus_mae": float(np.mean(np.abs(pred["fukui_plus"].numpy() - fukui_plus_true))),
                    "fukui_minus_mae": float(np.mean(np.abs(pred["fukui_minus"].numpy() - fukui_minus_true))),
                }
            )
        if keep_arrays and idx == 0:
            entry["rho_neutral_true"] = system.rho_diag
            entry["rho_neutral_pred"] = pred["rho_neutral"].numpy()
            if pair_density_feature_mode(config) == "fukui":
                entry["fukui_plus_true"] = system.rho_anion - system.rho_diag
                entry["fukui_plus_pred"] = pred["fukui_plus"].numpy()
                entry["fukui_minus_true"] = system.rho_diag - system.rho_cation
                entry["fukui_minus_pred"] = pred["fukui_minus"].numpy()
        per_system.append(entry)
    scalar_keys = [
        key for key in per_system[0]
        if key.endswith("_mae") or key.endswith("_rel_l1")
    ]
    summary = {key: float(np.mean([entry[key] for entry in per_system])) for key in scalar_keys}
    if delta_diagnostics is not None:
        summary.update(delta_diagnostic_scalars(delta_diagnostics))
    summary["per_system"] = per_system
    return summary


def pretrain_point_model(
    config: ExperimentConfig,
    split: DatasetSplit,
    models: ModelBundle,
) -> tuple[PointPretrainHistory, dict[str, object]]:
    """Fit density heads first, restore the best validation weights, then freeze them."""
    if density_source_mode(config) == "true":
        history = PointPretrainHistory()
        models.point_model.trainable = False
        clear_frozen_density_state_cache()
        summary = {
            "train": evaluate_point_model(split.train_systems, models, config),
            "val": evaluate_point_model(split.val_systems, models, config, keep_arrays=True),
        }
        if split.test_systems:
            summary["test"] = evaluate_point_model(split.test_systems, models, config)
        print_block(
            "Point density pretrain",
            [
                ("density source", "true (oracle)"),
                ("pretraining", "skipped"),
                ("val rho_N MAE", f"{summary['val']['rho_neutral_mae']:.6e}"),
                ("val rho_N relative L1", f"{summary['val']['rho_neutral_rel_l1']:.6e}"),
                ("point trainable in pair stage", models.point_model.trainable),
            ],
        )
        return history, summary
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.point_pretrain_lr)
    history = PointPretrainHistory()
    rng = np.random.default_rng(config.seed + 71)
    best_val = np.inf
    best_val_for_lr = np.inf
    best_weights = None
    stale_epochs = 0
    stale_lr_epochs = 0
    previous_stage = None
    previous_fukui_weight = 0.0
    print_block(
        "Point density pretrain",
        [
            ("pair density feature mode", pair_density_feature_mode(config)),
            ("density baseline mode", config.density_baseline_mode),
            ("epochs", config.point_pretrain_epochs),
            ("steps/epoch", config.point_pretrain_steps_per_epoch),
            ("charged weight/start", f"{config.point_charged_weight:g} / {config.point_charged_start_epoch}"),
            (
                "Fukui weight/start/ramp",
                f"{config.point_fukui_weight:g} / {config.point_fukui_start_epoch} / {config.point_fukui_ramp_epochs}",
            ),
            ("density scale floor", f"{config.point_density_scale_floor:g}"),
            (
                "density loss weights",
                f"scaled-MSE={config.point_density_mse_weight:g}, "
                f"relative-L1={config.point_density_rel_l1_weight:g}, "
                f"log-rho-MSE={config.point_density_log_weight:g}",
            ),
            (
                "SAD delta residual loss",
                f"weight={config.point_delta_weight:g}, Huber={config.point_delta_huber:g}, eps={config.point_delta_eps:g}",
            ),
            ("density log epsilon", f"{config.point_density_log_eps:g}"),
            (
                "lr decay",
                f"factor={config.point_pretrain_lr_decay:g}, "
                f"patience={config.point_pretrain_lr_patience}, min={config.point_pretrain_min_lr:g}",
            ),
            ("freeze after pretrain", config.freeze_point_after_pretrain),
        ],
    )
    for epoch in range(config.point_pretrain_epochs):
        uses_fukui = pair_density_feature_mode(config) == "fukui"
        stage = (
            uses_fukui and epoch >= config.point_charged_start_epoch,
            uses_fukui and epoch >= config.point_fukui_start_epoch,
        )
        if previous_stage is not None and stage != previous_stage:
            best_val = np.inf
            best_val_for_lr = np.inf
            best_weights = None
            stale_epochs = 0
            stale_lr_epochs = 0
            print(f"Point pretrain stage transition at epoch {epoch}: charged={stage[0]} fukui={stage[1]}")
        previous_stage = stage
        current_fukui_weight = point_fukui_weight_at_epoch(config, epoch)
        if current_fukui_weight >= config.point_fukui_weight and previous_fukui_weight < config.point_fukui_weight:
            best_val = np.inf
            best_val_for_lr = np.inf
            best_weights = None
            stale_epochs = 0
            stale_lr_epochs = 0
            print(f"Point pretrain Fukui ramp completed at epoch {epoch}.")
        previous_fukui_weight = current_fukui_weight
        running = 0.0
        running_components = {
            "neutral": 0.0,
            "delta": 0.0,
            "charged": 0.0,
            "fukui": 0.0,
            "neutral_scaled_mse": 0.0,
            "neutral_relative_l1": 0.0,
            "neutral_log_mse": 0.0,
        }
        last_delta_raw = None
        for _ in range(config.point_pretrain_steps_per_epoch):
            system = choose_system(split.train_systems, rng)
            with tf.GradientTape() as tape:
                losses = point_pretrain_losses(system, models, config, epoch)
                loss = losses["total"]
            grads = tape.gradient(loss, models.point_model.trainable_variables)
            grads_and_vars = [(grad, var) for grad, var in zip(grads, models.point_model.trainable_variables) if grad is not None]
            optimizer.apply_gradients(grads_and_vars)
            running += float(loss.numpy())
            for key in running_components:
                running_components[key] += float(losses[key].numpy())
            if density_baseline_mode(config) == "sad-multiplicative":
                last_delta_raw = losses["delta_raw"]
        train_loss = running / max(config.point_pretrain_steps_per_epoch, 1)
        history.train_loss.append(train_loss)
        history.learning_rate.append(float(optimizer.learning_rate.numpy()))
        validation_ran = epoch % max(config.point_pretrain_val_every, 1) == 0 or epoch == config.point_pretrain_epochs - 1
        if validation_ran:
            val_losses = [
                float(point_pretrain_losses(system, models, config, epoch)["total"].numpy())
                for system in split.val_systems
            ]
            val_loss = float(np.mean(val_losses))
            ramp_active = point_fukui_ramp_active(config, epoch)
            if ramp_active:
                best_val = val_loss
                best_val_for_lr = val_loss
                best_weights = models.point_model.get_weights()
                stale_epochs = 0
                stale_lr_epochs = 0
            elif val_loss < best_val - 1e-9:
                best_val = val_loss
                best_weights = models.point_model.get_weights()
                stale_epochs = 0
            else:
                stale_epochs += max(config.point_pretrain_val_every, 1)
            if not ramp_active:
                if val_loss < best_val_for_lr - 1e-9:
                    best_val_for_lr = val_loss
                    stale_lr_epochs = 0
                else:
                    stale_lr_epochs += max(config.point_pretrain_val_every, 1)
                if stale_lr_epochs >= config.point_pretrain_lr_patience:
                    old_lr = float(optimizer.learning_rate.numpy())
                    new_lr = max(old_lr * config.point_pretrain_lr_decay, config.point_pretrain_min_lr)
                    if new_lr < old_lr:
                        optimizer.learning_rate.assign(new_lr)
                        print(f"Point pretrain reduce lr {old_lr:.3e} -> {new_lr:.3e}")
                    stale_lr_epochs = 0
        else:
            val_loss = history.val_loss[-1] if history.val_loss else float("nan")
        history.val_loss.append(val_loss)
        if epoch % max(config.log_every, 1) == 0 or epoch == config.point_pretrain_epochs - 1:
            denom = max(config.point_pretrain_steps_per_epoch, 1)
            print(
                f"Point epoch {epoch:4d} | train={train_loss:.6e} val={val_loss:.6e} "
                f"rho_N={running_components['neutral'] / denom:.3e} "
                f"delta={running_components['delta'] / denom:.3e} "
                f"charged={running_components['charged'] / denom:.3e} "
                f"fukui={running_components['fukui'] / denom:.3e} "
                f"w(fukui)={point_fukui_weight_at_epoch(config, epoch):.3e} "
                f"lr={float(optimizer.learning_rate.numpy()):.3e}"
            )
            print(
                f"              rho_N terms scaled-MSE={running_components['neutral_scaled_mse'] / denom:.3e} "
                f"relative-L1={running_components['neutral_relative_l1'] / denom:.3e} "
                f"log-rho-MSE={running_components['neutral_log_mse'] / denom:.3e}"
            )
            if last_delta_raw is not None:
                delta_diagnostics = empty_delta_diagnostics(int(last_delta_raw.shape[1]))
                update_delta_diagnostics(delta_diagnostics, last_delta_raw, config.sad_residual_clip)
                print(f"              delta raw sample {format_delta_diagnostics(delta_diagnostics)}")
        if stale_epochs >= config.point_pretrain_patience:
            print(f"Point pretrain early stopping at epoch {epoch}.")
            break
    if best_weights is not None:
        models.point_model.set_weights(best_weights)
    models.point_model.trainable = not config.freeze_point_after_pretrain
    clear_frozen_density_state_cache()
    summary = {
        "train": evaluate_point_model(split.train_systems, models, config),
        "val": evaluate_point_model(split.val_systems, models, config, keep_arrays=True),
    }
    if split.test_systems:
        summary["test"] = evaluate_point_model(split.test_systems, models, config)
    print_block(
        "Point density pretrain summary",
        [
            ("val rho_N MAE", f"{summary['val']['rho_neutral_mae']:.6e}"),
            ("val rho_N relative L1", f"{summary['val']['rho_neutral_rel_l1']:.6e}"),
            ("val SAD relative L1", f"{summary['val'].get('rho_sad_rel_l1', float('nan')):.6e}"),
            (
                "val rho_N relL1 improvement",
                f"{summary['val'].get('rho_sad_rel_l1', float('nan')) - summary['val']['rho_neutral_rel_l1']:.6e}",
            ),
            ("val Fukui+ MAE", f"{summary['val'].get('fukui_plus_mae', float('nan')):.6e}"),
            ("val Fukui- MAE", f"{summary['val'].get('fukui_minus_mae', float('nan')):.6e}"),
            ("point trainable in pair stage", models.point_model.trainable),
        ],
    )
    if density_baseline_mode(config) == "sad-multiplicative":
        print(f"val delta raw               : {format_delta_scalar_summary(summary['val'])}")
    return history, summary


def kinetic_energy_reference(
    system: SystemRecord,
    config: ExperimentConfig | None = None,
) -> float:
    if config is not None:
        _, tau_target = physics_stencil_targets(system, config)
        return float(np.sum(tau_target, dtype=np.float64) * system.cell_volume)
    kinetic_ref = float(system.metadata.get("kinetic_energy_hartree", np.nan))
    if np.isfinite(kinetic_ref):
        return kinetic_ref
    return float(np.sum(system.tau_true) * system.cell_volume)


def kinetic_prefactor(system: SystemRecord) -> float:
    return float(system.metadata.get("kinetic_prefactor", 0.5))


def physics_tau_integral(system: SystemRecord, config: ExperimentConfig) -> float:
    key = (
        id(system),
        physics_target_mode(config),
        int(system.stencil_left.shape[2]),
        float(system.step),
    )
    cached = _PHYSICS_TAU_INTEGRAL_CACHE.get(key)
    if cached is None:
        _, tau_target = physics_stencil_targets(system, config)
        cached = float(np.sum(tau_target, dtype=np.float64) * system.cell_volume)
        _PHYSICS_TAU_INTEGRAL_CACHE[key] = cached
    return cached


def local_curvature_basis_scale(system: SystemRecord, config: ExperimentConfig) -> float:
    """Undo pair-feature length normalization for local near-diagonal curvature terms."""
    if config.local_curvature_basis_scale > 0.0:
        return float(config.local_curvature_basis_scale)
    domain_scale = max(float(np.max(np.abs(system.axis))), 1e-6)
    if config.local_curvature_form.strip().lower() == "quadratic":
        return domain_scale**2
    step = max(float(system.step), 1e-8)
    return (domain_scale / step) ** 2


def kinetic_energy_loss_from_tau(
    system: SystemRecord,
    tau_pred: tf.Tensor,
    *,
    config: ExperimentConfig | None = None,
    integration_multiplier: float = 1.0,
    tau_target: tf.Tensor | None = None,
    target_integral: float | None = None,
    control_variate: bool = True,
) -> tuple[tf.Tensor, tf.Tensor, float]:
    if control_variate and tau_target is not None and target_integral is not None:
        residual_integral = (
            tf.reduce_sum(tau_pred - tau_target)
            * float(integration_multiplier)
            * system.cell_volume
        )
        kinetic_pred = tf.cast(target_integral, tau_pred.dtype) + residual_integral
    else:
        kinetic_pred = tf.reduce_sum(tau_pred) * float(integration_multiplier) * system.cell_volume
    kinetic_ref = kinetic_energy_reference(system, config)
    scale = max(abs(kinetic_ref), 1.0)
    loss = tf.square((kinetic_pred - kinetic_ref) / scale)
    return loss, kinetic_pred, kinetic_ref


def ion_ion_energy_hartree(system: SystemRecord) -> float:
    if system.family in {"ks_like", "toy_dimensional"}:
        return 0.0
    symbols = system.metadata.get("atom_symbols", [])
    coords = np.asarray(system.metadata.get("atom_coords_bohr", np.empty((0, 3))), dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(symbols) != len(coords):
        return float("nan")
    charges = np.asarray([_ELEMENT_Z.get(str(symbol), 0) for symbol in symbols], dtype=np.float64)
    if len(charges) == 0 or np.any(charges <= 0):
        return float("nan")
    energy = 0.0
    for i in range(len(charges)):
        delta = coords[i + 1 :] - coords[i]
        distances = np.linalg.norm(delta, axis=1)
        valid = distances > 1e-12
        if np.any(valid):
            energy += float(np.sum(charges[i] * charges[i + 1 :][valid] / distances[valid]))
    return energy


def coulomb_kernel_fft(n_axis: int, step: float) -> np.ndarray:
    key = (int(n_axis), float(step))
    cached = _COULOMB_KERNEL_FFT_CACHE.get(key)
    if cached is not None:
        _COULOMB_KERNEL_FFT_CACHE.move_to_end(key)
        return cached

    shape = (2 * n_axis - 1, 2 * n_axis - 1, 2 * n_axis - 1)
    axes = []
    for size in shape:
        idx = np.arange(size, dtype=np.float64)
        idx[idx >= n_axis] -= size
        axes.append(idx * float(step))
    dx, dy, dz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    radius = np.sqrt(dx * dx + dy * dy + dz * dz)
    kernel = np.zeros(shape, dtype=np.float64)
    mask = radius > 1e-12
    kernel[mask] = 1.0 / radius[mask]
    kernel_fft = np.fft.fftn(kernel)

    while len(_COULOMB_KERNEL_FFT_CACHE) >= 2:
        _COULOMB_KERNEL_FFT_CACHE.popitem(last=False)
    _COULOMB_KERNEL_FFT_CACHE[key] = kernel_fft
    return kernel_fft


def hartree_energy_from_density(system: SystemRecord, rho: np.ndarray) -> float:
    n_axis = len(system.axis)
    rho_grid = np.asarray(rho, dtype=np.float64).reshape(n_axis, n_axis, n_axis)
    rho_grid = np.maximum(rho_grid, 0.0)
    shape = (2 * n_axis - 1, 2 * n_axis - 1, 2 * n_axis - 1)
    padded = np.zeros(shape, dtype=np.float64)
    padded[:n_axis, :n_axis, :n_axis] = rho_grid
    potential = np.fft.ifftn(np.fft.fftn(padded) * coulomb_kernel_fft(n_axis, system.step)).real
    potential = potential[:n_axis, :n_axis, :n_axis] * system.cell_volume
    return float(0.5 * np.sum(rho_grid * potential) * system.cell_volume)


def lda_xc_energy_from_density(system: SystemRecord, rho: np.ndarray) -> float:
    rho_clipped = np.maximum(np.asarray(rho, dtype=np.float64).reshape(-1), 0.0)
    nonzero = rho_clipped > 1e-14
    if not np.any(nonzero):
        return 0.0

    rho_nz = rho_clipped[nonzero]
    exchange_eps = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * rho_nz ** (1.0 / 3.0)

    rs = (3.0 / (4.0 * np.pi * rho_nz)) ** (1.0 / 3.0)
    corr_eps = np.empty_like(rs)
    high_density = rs < 1.0
    if np.any(high_density):
        rs_h = rs[high_density]
        corr_eps[high_density] = (
            0.0311 * np.log(rs_h)
            - 0.048
            + 0.0020 * rs_h * np.log(rs_h)
            - 0.0116 * rs_h
        )
    if np.any(~high_density):
        rs_l = rs[~high_density]
        corr_eps[~high_density] = -0.1423 / (1.0 + 1.0529 * np.sqrt(rs_l) + 0.3334 * rs_l)

    return float(np.sum(rho_nz * (exchange_eps + corr_eps)) * system.cell_volume)


def external_energy_from_density(system: SystemRecord, rho: np.ndarray) -> float:
    rho_clipped = np.maximum(np.asarray(rho, dtype=np.float64).reshape(-1, 1), 0.0)
    potential = np.asarray(system.potential, dtype=np.float64).reshape(-1, 1)
    return float(np.sum(rho_clipped * potential) * system.cell_volume)


def dft_component_energies(
    system: SystemRecord,
    rho: np.ndarray,
    kinetic: float,
    *,
    ion_ion: float | None = None,
) -> dict[str, float]:
    ion = ion_ion_energy_hartree(system) if ion_ion is None else float(ion_ion)
    external = external_energy_from_density(system, rho)
    hartree = hartree_energy_from_density(system, rho)
    xc = lda_xc_energy_from_density(system, rho)
    total = float(kinetic + external + hartree + xc + ion) if np.isfinite(ion) else float("nan")
    return {
        "kinetic": float(kinetic),
        "external": external,
        "hartree": hartree,
        "xc_lda": xc,
        "ion_ion": ion,
        "total": total,
    }


def energy_diagnostics(
    system: SystemRecord,
    rho_ref: np.ndarray,
    rho_pred: np.ndarray,
    kinetic_ref: float,
    kinetic_pred: float,
) -> dict[str, float]:
    ion = ion_ion_energy_hartree(system)
    ref = dft_component_energies(system, rho_ref, kinetic_ref, ion_ion=ion)
    pred = dft_component_energies(system, rho_pred, kinetic_pred, ion_ion=ion)
    stored_total_ref = float(system.metadata.get("total_energy_hartree", np.nan))
    total_ref_for_error = stored_total_ref if np.isfinite(stored_total_ref) else ref["total"]
    total_error = float(total_ref_for_error - pred["total"])
    grid_total_error = float(ref["total"] - pred["total"])
    stored_minus_grid_ref = (
        float(stored_total_ref - ref["total"])
        if np.isfinite(stored_total_ref)
        else float("nan")
    )
    out: dict[str, float] = {
        "energy_total_ref": total_ref_for_error,
        "energy_total_grid_ref": ref["total"],
        "energy_total_pred": pred["total"],
        "energy_total_ref_minus_pred": total_error,
        "energy_total_grid_ref_minus_pred": grid_total_error,
        "energy_stored_minus_grid_ref": stored_minus_grid_ref,
        "energy_stored_total_available": float(np.isfinite(stored_total_ref)),
        "energy_total_abs_error": abs(total_error),
        "energy_total_sq_error": total_error**2,
        "energy_grid_total_abs_error": abs(grid_total_error),
        "energy_grid_total_sq_error": grid_total_error**2,
    }
    for name in ("kinetic", "external", "hartree", "xc_lda", "ion_ion"):
        component_error = float(ref[name] - pred[name])
        out[f"energy_{name}_ref"] = ref[name]
        out[f"energy_{name}_pred"] = pred[name]
        out[f"energy_{name}_ref_minus_pred"] = component_error
        out[f"energy_{name}_abs_error"] = abs(component_error)
    return out


def energy_component_summary(metrics: dict[str, object], suffix: str) -> str:
    return (
        f"T={float(metrics[f'energy_kinetic_{suffix}']):.6e}, "
        f"Vext={float(metrics[f'energy_external_{suffix}']):.6e}, "
        f"J={float(metrics[f'energy_hartree_{suffix}']):.6e}, "
        f"Exc={float(metrics[f'energy_xc_lda_{suffix}']):.6e}, "
        f"Enn={float(metrics[f'energy_ion_ion_{suffix}']):.6e}"
    )


def gather_density(rho_all: tf.Tensor, indices: np.ndarray) -> tf.Tensor:
    return tf.gather(rho_all, tf.convert_to_tensor(indices, dtype=tf.int64))


def diagonal_predictions(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
    rho_all: tf.Tensor | None = None,
    density_state: DensityFeatureState | None = None,
    diag_indices: np.ndarray | None = None,
) -> dict[str, tf.Tensor]:
    """r = r' diagonal prediction."""
    if density_state is None:
        _, density_state = point_output_and_state(system, models, config)
    if rho_all is None:
        rho_all = density_state.rho_neutral
    diag_idx = (
        np.arange(len(system.points), dtype=np.int64)
        if diag_indices is None
        else np.asarray(diag_indices, dtype=np.int64)
    )
    chunk_size = max(int(config.diagonal_prediction_chunk_size), 1)
    if len(diag_idx) > chunk_size:
        gamma_chunks: list[tf.Tensor] = []
        kernel_chunks: list[tf.Tensor] = []
        for start in range(0, len(diag_idx), chunk_size):
            chunk_outputs = diagonal_predictions(
                system,
                models,
                config,
                rho_all=rho_all,
                density_state=density_state,
                diag_indices=diag_idx[start : start + chunk_size],
            )
            gamma_chunks.append(chunk_outputs["gamma"])
            kernel_chunks.append(chunk_outputs["kernel"])
        return {
            "gamma": tf.concat(gamma_chunks, axis=0),
            "kernel": tf.concat(kernel_chunks, axis=0),
        }
    pair_feat = build_pair_features(system, diag_idx, diag_idx)
    rho_diag = gather_density(rho_all, diag_idx)
    outputs = predict_from_features(
        to_tensor(system.local_features[diag_idx]),
        to_tensor(system.local_features[diag_idx]),
        to_tensor(pair_feat),
        to_tensor(system.global_context),
        models,
        rho_r_override=rho_diag,
        rho_rp_override=rho_diag,
        pair_density_feat_t=pair_density_features(system, density_state, diag_idx, diag_idx, config),
        local_curvature_basis_scale=local_curvature_basis_scale(system, config),
    )
    return outputs


def clear_gpu_evaluation_caches() -> None:
    """Release per-system tensors before evaluating another large split."""
    clear_frozen_density_state_cache()
    _SYSTEM_TENSOR_CACHE.clear()
    gc.collect()


def select_diagonal_indices(
    system: SystemRecord,
    max_points: int | None,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """Return sampled diagonal point indices, or None when using all points."""
    n_points = int(len(system.points))
    if max_points is None or int(max_points) <= 0 or int(max_points) >= n_points:
        return None
    count = int(max_points)
    if rng is None:
        return np.arange(count, dtype=np.int64)
    return np.sort(rng.choice(n_points, size=count, replace=False)).astype(np.int64)


def select_stencil_center_indices(
    system: SystemRecord,
    max_centers: int | None,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """Return sampled stencil-center indices, or None when using all centers."""
    n_centers = int(system.stencil_left.shape[0])
    if max_centers is None or int(max_centers) <= 0 or int(max_centers) >= n_centers:
        return None
    count = int(max_centers)
    if rng is None:
        return np.arange(count, dtype=np.int64)
    return np.sort(rng.choice(n_centers, size=count, replace=False)).astype(np.int64)


def _stencil_cache_trim(max_centers: int) -> None:
    if max_centers <= 0:
        _STENCIL_PAIR_FEATURE_CACHE.clear()
        return
    while len(_STENCIL_PAIR_FEATURE_CACHE) > max_centers:
        _STENCIL_PAIR_FEATURE_CACHE.popitem(last=False)


def stencil_pair_features_for_centers(
    system: SystemRecord,
    center_indices: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    """CPU-side LRU cache for immutable base pair features on stencil centers."""
    center_indices = np.asarray(center_indices, dtype=np.int64)
    if center_indices.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    max_cache_centers = int(config.stencil_feature_cache_max_centers)
    stencil_shape = system.stencil_left.shape
    flat_per_center = int(np.prod(stencil_shape[1:]))
    cache_enabled = max_cache_centers > 0
    cache_key_prefix = id(system)

    cached_by_position: dict[int, np.ndarray] = {}
    missing_positions = []
    missing_centers = []
    if cache_enabled:
        for position, center in enumerate(center_indices):
            key = (cache_key_prefix, int(center))
            cached = _STENCIL_PAIR_FEATURE_CACHE.get(key)
            if cached is None:
                missing_positions.append(position)
                missing_centers.append(int(center))
                continue
            _STENCIL_PAIR_FEATURE_CACHE.move_to_end(key)
            cached_by_position[position] = cached
    else:
        missing_positions = list(range(len(center_indices)))
        missing_centers = [int(center) for center in center_indices]

    if missing_centers:
        missing_array = np.asarray(missing_centers, dtype=np.int64)
        left_idx = system.stencil_left[missing_array].reshape(-1)
        right_idx = system.stencil_right[missing_array].reshape(-1)
        missing_features = build_pair_features(system, left_idx, right_idx)
        missing_features = missing_features.reshape((len(missing_centers),) + stencil_shape[1:] + (-1,))
        for position, center, features in zip(missing_positions, missing_centers, missing_features):
            features = np.asarray(features, dtype=np.float32)
            cached_by_position[position] = features
            if cache_enabled:
                _STENCIL_PAIR_FEATURE_CACHE[(cache_key_prefix, int(center))] = features
        if cache_enabled:
            _stencil_cache_trim(max_cache_centers)

    ordered = [cached_by_position[position] for position in range(len(center_indices))]
    return np.stack(ordered, axis=0).reshape((len(center_indices) * flat_per_center, -1)).astype(np.float32)


def stencil_predictions(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
    rho_all: tf.Tensor | None = None,
    density_state: DensityFeatureState | None = None,
    max_centers: int | None = None,
    center_indices: np.ndarray | None = None,
    return_gamma_stencil: bool = False,
) -> tuple[tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """explicit near-diagonal mixed derivative prediction.

    Returns
    -------
    derivative_pred : (n_interior, 3)
    tau_pred        : (n_interior, 1)
    """
    if density_state is None:
        _, density_state = point_output_and_state(system, models, config)
    if rho_all is None:
        rho_all = density_state.rho_neutral
    stencil_order = int(system.stencil_left.shape[2])
    stencil_shape = system.stencil_left.shape
    if center_indices is None:
        selected_centers = select_stencil_center_indices(system, max_centers)
    else:
        selected_centers = np.asarray(center_indices, dtype=np.int64)
    n_centers = int(stencil_shape[0]) if selected_centers is None else int(len(selected_centers))
    flat_per_center = int(np.prod(stencil_shape[1:]))
    chunk_pairs = max(int(config.stencil_prediction_chunk_size), flat_per_center)
    chunk_centers = max(1, chunk_pairs // flat_per_center)

    derivative_chunks = []
    gamma_chunks = []
    for start in range(0, n_centers, chunk_centers):
        end = min(start + chunk_centers, n_centers)
        chunk_shape = (end - start,) + stencil_shape[1:]
        chunk_centers_idx = (
            np.arange(start, end, dtype=np.int64)
            if selected_centers is None
            else selected_centers[start:end]
        )
        left_idx = system.stencil_left[chunk_centers_idx].reshape(-1)
        right_idx = system.stencil_right[chunk_centers_idx].reshape(-1)
        pair_feat = stencil_pair_features_for_centers(system, chunk_centers_idx, config)
        outputs = predict_from_features(
            to_tensor(system.local_features[left_idx]),
            to_tensor(system.local_features[right_idx]),
            to_tensor(pair_feat),
            to_tensor(system.global_context),
            models,
            rho_r_override=gather_density(rho_all, left_idx),
            rho_rp_override=gather_density(rho_all, right_idx),
            pair_density_feat_t=pair_density_features(system, density_state, left_idx, right_idx, config),
            local_curvature_basis_scale=local_curvature_basis_scale(system, config),
        )
        gamma_stencil = tf.reshape(outputs["gamma"], chunk_shape)
        if return_gamma_stencil:
            gamma_chunks.append(gamma_stencil)
        d_h = (
            gamma_stencil[:, :, 0]
            - gamma_stencil[:, :, 1]
            - gamma_stencil[:, :, 2]
            + gamma_stencil[:, :, 3]
        ) / (4.0 * system.step * system.step)
        if stencil_order >= 8:
            d_2h = (
                gamma_stencil[:, :, 4]
                - gamma_stencil[:, :, 5]
                - gamma_stencil[:, :, 6]
                + gamma_stencil[:, :, 7]
            ) / (16.0 * system.step * system.step)
            derivative_chunks.append((4.0 * d_h - d_2h) / 3.0)
        else:
            derivative_chunks.append(d_h)
    derivative_pred = tf.concat(derivative_chunks, axis=0)
    tau_pred = float(kinetic_prefactor(system)) * tf.reduce_sum(derivative_pred, axis=1, keepdims=True)
    if return_gamma_stencil:
        return derivative_pred, tau_pred, tf.concat(gamma_chunks, axis=0)
    return derivative_pred, tau_pred


def kinetic_stencil_error_decomposition(
    system: SystemRecord,
    gamma_pred: np.ndarray,
    gamma_true: np.ndarray,
    *,
    integration_multiplier: float = 1.0,
) -> dict[str, float]:
    """Split signed kinetic error into diagonal and off-diagonal stencil terms."""
    error = np.asarray(gamma_pred, dtype=np.float64) - np.asarray(gamma_true, dtype=np.float64)
    stencil_order = int(error.shape[2])
    step_sq = float(system.step) ** 2

    diag_derivative_error = (error[:, :, 0] + error[:, :, 3]) / (4.0 * step_sq)
    offdiag_derivative_error = -(error[:, :, 1] + error[:, :, 2]) / (4.0 * step_sq)
    if stencil_order >= 8:
        diag_2h = (error[:, :, 4] + error[:, :, 7]) / (16.0 * step_sq)
        offdiag_2h = -(error[:, :, 5] + error[:, :, 6]) / (16.0 * step_sq)
        diag_derivative_error = (4.0 * diag_derivative_error - diag_2h) / 3.0
        offdiag_derivative_error = (4.0 * offdiag_derivative_error - offdiag_2h) / 3.0

    scale = float(kinetic_prefactor(system)) * float(integration_multiplier) * float(system.cell_volume)
    diag_error = scale * float(np.sum(diag_derivative_error, dtype=np.float64))
    offdiag_error = scale * float(np.sum(offdiag_derivative_error, dtype=np.float64))
    total_error = diag_error + offdiag_error
    return {
        "kinetic_stencil_diag_error": diag_error,
        "kinetic_stencil_offdiag_error": offdiag_error,
        "kinetic_stencil_total_error": total_error,
        "kinetic_stencil_diag_abs_error": abs(diag_error),
        "kinetic_stencil_offdiag_abs_error": abs(offdiag_error),
    }


def gamma_error_by_category(
    gamma_pred: np.ndarray,
    gamma_true: np.ndarray,
    categories: np.ndarray,
) -> dict[str, float]:
    """MAE/RMSE diagnostics for sampled pair categories."""
    pred = np.asarray(gamma_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(gamma_true, dtype=np.float64).reshape(-1)
    cats = np.asarray(categories, dtype=np.int64).reshape(-1)
    error = pred - true
    metrics: dict[str, float] = {}
    for category_id, name in enumerate(("diag", "near", "mid", "far")):
        mask = cats == category_id
        if np.any(mask):
            cat_error = error[mask]
            metrics[f"{name}_pair_mae"] = float(np.mean(np.abs(cat_error)))
            metrics[f"{name}_pair_rmse"] = float(np.sqrt(np.mean(cat_error**2)))
        else:
            metrics[f"{name}_pair_mae"] = float("nan")
            metrics[f"{name}_pair_rmse"] = float("nan")
    metrics["near_diag_mae"] = metrics["near_pair_mae"]
    metrics["near_diag_rmse"] = metrics["near_pair_rmse"]
    metrics["far_offdiag_mae"] = metrics["far_pair_mae"]
    metrics["far_offdiag_rmse"] = metrics["far_pair_rmse"]
    return metrics


def gamma_stencil_error_metrics(
    gamma_pred: np.ndarray,
    gamma_true: np.ndarray,
    rho_left: np.ndarray,
    rho_right: np.ndarray,
    *,
    scale_floor: float,
    delta: float,
) -> dict[str, float]:
    """Raw and density-normalized diagnostics for stencil gamma values."""
    pred = np.asarray(gamma_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(gamma_true, dtype=np.float64).reshape(-1)
    rho_l = np.asarray(rho_left, dtype=np.float64).reshape(-1)
    rho_r = np.asarray(rho_right, dtype=np.float64).reshape(-1)
    error = pred - true
    scale = np.sqrt(np.maximum(rho_l * rho_r, 0.0)) + float(scale_floor)
    rel_error = error / scale
    abs_rel = np.abs(rel_error)
    quadratic = np.minimum(abs_rel, float(delta))
    linear = abs_rel - quadratic
    return {
        "stencil_gamma_huber": float(np.mean(0.5 * quadratic**2 + float(delta) * linear)),
        "stencil_gamma_mae": float(np.mean(np.abs(error))),
        "stencil_gamma_rmse": float(np.sqrt(np.mean(error**2))),
        "stencil_gamma_rel_mae": float(np.mean(abs_rel)),
        "stencil_gamma_rel_rmse": float(np.sqrt(np.mean(rel_error**2))),
    }


def gamma_stencil_targets(
    system: SystemRecord,
    *,
    center_indices: np.ndarray | None = None,
    chunk_centers: int = 8192,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Compute true-gamma mixed derivatives, preserving the diagnostic source."""
    selected = (
        np.arange(system.stencil_left.shape[0], dtype=np.int64)
        if center_indices is None
        else np.asarray(center_indices, dtype=np.int64)
    )
    if system.derivative_true_fd is not None and system.tau_true_fd is not None:
        return (
            np.asarray(system.derivative_true_fd[selected], dtype=np.float32),
            np.asarray(system.tau_true_fd[selected], dtype=np.float32),
            "stored_gamma_stencil",
        )

    stencil_order = int(system.stencil_left.shape[2])
    derivative_chunks = []
    for start in range(0, len(selected), max(int(chunk_centers), 1)):
        chunk_indices = selected[start : start + max(int(chunk_centers), 1)]
        chunk_shape = (len(chunk_indices),) + tuple(system.stencil_left.shape[1:])
        left_idx = system.stencil_left[chunk_indices].reshape(-1)
        right_idx = system.stencil_right[chunk_indices].reshape(-1)
        gamma_stencil = system.gamma_values(left_idx, right_idx).reshape(chunk_shape)
        d_h = (
            gamma_stencil[:, :, 0]
            - gamma_stencil[:, :, 1]
            - gamma_stencil[:, :, 2]
            + gamma_stencil[:, :, 3]
        ) / (4.0 * system.step * system.step)
        if stencil_order >= 8:
            d_2h = (
                gamma_stencil[:, :, 4]
                - gamma_stencil[:, :, 5]
                - gamma_stencil[:, :, 6]
                + gamma_stencil[:, :, 7]
            ) / (16.0 * system.step * system.step)
            derivative_chunks.append((4.0 * d_h - d_2h) / 3.0)
        else:
            derivative_chunks.append(d_h)
    derivative_fd = np.concatenate(derivative_chunks, axis=0).astype(np.float32)
    tau_fd = float(kinetic_prefactor(system)) * np.sum(derivative_fd, axis=1, keepdims=True)
    source = (
        "reconstructed_from_psi_occ"
        if (system.psi_occ is not None and system.psi_occ.size)
        or bool(system.metadata.get("has_psi_occ", False))
        else "reconstructed_from_gamma_matrix"
    )
    return derivative_fd, tau_fd.astype(np.float32), source


def true_gamma_stencil_targets(system: SystemRecord) -> tuple[np.ndarray, np.ndarray]:
    """Compute and cache full derivative/tau targets from true gamma."""
    cache_key = (id(system), int(system.stencil_left.shape[2]), float(system.step))
    cached = _TRUE_GAMMA_STENCIL_TARGET_CACHE.get(cache_key)
    if cached is not None:
        return cached
    derivative_fd, tau_fd, _ = gamma_stencil_targets(system)
    targets = (derivative_fd, tau_fd)
    _TRUE_GAMMA_STENCIL_TARGET_CACHE[cache_key] = targets
    return targets


def physics_target_mode(config: ExperimentConfig) -> str:
    mode = config.physics_target.strip().lower()
    if mode == "ao":
        mode = "orbital"
    if mode not in PHYSICS_TARGET_MODES:
        raise ValueError(
            f"Unknown RDM_PHYSICS_TARGET: {config.physics_target!r}. "
            "Choose 'orbital' (legacy alias: 'ao') or 'fd'."
        )
    return mode


def physics_stencil_targets(system: SystemRecord, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    if physics_target_mode(config) == "fd":
        return true_gamma_stencil_targets(system)
    return system.derivative_true, system.tau_true


def curvature_target_diagnostics(
    system: SystemRecord,
    derivative_target: np.ndarray,
    *,
    center_indices: np.ndarray | None = None,
    rho_cut_fraction: float = 1e-4,
) -> dict[str, float]:
    """Estimate axis-resolved effective kernel curvature target statistics."""
    rho = np.asarray(system.rho_diag, dtype=np.float32)
    rho_scale = max(float(np.max(rho)), 1e-30)
    interior_indices = (
        system.interior_point_indices
        if center_indices is None
        else system.interior_point_indices[np.asarray(center_indices, dtype=np.int64)]
    )
    mask = rho[interior_indices, 0] > rho_cut_fraction * rho_scale
    if not np.any(mask):
        return {
            "curvature_target_min": float("nan"),
            "curvature_target_p05": float("nan"),
            "curvature_target_p50": float("nan"),
            "curvature_target_p95": float("nan"),
            "curvature_target_max": float("nan"),
            "curvature_target_neg_frac": float("nan"),
        }
    grad_rho = richardson_gradient_3d(to_tensor(rho), len(system.axis), system.step).numpy()
    rho_interior = np.maximum(rho[interior_indices], 1e-30)
    grad_term = grad_rho[interior_indices] ** 2 / (4.0 * rho_interior)
    curvature_target = (derivative_target - grad_term) / rho_interior
    values = curvature_target[mask].reshape(-1)
    return {
        "curvature_target_min": float(np.min(values)),
        "curvature_target_p05": float(np.percentile(values, 5.0)),
        "curvature_target_p50": float(np.percentile(values, 50.0)),
        "curvature_target_p95": float(np.percentile(values, 95.0)),
        "curvature_target_max": float(np.max(values)),
        "curvature_target_neg_frac": float(np.mean(values < 0.0)),
    }


def np_rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value)))) if value.size else float("nan")


def safe_rms_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    denom = max(np_rms(denominator), 1e-30)
    return float(np_rms(numerator) / denom)


def spectral_occupation_penalty(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
    rho_all: tf.Tensor | None = None,
    density_state: DensityFeatureState | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """coarse operator spectrum이 물리 범위를 벗어나지 않게 벌점."""
    if density_state is None:
        _, density_state = point_output_and_state(system, models, config)
    if rho_all is None:
        rho_all = density_state.rho_neutral
    subset = system.spectral_subset
    left = np.repeat(subset, len(subset))
    right = np.tile(subset, len(subset))
    outputs = predict_from_features(
        to_tensor(system.local_features[left]),
        to_tensor(system.local_features[right]),
        to_tensor(build_pair_features(system, left, right)),
        to_tensor(system.global_context),
        models,
        rho_r_override=gather_density(rho_all, left),
        rho_rp_override=gather_density(rho_all, right),
        pair_density_feat_t=pair_density_features(system, density_state, left, right, config),
        local_curvature_basis_scale=local_curvature_basis_scale(system, config),
    )
    n_spec = len(subset)
    gamma_sub = tf.reshape(outputs["gamma"], (n_spec, n_spec))
    operator = system.cell_volume * 0.5 * (gamma_sub + tf.transpose(gamma_sub))
    eigvals = tf.linalg.eigvalsh(operator)
    penalty = tf.reduce_mean(tf.square(tf.nn.relu(-eigvals)) + tf.square(tf.nn.relu(eigvals - config.occ_max)))
    return penalty, eigvals


def optimizer_from_config(config: ExperimentConfig) -> tf.keras.optimizers.Optimizer:
    """가능하면 AdamW, 아니면 Adam."""
    try:
        return tf.keras.optimizers.AdamW(
            learning_rate=config.initial_lr,
            weight_decay=config.weight_decay,
        )
    except AttributeError:
        return tf.keras.optimizers.Adam(learning_rate=config.initial_lr)


def trainable_variables(models: ModelBundle) -> list[tf.Variable]:
    return (
        models.point_model.trainable_variables
        + models.mode_model.trainable_variables
        + models.pair_model.trainable_variables
        + models.context_model.trainable_variables
    )


def use_compiled_train_step(config: ExperimentConfig) -> bool:
    """The compiled path receives detached density tensors from Python."""
    density_is_fixed = config.freeze_point_after_pretrain or density_source_mode(config) == "true"
    return config.compile_train_step and density_is_fixed


def loss_enabled(config: ExperimentConfig, name: str) -> bool:
    if name in {"tau_mse", "kinetic_mse"}:
        return bool(getattr(config, f"use_{name}_loss"))
    preset = config.loss_preset.strip().lower()
    if preset in {"core5", "simple5"}:
        return name in {"gamma", "rho", "kernel", "trace", "mode"}
    if preset in {"staged-physics", "physics7"}:
        return name in {"gamma", "rho", "kernel", "deriv", "tau", "trace", "mode"}
    if preset in {"staged-physics-kinetic", "physics8"}:
        return name in {"gamma", "rho", "kernel", "deriv", "tau", "trace", "mode", "kinetic"}
    if preset in {"core7", "kerdf7"}:
        return name in {"gamma", "rho", "kernel", "trace", "mode", "kinetic"}
    if preset in {"all", "custom", "none"}:
        return bool(getattr(config, f"use_{name}_loss"))
    raise ValueError(f"Unknown RDM_LOSS_PRESET: {config.loss_preset}")


def loss_weight(config: ExperimentConfig, name: str) -> float:
    if not loss_enabled(config, name):
        return 0.0
    return float(getattr(config, f"lambda_{name}"))


def loss_schedule_multiplier(config: ExperimentConfig, name: str, epoch: int) -> float:
    """Epoch-dependent multiplier for staged loss terms."""
    if name not in {"deriv", "tau", "tau_mse", "stencil_gamma", "kinetic", "kinetic_mse"}:
        return 1.0
    schedule_name = {"tau_mse": "tau", "stencil_gamma": "tau", "kinetic_mse": "kinetic"}.get(name, name)
    start_epoch = max(int(getattr(config, f"{schedule_name}_start_epoch")), 0)
    ramp_epochs = max(int(getattr(config, f"{schedule_name}_ramp_epochs")), 0)

    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch - start_epoch + 1) / float(ramp_epochs))


def scheduled_loss_weight(config: ExperimentConfig, name: str, epoch: int | None = None) -> float:
    base_weight = loss_weight(config, name)
    if epoch is None or base_weight == 0.0:
        return base_weight
    return base_weight * loss_schedule_multiplier(config, name, epoch)


def loss_weight_rows(config: ExperimentConfig, epoch: int | None = None) -> list[tuple[str, str]]:
    return [(name, f"{scheduled_loss_weight(config, name, epoch):.6g}") for name in LOSS_NAMES]


def loss_schedule_rows(config: ExperimentConfig) -> list[tuple[str, str]]:
    return [
        ("deriv start/ramp epochs", f"{config.deriv_start_epoch}/{config.deriv_ramp_epochs}"),
        ("tau start/ramp epochs", f"{config.tau_start_epoch}/{config.tau_ramp_epochs}"),
        ("kinetic start/ramp epochs", f"{config.kinetic_start_epoch}/{config.kinetic_ramp_epochs}"),
    ]


def active_loss_summary(weights: dict[str, float]) -> str:
    active = [f"{name}={weight:.4g}" for name, weight in weights.items() if weight != 0.0]
    return ", ".join(active) if active else "none"


def loss_stage_value(config: ExperimentConfig, name: str, epoch: int) -> int:
    """0: inactive, 1: scheduled ramp, 2: fully active."""
    if scheduled_loss_weight(config, name, epoch) == 0.0:
        return 0
    if loss_schedule_multiplier(config, name, epoch) < 1.0:
        return 1
    return 2


def loss_stage_signature(config: ExperimentConfig, epoch: int) -> tuple[int, ...]:
    return tuple(loss_stage_value(config, name, epoch) for name in LOSS_NAMES)


def fully_active_schedule_epoch(config: ExperimentConfig) -> int:
    """First epoch where every enabled scheduled loss has reached full weight."""
    epochs = [0]
    for name in ("deriv", "tau", "tau_mse", "stencil_gamma", "kinetic", "kinetic_mse"):
        if loss_weight(config, name) == 0.0:
            continue
        schedule_name = {"tau_mse": "tau", "stencil_gamma": "tau", "kinetic_mse": "kinetic"}.get(name, name)
        start = max(int(getattr(config, f"{schedule_name}_start_epoch")), 0)
        ramp = max(int(getattr(config, f"{schedule_name}_ramp_epochs")), 0)
        epochs.append(start + max(ramp - 1, 0))
    return max(epochs)


def objective_from_metrics(metrics: dict[str, float], config: ExperimentConfig, epoch: int | None = None) -> float:
    return float(
        sum(scheduled_loss_weight(config, weight_name, epoch) * metrics[metric_name]
            for weight_name, metric_name in EVAL_OBJECTIVE_TERMS)
    )


def predict_full_gamma_matrix(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
    chunk_size: int = 16384,
) -> np.ndarray:
    """full grid gamma matrix prediction."""
    pieces = []
    _, density_state = point_output_and_state(system, models, config)
    rho_all = density_state.rho_neutral
    for left, right in full_pair_chunks(system, chunk_size):
        outputs = predict_from_features(
            to_tensor(system.local_features[left]),
            to_tensor(system.local_features[right]),
            to_tensor(build_pair_features(system, left, right)),
            to_tensor(system.global_context),
            models,
            rho_r_override=gather_density(rho_all, left),
            rho_rp_override=gather_density(rho_all, right),
            pair_density_feat_t=pair_density_features(system, density_state, left, right, config),
            local_curvature_basis_scale=local_curvature_basis_scale(system, config),
        )
        pieces.append(outputs["gamma"].numpy())
    gamma_pairs = np.concatenate(pieces, axis=0).astype(np.float32)
    return gamma_pairs.reshape(len(system.points), len(system.points))


def natural_occupation_spectrum(gamma_matrix: np.ndarray, cell_volume: float) -> np.ndarray:
    operator = cell_volume * 0.5 * (gamma_matrix + gamma_matrix.T)
    eigvals = np.linalg.eigvalsh(operator)
    return np.sort(eigvals)[::-1]


def predict_pair_values(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
    left: np.ndarray,
    right: np.ndarray,
    rho_all: tf.Tensor | None = None,
    density_state: DensityFeatureState | None = None,
) -> np.ndarray:
    if density_state is None:
        _, density_state = point_output_and_state(system, models, config)
    if rho_all is None:
        rho_all = density_state.rho_neutral
    outputs = predict_from_features(
        to_tensor(system.local_features[left]),
        to_tensor(system.local_features[right]),
        to_tensor(build_pair_features(system, left, right)),
        to_tensor(system.global_context),
        models,
        rho_r_override=gather_density(rho_all, left),
        rho_rp_override=gather_density(rho_all, right),
        pair_density_feat_t=pair_density_features(system, density_state, left, right, config),
        local_curvature_basis_scale=local_curvature_basis_scale(system, config),
    )
    return outputs["gamma"].numpy().astype(np.float32)


def gamma_anchor_slice(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
    *,
    rho_all: tf.Tensor,
    density_state: DensityFeatureState,
) -> dict[str, object]:
    """Evaluate gamma(r, r0) on one z-plane without constructing full gamma."""
    axis_points = int(len(system.axis))
    anchor_index = int(np.argmax(np.asarray(system.rho_diag).reshape(-1)))
    anchor_xyz = np.unravel_index(
        anchor_index,
        (axis_points, axis_points, axis_points),
    )
    index_cube = np.arange(len(system.points), dtype=np.int64).reshape(
        axis_points,
        axis_points,
        axis_points,
    )
    slice_indices = index_cube[:, :, anchor_xyz[2]].reshape(-1)
    anchor_indices = np.full(slice_indices.shape, anchor_index, dtype=np.int64)
    gamma_true = system.gamma_values(slice_indices, anchor_indices).reshape(
        axis_points,
        axis_points,
    )
    gamma_pred = predict_pair_values(
        system,
        models,
        config,
        slice_indices,
        anchor_indices,
        rho_all=rho_all,
        density_state=density_state,
    ).reshape(axis_points, axis_points)
    return {
        "gamma_anchor_true_slice": gamma_true.astype(np.float32),
        "gamma_anchor_pred_slice": gamma_pred.astype(np.float32),
        "gamma_anchor_index": anchor_index,
        "gamma_anchor_xyz_index": np.asarray(anchor_xyz, dtype=np.int64),
        "gamma_anchor_position_bohr": np.asarray(
            system.points[anchor_index],
            dtype=np.float32,
        ),
    }


def evaluate_system(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
    *,
    rng: np.random.Generator,
    keep_arrays: bool = False,
    epoch: int | None = None,
) -> dict[str, object]:
    """한 시스템에 대한 sampled evaluation.

    Large spacing-based grids make full G^2 evaluation too expensive. Scalar
    pair metrics are therefore estimated from a deterministic category-balanced
    sample. Stencil physics metrics are sampled during training epochs and run
    exactly for final summaries unless disabled by config.
    """
    eval_epoch = max(config.epochs - 1 if epoch is None else epoch, 0)
    left, right, categories = sample_pair_indices(
        system,
        config.eval_pair_count,
        epoch=eval_epoch,
        total_epochs=max(config.epochs, 1),
        rng=rng,
    )
    gamma_true_pairs = system.gamma_values(left, right)
    _, density_state = point_output_and_state(system, models, config)
    rho_all = density_state.rho_neutral
    gamma_pred_pairs = predict_pair_values(
        system, models, config, left, right, rho_all=rho_all, density_state=density_state
    )
    pair_weights = pair_weights_from_categories(categories)

    pair_loss = float(np.sum(pair_weights * (gamma_pred_pairs - gamma_true_pairs) ** 2) / np.sum(pair_weights))
    pair_mae = float(np.mean(np.abs(gamma_pred_pairs - gamma_true_pairs)))
    pair_rmse = float(np.sqrt(np.mean((gamma_pred_pairs - gamma_true_pairs) ** 2)))
    category_gamma_metrics = gamma_error_by_category(gamma_pred_pairs, gamma_true_pairs, categories)

    diag_outputs = diagonal_predictions(system, models, config, rho_all=rho_all, density_state=density_state)
    gamma_diag = diag_outputs["gamma"].numpy()
    kernel_diag = diag_outputs["kernel"].numpy()
    rho_loss = float(np.mean((gamma_diag - system.rho_diag) ** 2))
    density_mae = float(np.mean(np.abs(gamma_diag - system.rho_diag)))
    rho_point_mae = float(np.mean(np.abs(rho_all.numpy() - system.rho_diag)))
    kernel_loss = float(np.mean((kernel_diag - 1.0) ** 2))
    kernel_diag_error = float(np.mean(np.abs(kernel_diag - 1.0)))

    total_stencil_centers = int(system.stencil_left.shape[0])
    final_epoch = epoch is None or int(epoch) >= int(config.epochs) - 1
    full_stencil_eval = config.eval_stencil_centers <= 0 or (
        final_epoch and bool(config.eval_full_final)
    )
    eval_center_indices = None
    if not full_stencil_eval:
        eval_center_indices = select_stencil_center_indices(
            system,
            config.eval_stencil_centers,
            rng,
        )
    derivative_pred_t, tau_pred_t, gamma_stencil_pred_t = stencil_predictions(
        system,
        models,
        config,
        rho_all=rho_all,
        density_state=density_state,
        max_centers=None if full_stencil_eval else config.eval_stencil_centers,
        center_indices=eval_center_indices,
        return_gamma_stencil=True,
    )
    derivative_pred = derivative_pred_t.numpy()
    tau_pred = tau_pred_t.numpy()
    full_derivative_target, full_tau_target = physics_stencil_targets(system, config)
    if eval_center_indices is None:
        derivative_true_fd, tau_true_fd, gamma_fd_target_source = gamma_stencil_targets(system)
        derivative_target = full_derivative_target
        tau_target = full_tau_target
        derivative_true_orbital = system.derivative_true
        tau_true_orbital = system.tau_true
        stencil_integration_multiplier = 1.0
        stencil_eval_centers = total_stencil_centers
    else:
        derivative_true_fd, tau_true_fd, gamma_fd_target_source = gamma_stencil_targets(
            system,
            center_indices=eval_center_indices,
        )
        derivative_target = full_derivative_target[eval_center_indices]
        tau_target = full_tau_target[eval_center_indices]
        derivative_true_orbital = system.derivative_true[eval_center_indices]
        tau_true_orbital = system.tau_true[eval_center_indices]
        stencil_eval_centers = int(len(eval_center_indices))
        stencil_integration_multiplier = float(total_stencil_centers) / max(float(stencil_eval_centers), 1.0)
    deriv_raw_mse = float(np.mean((derivative_pred - derivative_target) ** 2))
    tau_raw_mse = float(np.mean((tau_pred - tau_target) ** 2))
    tau_rmse = float(np.sqrt(tau_raw_mse))
    tau_target_rms_sq = float(max(np.mean(tau_target**2), config.tau_scale_floor**2))
    tau_rel_mse_loss = float(tau_raw_mse / tau_target_rms_sq)
    tau_rel_rmse = float(np.sqrt(tau_rel_mse_loss))
    deriv_pred_ao_raw_mse = float(np.mean((derivative_pred - derivative_true_orbital) ** 2))
    deriv_pred_ao_mae = float(np.mean(np.abs(derivative_pred - derivative_true_orbital)))
    deriv_fd_ao_raw_mse = float(np.mean((derivative_true_fd - derivative_true_orbital) ** 2))
    deriv_fd_ao_mae = float(np.mean(np.abs(derivative_true_fd - derivative_true_orbital)))
    deriv_fd_ao_rms_ratio = safe_rms_ratio(derivative_true_fd, derivative_true_orbital)
    deriv_pred_fd_raw_mse = float(np.mean((derivative_pred - derivative_true_fd) ** 2))
    deriv_pred_fd_mae = float(np.mean(np.abs(derivative_pred - derivative_true_fd)))
    tau_pred_ao_raw_mse = float(np.mean((tau_pred - tau_true_orbital) ** 2))
    tau_pred_ao_mae = float(np.mean(np.abs(tau_pred - tau_true_orbital)))
    tau_fd_ao_raw_mse = float(np.mean((tau_true_fd - tau_true_orbital) ** 2))
    tau_fd_ao_mae = float(np.mean(np.abs(tau_true_fd - tau_true_orbital)))
    tau_fd_ao_rms_ratio = safe_rms_ratio(tau_true_fd, tau_true_orbital)
    tau_pred_fd_raw_mse = float(np.mean((tau_pred - tau_true_fd) ** 2))
    tau_pred_fd_mae = float(np.mean(np.abs(tau_pred - tau_true_fd)))
    deriv_loss = float(
        rms_normalized_huber(
            to_tensor(derivative_target),
            derivative_pred_t,
            scale_floor=config.deriv_scale_floor,
            delta=config.physics_huber_delta,
        ).numpy()
    )
    tau_loss = float(
        rms_normalized_huber(
            to_tensor(tau_target),
            tau_pred_t,
            scale_floor=config.tau_scale_floor,
            delta=config.physics_huber_delta,
        ).numpy()
    )
    deriv_mae = float(np.mean(np.abs(derivative_pred - derivative_target)))
    tau_mae = float(np.mean(np.abs(tau_pred - tau_target)))
    curvature_stats = curvature_target_diagnostics(
        system,
        derivative_target,
        center_indices=eval_center_indices,
    )
    kinetic_loss_t, kinetic_pred_t, kinetic_ref = kinetic_energy_loss_from_tau(
        system,
        tau_pred_t,
        config=config,
        integration_multiplier=stencil_integration_multiplier,
        tau_target=to_tensor(tau_target),
        target_integral=physics_tau_integral(system, config),
        control_variate=config.kinetic_control_variate,
    )
    kinetic_loss = float(kinetic_loss_t.numpy())
    kinetic_mse_loss = kinetic_loss
    kinetic_pred = float(kinetic_pred_t.numpy())
    kinetic_ref_error = float(kinetic_pred - kinetic_ref)
    kinetic_sq_error = float(kinetic_ref_error**2)
    kinetic_scale = max(abs(kinetic_ref), 1.0)
    kinetic_rel_abs_error = float(abs(kinetic_ref_error) / kinetic_scale)
    kinetic_rel_sq_error = float((kinetic_ref_error / kinetic_scale) ** 2)
    stencil_indices = (
        np.arange(total_stencil_centers, dtype=np.int64)
        if eval_center_indices is None
        else np.asarray(eval_center_indices, dtype=np.int64)
    )
    stencil_shape = (len(stencil_indices),) + tuple(system.stencil_left.shape[1:])
    gamma_stencil_true = system.gamma_values(
        system.stencil_left[stencil_indices].reshape(-1),
        system.stencil_right[stencil_indices].reshape(-1),
    ).reshape(stencil_shape)
    stencil_left_flat = system.stencil_left[stencil_indices].reshape(-1)
    stencil_right_flat = system.stencil_right[stencil_indices].reshape(-1)
    stencil_gamma_metrics = gamma_stencil_error_metrics(
        gamma_stencil_pred_t.numpy(),
        gamma_stencil_true,
        gather_density(rho_all, stencil_left_flat).numpy(),
        gather_density(rho_all, stencil_right_flat).numpy(),
        scale_floor=config.stencil_gamma_scale_floor,
        delta=config.physics_huber_delta,
    )
    kinetic_stencil_decomposition = kinetic_stencil_error_decomposition(
        system,
        gamma_stencil_pred_t.numpy(),
        gamma_stencil_true,
        integration_multiplier=stencil_integration_multiplier,
    )
    kinetic_stencil_decomposition["kinetic_stencil_reconstruction_residual"] = float(
        kinetic_stencil_decomposition["kinetic_stencil_total_error"]
        - (
            float(np.sum(tau_pred - tau_true_fd, dtype=np.float64))
            * stencil_integration_multiplier
            * system.cell_volume
        )
    )
    kinetic_stencil_decomposition["kinetic_stencil_reference_gap"] = float(
        kinetic_ref_error - kinetic_stencil_decomposition["kinetic_stencil_total_error"]
    )
    kinetic_energy_ref = float(system.metadata.get("kinetic_energy_hartree", np.nan))
    kinetic_energy_ref_error = float(kinetic_pred - kinetic_energy_ref) if np.isfinite(kinetic_energy_ref) else float("nan")
    energy_stats = energy_diagnostics(
        system,
        system.rho_diag,
        rho_all.numpy(),
        kinetic_ref,
        kinetic_pred,
    )
    trace_pred = float(np.sum(gamma_diag) * system.cell_volume)
    trace_true = float(system.electron_count)
    trace_scale = max(trace_true, 1.0)
    trace_rel_error = float((trace_pred - trace_true) / trace_scale)
    trace_loss = float(trace_rel_error**2)

    gamma_pred_reverse = predict_pair_values(
        system, models, config, right, left, rho_all=rho_all, density_state=density_state
    )
    symmetry_mae = float(np.mean(np.abs(gamma_pred_pairs - gamma_pred_reverse)))

    subset = system.spectral_subset
    gamma_true_sub = system.gamma_submatrix(subset)
    subset_eigs_true = natural_occupation_spectrum(gamma_true_sub, system.cell_volume)
    occ_penalty_t, occ_eigs_t = spectral_occupation_penalty(
        system, models, config, rho_all=rho_all, density_state=density_state
    )
    occ_penalty = float(occ_penalty_t.numpy())
    subset_eigs_pred = np.sort(occ_eigs_t.numpy())[::-1]
    min_eig_pred = float(np.min(occ_eigs_t.numpy()))
    top_mo_occ_true = topk_descending(system.occupancies, 6) if len(system.occupancies) else np.array([], dtype=np.float32)
    tau_true_integral = float(np.sum(system.tau_true) * system.cell_volume)
    tau_true_fd_integral = (
        float(np.sum(tau_true_fd) * system.cell_volume)
        if eval_center_indices is None
        else float("nan")
    )
    tau_pred_integral = float(kinetic_pred)

    metrics = {
        "system_id": system.system_id,
        "family": system.family,
        "toy_dimension": system.metadata.get("toy_dimension", ""),
        "particle_mass": float(system.metadata.get("particle_mass", 1.0)),
        "formula": str(system.metadata.get("formula", "")),
        "axis_points": int(len(system.axis)),
        "n_points": int(len(system.points)),
        "grid_spacing_bohr": float(system.step),
        "electron_count": float(system.electron_count),
        "stencil_eval_centers": float(stencil_eval_centers),
        "stencil_eval_total_centers": float(total_stencil_centers),
        "stencil_eval_sampled": float(eval_center_indices is not None),
        "gamma_fd_target_source": gamma_fd_target_source,
        "kinetic_evaluation_mode": (
            "full_grid"
            if eval_center_indices is None
            else (
                "sampled_control_variate"
                if config.kinetic_control_variate
                else "sampled_scaled_integral"
            )
        ),
        "pair_loss": pair_loss,
        "pair_mae": pair_mae,
        "pair_rmse": pair_rmse,
        **category_gamma_metrics,
        "rho_loss": rho_loss,
        "density_mae": density_mae,
        "rho_point_mae": rho_point_mae,
        "kernel_loss": kernel_loss,
        "kernel_diag_error": kernel_diag_error,
        "deriv_loss": deriv_loss,
        "deriv_raw_mse": deriv_raw_mse,
        "deriv_mae": deriv_mae,
        "deriv_pred_ao_raw_mse": deriv_pred_ao_raw_mse,
        "deriv_pred_ao_mae": deriv_pred_ao_mae,
        "deriv_fd_ao_raw_mse": deriv_fd_ao_raw_mse,
        "deriv_fd_ao_mae": deriv_fd_ao_mae,
        "deriv_fd_ao_rms_ratio": deriv_fd_ao_rms_ratio,
        "deriv_pred_fd_raw_mse": deriv_pred_fd_raw_mse,
        "deriv_pred_fd_mae": deriv_pred_fd_mae,
        "tau_loss": tau_loss,
        "tau_mse_loss": tau_rel_mse_loss,
        "tau_raw_mse": tau_raw_mse,
        "tau_rmse": tau_rmse,
        "tau_rel_mse_loss": tau_rel_mse_loss,
        "tau_rel_rmse": tau_rel_rmse,
        "tau_mae": tau_mae,
        "tau_pred_ao_raw_mse": tau_pred_ao_raw_mse,
        "tau_pred_ao_mae": tau_pred_ao_mae,
        "tau_fd_ao_raw_mse": tau_fd_ao_raw_mse,
        "tau_fd_ao_mae": tau_fd_ao_mae,
        "tau_fd_ao_rms_ratio": tau_fd_ao_rms_ratio,
        "tau_pred_fd_raw_mse": tau_pred_fd_raw_mse,
        "tau_pred_fd_mae": tau_pred_fd_mae,
        **stencil_gamma_metrics,
        "kinetic_loss": kinetic_loss,
        "kinetic_mse_loss": kinetic_mse_loss,
        "kinetic_pred": kinetic_pred,
        "kinetic_training_ref": kinetic_ref,
        "kinetic_ref_error": kinetic_ref_error,
        "kinetic_abs_error": abs(kinetic_ref_error),
        "kinetic_sq_error": kinetic_sq_error,
        "kinetic_rmse": abs(kinetic_ref_error),
        "kinetic_rel_abs_error": kinetic_rel_abs_error,
        "kinetic_rel_sq_error": kinetic_rel_sq_error,
        "kinetic_rel_rmse": abs(kinetic_ref_error) / kinetic_scale,
        **kinetic_stencil_decomposition,
        "trace_loss": trace_loss,
        "trace_rel_error": trace_rel_error,
        "trace_abs_rel_error": abs(trace_rel_error),
        "trace_true": trace_true,
        "trace_pred": trace_pred,
        "occ_penalty": occ_penalty,
        "symmetry_mae": symmetry_mae,
        "subset_eigs_true": subset_eigs_true,
        "subset_eigs_pred": subset_eigs_pred,
        "top_subset_eigs_true": topk_descending(subset_eigs_true, 6),
        "top_subset_eigs_pred": topk_descending(subset_eigs_pred, 6),
        "top_mo_occ_true": top_mo_occ_true,
        # Backward-compatible keys. These are subset operator eigenvalues, not
        # physically reliable full natural occupations.
        "natural_occ_true": subset_eigs_true,
        "natural_occ_pred": subset_eigs_pred,
        "top_occ_true": topk_descending(subset_eigs_true, 6),
        "top_occ_pred": topk_descending(subset_eigs_pred, 6),
        "min_eig_pred": min_eig_pred,
        "tau_true_integral": tau_true_integral,
        "tau_true_fd_integral": tau_true_fd_integral,
        "tau_fd_ao_integral_error": tau_true_fd_integral - tau_true_integral,
        "tau_pred_integral": tau_pred_integral,
        "ked_true_integral": tau_true_integral,
        "ked_true_fd_integral": tau_true_fd_integral,
        "ked_pred_integral": tau_pred_integral,
        "kinetic_energy_ref": kinetic_energy_ref,
        "kinetic_energy_ref_error": kinetic_energy_ref_error,
        **energy_stats,
        "physics_target": physics_target_mode(config),
        **curvature_stats,
        "rho_true_diag": system.rho_diag,
        "rho_pred_diag": rho_all.numpy(),
        "tau_true": system.tau_true,
        "tau_target_eval": tau_target,
        "tau_true_fd": tau_true_fd,
        "tau_pred": tau_pred,
        "gamma_true_sample": gamma_true_pairs,
        "gamma_pred_sample": gamma_pred_pairs,
        "occ_eigs_subset": occ_eigs_t.numpy(),
    }
    if density_state.rho_cation is not None and density_state.rho_anion is not None:
        metrics["rho_cation_mae"] = float(np.mean(np.abs(density_state.rho_cation.numpy() - system.rho_cation)))
        metrics["rho_anion_mae"] = float(np.mean(np.abs(density_state.rho_anion.numpy() - system.rho_anion)))
        metrics["fukui_plus_mae"] = float(
            np.mean(np.abs((density_state.rho_anion - density_state.rho_neutral).numpy() - (system.rho_anion - system.rho_diag)))
        )
        metrics["fukui_minus_mae"] = float(
            np.mean(np.abs((density_state.rho_neutral - density_state.rho_cation).numpy() - (system.rho_diag - system.rho_cation)))
        )
    if keep_arrays:
        metrics.update(
            gamma_anchor_slice(
                system,
                models,
                config,
                rho_all=rho_all,
                density_state=density_state,
            )
        )
    metrics["objective"] = objective_from_metrics(metrics, config, epoch=epoch)
    return metrics


def evaluate_systems(
    systems: list[SystemRecord],
    models: ModelBundle,
    config: ExperimentConfig,
    epoch: int | None = None,
    progress_label: str | None = None,
) -> dict[str, object]:
    """여러 시스템 평가 후 평균 지표 계산."""
    rng = np.random.default_rng(config.seed + 999)
    per_system = []
    start_time = time.perf_counter()
    for idx, system in enumerate(systems):
        per_system.append(
            evaluate_system(
                system,
                models,
                config,
                rng=rng,
                keep_arrays=(idx == 0),
                epoch=epoch,
            )
        )
        if progress_label:
            elapsed = time.perf_counter() - start_time
            print(
                f"[{progress_label}] {idx + 1}/{len(systems)} systems | "
                f"{system.system_id} | elapsed {elapsed:.1f}s"
            )

    scalar_keys = [
        "objective",
        "stencil_eval_centers",
        "stencil_eval_total_centers",
        "stencil_eval_sampled",
        "pair_loss",
        "pair_mae",
        "pair_rmse",
        "diag_pair_mae",
        "diag_pair_rmse",
        "near_diag_mae",
        "near_diag_rmse",
        "mid_pair_mae",
        "mid_pair_rmse",
        "far_offdiag_mae",
        "far_offdiag_rmse",
        "rho_loss",
        "density_mae",
        "rho_point_mae",
        "kernel_loss",
        "kernel_diag_error",
        "deriv_loss",
        "deriv_raw_mse",
        "deriv_mae",
        "deriv_pred_ao_raw_mse",
        "deriv_pred_ao_mae",
        "deriv_fd_ao_raw_mse",
        "deriv_fd_ao_mae",
        "deriv_fd_ao_rms_ratio",
        "deriv_pred_fd_raw_mse",
        "deriv_pred_fd_mae",
        "tau_loss",
        "tau_mse_loss",
        "tau_raw_mse",
        "tau_rmse",
        "tau_rel_mse_loss",
        "tau_rel_rmse",
        "tau_mae",
        "tau_pred_ao_raw_mse",
        "tau_pred_ao_mae",
        "tau_fd_ao_raw_mse",
        "tau_fd_ao_mae",
        "tau_fd_ao_rms_ratio",
        "tau_pred_fd_raw_mse",
        "tau_pred_fd_mae",
        "stencil_gamma_huber",
        "stencil_gamma_mae",
        "stencil_gamma_rmse",
        "stencil_gamma_rel_mae",
        "stencil_gamma_rel_rmse",
        "kinetic_loss",
        "kinetic_mse_loss",
        "kinetic_pred",
        "kinetic_training_ref",
        "kinetic_ref_error",
        "kinetic_abs_error",
        "kinetic_sq_error",
        "kinetic_rmse",
        "kinetic_rel_abs_error",
        "kinetic_rel_sq_error",
        "kinetic_rel_rmse",
        "kinetic_stencil_diag_error",
        "kinetic_stencil_diag_abs_error",
        "kinetic_stencil_offdiag_error",
        "kinetic_stencil_offdiag_abs_error",
        "kinetic_stencil_total_error",
        "kinetic_stencil_reconstruction_residual",
        "kinetic_stencil_reference_gap",
        "trace_loss",
        "trace_rel_error",
        "trace_abs_rel_error",
        "tau_true_integral",
        "tau_true_fd_integral",
        "tau_fd_ao_integral_error",
        "tau_pred_integral",
        "ked_true_integral",
        "ked_true_fd_integral",
        "ked_pred_integral",
        "kinetic_energy_ref",
        "kinetic_energy_ref_error",
        "energy_total_ref",
        "energy_total_grid_ref",
        "energy_total_pred",
        "energy_total_ref_minus_pred",
        "energy_total_grid_ref_minus_pred",
        "energy_stored_minus_grid_ref",
        "energy_stored_total_available",
        "energy_total_abs_error",
        "energy_total_sq_error",
        "energy_grid_total_abs_error",
        "energy_grid_total_sq_error",
        "energy_kinetic_ref",
        "energy_kinetic_pred",
        "energy_kinetic_ref_minus_pred",
        "energy_kinetic_abs_error",
        "energy_external_ref",
        "energy_external_pred",
        "energy_external_ref_minus_pred",
        "energy_external_abs_error",
        "energy_hartree_ref",
        "energy_hartree_pred",
        "energy_hartree_ref_minus_pred",
        "energy_hartree_abs_error",
        "energy_xc_lda_ref",
        "energy_xc_lda_pred",
        "energy_xc_lda_ref_minus_pred",
        "energy_xc_lda_abs_error",
        "energy_ion_ion_ref",
        "energy_ion_ion_pred",
        "energy_ion_ion_ref_minus_pred",
        "energy_ion_ion_abs_error",
        "occ_penalty",
        "symmetry_mae",
        "near_diag_mae",
        "far_offdiag_mae",
        "curvature_target_min",
        "curvature_target_p05",
        "curvature_target_p50",
        "curvature_target_p95",
        "curvature_target_max",
        "curvature_target_neg_frac",
    ]
    scalar_keys.extend(
        key
        for key in ("rho_cation_mae", "rho_anion_mae", "fukui_plus_mae", "fukui_minus_mae")
        if key in per_system[0]
    )
    averages = {}
    for key in scalar_keys:
        values = np.asarray([entry[key] for entry in per_system], dtype=np.float64)
        averages[key] = float(np.nan) if np.all(np.isnan(values)) else float(np.nanmean(values))
    for src_key, dst_key in (
        ("kinetic_abs_error", "kinetic_abs_error_p90"),
        ("kinetic_rel_abs_error", "kinetic_rel_abs_error_p90"),
    ):
        values = np.asarray([entry[src_key] for entry in per_system], dtype=np.float64)
        averages[dst_key] = float(np.nan) if np.all(np.isnan(values)) else float(np.nanpercentile(values, 90.0))
    for src_key, dst_key in (
        ("kinetic_sq_error", "kinetic_rmse"),
        ("kinetic_rel_sq_error", "kinetic_rel_rmse"),
        ("energy_total_sq_error", "energy_total_rmse"),
        ("energy_grid_total_sq_error", "energy_grid_total_rmse"),
    ):
        values = np.asarray([entry[src_key] for entry in per_system], dtype=np.float64)
        averages[dst_key] = float(np.nan) if np.all(np.isnan(values)) else float(np.sqrt(np.nanmean(values)))
    averages["gamma_fd_target_sources"] = sorted(
        {str(entry["gamma_fd_target_source"]) for entry in per_system}
    )
    averages["kinetic_evaluation_modes"] = sorted(
        {str(entry["kinetic_evaluation_mode"]) for entry in per_system}
    )
    averages["per_system"] = per_system
    return averages


def select_evaluation_systems(
    systems: list[SystemRecord],
    count: int,
) -> list[SystemRecord]:
    """Select a deterministic subset spanning the full split."""
    if count <= 0 or count >= len(systems):
        return systems
    indices = np.linspace(0, len(systems) - 1, num=count, dtype=np.int64)
    return [systems[int(index)] for index in indices]


def epoch_loss_weights(config: ExperimentConfig, epoch: int) -> dict[str, float]:
    return {name: scheduled_loss_weight(config, name, epoch) for name in LOSS_NAMES}


def weighted_training_objective(losses: dict[str, tf.Tensor], weights: dict[str, float]) -> tf.Tensor:
    total = tf.constant(0.0, dtype=tf.float32)
    for weight_name, loss_name in TRAIN_OBJECTIVE_TERMS:
        total = total + weights[weight_name] * losses[loss_name]
    return total


def make_compiled_train_step(
    models: ModelBundle,
    optimizer: tf.keras.optimizers.Optimizer,
    vars_all: list[tf.Variable],
    config: ExperimentConfig,
):
    """Compile the dense TF part of one training step.

    Python still samples indices and prepares immutable features, but the model
    forward/loss/backward/update path runs as one graph call.
    """

    @tf.function(reduce_retracing=True)
    def train_step(
        pair_point_r: tf.Tensor,
        pair_point_rp: tf.Tensor,
        pair_feat: tf.Tensor,
        pair_density_feat: tf.Tensor,
        pair_rho_r: tf.Tensor,
        pair_rho_rp: tf.Tensor,
        gamma_true: tf.Tensor,
        pair_weights: tf.Tensor,
        diag_point_feat: tf.Tensor,
        diag_pair_feat: tf.Tensor,
        diag_density_feat: tf.Tensor,
        diag_rho: tf.Tensor,
        rho_target: tf.Tensor,
        trace_multiplier: tf.Tensor,
        stencil_point_l: tf.Tensor,
        stencil_point_r: tf.Tensor,
        stencil_pair_feat: tf.Tensor,
        stencil_density_feat: tf.Tensor,
        stencil_rho_l: tf.Tensor,
        stencil_rho_r: tf.Tensor,
        stencil_gamma_true: tf.Tensor,
        derivative_target: tf.Tensor,
        tau_target: tf.Tensor,
        global_context: tf.Tensor,
        basis_scale: tf.Tensor,
        step_size: tf.Tensor,
        cell_volume: tf.Tensor,
        electron_count: tf.Tensor,
        kinetic_ref: tf.Tensor,
        kinetic_scale: tf.Tensor,
        kinetic_target_integral: tf.Tensor,
        kinetic_multiplier: tf.Tensor,
        kinetic_prefactor_t: tf.Tensor,
        stencil_order: int,
        gamma_weight: tf.Tensor,
        rho_weight: tf.Tensor,
        kernel_weight: tf.Tensor,
        deriv_weight: tf.Tensor,
        tau_weight: tf.Tensor,
        stencil_gamma_weight: tf.Tensor,
        trace_weight: tf.Tensor,
        occ_weight: tf.Tensor,
        mode_weight: tf.Tensor,
        kinetic_weight: tf.Tensor,
        tau_mse_weight: tf.Tensor,
        kinetic_mse_weight: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        del occ_weight
        with tf.GradientTape() as tape:
            pair_outputs = predict_from_features(
                pair_point_r,
                pair_point_rp,
                pair_feat,
                global_context,
                models,
                rho_r_override=pair_rho_r,
                rho_rp_override=pair_rho_rp,
                pair_density_feat_t=pair_density_feat,
                local_curvature_basis_scale=basis_scale,
            )
            pair_loss = weighted_mse(gamma_true, pair_outputs["gamma"], pair_weights)

            diag_outputs = predict_from_features(
                diag_point_feat,
                diag_point_feat,
                diag_pair_feat,
                global_context,
                models,
                rho_r_override=diag_rho,
                rho_rp_override=diag_rho,
                pair_density_feat_t=diag_density_feat,
                local_curvature_basis_scale=basis_scale,
            )
            gamma_diag = diag_outputs["gamma"]
            rho_loss = tf.reduce_mean(tf.square(gamma_diag - rho_target))
            kernel_loss = tf.reduce_mean(tf.square(diag_outputs["kernel"] - 1.0))
            trace_pred = tf.reduce_sum(gamma_diag) * trace_multiplier * cell_volume
            trace_scale = tf.maximum(electron_count, 1.0)
            trace_loss = tf.square((trace_pred - electron_count) / trace_scale)

            zero = tf.constant(0.0, dtype=tf.float32)

            def physics_losses() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
                stencil_outputs = predict_from_features(
                    stencil_point_l,
                    stencil_point_r,
                    stencil_pair_feat,
                    global_context,
                    models,
                    rho_r_override=stencil_rho_l,
                    rho_rp_override=stencil_rho_r,
                    pair_density_feat_t=stencil_density_feat,
                    local_curvature_basis_scale=basis_scale,
                )
                stencil_gamma_loss_t = density_normalized_huber(
                    stencil_gamma_true,
                    stencil_outputs["gamma"],
                    stencil_rho_l,
                    stencil_rho_r,
                    scale_floor=config.stencil_gamma_scale_floor,
                    delta=config.physics_huber_delta,
                )
                n_centers = tf.shape(derivative_target)[0]
                gamma_stencil = tf.reshape(stencil_outputs["gamma"], (n_centers, 3, stencil_order))
                d_h = (
                    gamma_stencil[:, :, 0]
                    - gamma_stencil[:, :, 1]
                    - gamma_stencil[:, :, 2]
                    + gamma_stencil[:, :, 3]
                ) / (4.0 * step_size * step_size)
                if stencil_order >= 8:
                    d_2h = (
                        gamma_stencil[:, :, 4]
                        - gamma_stencil[:, :, 5]
                        - gamma_stencil[:, :, 6]
                        + gamma_stencil[:, :, 7]
                    ) / (16.0 * step_size * step_size)
                    derivative_pred = (4.0 * d_h - d_2h) / 3.0
                else:
                    derivative_pred = d_h
                tau_pred = kinetic_prefactor_t * tf.reduce_sum(derivative_pred, axis=1, keepdims=True)
                deriv_loss_t = rms_normalized_huber(
                    derivative_target,
                    derivative_pred,
                    scale_floor=config.deriv_scale_floor,
                    delta=config.physics_huber_delta,
                )
                tau_loss_t = rms_normalized_huber(
                    tau_target,
                    tau_pred,
                    scale_floor=config.tau_scale_floor,
                    delta=config.physics_huber_delta,
                )
                tau_mse_loss_t = rms_normalized_mse(
                    tau_target,
                    tau_pred,
                    scale_floor=config.tau_scale_floor,
                )
                if config.kinetic_control_variate:
                    kinetic_pred = kinetic_target_integral + (
                        tf.reduce_sum(tau_pred - tau_target) * kinetic_multiplier * cell_volume
                    )
                else:
                    kinetic_pred = tf.reduce_sum(tau_pred) * kinetic_multiplier * cell_volume
                kinetic_loss_t = tf.square((kinetic_pred - kinetic_ref) / kinetic_scale)
                kinetic_mse_loss_t = kinetic_loss_t
                return (
                    deriv_loss_t,
                    tau_loss_t,
                    kinetic_loss_t,
                    tau_mse_loss_t,
                    kinetic_mse_loss_t,
                    stencil_gamma_loss_t,
                )

            deriv_loss, tau_loss, kinetic_loss, tau_mse_loss, kinetic_mse_loss, stencil_gamma_loss = tf.cond(
                tf.shape(derivative_target)[0] > 0,
                physics_losses,
                lambda: (zero, zero, zero, zero, zero, zero),
            )
            mode_reg = tf.reduce_mean(pair_outputs["mode_weights"])

            total_loss = (
                gamma_weight * pair_loss
                + rho_weight * rho_loss
                + kernel_weight * kernel_loss
                + deriv_weight * deriv_loss
                + tau_weight * tau_loss
                + stencil_gamma_weight * stencil_gamma_loss
                + trace_weight * trace_loss
                + mode_weight * mode_reg
                + kinetic_weight * kinetic_loss
                + tau_mse_weight * tau_mse_loss
                + kinetic_mse_weight * kinetic_mse_loss
            )

        grads = tape.gradient(total_loss, vars_all)
        grads_and_vars = [(grad, var) for grad, var in zip(grads, vars_all) if grad is not None]
        grads_filtered = [grad for grad, _ in grads_and_vars]
        vars_filtered = [var for _, var in grads_and_vars]
        grads_filtered, _ = tf.clip_by_global_norm(grads_filtered, 5.0)
        optimizer.apply_gradients(zip(grads_filtered, vars_filtered))
        return (
            total_loss,
            pair_loss,
            rho_loss,
            kernel_loss,
            deriv_loss,
            tau_loss,
            tau_mse_loss,
            stencil_gamma_loss,
            trace_loss,
            tf.constant(0.0, dtype=tf.float32),
            mode_reg,
            kinetic_loss,
            kinetic_mse_loss,
        )

    return train_step


def compute_training_losses(
    system: SystemRecord,
    batch: PairBatch,
    models: ModelBundle,
    config: ExperimentConfig,
    weights: dict[str, float],
    stencil_center_limit: int | None = None,
    stencil_center_indices: np.ndarray | None = None,
    diagonal_indices: np.ndarray | None = None,
) -> dict[str, tf.Tensor]:
    """Compute all train-step losses, skipping expensive inactive terms."""
    _, density_state = point_output_and_state(system, models, config)
    rho_all = density_state.rho_neutral
    pair_outputs = predict_from_features(
        to_tensor(batch.point_feat_r),
        to_tensor(batch.point_feat_rp),
        to_tensor(batch.pair_feat),
        to_tensor(batch.global_context),
        models,
        rho_r_override=gather_density(rho_all, batch.left_idx),
        rho_rp_override=gather_density(rho_all, batch.right_idx),
        pair_density_feat_t=pair_density_features(system, density_state, batch.left_idx, batch.right_idx, config),
        local_curvature_basis_scale=local_curvature_basis_scale(system, config),
    )
    pair_loss = weighted_mse(to_tensor(batch.gamma_true), pair_outputs["gamma"], to_tensor(batch.weights))

    diag_outputs = diagonal_predictions(
        system,
        models,
        config,
        rho_all=rho_all,
        density_state=density_state,
        diag_indices=diagonal_indices,
    )
    gamma_diag = diag_outputs["gamma"]
    rho_target = system.rho_diag if diagonal_indices is None else system.rho_diag[np.asarray(diagonal_indices, dtype=np.int64)]
    rho_loss = tf.reduce_mean(tf.square(gamma_diag - to_tensor(rho_target)))
    kernel_loss = tf.reduce_mean(tf.square(diag_outputs["kernel"] - 1.0))

    zero = tf.constant(0.0, dtype=tf.float32)
    if (
        weights["deriv"] != 0.0
        or weights["tau"] != 0.0
        or weights["tau_mse"] != 0.0
        or weights["stencil_gamma"] != 0.0
        or weights["kinetic"] != 0.0
        or weights["kinetic_mse"] != 0.0
    ):
        derivative_pred, tau_pred, gamma_stencil_pred = stencil_predictions(
            system,
            models,
            config,
            rho_all=rho_all,
            density_state=density_state,
            max_centers=stencil_center_limit,
            center_indices=stencil_center_indices,
            return_gamma_stencil=True,
        )
        derivative_target, tau_target = physics_stencil_targets(system, config)
        selected_stencil_indices = None
        if stencil_center_indices is not None:
            stencil_center_indices = np.asarray(stencil_center_indices, dtype=np.int64)
            selected_stencil_indices = stencil_center_indices
            derivative_target = derivative_target[stencil_center_indices]
            tau_target = tau_target[stencil_center_indices]
        elif stencil_center_limit is not None and int(stencil_center_limit) > 0:
            selected_stencil_indices = select_stencil_center_indices(system, stencil_center_limit)
            if selected_stencil_indices is not None:
                derivative_target = derivative_target[selected_stencil_indices]
                tau_target = tau_target[selected_stencil_indices]
            else:
                derivative_target = derivative_target[: int(stencil_center_limit)]
                tau_target = tau_target[: int(stencil_center_limit)]
        if selected_stencil_indices is None:
            stencil_indices_for_gamma = np.arange(int(system.stencil_left.shape[0]), dtype=np.int64)
        else:
            stencil_indices_for_gamma = np.asarray(selected_stencil_indices, dtype=np.int64)
        stencil_left = system.stencil_left[stencil_indices_for_gamma].reshape(-1)
        stencil_right = system.stencil_right[stencil_indices_for_gamma].reshape(-1)
        stencil_gamma_true = system.gamma_values(stencil_left, stencil_right).reshape(-1, 1)
        stencil_gamma_loss = density_normalized_huber(
            to_tensor(stencil_gamma_true),
            tf.reshape(gamma_stencil_pred, (-1, 1)),
            gather_density(rho_all, stencil_left),
            gather_density(rho_all, stencil_right),
            scale_floor=config.stencil_gamma_scale_floor,
            delta=config.physics_huber_delta,
        )
        deriv_loss = rms_normalized_huber(
            to_tensor(derivative_target),
            derivative_pred,
            scale_floor=config.deriv_scale_floor,
            delta=config.physics_huber_delta,
        )
        tau_loss = rms_normalized_huber(
            to_tensor(tau_target),
            tau_pred,
            scale_floor=config.tau_scale_floor,
            delta=config.physics_huber_delta,
        )
        tau_mse_loss = rms_normalized_mse(
            to_tensor(tau_target),
            tau_pred,
            scale_floor=config.tau_scale_floor,
        )
        if stencil_center_indices is not None:
            n_sampled_centers = max(int(len(stencil_center_indices)), 1)
            kinetic_multiplier = float(system.stencil_left.shape[0]) / float(n_sampled_centers)
        elif stencil_center_limit is not None and int(stencil_center_limit) > 0:
            n_sampled_centers = max(min(int(stencil_center_limit), int(system.stencil_left.shape[0])), 1)
            kinetic_multiplier = float(system.stencil_left.shape[0]) / float(n_sampled_centers)
        else:
            kinetic_multiplier = 1.0
        kinetic_loss, _, _ = kinetic_energy_loss_from_tau(
            system,
            tau_pred,
            config=config,
            integration_multiplier=kinetic_multiplier,
            tau_target=to_tensor(tau_target),
            target_integral=physics_tau_integral(system, config),
            control_variate=config.kinetic_control_variate,
        )
        kinetic_mse_loss = kinetic_loss
    else:
        deriv_loss = zero
        tau_loss = zero
        tau_mse_loss = zero
        stencil_gamma_loss = zero
        kinetic_loss = zero
        kinetic_mse_loss = zero

    if diagonal_indices is None:
        trace_pred = tf.reduce_sum(gamma_diag) * system.cell_volume
    else:
        trace_pred = tf.reduce_mean(gamma_diag) * float(len(system.points)) * system.cell_volume
    trace_scale = max(system.electron_count, 1.0)
    trace_loss = tf.square((trace_pred - system.electron_count) / trace_scale)

    if weights["occ"] != 0.0:
        occ_penalty, _ = spectral_occupation_penalty(
            system, models, config, rho_all=rho_all, density_state=density_state
        )
    else:
        occ_penalty = zero
    mode_reg = tf.reduce_mean(pair_outputs["mode_weights"]) if weights["mode"] != 0.0 else zero
    return {
        "pair_loss": pair_loss,
        "rho_loss": rho_loss,
        "kernel_loss": kernel_loss,
        "deriv_loss": deriv_loss,
        "tau_loss": tau_loss,
        "tau_mse_loss": tau_mse_loss,
        "stencil_gamma_loss": stencil_gamma_loss,
        "trace_loss": trace_loss,
        "occ_penalty": occ_penalty,
        "mode_reg": mode_reg,
        "kinetic_loss": kinetic_loss,
        "kinetic_mse_loss": kinetic_mse_loss,
    }


def gradient_global_norm(grads: list[tf.Tensor | None]) -> float:
    non_null = [grad for grad in grads if grad is not None]
    if not non_null:
        return 0.0
    return float(tf.linalg.global_norm(non_null).numpy())


def tensor_rms(value: tf.Tensor) -> float:
    return float(tf.sqrt(tf.reduce_mean(tf.square(value))).numpy())


def print_gradient_diagnostics(
    system: SystemRecord,
    batch: PairBatch,
    models: ModelBundle,
    config: ExperimentConfig,
    weights: dict[str, float],
) -> None:
    """Print loss-specific gradient paths on one fixed train-system batch."""
    variable_groups = {
        "point": models.point_model.trainable_variables,
        "mode": models.mode_model.trainable_variables,
        "pair": models.pair_model.trainable_variables,
        "context": models.context_model.trainable_variables,
    }
    all_vars = [var for variables in variable_groups.values() for var in variables]
    diagnostic_weights = dict(weights)
    diagnostic_weights["deriv"] = 1.0
    diagnostic_weights["tau"] = 1.0
    diagnostic_weights["tau_mse"] = 1.0
    diagnostic_weights["kinetic"] = 1.0
    diagnostic_weights["kinetic_mse"] = 1.0
    diagnostic_stencil_centers = int(config.gradient_diagnostic_stencil_centers)
    diagnostic_center_indices = select_stencil_center_indices(
        system,
        diagnostic_stencil_centers,
        rng=np.random.default_rng(config.seed + 4401),
    )
    diagnostic_mode = str(config.gradient_diagnostic_mode).strip().lower()
    if diagnostic_mode not in {"fast", "full"}:
        raise ValueError("RDM_GRADIENT_DIAGNOSTIC_MODE must be 'fast' or 'full'.")
    diagnostic_diagonal_indices = select_diagonal_indices(
        system,
        config.train_diagonal_points,
        rng=np.random.default_rng(config.seed + 4402),
    )
    diagnostic_derivative_fd, diagnostic_tau_fd, diagnostic_gamma_fd_source = gamma_stencil_targets(
        system,
        center_indices=diagnostic_center_indices,
    )

    with tf.GradientTape(persistent=True) as tape:
        losses = compute_training_losses(
            system,
            batch,
            models,
            config,
            diagnostic_weights,
            stencil_center_limit=diagnostic_stencil_centers,
            stencil_center_indices=diagnostic_center_indices,
            diagonal_indices=diagnostic_diagonal_indices,
        )

    rows: list[tuple[str, str]] = [
        ("system", system.system_id),
        ("mode", diagnostic_mode),
        (
            "diagnostic diagonal points",
            (
                "full"
                if diagnostic_diagonal_indices is None
                else f"{len(diagnostic_diagonal_indices)}/{len(system.points)}"
            ),
        ),
        (
            "diagnostic stencil centers",
            f"{min(diagnostic_stencil_centers, int(system.stencil_left.shape[0]))}/{int(system.stencil_left.shape[0])}",
        ),
        ("local curvature basis scale", f"{local_curvature_basis_scale(system, config):.6e}"),
        ("local curvature sigma", f"{config.local_curvature_sigma:.6g} normalized domain units"),
        ("gamma-FD diagnostic source", diagnostic_gamma_fd_source),
        (
            "scheduled weights gamma/deriv/tau/tau_mse/kinetic/kinetic_mse",
            (
                f"{weights['gamma']:.3e} / {weights['deriv']:.3e} / "
                f"{weights['tau']:.3e} / {weights['tau_mse']:.3e} / "
                f"{weights['kinetic']:.3e} / {weights['kinetic_mse']:.3e}"
            ),
        ),
        ("kinetic control variate", config.kinetic_control_variate),
        (
            "T reference loss-target/physics-target/orbital",
            (
                f"{kinetic_energy_reference(system, config):.6e} / "
                f"{physics_tau_integral(system, config):.6e} / "
                f"{float(np.sum(system.tau_true, dtype=np.float64) * system.cell_volume):.6e}"
            ),
        ),
    ]
    effective_norms = {}
    for label, loss_name, weight_name in (
        ("gamma", "pair_loss", "gamma"),
        ("deriv", "deriv_loss", "deriv"),
        ("tau", "tau_loss", "tau"),
        ("kinetic", "kinetic_loss", "kinetic"),
    ):
        raw_total = gradient_global_norm(tape.gradient(losses[loss_name], all_vars))
        effective_total = raw_total * weights[weight_name]
        effective_norms[label] = effective_total
        rows.extend(
            [
                (f"{label} loss", f"{float(losses[loss_name].numpy()):.6e}"),
                (f"{label} grad raw/effective", f"{raw_total:.6e} / {effective_total:.6e}"),
            ]
        )
        if diagnostic_mode == "full":
            group_norms = {
                group_name: gradient_global_norm(tape.gradient(losses[loss_name], variables)) if variables else 0.0
                for group_name, variables in variable_groups.items()
            }
            rows.append(
                (
                    f"{label} grad point/mode/pair/context",
                    " / ".join(f"{group_norms[name]:.3e}" for name in ("point", "mode", "pair", "context")),
                )
            )
    gamma_effective = max(effective_norms["gamma"], 1e-30)
    rows.append(
        (
            "effective grad ratio deriv/tau/kinetic vs gamma",
            (
                f"{effective_norms['deriv'] / gamma_effective:.6e} / "
                f"{effective_norms['tau'] / gamma_effective:.6e} / "
                f"{effective_norms['kinetic'] / gamma_effective:.6e}"
            ),
        )
    )
    del tape

    if diagnostic_mode == "fast":
        rows.extend(
            [
                ("physics target", physics_target_mode(config)),
                ("RMS diagnostics", "skipped in fast mode"),
            ]
        )
        print_block("Gradient diagnostics", rows)
        return

    _, density_state = point_output_and_state(system, models, config)
    derivative_pred, tau_pred = stencil_predictions(
        system,
        models,
        config,
        rho_all=density_state.rho_neutral,
        density_state=density_state,
        max_centers=diagnostic_stencil_centers,
        center_indices=diagnostic_center_indices,
    )
    derivative_true_fd = diagnostic_derivative_fd
    tau_true_fd = diagnostic_tau_fd
    derivative_target, tau_target = physics_stencil_targets(system, config)
    if diagnostic_center_indices is not None:
        derivative_target = derivative_target[diagnostic_center_indices]
        tau_target = tau_target[diagnostic_center_indices]
    elif diagnostic_stencil_centers > 0:
        derivative_true_fd = derivative_true_fd[:diagnostic_stencil_centers]
        tau_true_fd = tau_true_fd[:diagnostic_stencil_centers]
        derivative_target = derivative_target[:diagnostic_stencil_centers]
        tau_target = tau_target[:diagnostic_stencil_centers]
    rows.extend(
        [
            ("physics target", physics_target_mode(config)),
            (
                "deriv RMS target/pred",
                f"{np_rms(derivative_target):.6e} / {tensor_rms(derivative_pred):.6e}",
            ),
            (
                "deriv RMS gamma-FD/orbital",
                f"{np_rms(derivative_true_fd):.6e} / {np_rms(system.derivative_true):.6e}",
            ),
            (
                "tau RMS target/pred",
                f"{np_rms(tau_target):.6e} / {tensor_rms(tau_pred):.6e}",
            ),
            (
                "tau RMS gamma-FD/orbital",
                f"{np_rms(tau_true_fd):.6e} / {np_rms(system.tau_true):.6e}",
            ),
        ]
    )
    print_block("Gradient diagnostics", rows)


def train_models(
    config: ExperimentConfig,
    split: DatasetSplit,
    models: ModelBundle,
    *,
    initialize_best_from_current: bool = False,
) -> tuple[TrainingHistory, dict[str, object]]:
    """multi-system transferable training."""
    optimizer = optimizer_from_config(config)
    history = TrainingHistory()
    rng = np.random.default_rng(config.seed + 123)

    best_val_objective = np.inf
    best_weights = None
    epochs_without_improvement = 0
    epochs_since_lr_drop = 0
    best_val_for_lr = np.inf

    vars_all = trainable_variables(models)
    compile_train_step = use_compiled_train_step(config)
    compiled_train_step = (
        make_compiled_train_step(models, optimizer, vars_all, config)
        if compile_train_step
        else None
    )
    if config.compile_train_step and not compile_train_step:
        print(
            "Joint point training disables the compiled train step because "
            "precomputed density tensors would detach point-model gradients."
        )
    diagnostic_system = split.train_systems[0]
    diagnostic_batch = (
        sample_pair_batch(
            diagnostic_system,
            config,
            epoch=0,
            rng=np.random.default_rng(config.seed + 2026),
        )
        if config.gradient_diagnostics
        else None
    )
    last_stage_signature: tuple[int, ...] | None = None
    last_epoch = 0
    best_selection_epoch = fully_active_schedule_epoch(config) if initialize_best_from_current else 0
    print_block("Base loss weights", loss_weight_rows(config))
    print_block("Loss schedule", loss_schedule_rows(config))
    print_block(
        "Physics loss",
        [
            ("physics target", physics_target_mode(config)),
            ("deriv/tau loss", "target-RMS normalized Huber"),
            ("Huber delta", f"{config.physics_huber_delta:.6g}"),
            ("deriv/tau scale floor", f"{config.deriv_scale_floor:.6g} / {config.tau_scale_floor:.6g}"),
            ("kinetic control variate", config.kinetic_control_variate),
            ("compiled train step", compile_train_step),
            ("active system tensor cache", config.active_system_tensor_cache_size),
            ("train diagonal points", "full" if config.train_diagonal_points <= 0 else config.train_diagonal_points),
            ("train stencil centers", "full" if config.train_stencil_centers <= 0 else config.train_stencil_centers),
            ("validation every epochs", max(config.val_every, 1)),
            ("eval pair samples/system", config.eval_pair_count),
            ("eval stencil centers", "full" if config.eval_stencil_centers <= 0 else config.eval_stencil_centers),
            ("full final eval", config.eval_full_final),
            (
                "periodic val systems",
                "full" if config.val_eval_system_count <= 0 else config.val_eval_system_count,
            ),
            (
                "final train/val/test systems",
                " / ".join(
                    "full" if count <= 0 else str(count)
                    for count in (
                        config.final_train_eval_system_count,
                        config.final_val_eval_system_count,
                        config.final_test_eval_system_count,
                    )
                ),
            ),
            (
                "stencil feature cache centers",
                "disabled" if config.stencil_feature_cache_max_centers <= 0 else config.stencil_feature_cache_max_centers,
            ),
            ("diagonal prediction chunk", config.diagonal_prediction_chunk_size),
            ("kinetic integral active", loss_enabled(config, "kinetic")),
        ],
    )
    print_block(
        "Density constraint",
        [
            ("density source", density_source_mode(config)),
            ("normalize_rho", config.normalize_rho),
            ("frozen density-state cache", config.freeze_point_after_pretrain),
        ],
    )
    print_block(
        "Gradient diagnostics",
        [
            ("enabled", config.gradient_diagnostics),
            ("every epochs", max(config.gradient_diagnostics_every, 1)),
            ("start epoch", max(config.gradient_diagnostics_start_epoch, 0)),
            ("mode", config.gradient_diagnostic_mode),
            ("fixed train system", diagnostic_system.system_id),
            ("stencil centers", config.gradient_diagnostic_stencil_centers),
        ],
    )

    if initialize_best_from_current:
        initial_weights = epoch_loss_weights(config, best_selection_epoch)
        initial_val_systems = select_evaluation_systems(
            split.val_systems,
            config.val_eval_system_count,
        )
        clear_gpu_evaluation_caches()
        initial_val = evaluate_systems(
            initial_val_systems,
            models,
            config,
            epoch=best_selection_epoch,
        )
        clear_gpu_evaluation_caches()
        best_val_objective = float(initial_val["objective"])
        best_val_for_lr = best_val_objective
        best_weights = {
            "point": models.point_model.get_weights(),
            "mode": models.mode_model.get_weights(),
            "pair": models.pair_model.get_weights(),
            "context": models.context_model.get_weights(),
        }
        print_block(
            "Fine-tune baseline",
            [
                ("validation systems", len(initial_val_systems)),
                ("objective", f"{best_val_objective:.6e}"),
                ("tau MAE", f"{initial_val['tau_mae']:.6e}"),
                ("kinetic abs error", f"{initial_val['kinetic_abs_error']:.6e} Ha"),
                ("best-selection starts", f"epoch {best_selection_epoch}"),
                ("active losses", active_loss_summary(initial_weights)),
            ],
        )

    for epoch in range(config.epochs):
        last_epoch = epoch
        weights = epoch_loss_weights(config, epoch)
        stage_signature = loss_stage_signature(config, epoch)
        if (
            not initialize_best_from_current
            and last_stage_signature is not None
            and stage_signature != last_stage_signature
        ):
            best_val_objective = np.inf
            best_weights = None
            best_val_for_lr = np.inf
            epochs_without_improvement = 0
            epochs_since_lr_drop = 0
            print(f"Epoch {epoch:4d} | loss schedule changed: {active_loss_summary(weights)}")
        last_stage_signature = stage_signature

        running_total = 0.0
        running_components = {key: 0.0 for key in TRAIN_HISTORY_KEYS}

        for _ in range(config.steps_per_epoch):
            system = choose_system(split.train_systems, rng)
            batch = sample_pair_batch(system, config, epoch, rng)
            stencil_center_indices = (
                select_stencil_center_indices(system, config.train_stencil_centers, rng)
                if (
                    weights["deriv"] != 0.0
                    or weights["tau"] != 0.0
                    or weights["tau_mse"] != 0.0
                    or weights["stencil_gamma"] != 0.0
                    or weights["kinetic"] != 0.0
                    or weights["kinetic_mse"] != 0.0
                )
                else None
            )
            diagonal_indices = select_diagonal_indices(system, config.train_diagonal_points, rng)

            if compiled_train_step is not None:
                _, density_state = point_output_and_state(system, models, config)
                system_t = active_system_tensors(system, config)
                left_idx_t = tf.convert_to_tensor(batch.left_idx, dtype=tf.int64)
                right_idx_t = tf.convert_to_tensor(batch.right_idx, dtype=tf.int64)

                diag_idx = (
                    np.arange(len(system.points), dtype=np.int64)
                    if diagonal_indices is None
                    else np.asarray(diagonal_indices, dtype=np.int64)
                )
                diag_idx_t = tf.convert_to_tensor(diag_idx, dtype=tf.int64)
                trace_multiplier = float(len(system.points)) / max(float(len(diag_idx)), 1.0)

                physics_active = (
                    weights["deriv"] != 0.0
                    or weights["tau"] != 0.0
                    or weights["tau_mse"] != 0.0
                    or weights["stencil_gamma"] != 0.0
                    or weights["kinetic"] != 0.0
                    or weights["kinetic_mse"] != 0.0
                )
                if physics_active:
                    stencil_idx = (
                        np.arange(int(system.stencil_left.shape[0]), dtype=np.int64)
                        if stencil_center_indices is None
                        else np.asarray(stencil_center_indices, dtype=np.int64)
                    )
                    stencil_left = system.stencil_left[stencil_idx].reshape(-1)
                    stencil_right = system.stencil_right[stencil_idx].reshape(-1)
                    stencil_gamma_true = system.gamma_values(stencil_left, stencil_right).reshape(-1, 1)
                    stencil_pair_feat = stencil_pair_features_for_centers(system, stencil_idx, config)
                    derivative_target, tau_target = physics_stencil_targets(system, config)
                    derivative_target = derivative_target[stencil_idx]
                    tau_target = tau_target[stencil_idx]
                    kinetic_multiplier = float(system.stencil_left.shape[0]) / max(float(len(stencil_idx)), 1.0)
                else:
                    stencil_left = np.zeros((0,), dtype=np.int64)
                    stencil_right = np.zeros((0,), dtype=np.int64)
                    stencil_gamma_true = np.zeros((0, 1), dtype=np.float32)
                    stencil_pair_feat = np.zeros((0, batch.pair_feat.shape[1]), dtype=np.float32)
                    derivative_target = np.zeros((0, 3), dtype=np.float32)
                    tau_target = np.zeros((0, 1), dtype=np.float32)
                    kinetic_multiplier = 1.0
                stencil_left_t = tf.convert_to_tensor(stencil_left, dtype=tf.int64)
                stencil_right_t = tf.convert_to_tensor(stencil_right, dtype=tf.int64)
                kinetic_ref = kinetic_energy_reference(system, config)

                loss_values = compiled_train_step(
                    tf.gather(system_t.local_features, left_idx_t),
                    tf.gather(system_t.local_features, right_idx_t),
                    to_tensor(batch.pair_feat),
                    pair_density_features(system, density_state, batch.left_idx, batch.right_idx, config),
                    tf.gather(density_state.rho_neutral, left_idx_t),
                    tf.gather(density_state.rho_neutral, right_idx_t),
                    to_tensor(batch.gamma_true),
                    to_tensor(batch.weights),
                    tf.gather(system_t.local_features, diag_idx_t),
                    to_tensor(build_pair_features(system, diag_idx, diag_idx)),
                    pair_density_features(system, density_state, diag_idx, diag_idx, config),
                    tf.gather(density_state.rho_neutral, diag_idx_t),
                    tf.gather(system_t.rho_diag, diag_idx_t),
                    tf.constant(trace_multiplier, dtype=tf.float32),
                    tf.gather(system_t.local_features, stencil_left_t),
                    tf.gather(system_t.local_features, stencil_right_t),
                    to_tensor(stencil_pair_feat),
                    pair_density_features(system, density_state, stencil_left, stencil_right, config),
                    tf.gather(density_state.rho_neutral, stencil_left_t),
                    tf.gather(density_state.rho_neutral, stencil_right_t),
                    to_tensor(stencil_gamma_true),
                    to_tensor(derivative_target),
                    to_tensor(tau_target),
                    system_t.global_context,
                    tf.constant(local_curvature_basis_scale(system, config), dtype=tf.float32),
                    tf.constant(float(system.step), dtype=tf.float32),
                    tf.constant(float(system.cell_volume), dtype=tf.float32),
                    tf.constant(float(system.electron_count), dtype=tf.float32),
                    tf.constant(float(kinetic_ref), dtype=tf.float32),
                    tf.constant(float(max(abs(kinetic_ref), 1.0)), dtype=tf.float32),
                    tf.constant(float(physics_tau_integral(system, config)), dtype=tf.float32),
                    tf.constant(float(kinetic_multiplier), dtype=tf.float32),
                    tf.constant(float(kinetic_prefactor(system)), dtype=tf.float32),
                    int(system.stencil_left.shape[2]),
                    tf.constant(float(weights["gamma"]), dtype=tf.float32),
                    tf.constant(float(weights["rho"]), dtype=tf.float32),
                    tf.constant(float(weights["kernel"]), dtype=tf.float32),
                    tf.constant(float(weights["deriv"]), dtype=tf.float32),
                    tf.constant(float(weights["tau"]), dtype=tf.float32),
                    tf.constant(float(weights["stencil_gamma"]), dtype=tf.float32),
                    tf.constant(float(weights["trace"]), dtype=tf.float32),
                    tf.constant(float(weights["occ"]), dtype=tf.float32),
                    tf.constant(float(weights["mode"]), dtype=tf.float32),
                    tf.constant(float(weights["kinetic"]), dtype=tf.float32),
                    tf.constant(float(weights["tau_mse"]), dtype=tf.float32),
                    tf.constant(float(weights["kinetic_mse"]), dtype=tf.float32),
                )
                running_total += float(loss_values[0].numpy())
                for key, value in zip(TRAIN_HISTORY_KEYS, loss_values[1:]):
                    running_components[key] += float(value.numpy())
            else:
                with tf.GradientTape() as tape:
                    losses = compute_training_losses(
                        system,
                        batch,
                        models,
                        config,
                        weights,
                        stencil_center_limit=config.train_stencil_centers,
                        stencil_center_indices=stencil_center_indices,
                        diagonal_indices=diagonal_indices,
                    )
                    total_loss = weighted_training_objective(losses, weights)

                grads = tape.gradient(total_loss, vars_all)
                grads_and_vars = [(g, v) for g, v in zip(grads, vars_all) if g is not None]
                grads_filtered = [g for g, _ in grads_and_vars]
                vars_filtered = [v for _, v in grads_and_vars]
                grads_filtered, _ = tf.clip_by_global_norm(grads_filtered, 5.0)
                optimizer.apply_gradients(zip(grads_filtered, vars_filtered))
                running_total += float(total_loss.numpy())
                for key in TRAIN_HISTORY_KEYS:
                    running_components[key] += float(losses[key].numpy())

        train_objective = running_total / max(config.steps_per_epoch, 1)
        history.train_objective.append(train_objective)
        history.learning_rate.append(float(optimizer.learning_rate.numpy()))
        history.kinetic_weight.append(weights["kinetic"])
        history.validation_ran.append(0)
        for key in LOSS_NAMES:
            history.loss_weights[key].append(weights[key])
        for key in TRAIN_HISTORY_KEYS:
            history.train_components[key].append(running_components[key] / max(config.steps_per_epoch, 1))

        validation_ran = epoch % config.val_every == 0 or epoch == config.epochs - 1
        if validation_ran:
            periodic_val_systems = select_evaluation_systems(
                split.val_systems,
                config.val_eval_system_count,
            )
            val_metrics = evaluate_systems(periodic_val_systems, models, config, epoch=epoch)
            val_objective = float(val_metrics["objective"])
            history.validation_ran[-1] = 1
            for key in VAL_HISTORY_KEYS:
                history.val_components[key].append(float(val_metrics[key]))
        else:
            val_metrics = {}
            val_objective = history.val_objective[-1] if history.val_objective else np.inf
            for key in VAL_HISTORY_KEYS:
                history.val_components[key].append(float("nan"))

        history.val_objective.append(val_objective)

        if (
            validation_ran
            and config.gradient_diagnostics
            and epoch >= max(config.gradient_diagnostics_start_epoch, 0)
            and epoch % max(config.gradient_diagnostics_every, 1) == 0
        ):
            assert diagnostic_batch is not None
            print_gradient_diagnostics(diagnostic_system, diagnostic_batch, models, config, weights)

        selection_eligible = validation_ran and epoch >= best_selection_epoch
        if selection_eligible:
            if val_objective < best_val_objective - 1e-9:
                best_val_objective = val_objective
                best_weights = {
                    "point": models.point_model.get_weights(),
                    "mode": models.mode_model.get_weights(),
                    "pair": models.pair_model.get_weights(),
                    "context": models.context_model.get_weights(),
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += config.val_every

            if val_objective < best_val_for_lr - config.lr_min_improvement:
                best_val_for_lr = val_objective
                epochs_since_lr_drop = 0
            else:
                epochs_since_lr_drop += config.val_every

        current_lr = float(optimizer.learning_rate.numpy())
        if validation_ran and epochs_since_lr_drop >= config.lr_patience:
            new_lr = max(config.min_lr, current_lr * config.lr_decay)
            if new_lr < current_lr - 1e-15:
                optimizer.learning_rate.assign(new_lr)
                print(f"Epoch {epoch:4d} | reducing learning rate: {current_lr:.6e} -> {new_lr:.6e}")
            epochs_since_lr_drop = 0

        log_every = max(int(config.log_every), 1)
        if epoch % log_every == 0 or epoch == config.epochs - 1:
            print(
                f"Epoch {epoch:4d} | "
                f"train obj = {train_objective:.6e} | "
                f"val obj = {val_objective:.6e} | "
                f"lr = {float(optimizer.learning_rate.numpy()):.3e} | "
                f"w(T)={weights['kinetic']:.3g}"
            )
            if val_metrics:
                print(
                    " " * 14
                    + f"held-out systems={len(val_metrics['per_system'])} "
                    + f"stencil eval centers={val_metrics['stencil_eval_centers']:.0f}/"
                    + f"{val_metrics['stencil_eval_total_centers']:.0f} "
                    + f"sampled={bool(val_metrics['stencil_eval_sampled'])} "
                    + f"kinetic={','.join(val_metrics['kinetic_evaluation_modes'])}"
                )
                print(
                    " " * 14
                    + f"held-out mae gamma_pair={val_metrics['pair_mae']:.3e} "
                    + f"rho={val_metrics['density_mae']:.3e} "
                    + f"deriv={val_metrics['deriv_mae']:.3e} "
                    + f"tau={val_metrics['tau_mae']:.3e}"
                )
                print(
                    " " * 14
                    + f"held-out rms T={val_metrics['kinetic_rmse']:.3e} "
                    + f"T_rel={val_metrics['kinetic_rel_rmse']:.3e} "
                    + f"tau={val_metrics['tau_rmse']:.3e} "
                    + f"tau_rel={val_metrics['tau_rel_rmse']:.3e}"
                )
                print(
                    " " * 14
                    + f"held-out loss gamma_pair={val_metrics['pair_loss']:.3e} "
                    + f"rho={val_metrics['rho_loss']:.3e} "
                    + f"T={val_metrics['kinetic_loss']:.3e} "
                    + f"T_mse={val_metrics['kinetic_mse_loss']:.3e} "
                    + f"stencil_gamma={val_metrics['stencil_gamma_huber']:.3e} "
                    + f"deriv_huber={val_metrics['deriv_loss']:.3e} "
                    + f"tau_huber={val_metrics['tau_loss']:.3e} "
                    + f"tau_mse={val_metrics['tau_mse_loss']:.3e} "
                    + f"trace_rel={val_metrics['trace_abs_rel_error']:.3e} "
                    + f"occ_pen={val_metrics['occ_penalty']:.3e}"
                )
                print(
                    " " * 14
                    + f"held-out raw mse deriv={val_metrics['deriv_raw_mse']:.3e} "
                    + f"tau={val_metrics['tau_raw_mse']:.3e} "
                    + f"gamma_rmse={val_metrics['pair_rmse']:.3e} "
                    + f"stencil_gamma_rmse={val_metrics['stencil_gamma_rmse']:.3e}"
                )
                print(
                    " " * 14
                    + f"held-out gamma-FD-vs-orbital tau_mae={val_metrics['tau_fd_ao_mae']:.3e} "
                    + f"pred-vs-FD tau_mae={val_metrics['tau_pred_fd_mae']:.3e} "
                    + f"tau_fd/orbital_rms={val_metrics['tau_fd_ao_rms_ratio']:.3e} "
                    + f"source={','.join(val_metrics['gamma_fd_target_sources'])}"
                )
                if val_metrics["energy_stored_total_available"] >= 1.0:
                    print(
                        " " * 14
                        + f"held-out E stored-GPAW minus grid-pred={val_metrics['energy_total_ref_minus_pred']:.3e} "
                        + f"MAE={val_metrics['energy_total_abs_error']:.3e} "
                        + f"RMSE={val_metrics['energy_total_rmse']:.3e} diagnostic-only"
                    )
                    print(
                        " " * 14
                        + f"held-out E stored-GPAW minus grid-ref={val_metrics['energy_stored_minus_grid_ref']:.3e}"
                    )
                else:
                    print(" " * 14 + "held-out stored GPAW total unavailable; reporting grid energy only")
                print(
                    " " * 14
                    + f"held-out E grid ref-pred={val_metrics['energy_total_grid_ref_minus_pred']:.3e} "
                    + f"T={val_metrics['energy_kinetic_ref_minus_pred']:.3e} "
                    + f"Vext={val_metrics['energy_external_ref_minus_pred']:.3e} "
                    + f"J={val_metrics['energy_hartree_ref_minus_pred']:.3e} "
                    + f"Exc={val_metrics['energy_xc_lda_ref_minus_pred']:.3e}"
                )
                print(
                    " " * 14
                    + f"held-out E ref components T={val_metrics['energy_kinetic_ref']:.3e} "
                    + f"Vext={val_metrics['energy_external_ref']:.3e} "
                    + f"J={val_metrics['energy_hartree_ref']:.3e} "
                    + f"Exc={val_metrics['energy_xc_lda_ref']:.3e} "
                    + f"Enn={val_metrics['energy_ion_ion_ref']:.3e}"
                )
                print(
                    " " * 14
                    + f"held-out E pred components T={val_metrics['energy_kinetic_pred']:.3e} "
                    + f"Vext={val_metrics['energy_external_pred']:.3e} "
                    + f"J={val_metrics['energy_hartree_pred']:.3e} "
                    + f"Exc={val_metrics['energy_xc_lda_pred']:.3e} "
                    + f"Enn={val_metrics['energy_ion_ion_pred']:.3e}"
                )

        if selection_eligible and epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_weights is not None:
        models.point_model.set_weights(best_weights["point"])
        models.mode_model.set_weights(best_weights["mode"])
        models.pair_model.set_weights(best_weights["pair"])
        models.context_model.set_weights(best_weights["context"])

    final_train_systems = select_evaluation_systems(
        split.train_systems,
        config.final_train_eval_system_count,
    )
    final_val_systems = select_evaluation_systems(
        split.val_systems,
        config.final_val_eval_system_count,
    )
    clear_gpu_evaluation_caches()
    final_train = evaluate_systems(final_train_systems, models, config, epoch=None)
    clear_gpu_evaluation_caches()
    final_val = evaluate_systems(final_val_systems, models, config, epoch=None)
    final_train["evaluated_system_count"] = len(final_train_systems)
    final_train["available_system_count"] = len(split.train_systems)
    final_val["evaluated_system_count"] = len(final_val_systems)
    final_val["available_system_count"] = len(split.val_systems)
    summary = {"train": final_train, "val": final_val}
    if split.test_systems:
        final_test_systems = select_evaluation_systems(
            split.test_systems,
            config.final_test_eval_system_count,
        )
        clear_gpu_evaluation_caches()
        summary["test"] = evaluate_systems(final_test_systems, models, config, epoch=None)
        summary["test"]["evaluated_system_count"] = len(final_test_systems)
        summary["test"]["available_system_count"] = len(split.test_systems)
    clear_gpu_evaluation_caches()
    summary["evaluation_metadata"] = {
        "density_source": (
            "true oracle"
            if density_source_mode(config) == "true"
            else density_source_mode(config)
        ),
        "kinetic_integral_active": bool(loss_enabled(config, "kinetic")),
        "eval_full_final": bool(config.eval_full_final),
    }

    rows = [
        ("train objective", f"{final_train['objective']:.6e}"),
        ("val objective", f"{final_val['objective']:.6e}"),
        ("held-out gamma_pair loss", f"{final_val['pair_loss']:.6e}"),
        ("held-out gamma_pair MAE/RMSE", f"{final_val['pair_mae']:.6e} / {final_val['pair_rmse']:.6e}"),
        ("held-out gamma diag/near/mid/far MAE", (
            f"{final_val['diag_pair_mae']:.3e} / {final_val['near_diag_mae']:.3e} / "
            f"{final_val['mid_pair_mae']:.3e} / {final_val['far_offdiag_mae']:.3e}"
        )),
        ("held-out stencil gamma Huber/MAE/RMSE", (
            f"{final_val['stencil_gamma_huber']:.6e} / "
            f"{final_val['stencil_gamma_mae']:.6e} / "
            f"{final_val['stencil_gamma_rmse']:.6e}"
        )),
        ("held-out density MAE", f"{final_val['density_mae']:.6e}"),
        ("held-out kinetic loss", f"{final_val['kinetic_loss']:.6e}"),
        ("held-out kinetic MSE loss", f"{final_val['kinetic_mse_loss']:.6e}"),
        ("held-out kinetic abs err", f"{final_val['kinetic_abs_error']:.6e}"),
        ("held-out kinetic abs err P90", f"{final_val['kinetic_abs_error_p90']:.6e}"),
        ("held-out kinetic RMSE", f"{final_val['kinetic_rmse']:.6e}"),
        ("held-out kinetic stencil diag/offdiag", (
            f"{final_val['kinetic_stencil_diag_error']:.6e} / "
            f"{final_val['kinetic_stencil_offdiag_error']:.6e}"
        )),
        ("held-out kinetic rel RMSE", f"{final_val['kinetic_rel_rmse']:.6e}"),
        (
            "held-out systems evaluated",
            f"{final_val['evaluated_system_count']} / {final_val['available_system_count']}",
        ),
        (
            "held-out stencil evaluation",
            (
                f"{final_val['stencil_eval_centers']:.0f} / "
                f"{final_val['stencil_eval_total_centers']:.0f}; "
                f"{','.join(final_val['kinetic_evaluation_modes'])}"
            ),
        ),
        ("held-out gamma-FD source", ",".join(final_val["gamma_fd_target_sources"])),
        ("held-out grid E ref-pred", f"{final_val['energy_total_grid_ref_minus_pred']:.6e}"),
        ("held-out grid E MAE", f"{final_val['energy_grid_total_abs_error']:.6e}"),
        ("held-out grid E RMSE", f"{final_val['energy_grid_total_rmse']:.6e}"),
        ("held-out E ref T/Vext/J/Exc/Enn", energy_component_summary(final_val, "ref")),
        ("held-out E pred T/Vext/J/Exc/Enn", energy_component_summary(final_val, "pred")),
        ("held-out trace rel err", f"{final_val['trace_abs_rel_error']:.6e}"),
        ("held-out tau MAE", f"{final_val['tau_mae']:.6e}"),
        ("held-out tau RMSE", f"{final_val['tau_rmse']:.6e}"),
        ("held-out tau rel MSE loss", f"{final_val['tau_rel_mse_loss']:.6e}"),
        ("held-out tau rel RMSE", f"{final_val['tau_rel_rmse']:.6e}"),
        ("held-out tau gamma-FD-vs-orbital MAE", f"{final_val['tau_fd_ao_mae']:.6e}"),
        ("held-out tau pred-vs-FD MAE", f"{final_val['tau_pred_fd_mae']:.6e}"),
        ("held-out tau gamma-FD/orbital RMS", f"{final_val['tau_fd_ao_rms_ratio']:.6e}"),
        ("held-out symmetry MAE", f"{final_val['symmetry_mae']:.6e}"),
        ("held-out kernel diag err", f"{final_val['kernel_diag_error']:.6e}"),
    ]
    if final_val["energy_stored_total_available"] >= 1.0:
        rows[9:9] = [
            (
                "held-out stored GPAW E - grid pred",
                f"{final_val['energy_total_ref_minus_pred']:.6e} (diagnostic only)",
            ),
            (
                "held-out stored GPAW E - grid ref",
                f"{final_val['energy_stored_minus_grid_ref']:.6e}",
            ),
        ]
    else:
        rows.insert(9, ("held-out stored GPAW total", "unavailable; grid energy only"))
    if "test" in summary:
        final_test = summary["test"]
        test_rows = [
                ("test objective", f"{final_test['objective']:.6e}"),
                ("test gamma_pair loss", f"{final_test['pair_loss']:.6e}"),
                ("test gamma_pair MAE/RMSE", f"{final_test['pair_mae']:.6e} / {final_test['pair_rmse']:.6e}"),
                ("test gamma diag/near/mid/far MAE", (
                    f"{final_test['diag_pair_mae']:.3e} / {final_test['near_diag_mae']:.3e} / "
                    f"{final_test['mid_pair_mae']:.3e} / {final_test['far_offdiag_mae']:.3e}"
                )),
                ("test stencil gamma Huber/MAE/RMSE", (
                    f"{final_test['stencil_gamma_huber']:.6e} / "
                    f"{final_test['stencil_gamma_mae']:.6e} / "
                    f"{final_test['stencil_gamma_rmse']:.6e}"
                )),
                ("test density MAE", f"{final_test['density_mae']:.6e}"),
                ("test kinetic loss", f"{final_test['kinetic_loss']:.6e}"),
                ("test kinetic MSE loss", f"{final_test['kinetic_mse_loss']:.6e}"),
                ("test kinetic abs err", f"{final_test['kinetic_abs_error']:.6e}"),
                ("test kinetic abs err P90", f"{final_test['kinetic_abs_error_p90']:.6e}"),
                ("test kinetic RMSE", f"{final_test['kinetic_rmse']:.6e}"),
                ("test kinetic stencil diag/offdiag", (
                    f"{final_test['kinetic_stencil_diag_error']:.6e} / "
                    f"{final_test['kinetic_stencil_offdiag_error']:.6e}"
                )),
                ("test kinetic rel RMSE", f"{final_test['kinetic_rel_rmse']:.6e}"),
                (
                    "test systems evaluated",
                    f"{final_test['evaluated_system_count']} / {final_test['available_system_count']}",
                ),
                (
                    "test stencil evaluation",
                    (
                        f"{final_test['stencil_eval_centers']:.0f} / "
                        f"{final_test['stencil_eval_total_centers']:.0f}; "
                        f"{','.join(final_test['kinetic_evaluation_modes'])}"
                    ),
                ),
                ("test gamma-FD source", ",".join(final_test["gamma_fd_target_sources"])),
                ("test grid E ref-pred", f"{final_test['energy_total_grid_ref_minus_pred']:.6e}"),
                ("test grid E MAE", f"{final_test['energy_grid_total_abs_error']:.6e}"),
                ("test grid E RMSE", f"{final_test['energy_grid_total_rmse']:.6e}"),
                ("test E ref T/Vext/J/Exc/Enn", energy_component_summary(final_test, "ref")),
                ("test E pred T/Vext/J/Exc/Enn", energy_component_summary(final_test, "pred")),
                ("test tau MAE", f"{final_test['tau_mae']:.6e}"),
                ("test tau RMSE", f"{final_test['tau_rmse']:.6e}"),
                ("test tau rel MSE loss", f"{final_test['tau_rel_mse_loss']:.6e}"),
                ("test tau rel RMSE", f"{final_test['tau_rel_rmse']:.6e}"),
                ("test tau gamma-FD-vs-orbital MAE", f"{final_test['tau_fd_ao_mae']:.6e}"),
                ("test tau pred-vs-FD MAE", f"{final_test['tau_pred_fd_mae']:.6e}"),
        ]
        if final_test["energy_stored_total_available"] >= 1.0:
            test_rows[8:8] = [
                (
                    "test stored GPAW E - grid pred",
                    f"{final_test['energy_total_ref_minus_pred']:.6e} (diagnostic only)",
                ),
                ("test stored GPAW E - grid ref", f"{final_test['energy_stored_minus_grid_ref']:.6e}"),
            ]
        else:
            test_rows.insert(8, ("test stored GPAW total", "unavailable; grid energy only"))
        rows.extend(test_rows)
    print_block("Final transferable summary", rows)
    return history, summary
