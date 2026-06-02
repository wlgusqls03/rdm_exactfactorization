from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from .config import ExperimentConfig
from .density_features import (
    DensityFeatureState,
    build_density_feature_state,
    cache_frozen_density_state,
    cached_frozen_density_state,
    clear_frozen_density_state_cache,
    density_baseline_mode,
    normalized_density_head,
    pair_density_features,
    pair_density_feature_mode,
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


LOSS_NAMES = ("gamma", "rho", "kernel", "deriv", "tau", "trace", "occ", "mode", "kinetic")
TRAIN_HISTORY_KEYS = (
    "pair_loss",
    "rho_loss",
    "kernel_loss",
    "deriv_loss",
    "tau_loss",
    "trace_loss",
    "occ_penalty",
    "mode_reg",
    "kinetic_loss",
)
VAL_HISTORY_KEYS = (
    "pair_loss",
    "pair_mae",
    "rho_loss",
    "density_mae",
    "kernel_loss",
    "kernel_diag_error",
    "deriv_loss",
    "deriv_raw_mse",
    "deriv_mae",
    "tau_loss",
    "tau_raw_mse",
    "tau_mae",
    "kinetic_loss",
    "kinetic_abs_error",
    "trace_loss",
    "trace_abs_rel_error",
    "occ_penalty",
    "symmetry_mae",
    "near_diag_mae",
    "far_offdiag_mae",
)
TRAIN_OBJECTIVE_TERMS = (
    ("gamma", "pair_loss"),
    ("rho", "rho_loss"),
    ("kernel", "kernel_loss"),
    ("deriv", "deriv_loss"),
    ("tau", "tau_loss"),
    ("trace", "trace_loss"),
    ("occ", "occ_penalty"),
    ("mode", "mode_reg"),
    ("kinetic", "kinetic_loss"),
)
EVAL_OBJECTIVE_TERMS = (
    ("gamma", "pair_loss"),
    ("rho", "rho_loss"),
    ("kernel", "kernel_loss"),
    ("deriv", "deriv_loss"),
    ("tau", "tau_loss"),
    ("trace", "trace_loss"),
    ("occ", "occ_penalty"),
    ("kinetic", "kinetic_loss"),
)


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
    total = neutral_loss + config.point_charged_weight * charged_loss + fukui_weight * fukui_loss
    return {
        "total": total,
        "neutral": neutral_loss,
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
        if density_baseline_mode(config) == "sad-multiplicative":
            if delta_diagnostics is None:
                delta_diagnostics = empty_delta_diagnostics(int(pred["delta_raw"].shape[1]))
            update_delta_diagnostics(delta_diagnostics, pred["delta_raw"], config.sad_residual_clip)
        entry: dict[str, object] = {
            "system_id": system.system_id,
            "rho_neutral_mae": float(np.mean(np.abs(pred["rho_neutral"].numpy() - system.rho_diag))),
            "rho_neutral_rel_l1": float(
                np.sum(np.abs(pred["rho_neutral"].numpy() - system.rho_diag))
                * system.cell_volume
                / max(system.electron_count, 1e-12)
            ),
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
            ("val Fukui+ MAE", f"{summary['val'].get('fukui_plus_mae', float('nan')):.6e}"),
            ("val Fukui- MAE", f"{summary['val'].get('fukui_minus_mae', float('nan')):.6e}"),
            ("point trainable in pair stage", models.point_model.trainable),
        ],
    )
    if density_baseline_mode(config) == "sad-multiplicative":
        print(f"val delta raw               : {format_delta_scalar_summary(summary['val'])}")
    return history, summary


def kinetic_energy_reference(system: SystemRecord) -> float:
    kinetic_ref = float(system.metadata.get("kinetic_energy_hartree", np.nan))
    if np.isfinite(kinetic_ref):
        return kinetic_ref
    return float(np.sum(system.tau_true) * system.cell_volume)


def kinetic_energy_loss_from_tau(system: SystemRecord, tau_pred: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, float]:
    kinetic_pred = tf.reduce_sum(tau_pred) * system.cell_volume
    kinetic_ref = kinetic_energy_reference(system)
    scale = max(abs(kinetic_ref), 1.0)
    loss = tf.square((kinetic_pred - kinetic_ref) / scale)
    return loss, kinetic_pred, kinetic_ref


def gather_density(rho_all: tf.Tensor, indices: np.ndarray) -> tf.Tensor:
    return tf.gather(rho_all, tf.convert_to_tensor(indices, dtype=tf.int64))


def diagonal_predictions(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
    rho_all: tf.Tensor | None = None,
    density_state: DensityFeatureState | None = None,
) -> dict[str, tf.Tensor]:
    """r = r' diagonal prediction."""
    if density_state is None:
        _, density_state = point_output_and_state(system, models, config)
    if rho_all is None:
        rho_all = density_state.rho_neutral
    diag_idx = np.arange(len(system.points), dtype=np.int64)
    pair_feat = build_pair_features(system, diag_idx, diag_idx)
    rho_diag = gather_density(rho_all, diag_idx)
    outputs = predict_from_features(
        to_tensor(system.local_features),
        to_tensor(system.local_features),
        to_tensor(pair_feat),
        to_tensor(system.global_context),
        models,
        rho_r_override=rho_diag,
        rho_rp_override=rho_diag,
        pair_density_feat_t=pair_density_features(system, density_state, diag_idx, diag_idx, config),
    )
    return outputs


def stencil_predictions(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
    rho_all: tf.Tensor | None = None,
    density_state: DensityFeatureState | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
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
    left_idx = system.stencil_left.reshape(-1)
    right_idx = system.stencil_right.reshape(-1)
    outputs = predict_from_features(
        to_tensor(system.local_features[left_idx]),
        to_tensor(system.local_features[right_idx]),
        to_tensor(build_pair_features(system, left_idx, right_idx)),
        to_tensor(system.global_context),
        models,
        rho_r_override=gather_density(rho_all, left_idx),
        rho_rp_override=gather_density(rho_all, right_idx),
        pair_density_feat_t=pair_density_features(system, density_state, left_idx, right_idx, config),
    )
    gamma_stencil = tf.reshape(outputs["gamma"], stencil_shape)
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
        derivative_pred = (4.0 * d_h - d_2h) / 3.0
    else:
        derivative_pred = d_h
    tau_pred = 0.5 * tf.reduce_sum(derivative_pred, axis=1, keepdims=True)
    return derivative_pred, tau_pred


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


def loss_enabled(config: ExperimentConfig, name: str) -> bool:
    preset = config.loss_preset.strip().lower()
    if preset in {"core5", "simple5"}:
        return name in {"gamma", "rho", "kernel", "trace", "mode"}
    if preset in {"staged-physics", "physics7"}:
        return name in {"gamma", "rho", "kernel", "deriv", "tau", "trace", "mode"}
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
    if name not in {"deriv", "tau", "kinetic"}:
        return 1.0
    start_epoch = max(int(getattr(config, f"{name}_start_epoch")), 0)
    ramp_epochs = max(int(getattr(config, f"{name}_ramp_epochs")), 0)

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
    )
    return outputs["gamma"].numpy().astype(np.float32)


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
    sample, while diagonal density and stencil tau are still evaluated exactly.
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

    diag_outputs = diagonal_predictions(system, models, config, rho_all=rho_all, density_state=density_state)
    gamma_diag = diag_outputs["gamma"].numpy()
    kernel_diag = diag_outputs["kernel"].numpy()
    rho_loss = float(np.mean((gamma_diag - system.rho_diag) ** 2))
    density_mae = float(np.mean(np.abs(gamma_diag - system.rho_diag)))
    rho_point_mae = float(np.mean(np.abs(rho_all.numpy() - system.rho_diag)))
    kernel_loss = float(np.mean((kernel_diag - 1.0) ** 2))
    kernel_diag_error = float(np.mean(np.abs(kernel_diag - 1.0)))

    derivative_pred_t, tau_pred_t = stencil_predictions(
        system, models, config, rho_all=rho_all, density_state=density_state
    )
    derivative_pred = derivative_pred_t.numpy()
    tau_pred = tau_pred_t.numpy()
    deriv_raw_mse = float(np.mean((derivative_pred - system.derivative_true) ** 2))
    tau_raw_mse = float(np.mean((tau_pred - system.tau_true) ** 2))
    deriv_loss = float(
        rms_normalized_huber(
            to_tensor(system.derivative_true),
            derivative_pred_t,
            scale_floor=config.deriv_scale_floor,
            delta=config.physics_huber_delta,
        ).numpy()
    )
    tau_loss = float(
        rms_normalized_huber(
            to_tensor(system.tau_true),
            tau_pred_t,
            scale_floor=config.tau_scale_floor,
            delta=config.physics_huber_delta,
        ).numpy()
    )
    deriv_mae = float(np.mean(np.abs(derivative_pred - system.derivative_true)))
    tau_mae = float(np.mean(np.abs(tau_pred - system.tau_true)))
    kinetic_loss_t, kinetic_pred_t, kinetic_ref = kinetic_energy_loss_from_tau(system, tau_pred_t)
    kinetic_loss = float(kinetic_loss_t.numpy())
    kinetic_pred = float(kinetic_pred_t.numpy())
    kinetic_ref_error = float(kinetic_pred - kinetic_ref)
    kinetic_energy_ref = float(system.metadata.get("kinetic_energy_hartree", np.nan))
    kinetic_energy_ref_error = float(kinetic_pred - kinetic_energy_ref) if np.isfinite(kinetic_energy_ref) else float("nan")
    trace_pred = float(np.sum(gamma_diag) * system.cell_volume)
    trace_true = float(system.electron_count)
    trace_scale = max(trace_true, 1.0)
    trace_rel_error = float((trace_pred - trace_true) / trace_scale)
    trace_loss = float(trace_rel_error**2)

    near_mask = categories == 1
    far_mask = categories == 3
    near_diag_mae = float(np.mean(np.abs(gamma_pred_pairs[near_mask] - gamma_true_pairs[near_mask]))) if np.any(near_mask) else float("nan")
    far_offdiag_mae = float(np.mean(np.abs(gamma_pred_pairs[far_mask] - gamma_true_pairs[far_mask]))) if np.any(far_mask) else float("nan")

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
    tau_pred_integral = float(np.sum(tau_pred) * system.cell_volume)

    metrics = {
        "system_id": system.system_id,
        "formula": str(system.metadata.get("formula", "")),
        "axis_points": int(len(system.axis)),
        "n_points": int(len(system.points)),
        "grid_spacing_bohr": float(system.step),
        "electron_count": float(system.electron_count),
        "pair_loss": pair_loss,
        "pair_mae": pair_mae,
        "rho_loss": rho_loss,
        "density_mae": density_mae,
        "rho_point_mae": rho_point_mae,
        "kernel_loss": kernel_loss,
        "kernel_diag_error": kernel_diag_error,
        "deriv_loss": deriv_loss,
        "deriv_raw_mse": deriv_raw_mse,
        "deriv_mae": deriv_mae,
        "tau_loss": tau_loss,
        "tau_raw_mse": tau_raw_mse,
        "tau_mae": tau_mae,
        "kinetic_loss": kinetic_loss,
        "kinetic_pred": kinetic_pred,
        "kinetic_training_ref": kinetic_ref,
        "kinetic_ref_error": kinetic_ref_error,
        "kinetic_abs_error": abs(kinetic_ref_error),
        "trace_loss": trace_loss,
        "trace_rel_error": trace_rel_error,
        "trace_abs_rel_error": abs(trace_rel_error),
        "trace_true": trace_true,
        "trace_pred": trace_pred,
        "occ_penalty": occ_penalty,
        "near_diag_mae": near_diag_mae,
        "far_offdiag_mae": far_offdiag_mae,
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
        "tau_pred_integral": tau_pred_integral,
        "ked_true_integral": tau_true_integral,
        "ked_pred_integral": tau_pred_integral,
        "kinetic_energy_ref": kinetic_energy_ref,
        "kinetic_energy_ref_error": kinetic_energy_ref_error,
        "rho_true_diag": system.rho_diag,
        "rho_pred_diag": rho_all.numpy(),
        "tau_true": system.tau_true,
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
    if keep_arrays and len(system.points) <= config.full_eval_max_points:
        gamma_pred_matrix = predict_full_gamma_matrix(system, models, config)
        metrics["gamma_pred_matrix"] = gamma_pred_matrix
        metrics["gamma_true_matrix"] = system.load_gamma_matrix()
    metrics["objective"] = objective_from_metrics(metrics, config, epoch=epoch)
    return metrics


def evaluate_systems(
    systems: list[SystemRecord],
    models: ModelBundle,
    config: ExperimentConfig,
    epoch: int | None = None,
) -> dict[str, object]:
    """여러 시스템 평가 후 평균 지표 계산."""
    rng = np.random.default_rng(config.seed + 999)
    per_system = [
        evaluate_system(system, models, config, rng=rng, keep_arrays=(idx == 0), epoch=epoch)
        for idx, system in enumerate(systems)
    ]

    scalar_keys = [
        "objective",
        "pair_loss",
        "pair_mae",
        "rho_loss",
        "density_mae",
        "rho_point_mae",
        "kernel_loss",
        "kernel_diag_error",
        "deriv_loss",
        "deriv_raw_mse",
        "deriv_mae",
        "tau_loss",
        "tau_raw_mse",
        "tau_mae",
        "kinetic_loss",
        "kinetic_pred",
        "kinetic_training_ref",
        "kinetic_ref_error",
        "kinetic_abs_error",
        "trace_loss",
        "trace_rel_error",
        "trace_abs_rel_error",
        "tau_true_integral",
        "tau_pred_integral",
        "ked_true_integral",
        "ked_pred_integral",
        "kinetic_energy_ref",
        "kinetic_energy_ref_error",
        "occ_penalty",
        "symmetry_mae",
        "near_diag_mae",
        "far_offdiag_mae",
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
    averages["per_system"] = per_system
    return averages


def epoch_loss_weights(config: ExperimentConfig, epoch: int) -> dict[str, float]:
    return {name: scheduled_loss_weight(config, name, epoch) for name in LOSS_NAMES}


def weighted_training_objective(losses: dict[str, tf.Tensor], weights: dict[str, float]) -> tf.Tensor:
    total = tf.constant(0.0, dtype=tf.float32)
    for weight_name, loss_name in TRAIN_OBJECTIVE_TERMS:
        total = total + weights[weight_name] * losses[loss_name]
    return total


def compute_training_losses(
    system: SystemRecord,
    batch: PairBatch,
    models: ModelBundle,
    config: ExperimentConfig,
    weights: dict[str, float],
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
    )
    pair_loss = weighted_mse(to_tensor(batch.gamma_true), pair_outputs["gamma"], to_tensor(batch.weights))

    diag_outputs = diagonal_predictions(system, models, config, rho_all=rho_all, density_state=density_state)
    gamma_diag = diag_outputs["gamma"]
    rho_loss = tf.reduce_mean(tf.square(gamma_diag - to_tensor(system.rho_diag)))
    kernel_loss = tf.reduce_mean(tf.square(diag_outputs["kernel"] - 1.0))

    zero = tf.constant(0.0, dtype=tf.float32)
    if weights["deriv"] != 0.0 or weights["tau"] != 0.0 or weights["kinetic"] != 0.0:
        derivative_pred, tau_pred = stencil_predictions(
            system, models, config, rho_all=rho_all, density_state=density_state
        )
        deriv_loss = rms_normalized_huber(
            to_tensor(system.derivative_true),
            derivative_pred,
            scale_floor=config.deriv_scale_floor,
            delta=config.physics_huber_delta,
        )
        tau_loss = rms_normalized_huber(
            to_tensor(system.tau_true),
            tau_pred,
            scale_floor=config.tau_scale_floor,
            delta=config.physics_huber_delta,
        )
        kinetic_loss, _, _ = kinetic_energy_loss_from_tau(system, tau_pred)
    else:
        deriv_loss = zero
        tau_loss = zero
        kinetic_loss = zero

    trace_pred = tf.reduce_sum(gamma_diag) * system.cell_volume
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
        "trace_loss": trace_loss,
        "occ_penalty": occ_penalty,
        "mode_reg": mode_reg,
        "kinetic_loss": kinetic_loss,
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
    diagnostic_weights["kinetic"] = 0.0

    with tf.GradientTape(persistent=True) as tape:
        losses = compute_training_losses(system, batch, models, config, diagnostic_weights)

    rows: list[tuple[str, str]] = [
        ("system", system.system_id),
        (
            "scheduled weights gamma/deriv/tau",
            f"{weights['gamma']:.3e} / {weights['deriv']:.3e} / {weights['tau']:.3e}",
        ),
    ]
    effective_norms = {}
    for label, loss_name, weight_name in (
        ("gamma", "pair_loss", "gamma"),
        ("deriv", "deriv_loss", "deriv"),
        ("tau", "tau_loss", "tau"),
    ):
        raw_total = gradient_global_norm(tape.gradient(losses[loss_name], all_vars))
        effective_total = raw_total * weights[weight_name]
        effective_norms[label] = effective_total
        group_norms = {
            group_name: gradient_global_norm(tape.gradient(losses[loss_name], variables)) if variables else 0.0
            for group_name, variables in variable_groups.items()
        }
        rows.extend(
            [
                (f"{label} loss", f"{float(losses[loss_name].numpy()):.6e}"),
                (f"{label} grad raw/effective", f"{raw_total:.6e} / {effective_total:.6e}"),
                (
                    f"{label} grad point/mode/pair/context",
                    " / ".join(f"{group_norms[name]:.3e}" for name in ("point", "mode", "pair", "context")),
                ),
            ]
        )
    gamma_effective = max(effective_norms["gamma"], 1e-30)
    rows.append(
        (
            "effective grad ratio deriv/tau vs gamma",
            f"{effective_norms['deriv'] / gamma_effective:.6e} / {effective_norms['tau'] / gamma_effective:.6e}",
        )
    )
    del tape

    _, density_state = point_output_and_state(system, models, config)
    derivative_pred, tau_pred = stencil_predictions(
        system, models, config, rho_all=density_state.rho_neutral, density_state=density_state
    )
    rows.extend(
        [
            (
                "deriv RMS target/pred",
                f"{tensor_rms(to_tensor(system.derivative_true)):.6e} / {tensor_rms(derivative_pred):.6e}",
            ),
            (
                "tau RMS target/pred",
                f"{tensor_rms(to_tensor(system.tau_true)):.6e} / {tensor_rms(tau_pred):.6e}",
            ),
        ]
    )
    print_block("Gradient diagnostics", rows)


def train_models(config: ExperimentConfig, split: DatasetSplit, models: ModelBundle) -> tuple[TrainingHistory, dict[str, object]]:
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
    print_block("Base loss weights", loss_weight_rows(config))
    print_block("Loss schedule", loss_schedule_rows(config))
    print_block(
        "Physics loss",
        [
            ("deriv/tau loss", "target-RMS normalized Huber"),
            ("Huber delta", f"{config.physics_huber_delta:.6g}"),
            ("deriv/tau scale floor", f"{config.deriv_scale_floor:.6g} / {config.tau_scale_floor:.6g}"),
            ("kinetic integral active", loss_enabled(config, "kinetic")),
        ],
    )
    print_block(
        "Density constraint",
        [
            ("normalize_rho", config.normalize_rho),
            ("frozen density-state cache", config.freeze_point_after_pretrain),
        ],
    )
    print_block(
        "Gradient diagnostics",
        [
            ("enabled", config.gradient_diagnostics),
            ("every epochs", max(config.gradient_diagnostics_every, 1)),
            ("fixed train system", diagnostic_system.system_id),
        ],
    )

    for epoch in range(config.epochs):
        last_epoch = epoch
        weights = epoch_loss_weights(config, epoch)
        stage_signature = loss_stage_signature(config, epoch)
        if last_stage_signature is not None and stage_signature != last_stage_signature:
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

            with tf.GradientTape() as tape:
                losses = compute_training_losses(system, batch, models, config, weights)
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
            val_metrics = evaluate_systems(split.val_systems, models, config, epoch=epoch)
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
            and epoch % max(config.gradient_diagnostics_every, 1) == 0
        ):
            assert diagnostic_batch is not None
            print_gradient_diagnostics(diagnostic_system, diagnostic_batch, models, config, weights)

        if validation_ran:
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
                    + f"held-out mae gamma_pair={val_metrics['pair_mae']:.3e} "
                    + f"rho={val_metrics['density_mae']:.3e} "
                    + f"deriv={val_metrics['deriv_mae']:.3e} "
                    + f"tau={val_metrics['tau_mae']:.3e}"
                )
                print(
                    " " * 14
                    + f"held-out loss gamma_pair={val_metrics['pair_loss']:.3e} "
                    + f"rho={val_metrics['rho_loss']:.3e} "
                    + f"T={val_metrics['kinetic_loss']:.3e} "
                    + f"deriv_huber={val_metrics['deriv_loss']:.3e} "
                    + f"tau_huber={val_metrics['tau_loss']:.3e} "
                    + f"trace_rel={val_metrics['trace_abs_rel_error']:.3e} "
                    + f"occ_pen={val_metrics['occ_penalty']:.3e}"
                )
                print(
                    " " * 14
                    + f"held-out raw mse deriv={val_metrics['deriv_raw_mse']:.3e} "
                    + f"tau={val_metrics['tau_raw_mse']:.3e}"
                )

        if validation_ran and epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_weights is not None:
        models.point_model.set_weights(best_weights["point"])
        models.mode_model.set_weights(best_weights["mode"])
        models.pair_model.set_weights(best_weights["pair"])
        models.context_model.set_weights(best_weights["context"])

    final_train = evaluate_systems(split.train_systems, models, config, epoch=last_epoch)
    final_val = evaluate_systems(split.val_systems, models, config, epoch=last_epoch)
    summary = {"train": final_train, "val": final_val}
    if split.test_systems:
        summary["test"] = evaluate_systems(split.test_systems, models, config, epoch=last_epoch)

    rows = [
        ("train objective", f"{final_train['objective']:.6e}"),
        ("val objective", f"{final_val['objective']:.6e}"),
        ("held-out gamma_pair loss", f"{final_val['pair_loss']:.6e}"),
        ("held-out density MAE", f"{final_val['density_mae']:.6e}"),
        ("held-out kinetic loss", f"{final_val['kinetic_loss']:.6e}"),
        ("held-out kinetic abs err", f"{final_val['kinetic_abs_error']:.6e}"),
        ("held-out trace rel err", f"{final_val['trace_abs_rel_error']:.6e}"),
        ("held-out tau MAE", f"{final_val['tau_mae']:.6e}"),
        ("held-out symmetry MAE", f"{final_val['symmetry_mae']:.6e}"),
        ("held-out kernel diag err", f"{final_val['kernel_diag_error']:.6e}"),
    ]
    if "test" in summary:
        final_test = summary["test"]
        rows.extend(
            [
                ("test objective", f"{final_test['objective']:.6e}"),
                ("test gamma_pair loss", f"{final_test['pair_loss']:.6e}"),
                ("test density MAE", f"{final_test['density_mae']:.6e}"),
                ("test kinetic loss", f"{final_test['kinetic_loss']:.6e}"),
                ("test kinetic abs err", f"{final_test['kinetic_abs_error']:.6e}"),
                ("test tau MAE", f"{final_test['tau_mae']:.6e}"),
            ]
        )
    print_block("Final transferable summary", rows)
    return history, summary
