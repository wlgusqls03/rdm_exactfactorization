from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from .config import ExperimentConfig
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


LOSS_NAMES = ("gamma", "rho", "kernel", "deriv", "tau", "trace", "occ", "mode", "kinetic", "kp")
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
    "kp_loss",
)
VAL_HISTORY_KEYS = (
    "pair_loss",
    "pair_mae",
    "rho_loss",
    "density_mae",
    "kernel_loss",
    "kernel_diag_error",
    "deriv_loss",
    "deriv_mae",
    "tau_loss",
    "tau_mae",
    "kinetic_loss",
    "kinetic_abs_error",
    "kp_loss",
    "kp_mae",
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
    ("kp", "kp_loss"),
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
    ("kp", "kp_loss"),
)


@dataclass
class TrainingHistory:
    train_objective: list[float] = field(default_factory=list)
    val_objective: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    kinetic_weight: list[float] = field(default_factory=list)
    kp_weight: list[float] = field(default_factory=list)
    validation_ran: list[int] = field(default_factory=list)
    loss_weights: dict[str, list[float]] = field(default_factory=lambda: {key: [] for key in LOSS_NAMES})
    train_components: dict[str, list[float]] = field(
        default_factory=lambda: {key: [] for key in TRAIN_HISTORY_KEYS}
    )
    val_components: dict[str, list[float]] = field(
        default_factory=lambda: {key: [] for key in VAL_HISTORY_KEYS}
    )


def weighted_mse(y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(weights * tf.square(y_true - y_pred)) / tf.reduce_sum(weights)


def to_tensor(array: np.ndarray) -> tf.Tensor:
    return tf.convert_to_tensor(array, dtype=tf.float32)


def point_model_outputs(system: SystemRecord, models: ModelBundle) -> tf.Tensor:
    point_features = to_tensor(system.local_features)
    global_context_t = tf.reshape(to_tensor(system.global_context), (1, -1))
    tiled_global = tf.repeat(global_context_t, repeats=tf.shape(point_features)[0], axis=0)
    point_input = tf.concat([point_features, tiled_global], axis=1)
    return models.point_model(point_input)


def density_from_point_output(system: SystemRecord, point_out: tf.Tensor, config: ExperimentConfig) -> tf.Tensor:
    """Predict rho on the full grid, optionally enforcing integral rho = N."""
    rho_raw = tf.nn.softplus(point_out[:, :1]) + 1e-6
    if not config.normalize_rho:
        return rho_raw
    normalizer = tf.reduce_sum(rho_raw) * system.cell_volume
    scale = system.electron_count / tf.maximum(normalizer, 1e-12)
    return rho_raw * scale


def point_output_and_density(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
) -> tuple[tf.Tensor, tf.Tensor]:
    point_out = point_model_outputs(system, models)
    return point_out, density_from_point_output(system, point_out, config)


def normalized_density(system: SystemRecord, models: ModelBundle, config: ExperimentConfig) -> tf.Tensor:
    return point_output_and_density(system, models, config)[1]


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


def kinetic_potential_loss(
    system: SystemRecord,
    models: ModelBundle,
    rho_all: tf.Tensor | None = None,
    point_out: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Density-weighted MSE for centered kinetic potential head."""
    target_np = np.asarray(system.kinetic_potential_centered, dtype=np.float32)
    finite_mask_np = np.isfinite(target_np).astype(np.float32)
    if not np.any(finite_mask_np):
        zero = tf.constant(0.0, dtype=tf.float32)
        return zero, zero, zero

    if point_out is None:
        point_out = point_model_outputs(system, models)
    if rho_all is None:
        # KP loss should not decide the normalization path; rho only supplies weights.
        rho_all = tf.nn.softplus(point_out[:, :1]) + 1e-6
    kp_pred = point_out[:, 1:2]
    target = to_tensor(target_np)
    mask = to_tensor(finite_mask_np)

    rho_weight = tf.stop_gradient(tf.maximum(rho_all, 0.0)) * mask
    norm = tf.reduce_sum(rho_weight) * system.cell_volume
    norm = tf.maximum(norm, 1e-12)

    pred_center = kp_pred - tf.reduce_sum(rho_weight * kp_pred) * system.cell_volume / norm
    target_center = target - tf.reduce_sum(rho_weight * target) * system.cell_volume / norm
    diff = pred_center - target_center
    target_rms = tf.sqrt(tf.reduce_sum(rho_weight * tf.square(target_center)) * system.cell_volume / norm)
    scale = tf.maximum(target_rms, 1e-3)
    mse = tf.reduce_sum(rho_weight * tf.square(diff)) * system.cell_volume / norm
    mae = tf.reduce_sum(rho_weight * tf.abs(diff)) * system.cell_volume / norm
    return mse / tf.square(scale), mae, scale


def kinetic_potential_centered_arrays(
    system: SystemRecord,
    models: ModelBundle,
    rho_all: tf.Tensor | None = None,
    point_out: tf.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    target_np = np.asarray(system.kinetic_potential_centered, dtype=np.float32)
    finite_mask_np = np.isfinite(target_np).astype(np.float32)
    if not np.any(finite_mask_np):
        nan_values = np.full((len(system.points), 1), np.nan, dtype=np.float32)
        return nan_values, nan_values
    if point_out is None:
        point_out = point_model_outputs(system, models)
    if rho_all is None:
        rho_all = tf.nn.softplus(point_out[:, :1]) + 1e-6
    kp_pred = point_out[:, 1:2]
    target = to_tensor(target_np)
    mask = to_tensor(finite_mask_np)
    rho_weight = tf.stop_gradient(tf.maximum(rho_all, 0.0)) * mask
    norm = tf.maximum(tf.reduce_sum(rho_weight) * system.cell_volume, 1e-12)
    pred_center = kp_pred - tf.reduce_sum(rho_weight * kp_pred) * system.cell_volume / norm
    target_center = target - tf.reduce_sum(rho_weight * target) * system.cell_volume / norm
    return target_center.numpy().astype(np.float32), pred_center.numpy().astype(np.float32)


def gather_density(rho_all: tf.Tensor, indices: np.ndarray) -> tf.Tensor:
    return tf.gather(rho_all, tf.convert_to_tensor(indices, dtype=tf.int64))


def diagonal_predictions(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
    rho_all: tf.Tensor | None = None,
) -> dict[str, tf.Tensor]:
    """r = r' diagonal prediction."""
    if rho_all is None:
        rho_all = normalized_density(system, models, config)
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
    )
    return outputs


def stencil_predictions(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
    rho_all: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """explicit near-diagonal mixed derivative prediction.

    Returns
    -------
    derivative_pred : (n_interior, 3)
    tau_pred        : (n_interior, 1)
    """
    if rho_all is None:
        rho_all = normalized_density(system, models, config)
    per_dim = []
    stencil_order = int(system.stencil_left.shape[2])
    for dim in range(3):
        left = system.stencil_left[:, dim, :]    # (n_interior, 4 or 8)
        right = system.stencil_right[:, dim, :]  # (n_interior, 4 or 8)

        gamma_terms = []
        for variant in range(stencil_order):
            left_idx = left[:, variant]
            right_idx = right[:, variant]
            outputs = predict_from_features(
                to_tensor(system.local_features[left_idx]),
                to_tensor(system.local_features[right_idx]),
                to_tensor(build_pair_features(system, left_idx, right_idx)),
                to_tensor(system.global_context),
                models,
                rho_r_override=gather_density(rho_all, left_idx),
                rho_rp_override=gather_density(rho_all, right_idx),
            )
            gamma_terms.append(outputs["gamma"])

        d_h = (gamma_terms[0] - gamma_terms[1] - gamma_terms[2] + gamma_terms[3]) / (
            4.0 * system.step * system.step
        )
        if stencil_order >= 8:
            d_2h = (gamma_terms[4] - gamma_terms[5] - gamma_terms[6] + gamma_terms[7]) / (
                16.0 * system.step * system.step
            )
            deriv_dim = (4.0 * d_h - d_2h) / 3.0
        else:
            deriv_dim = d_h
        per_dim.append(deriv_dim)

    derivative_pred = tf.concat(per_dim, axis=1)              # (n_interior, 3)
    tau_pred = 0.5 * tf.reduce_sum(derivative_pred, axis=1, keepdims=True)
    return derivative_pred, tau_pred


def spectral_occupation_penalty(
    system: SystemRecord,
    models: ModelBundle,
    config: ExperimentConfig,
    rho_all: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """coarse operator spectrum이 물리 범위를 벗어나지 않게 벌점."""
    if rho_all is None:
        rho_all = normalized_density(system, models, config)
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
        + models.pair_model.trainable_variables
        + models.context_model.trainable_variables
    )


def loss_enabled(config: ExperimentConfig, name: str) -> bool:
    preset = config.loss_preset.strip().lower()
    if preset in {"core5", "simple5"}:
        return name in {"gamma", "rho", "kernel", "trace", "mode"}
    if preset in {"core7", "kerdf7"}:
        return name in {"gamma", "rho", "kernel", "trace", "mode", "kinetic", "kp"}
    if preset in {"all", "custom", "none"}:
        return bool(getattr(config, f"use_{name}_loss"))
    raise ValueError(f"Unknown RDM_LOSS_PRESET: {config.loss_preset}")


def loss_weight(config: ExperimentConfig, name: str) -> float:
    if not loss_enabled(config, name):
        return 0.0
    return float(getattr(config, f"lambda_{name}"))


def loss_schedule_multiplier(config: ExperimentConfig, name: str, epoch: int) -> float:
    """Epoch-dependent multiplier for staged loss terms."""
    if name == "kp":
        start_epoch = max(int(config.kp_start_epoch), 0)
        ramp_epochs = max(int(config.kp_ramp_epochs), 0)
    elif name == "kinetic":
        start_epoch = max(int(config.kinetic_start_epoch), 0)
        ramp_epochs = max(int(config.kinetic_ramp_epochs), 0)
    else:
        return 1.0

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
        ("kp start/ramp epochs", f"{config.kp_start_epoch}/{config.kp_ramp_epochs}"),
        ("kinetic start/ramp epochs", f"{config.kinetic_start_epoch}/{config.kinetic_ramp_epochs}"),
    ]


def active_loss_summary(weights: dict[str, float]) -> str:
    active = [f"{name}={weight:.4g}" for name, weight in weights.items() if weight != 0.0]
    return ", ".join(active) if active else "none"


def loss_stage_value(config: ExperimentConfig, name: str, epoch: int) -> int:
    """0: inactive, 1: scheduled ramp, 2: fully active."""
    if scheduled_loss_weight(config, name, epoch) == 0.0:
        return 0
    if name in {"kinetic", "kp"} and loss_schedule_multiplier(config, name, epoch) < 1.0:
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
    point_out_all, rho_all = point_output_and_density(system, models, config)
    for left, right in full_pair_chunks(system, chunk_size):
        outputs = predict_from_features(
            to_tensor(system.local_features[left]),
            to_tensor(system.local_features[right]),
            to_tensor(build_pair_features(system, left, right)),
            to_tensor(system.global_context),
            models,
            rho_r_override=gather_density(rho_all, left),
            rho_rp_override=gather_density(rho_all, right),
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
) -> np.ndarray:
    if rho_all is None:
        rho_all = normalized_density(system, models, config)
    outputs = predict_from_features(
        to_tensor(system.local_features[left]),
        to_tensor(system.local_features[right]),
        to_tensor(build_pair_features(system, left, right)),
        to_tensor(system.global_context),
        models,
        rho_r_override=gather_density(rho_all, left),
        rho_rp_override=gather_density(rho_all, right),
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
    point_out_all, rho_all = point_output_and_density(system, models, config)
    gamma_pred_pairs = predict_pair_values(system, models, config, left, right, rho_all=rho_all)
    pair_weights = pair_weights_from_categories(categories)

    pair_loss = float(np.sum(pair_weights * (gamma_pred_pairs - gamma_true_pairs) ** 2) / np.sum(pair_weights))
    pair_mae = float(np.mean(np.abs(gamma_pred_pairs - gamma_true_pairs)))

    diag_outputs = diagonal_predictions(system, models, config, rho_all=rho_all)
    gamma_diag = diag_outputs["gamma"].numpy()
    kernel_diag = diag_outputs["kernel"].numpy()
    rho_loss = float(np.mean((gamma_diag - system.rho_diag) ** 2))
    density_mae = float(np.mean(np.abs(gamma_diag - system.rho_diag)))
    kernel_loss = float(np.mean((kernel_diag - 1.0) ** 2))
    kernel_diag_error = float(np.mean(np.abs(kernel_diag - 1.0)))

    derivative_pred_t, tau_pred_t = stencil_predictions(system, models, config, rho_all=rho_all)
    derivative_pred = derivative_pred_t.numpy()
    tau_pred = tau_pred_t.numpy()
    deriv_loss = float(np.mean((derivative_pred - system.derivative_true) ** 2))
    tau_loss = float(np.mean((tau_pred - system.tau_true) ** 2))
    deriv_mae = float(np.mean(np.abs(derivative_pred - system.derivative_true)))
    tau_mae = float(np.mean(np.abs(tau_pred - system.tau_true)))
    kinetic_loss_t, kinetic_pred_t, kinetic_ref = kinetic_energy_loss_from_tau(system, tau_pred_t)
    kinetic_loss = float(kinetic_loss_t.numpy())
    kinetic_pred = float(kinetic_pred_t.numpy())
    kinetic_ref_error = float(kinetic_pred - kinetic_ref)
    kinetic_energy_ref = float(system.metadata.get("kinetic_energy_hartree", np.nan))
    kinetic_energy_ref_error = float(kinetic_pred - kinetic_energy_ref) if np.isfinite(kinetic_energy_ref) else float("nan")
    kp_loss_t, kp_mae_t, kp_scale_t = kinetic_potential_loss(
        system,
        models,
        rho_all=rho_all,
        point_out=point_out_all,
    )
    kp_loss = float(kp_loss_t.numpy())
    kp_mae = float(kp_mae_t.numpy())
    kp_scale = float(kp_scale_t.numpy())
    kp_true_centered, kp_pred_centered = kinetic_potential_centered_arrays(
        system,
        models,
        rho_all=rho_all,
        point_out=point_out_all,
    )

    trace_pred = float(np.sum(gamma_diag) * system.cell_volume)
    trace_true = float(system.electron_count)
    trace_scale = max(trace_true, 1.0)
    trace_rel_error = float((trace_pred - trace_true) / trace_scale)
    trace_loss = float(trace_rel_error**2)

    near_mask = categories == 1
    far_mask = categories == 3
    near_diag_mae = float(np.mean(np.abs(gamma_pred_pairs[near_mask] - gamma_true_pairs[near_mask]))) if np.any(near_mask) else float("nan")
    far_offdiag_mae = float(np.mean(np.abs(gamma_pred_pairs[far_mask] - gamma_true_pairs[far_mask]))) if np.any(far_mask) else float("nan")

    gamma_pred_reverse = predict_pair_values(system, models, config, right, left, rho_all=rho_all)
    symmetry_mae = float(np.mean(np.abs(gamma_pred_pairs - gamma_pred_reverse)))

    subset = system.spectral_subset
    gamma_true_sub = system.gamma_submatrix(subset)
    subset_eigs_true = natural_occupation_spectrum(gamma_true_sub, system.cell_volume)
    occ_penalty_t, occ_eigs_t = spectral_occupation_penalty(system, models, config, rho_all=rho_all)
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
        "kernel_loss": kernel_loss,
        "kernel_diag_error": kernel_diag_error,
        "deriv_loss": deriv_loss,
        "deriv_mae": deriv_mae,
        "tau_loss": tau_loss,
        "tau_mae": tau_mae,
        "kinetic_loss": kinetic_loss,
        "kinetic_pred": kinetic_pred,
        "kinetic_training_ref": kinetic_ref,
        "kinetic_ref_error": kinetic_ref_error,
        "kinetic_abs_error": abs(kinetic_ref_error),
        "kp_loss": kp_loss,
        "kp_mae": kp_mae,
        "kp_scale": kp_scale,
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
        "kinetic_potential_reference": str(system.metadata.get("kinetic_potential_reference", "")),
        "rho_true_diag": system.rho_diag,
        "rho_pred_diag": gamma_diag,
        "tau_true": system.tau_true,
        "tau_pred": tau_pred,
        "kp_true_centered": kp_true_centered,
        "kp_pred_centered": kp_pred_centered,
        "gamma_true_sample": gamma_true_pairs,
        "gamma_pred_sample": gamma_pred_pairs,
        "occ_eigs_subset": occ_eigs_t.numpy(),
    }
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
        "kernel_loss",
        "kernel_diag_error",
        "deriv_loss",
        "deriv_mae",
        "tau_loss",
        "tau_mae",
        "kinetic_loss",
        "kinetic_pred",
        "kinetic_training_ref",
        "kinetic_ref_error",
        "kinetic_abs_error",
        "kp_loss",
        "kp_mae",
        "kp_scale",
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
    point_out_all, rho_all = point_output_and_density(system, models, config)
    pair_outputs = predict_from_features(
        to_tensor(batch.point_feat_r),
        to_tensor(batch.point_feat_rp),
        to_tensor(batch.pair_feat),
        to_tensor(batch.global_context),
        models,
        rho_r_override=gather_density(rho_all, batch.left_idx),
        rho_rp_override=gather_density(rho_all, batch.right_idx),
    )
    pair_loss = weighted_mse(to_tensor(batch.gamma_true), pair_outputs["gamma"], to_tensor(batch.weights))

    diag_outputs = diagonal_predictions(system, models, config, rho_all=rho_all)
    gamma_diag = diag_outputs["gamma"]
    rho_loss = tf.reduce_mean(tf.square(gamma_diag - to_tensor(system.rho_diag)))
    kernel_loss = tf.reduce_mean(tf.square(diag_outputs["kernel"] - 1.0))

    zero = tf.constant(0.0, dtype=tf.float32)
    if weights["deriv"] != 0.0 or weights["tau"] != 0.0 or weights["kinetic"] != 0.0:
        derivative_pred, tau_pred = stencil_predictions(system, models, config, rho_all=rho_all)
        deriv_loss = tf.reduce_mean(tf.square(derivative_pred - to_tensor(system.derivative_true)))
        tau_loss = tf.reduce_mean(tf.square(tau_pred - to_tensor(system.tau_true)))
        kinetic_loss, _, _ = kinetic_energy_loss_from_tau(system, tau_pred)
    else:
        deriv_loss = zero
        tau_loss = zero
        kinetic_loss = zero

    trace_pred = tf.reduce_sum(gamma_diag) * system.cell_volume
    trace_scale = max(system.electron_count, 1.0)
    trace_loss = tf.square((trace_pred - system.electron_count) / trace_scale)

    if weights["occ"] != 0.0:
        occ_penalty, _ = spectral_occupation_penalty(system, models, config, rho_all=rho_all)
    else:
        occ_penalty = zero
    mode_reg = tf.reduce_mean(pair_outputs["mode_weights"]) if weights["mode"] != 0.0 else zero
    kp_loss = (
        kinetic_potential_loss(system, models, rho_all=rho_all, point_out=point_out_all)[0]
        if weights["kp"] != 0.0
        else zero
    )

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
        "kp_loss": kp_loss,
    }


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
    last_stage_signature: tuple[int, ...] | None = None
    last_epoch = 0
    print_block("Base loss weights", loss_weight_rows(config))
    print_block("Loss schedule", loss_schedule_rows(config))
    print_block("Density constraint", [("normalize_rho", config.normalize_rho)])

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
        history.kp_weight.append(weights["kp"])
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

        if validation_ran:
            if val_objective < best_val_objective - 1e-9:
                best_val_objective = val_objective
                best_weights = {
                    "point": models.point_model.get_weights(),
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
                f"w(T)={weights['kinetic']:.3g} "
                f"w(KP)={weights['kp']:.3g}"
            )
            if val_metrics:
                print(
                    " " * 14
                    + f"held-out mae pair={val_metrics['pair_mae']:.3e} "
                    + f"rho={val_metrics['density_mae']:.3e} "
                    + f"deriv={val_metrics['deriv_mae']:.3e} "
                    + f"tau={val_metrics['tau_mae']:.3e}"
                )
                print(
                    " " * 14
                    + f"held-out loss pair={val_metrics['pair_loss']:.3e} "
                    + f"rho={val_metrics['rho_loss']:.3e} "
                    + f"T={val_metrics['kinetic_loss']:.3e} "
                    + f"KP={val_metrics['kp_loss']:.3e} "
                    + f"deriv={val_metrics['deriv_loss']:.3e} "
                    + f"tau={val_metrics['tau_loss']:.3e} "
                    + f"trace_rel={val_metrics['trace_abs_rel_error']:.3e} "
                    + f"occ_pen={val_metrics['occ_penalty']:.3e}"
                )

        if validation_ran and epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_weights is not None:
        models.point_model.set_weights(best_weights["point"])
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
        ("held-out pair loss", f"{final_val['pair_loss']:.6e}"),
        ("held-out density MAE", f"{final_val['density_mae']:.6e}"),
        ("held-out kinetic loss", f"{final_val['kinetic_loss']:.6e}"),
        ("held-out kinetic abs err", f"{final_val['kinetic_abs_error']:.6e}"),
        ("held-out KP loss", f"{final_val['kp_loss']:.6e}"),
        ("held-out KP MAE", f"{final_val['kp_mae']:.6e}"),
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
                ("test pair loss", f"{final_test['pair_loss']:.6e}"),
                ("test density MAE", f"{final_test['density_mae']:.6e}"),
                ("test kinetic loss", f"{final_test['kinetic_loss']:.6e}"),
                ("test kinetic abs err", f"{final_test['kinetic_abs_error']:.6e}"),
                ("test KP loss", f"{final_test['kp_loss']:.6e}"),
                ("test tau MAE", f"{final_test['tau_mae']:.6e}"),
            ]
        )
    print_block("Final transferable summary", rows)
    return history, summary
