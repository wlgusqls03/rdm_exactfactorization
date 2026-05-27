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

    eval_pair_count: int = 8192
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
    pair_rho_eps: float = 1e-14
    pair_rho_log_scale: float = 8.0
    pair_rho_log_clip: float = 4.0
    pair_rho_scaled_clip: float = 20.0
    pair_rho_product_transform: str = "log1p"

    save_weights: bool = True


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


def to_tensor(array: np.ndarray) -> tf.Tensor:
    return tf.convert_to_tensor(array, dtype=tf.float32)


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
    return names


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

    if not features:
        return np.empty((len(left_idx), 0), dtype=np.float32)
    return np.concatenate(features, axis=1).astype(np.float32)


def build_v2_pair_features(
    system: SystemRecord,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    config: V2Config,
) -> np.ndarray:
    base = build_pair_features(system, left_idx, right_idx)
    if not active_pair_rho_feature_names(config):
        return base.astype(np.float32)
    rho_features = true_rho_pair_features(system, left_idx, right_idx, config)
    return np.concatenate([base, rho_features], axis=1).astype(np.float32)


def build_v2_models(point_dim: int, pair_dim: int, global_dim: int, config: V2Config) -> V2Models:
    if config.experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown V2 experiment: {config.experiment}")
    pair_rho_names = active_pair_rho_feature_names(config)
    if pair_rho_names and config.experiment != "k-only":
        raise ValueError("True-rho pair features are currently allowed only for k-only oracle ablations.")

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
    print_block(
        "V2 model",
        [
            ("experiment", config.experiment),
            ("point input dim", point_dim + global_dim if point_model is not None else "off"),
            ("pair input dim", pair_dim + global_dim if pair_model is not None else "off"),
            ("global dim", global_dim),
            ("rank", config.rank if uses_residual(config.experiment) else 0),
            ("kernel base alpha", config.kernel_base_alpha if pair_model is not None else "off"),
            ("pair rho features", ", ".join(pair_rho_names) if pair_rho_names else "off"),
            ("context RFF", config.context_rff if context_model is not None else "off"),
            ("point params", point_model.count_params() if point_model is not None else 0),
            ("pair params", pair_model.count_params() if pair_model is not None else 0),
            ("context params", context_model.count_params() if context_model is not None else 0),
        ],
    )
    return models


def tile_global(global_context: tf.Tensor, count: tf.Tensor) -> tf.Tensor:
    global_context = tf.reshape(global_context, (1, -1))
    return tf.repeat(global_context, repeats=count, axis=0)


def point_inputs(system: SystemRecord) -> tf.Tensor:
    local = to_tensor(system.local_features)
    tiled = tile_global(to_tensor(system.global_context), tf.shape(local)[0])
    return tf.concat([local, tiled], axis=1)


def predict_rho_and_modes(system: SystemRecord, models: V2Models, config: V2Config) -> tuple[tf.Tensor, tf.Tensor | None]:
    if models.point is None:
        rho_true = to_tensor(system.rho_diag)
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


def gather(values: tf.Tensor, indices: np.ndarray) -> tf.Tensor:
    return tf.gather(values, tf.convert_to_tensor(indices, dtype=tf.int32))


def separation_factor(pair_feat: tf.Tensor, config: V2Config) -> tf.Tensor:
    sep_sq = tf.maximum(pair_feat[:, 10:11], 0.0)
    return 1.0 - tf.exp(-sep_sq / max(config.sep_factor_scale, 1e-8))


def kernel_base(pair_feat: tf.Tensor, config: V2Config) -> tf.Tensor:
    sep_sq = tf.maximum(pair_feat[:, 10:11], 0.0)
    alpha = tf.constant(max(config.kernel_base_alpha, 0.0), dtype=tf.float32)
    return tf.exp(-alpha * sep_sq)


def predict_kernel(
    system: SystemRecord,
    batch: dict[str, np.ndarray],
    models: V2Models,
    config: V2Config,
    modes_all: tf.Tensor | None,
) -> tf.Tensor:
    if models.pair is None:
        raise RuntimeError("This experiment has no pair model.")

    pair_feat = to_tensor(batch["pair_feat"])
    global_tiled = tile_global(to_tensor(system.global_context), tf.shape(pair_feat)[0])
    pair_input = tf.concat([pair_feat, global_tiled], axis=1)
    delta_pair = models.pair(pair_input)
    sep_factor = separation_factor(pair_feat, config)
    kernel = kernel_base(pair_feat, config) + sep_factor * delta_pair

    if uses_residual(config.experiment):
        if modes_all is None:
            raise RuntimeError("Residual experiment requires point modes.")
        left_modes = gather(modes_all, batch["left"])
        right_modes = gather(modes_all, batch["right"])

        if models.context is not None:
            context_weights = tf.nn.softplus(models.context(tf.reshape(to_tensor(system.global_context), (1, -1)))) + 1e-6
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
) -> dict[str, np.ndarray]:
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
    pair_feat = build_v2_pair_features(system, left, right, config)
    return {
        "left": left.astype(np.int64),
        "right": right.astype(np.int64),
        "pair_feat": pair_feat.astype(np.float32),
        "gamma_true": gamma_true.astype(np.float32),
        "rho_left_true": rho_left,
        "rho_right_true": rho_right,
        "weights": pair_weights_from_categories(categories, config.pair_category_weights),
    }


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
    batch: dict[str, np.ndarray],
    gamma_pred: tf.Tensor,
    config: V2Config,
) -> tuple[tf.Tensor, tf.Tensor]:
    rho_left = to_tensor(batch["rho_left_true"])
    rho_right = to_tensor(batch["rho_right_true"])
    denom = tf.sqrt(tf.maximum(rho_left * rho_right, 0.0))
    mask = tf.cast(denom > config.k_highrho_cut, tf.float32)
    weights = to_tensor(batch["weights"]) * mask
    scaled_err = (gamma_pred - to_tensor(batch["gamma_true"])) / tf.maximum(denom, config.k_highrho_eps)
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
    rho_loss = tf.reduce_mean(tf.square(rho_pred - to_tensor(system.rho_diag)))
    trace_pred = tf.reduce_sum(rho_pred) * system.cell_volume
    trace_loss = tf.square((trace_pred - system.electron_count) / max(system.electron_count, 1.0))
    return rho_loss, trace_loss


def compute_step_loss(
    system: SystemRecord,
    batch: dict[str, np.ndarray] | None,
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

    if config.experiment == "k-only":
        rho_left = to_tensor(batch["rho_left_true"])
        rho_right = to_tensor(batch["rho_right_true"])
    else:
        rho_left = gather(rho_pred, batch["left"])
        rho_right = gather(rho_pred, batch["right"])

    kernel = predict_kernel(system, batch, models, config, modes_all)
    gamma_pred = gamma_from_rho_kernel(rho_left, rho_right, kernel)
    gamma_loss = weighted_mse(to_tensor(batch["gamma_true"]), gamma_pred, to_tensor(batch["weights"]))

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
    batch = make_batch(system, config.eval_pair_count, 0, 1, rng, config)

    rho_pred_np: np.ndarray
    gamma_pred_np: np.ndarray | None = None
    kernel_pred_np: np.ndarray | None = None

    if config.experiment == "baseline":
        rho_pred_np = system.rho_diag.astype(np.float32)
        gamma_pred_np = baseline_gamma(
            system,
            batch["left"],
            batch["right"],
            alpha=0.0 if alpha is None else alpha,
            density_power=config.baseline_density_power,
        )
    else:
        rho_pred, modes_all = predict_rho_and_modes(system, models, config)
        rho_pred_np = rho_pred.numpy().astype(np.float32)
        if config.experiment != "rho-only":
            if config.experiment == "k-only":
                rho_left = to_tensor(batch["rho_left_true"])
                rho_right = to_tensor(batch["rho_right_true"])
            else:
                rho_left = gather(rho_pred, batch["left"])
                rho_right = gather(rho_pred, batch["right"])
            kernel_pred = predict_kernel(system, batch, models, config, modes_all)
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


def train_v2(config: V2Config, split: DatasetSplit, point_dim: int, pair_dim: int, global_dim: int) -> dict[str, object]:
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

    print_block(
        "V2 training",
        [
            ("experiment", config.experiment),
            ("epochs", config.epochs),
            ("steps/epoch", config.steps_per_epoch),
            ("batch", config.batch_size),
            ("learning rate", config.learning_rate),
            ("k-only objective", "true-rho gamma loss" if config.experiment == "k-only" else "n/a"),
            ("kernel form", "exp(-alpha d^2) + sep * deltaK_pair"),
            ("kernel base alpha", config.kernel_base_alpha),
            ("k high-rho lambda", config.lambda_k_highrho if config.experiment == "k-only" else "n/a"),
            ("k high-rho cut", config.k_highrho_cut if config.experiment == "k-only" else "n/a"),
            ("k high-rho eps", config.k_highrho_eps if config.experiment == "k-only" else "n/a"),
            ("pair rho features", ", ".join(active_pair_rho_feature_names(config)) or "off"),
            (
                "pair rho normalization",
                (
                    f"eps={config.pair_rho_eps:g}, log/scale={config.pair_rho_log_scale:g}, "
                    f"log clip={config.pair_rho_log_clip:g}, product={config.pair_rho_product_transform}"
                )
                if active_pair_rho_feature_names(config)
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
        for _ in range(config.steps_per_epoch):
            system = choose_system(split.train_systems, rng)
            batch = None
            if config.experiment != "rho-only":
                batch = make_batch(system, config.batch_size, epoch, config.epochs, rng, config)
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
            if np.isfinite(val_objective) and val_objective < best_val:
                best_val = val_objective
                best_summary = {"epoch": epoch, "val": val_avg}
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
                f"{train_highrho_text}"
                f"{val_text}",
                flush=True,
            )

    train_avg, train_rows = evaluate_split(split.train_systems, "train", models, config, None, seed_offset=1001)
    val_avg, val_rows = evaluate_split(split.val_systems, "val", models, config, None, seed_offset=1101)
    test_avg, test_rows = evaluate_split(split.test_systems, "test", models, config, None, seed_offset=1201)
    summary = {
        "config": asdict(config),
        "best": best_summary,
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
