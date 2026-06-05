from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .config import ExperimentConfig
from .density_features import density_head_count, pair_density_feature_dim, pair_density_feature_mode
from .utils import print_block


@dataclass
class ModelBundle:
    """전체 1-RDM surrogate를 구성하는 서브모델 묶음."""

    point_model: tf.keras.Model
    mode_model: tf.keras.Model
    pair_model: tf.keras.Model
    context_model: tf.keras.Model


class RandomFourierFeatures(tf.keras.layers.Layer):
    """고정 random Fourier feature layer."""

    def __init__(self, n_features: int, scale: float, seed: int, include_input: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.scale = scale
        self.seed = seed
        self.include_input = include_input

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        rng = np.random.default_rng(self.seed)
        projection = rng.normal(0.0, self.scale, size=(input_dim, self.n_features)).astype(np.float32)
        self.projection = tf.constant(projection, dtype=tf.float32)
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        phase = 2.0 * np.pi * tf.matmul(x, self.projection)
        fourier = tf.concat([tf.sin(phase), tf.cos(phase)], axis=-1)
        if self.include_input:
            return tf.concat([x, fourier], axis=-1)
        return fourier


def build_mlp(
    *,
    input_dim: int,
    output_dim: int,
    width: int,
    depth: int,
    seed: int,
    rff_features: int,
    rff_scale: float,
) -> tf.keras.Model:
    """RFF + SiLU MLP."""
    inputs = tf.keras.Input(shape=(input_dim,))
    x = RandomFourierFeatures(
        n_features=rff_features,
        scale=rff_scale,
        seed=seed,
        include_input=True,
    )(inputs)

    for offset in range(depth):
        x = tf.keras.layers.Dense(
            width,
            activation=tf.nn.silu,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=seed + offset),
            bias_initializer="zeros",
        )(x)

    outputs = tf.keras.layers.Dense(
        output_dim,
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed + depth + 99),
        bias_initializer="zeros",
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_models(config: ExperimentConfig, point_feat_dim: int, pair_feat_dim: int, global_dim: int) -> ModelBundle:
    """모델 구성.

    point_model
        local point feature + global context -> density logits

    mode_model
        local point feature + global context -> raw residual mode amplitudes

    pair_model
        pair feature + global context -> baseline width(alpha) + baseline/residual gate
        (+ optional axis-resolved local curvature correction)

    context_model
        global context -> system-specific mode scales
    """
    n_density_heads = density_head_count(config)
    density_feature_dim = pair_density_feature_dim(config)
    point_model = build_mlp(
        input_dim=point_feat_dim + global_dim,
        output_dim=n_density_heads,
        width=config.model_width,
        depth=config.point_model_depth,
        seed=config.seed + 11,
        rff_features=config.rff_features,
        rff_scale=config.rff_scale,
    )
    mode_model = build_mlp(
        input_dim=point_feat_dim + global_dim,
        output_dim=config.learned_rank,
        width=config.model_width,
        depth=config.point_model_depth,
        seed=config.seed + 17,
        rff_features=config.rff_features,
        rff_scale=config.rff_scale,
    )
    pair_model = build_mlp(
        input_dim=pair_feat_dim + density_feature_dim + global_dim,
        output_dim=5 if config.use_local_curvature_kernel else 2,
        width=config.model_width,
        depth=2,
        seed=config.seed + 23,
        rff_features=config.rff_features,
        rff_scale=config.rff_scale,
    )
    context_model = build_mlp(
        input_dim=global_dim,
        output_dim=config.learned_rank,
        width=max(64, config.model_width // 2),
        depth=2,
        seed=config.seed + 37,
        rff_features=max(8, config.rff_features // 2),
        rff_scale=config.rff_scale,
    )

    dummy_point = tf.zeros((2, point_feat_dim + global_dim), dtype=tf.float32)
    dummy_pair = tf.zeros((2, pair_feat_dim + density_feature_dim + global_dim), dtype=tf.float32)
    dummy_context = tf.zeros((1, global_dim), dtype=tf.float32)
    _ = point_model(dummy_point)
    _ = mode_model(dummy_point)
    _ = pair_model(dummy_pair)
    _ = context_model(dummy_context)
    if config.use_local_curvature_kernel:
        weights = pair_model.get_weights()
        weights[-2][:, 2:] = 0.0
        if config.local_curvature_form.strip().lower() == "quadratic":
            weights[-1][2:] = float(config.local_curvature_init_bias)
        else:
            weights[-1][2:] = 0.0
        pair_model.set_weights(weights)
    local_basis_scale = (
        float(config.local_curvature_basis_scale)
        if config.local_curvature_basis_scale > 0.0
        else (
            float(config.domain_radius) ** 2
            if config.local_curvature_form.strip().lower() == "quadratic"
            else (float(config.domain_radius) / max(float(config.step), 1e-8)) ** 2
        )
    )
    local_basis_label = f"{local_basis_scale:.6g}" if config.local_curvature_basis_scale > 0.0 else "system-specific"

    print_block(
        "Model dimensions",
        [
            ("point input dim", point_feat_dim + global_dim),
            ("pair input dim", pair_feat_dim + density_feature_dim + global_dim),
            ("global dim", global_dim),
            ("point density heads", n_density_heads),
            ("pair density features", pair_density_feature_mode(config)),
            ("pair density feature dim", density_feature_dim),
            ("symmetric kernel output", config.symmetrize_kernel_output),
            ("local curvature kernel", config.use_local_curvature_kernel),
            ("local curvature form", config.local_curvature_form if config.use_local_curvature_kernel else "off"),
            ("local curvature scale", f"{config.local_curvature_scale:.6g}" if config.use_local_curvature_kernel else "off"),
            ("local curvature sigma", f"{config.local_curvature_sigma:.6g}" if config.use_local_curvature_kernel else "off"),
            ("local curvature init bias", f"{config.local_curvature_init_bias:.6g}" if config.use_local_curvature_kernel else "off"),
            ("local curvature basis scale", local_basis_label if config.use_local_curvature_kernel else "off"),
            ("point output dim", n_density_heads),
            ("learned_rank", config.learned_rank),
            ("point params", point_model.count_params()),
            ("mode params", mode_model.count_params()),
            ("pair params", pair_model.count_params()),
            ("context params", context_model.count_params()),
        ],
    )
    bundle = ModelBundle(
        point_model=point_model,
        mode_model=mode_model,
        pair_model=pair_model,
        context_model=context_model,
    )
    bundle._local_curvature_scale = float(config.local_curvature_scale)
    bundle._local_curvature_sigma = float(config.local_curvature_sigma)
    bundle._local_curvature_diag_eps = float(config.local_curvature_diag_eps)
    bundle._local_curvature_basis_scale = local_basis_scale
    bundle._local_curvature_form = config.local_curvature_form.strip().lower()
    bundle._symmetrize_kernel_output = bool(config.symmetrize_kernel_output)
    bundle._pair_density_symmetric = bool(config.pair_density_symmetric)
    return bundle


def initialize_point_model_density_bias(
    point_model: tf.keras.Model,
    rho_mean: float,
    *,
    residual_baseline: bool = False,
) -> None:
    """density head bias를 mean rho 근처로 맞춰 초기 폭주를 줄인다."""
    weights = point_model.get_weights()
    if residual_baseline:
        # delta=0 makes the first prediction exactly the normalized SAD density.
        weights[-2] = np.zeros_like(weights[-2])
        weights[-1] = np.zeros_like(weights[-1])
        point_model.set_weights(weights)
        return

    rho_mean = max(float(rho_mean), 1e-4)
    bias_value = np.log(np.expm1(rho_mean)).astype(np.float32)
    last_bias = weights[-1].copy()
    last_bias[:] = bias_value
    weights[-1] = last_bias
    point_model.set_weights(weights)


def make_mode_weights(global_context_t: tf.Tensor, models: ModelBundle) -> tf.Tensor:
    """global context -> descending positive mode weights.

    여기서는 explicit occupation-like ordering을 조금 흉내 내기 위해,
    residual mode weight를 cumulative descending 형태로 만든다.
    """
    global_context_t = tf.reshape(global_context_t, (1, -1))               # (1, d_global)
    raw_residual = tf.nn.softplus(models.context_model(global_context_t))  # (1, rank)
    descending = tf.reverse(tf.math.cumsum(tf.reverse(raw_residual, axis=[1]), axis=1), axis=[1])
    anchor = tf.ones((1, 1), dtype=tf.float32)                             # (1, 1)
    return tf.concat([anchor, descending + 1e-6], axis=1)                 # (1, rank_total)


def reverse_pair_density_features(pair_density_feat_t: tf.Tensor, models: ModelBundle) -> tf.Tensor:
    """Swap left/right endpoint descriptor slots for K_theta(r', r)."""
    if bool(getattr(models, "_pair_density_symmetric", False)):
        return pair_density_feat_t
    n_features = pair_density_feat_t.shape[-1]
    if n_features is None or int(n_features) == 0:
        return pair_density_feat_t
    if int(n_features) % 6 != 0:
        return pair_density_feat_t
    reshaped = tf.reshape(pair_density_feat_t, (tf.shape(pair_density_feat_t)[0], int(n_features) // 6, 6))
    reversed_slots = tf.gather(reshaped, [1, 0, 3, 2, 5, 4], axis=2)
    return tf.reshape(reversed_slots, tf.shape(pair_density_feat_t))


def local_curvature_window(sep_sq: tf.Tensor, models: ModelBundle) -> tf.Tensor:
    sigma_sq = max(float(getattr(models, "_local_curvature_sigma", 0.0)), 0.0) ** 2
    if sigma_sq <= 0.0:
        sigma_sq = 1.0
    if str(getattr(models, "_local_curvature_form", "quadratic")) == "quadratic":
        return tf.exp(-tf.maximum(sep_sq, 0.0) / sigma_sq)
    diag_eps = max(float(getattr(models, "_local_curvature_diag_eps", 1e-8)), 1e-12)
    return sep_sq / (sep_sq + diag_eps) * tf.exp(-tf.maximum(sep_sq, 0.0) / sigma_sq)


def kernel_from_pair_output(
    pair_out: tf.Tensor,
    residual_kernel: tf.Tensor,
    sep_sq: tf.Tensor,
    sep_sq_components: tf.Tensor,
    models: ModelBundle,
) -> dict[str, tf.Tensor]:
    alpha = tf.nn.softplus(pair_out[:, :1]) + 1e-6
    gate = tf.sigmoid(pair_out[:, 1:2])
    baseline_kernel = tf.exp(-alpha * tf.maximum(sep_sq, 0.0))
    gated_kernel = baseline_kernel * ((1.0 - gate) + gate * residual_kernel)

    if pair_out.shape[-1] is not None and int(pair_out.shape[-1]) >= 5:
        local_window = local_curvature_window(sep_sq, models)
        local_scale = float(getattr(models, "_local_curvature_scale", 0.0))
        curvature_form = str(getattr(models, "_local_curvature_form", "quadratic"))
        if curvature_form == "quadratic":
            local_axis_coeff = tf.nn.softplus(pair_out[:, 2:5])
            local_correction = (
                -0.5
                * local_scale
                * local_window
                * tf.reduce_sum(local_axis_coeff * sep_sq_components, axis=1, keepdims=True)
            )
        elif curvature_form in {"legacy", "signed"}:
            local_axis_coeff = tf.tanh(pair_out[:, 2:5])
            local_correction = (
                local_scale
                * local_window
                * tf.reduce_sum(local_axis_coeff * sep_sq_components, axis=1, keepdims=True)
            )
        else:
            raise ValueError(f"Unknown local curvature form: {curvature_form!r}. Choose 'quadratic' or 'legacy'.")
    elif pair_out.shape[-1] is not None and int(pair_out.shape[-1]) >= 3:
        local_window = local_curvature_window(sep_sq, models)
        local_axis_coeff = tf.zeros((tf.shape(pair_out)[0], 3), dtype=tf.float32)
        local_correction = (
            float(getattr(models, "_local_curvature_scale", 0.0))
            * local_window
            * tf.tanh(pair_out[:, 2:3])
        )
    else:
        local_window = tf.zeros_like(gated_kernel)
        local_axis_coeff = tf.zeros((tf.shape(pair_out)[0], 3), dtype=tf.float32)
        local_correction = tf.zeros_like(gated_kernel)

    kernel = gated_kernel + local_correction
    return {
        "kernel": kernel,
        "gated_kernel": gated_kernel,
        "baseline_kernel": baseline_kernel,
        "local_window": local_window,
        "local_axis_coeff": local_axis_coeff,
        "local_correction": local_correction,
        "alpha": alpha,
        "gate": gate,
    }


def average_kernel_outputs(
    forward: dict[str, tf.Tensor],
    reverse: dict[str, tf.Tensor],
) -> dict[str, tf.Tensor]:
    return {key: 0.5 * (forward[key] + reverse[key]) for key in forward}


def predict_from_features(
    point_feat_r_t: tf.Tensor,
    point_feat_rp_t: tf.Tensor,
    pair_feat_t: tf.Tensor,
    global_context_t: tf.Tensor,
    models: ModelBundle,
    eps: float = 1e-8,
    rho_r_override: tf.Tensor | None = None,
    rho_rp_override: tf.Tensor | None = None,
    pair_density_feat_t: tf.Tensor | None = None,
    local_curvature_basis_scale: float | None = None,
) -> dict[str, tf.Tensor]:
    """feature array로부터 gamma / rho / kernel / diagnostics 계산.

    Shapes
    ------
    point_feat_r_t, point_feat_rp_t : (batch, d_point)
    pair_feat_t                     : (batch, d_pair)
    global_context_t                : (d_global,) or (1, d_global)
    """
    global_context_t = tf.reshape(global_context_t, (1, -1))                      # (1, d_global)
    batch_size = tf.shape(point_feat_r_t)[0]
    tiled_global = tf.repeat(global_context_t, repeats=batch_size, axis=0)        # (batch, d_global)

    point_input_r = tf.concat([point_feat_r_t, tiled_global], axis=1)             # (batch, d_point+d_global)
    point_input_rp = tf.concat([point_feat_rp_t, tiled_global], axis=1)           # (batch, d_point+d_global)
    base_pair_feat_t = pair_feat_t
    if pair_density_feat_t is not None:
        pair_feat_t = tf.concat([base_pair_feat_t, pair_density_feat_t], axis=1)
    pair_input = tf.concat([pair_feat_t, tiled_global], axis=1)                   # (batch, d_pair+d_global)

    point_out_r = models.point_model(point_input_r)                               # (batch, density heads)
    point_out_rp = models.point_model(point_input_rp)                             # (batch, density heads)
    raw_modes_r = models.mode_model(point_input_r)                                # (batch, rank)
    raw_modes_rp = models.mode_model(point_input_rp)                              # (batch, rank)
    pair_out = models.pair_model(pair_input)                                      # (batch, 2 or 5)

    rho_raw_r = tf.nn.softplus(point_out_r[:, :1]) + 1e-6                         # (batch, 1)
    rho_raw_rp = tf.nn.softplus(point_out_rp[:, :1]) + 1e-6                       # (batch, 1)
    rho_r = rho_raw_r if rho_r_override is None else rho_r_override
    rho_rp = rho_raw_rp if rho_rp_override is None else rho_rp_override

    feat_r = tf.concat([tf.ones((batch_size, 1), dtype=tf.float32), raw_modes_r], axis=1)       # (batch, rank_total)
    feat_rp = tf.concat([tf.ones((batch_size, 1), dtype=tf.float32), raw_modes_rp], axis=1)     # (batch, rank_total)

    mode_weights = make_mode_weights(global_context_t, models)                    # (1, rank_total)
    sqrt_weights = tf.sqrt(mode_weights)                                          # (1, rank_total)

    weighted_feat_r = feat_r * sqrt_weights                                       # (batch, rank_total)
    weighted_feat_rp = feat_rp * sqrt_weights                                     # (batch, rank_total)

    unit_r = weighted_feat_r / tf.sqrt(tf.reduce_sum(weighted_feat_r**2, axis=1, keepdims=True) + eps)
    unit_rp = weighted_feat_rp / tf.sqrt(tf.reduce_sum(weighted_feat_rp**2, axis=1, keepdims=True) + eps)
    residual_kernel = tf.reduce_sum(unit_r * unit_rp, axis=1, keepdims=True)      # (batch, 1)

    basis_scale = (
        tf.constant(float(getattr(models, "_local_curvature_basis_scale", 1.0)), dtype=tf.float32)
        if local_curvature_basis_scale is None
        else tf.cast(local_curvature_basis_scale, tf.float32)
    )
    sep_sq_components = pair_feat_t[:, 6:9] * basis_scale                         # (batch, 3)
    sep_sq = pair_feat_t[:, 10:11]                                                # (batch, 1)
    kernel_outputs = kernel_from_pair_output(pair_out, residual_kernel, sep_sq, sep_sq_components, models)
    if bool(getattr(models, "_symmetrize_kernel_output", False)):
        if pair_density_feat_t is not None:
            reverse_density_feat_t = reverse_pair_density_features(pair_density_feat_t, models)
            reverse_pair_feat_t = tf.concat([base_pair_feat_t, reverse_density_feat_t], axis=1)
        else:
            reverse_pair_feat_t = base_pair_feat_t
        reverse_pair_input = tf.concat([reverse_pair_feat_t, tiled_global], axis=1)
        reverse_pair_out = models.pair_model(reverse_pair_input)
        reverse_kernel_outputs = kernel_from_pair_output(
            reverse_pair_out, residual_kernel, sep_sq, sep_sq_components, models
        )
        kernel_outputs = average_kernel_outputs(kernel_outputs, reverse_kernel_outputs)

    kernel = kernel_outputs["kernel"]                                             # (batch, 1)
    gamma = tf.sqrt(rho_r * rho_rp) * kernel                                      # (batch, 1)

    return {
        "gamma": gamma,
        "rho_r": rho_r,
        "rho_rp": rho_rp,
        "rho_raw_r": rho_raw_r,
        "rho_raw_rp": rho_raw_rp,
        "kernel": kernel,
        "gated_kernel": kernel_outputs["gated_kernel"],
        "baseline_kernel": kernel_outputs["baseline_kernel"],
        "residual_kernel": residual_kernel,
        "local_window": kernel_outputs["local_window"],
        "local_axis_coeff": kernel_outputs["local_axis_coeff"],
        "local_correction": kernel_outputs["local_correction"],
        "alpha": kernel_outputs["alpha"],
        "gate": kernel_outputs["gate"],
        "mode_weights": mode_weights,
    }
