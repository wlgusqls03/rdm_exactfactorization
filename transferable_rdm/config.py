from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no", "n"}


def env_float_any(names: tuple[str, ...], default: float) -> float:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return float(value)
    return default


@dataclass(frozen=True)
class ExperimentConfig:
    """Transferable 1-RDM 실험 전체 설정.

    이 설정은 "단일 toy를 외우는 실험"이 아니라, 여러 시스템을 함께 학습하고
    held-out system에서 일반화 성능을 보는 실험을 목표로 한다.
    """

    # ------------------------------------------------------------------
    # 데이터 모드
    # ------------------------------------------------------------------
    dataset_mode: str = os.environ.get("RDM_DATASET_MODE", "ks_like")
    # "ks_like"  : synthetic separable KS-like systems 생성
    # "toy"      : 1D/2D/3D active-axis toy systems 생성
    # "npz"      : 외부 NPZ dataset 로드
    # "mixed"    : npz가 있으면 같이 섞고, 없으면 ks_like만 사용
    npz_glob: str = os.environ.get("RDM_NPZ_GLOB", "")
    toy_dimensions: str = os.environ.get("RDM_TOY_DIMENSIONS", "1,2,3")
    phase: str = os.environ.get("RDM_PHASE", "none")
    # "none"    : 기존 설정 사용
    # "phase1a" : QMugs-NPZ prototype, 300 train / 100 val
    # "phase1b" : QMugs-NPZ generalization check, 800 train / 100 val / 100 test

    # ------------------------------------------------------------------
    # 시스템 / grid 설정
    # ------------------------------------------------------------------
    seed: int = int(os.environ.get("RDM_SEED", 0))
    num_systems: int = int(os.environ.get("RDM_NUM_SYSTEMS", 500))
    train_system_fraction: float = float(os.environ.get("RDM_TRAIN_SYSTEM_FRACTION", 0.75))
    train_system_count: int = int(os.environ.get("RDM_TRAIN_SYSTEM_COUNT", 0))
    val_system_count: int = int(os.environ.get("RDM_VAL_SYSTEM_COUNT", 0))
    test_system_count: int = int(os.environ.get("RDM_TEST_SYSTEM_COUNT", 0))

    axis_points: int = int(os.environ.get("RDM_AXIS_POINTS", 7))
    domain_radius: float = float(os.environ.get("RDM_DOMAIN_RADIUS", 4.0))
    max_wells: int = int(os.environ.get("RDM_MAX_WELLS", 2))
    max_orbitals: int = int(os.environ.get("RDM_MAX_ORBITALS", 8))
    toy_inactive_omega: float = float(os.environ.get("RDM_TOY_INACTIVE_OMEGA", 0.8))
    spectral_subset_points: int = int(os.environ.get("RDM_SPECTRAL_SUBSET_POINTS", 64))
    eval_pair_count: int = int(os.environ.get("RDM_EVAL_PAIR_COUNT", 32768))
    full_eval_max_points: int = int(os.environ.get("RDM_FULL_EVAL_MAX_POINTS", 2500))

    # ------------------------------------------------------------------
    # 모델 설정
    # ------------------------------------------------------------------
    model_width: int = int(os.environ.get("RDM_MODEL_WIDTH", 192))
    point_model_depth: int = int(os.environ.get("RDM_POINT_MODEL_DEPTH", 3))
    learned_rank: int = int(os.environ.get("RDM_LEARNED_RANK", 8))
    rff_features: int = int(os.environ.get("RDM_RFF_FEATURES", 32))
    rff_scale: float = float(os.environ.get("RDM_RFF_SCALE", 2.0))
    occ_max: float = float(os.environ.get("RDM_OCC_MAX", 1.0))
    normalize_rho: bool = env_flag("RDM_NORMALIZE_RHO", True)
    density_source: str = os.environ.get("RDM_DENSITY_SOURCE", "predicted")
    tau_stencil: str = os.environ.get("RDM_TAU_STENCIL", "central2")
    pair_density_feature_mode: str = os.environ.get("RDM_PAIR_DENSITY_FEATURE_MODE", "off")
    pair_density_symmetric: bool = env_flag("RDM_PAIR_DENSITY_SYMMETRIC", True)
    pair_density_hessian: bool = env_flag("RDM_PAIR_DENSITY_HESSIAN", False)
    pair_density_eps: float = float(os.environ.get("RDM_PAIR_DENSITY_EPS", 1e-14))
    pair_density_value_clip: float = float(os.environ.get("RDM_PAIR_DENSITY_VALUE_CLIP", 8.0))
    pair_density_laplacian_clip: float = float(os.environ.get("RDM_PAIR_DENSITY_LAPLACIAN_CLIP", 8.0))
    pair_density_hessian_clip: float = float(os.environ.get("RDM_PAIR_DENSITY_HESSIAN_CLIP", 8.0))
    use_potential_laplacian_feature: bool = env_flag("RDM_USE_POTENTIAL_LAPLACIAN_FEATURE", True)
    potential_laplacian_clip: float = float(os.environ.get("RDM_POTENTIAL_LAPLACIAN_CLIP", 8.0))
    density_baseline_mode: str = os.environ.get("RDM_DENSITY_BASELINE_MODE", "learned")
    sad_density_floor: float = float(os.environ.get("RDM_SAD_DENSITY_FLOOR", 1e-8))
    sad_residual_clip: float = float(os.environ.get("RDM_SAD_RESIDUAL_CLIP", 4.0))
    symmetrize_kernel_output: bool = env_flag("RDM_SYMMETRIZE_KERNEL_OUTPUT", True)
    use_local_curvature_kernel: bool = env_flag("RDM_USE_LOCAL_CURVATURE_KERNEL", True)
    local_curvature_form: str = os.environ.get("RDM_LOCAL_CURVATURE_FORM", "quadratic")
    local_curvature_scale: float = float(os.environ.get("RDM_LOCAL_CURVATURE_SCALE", 0.5))
    # Width in normalized separation units, where 1.0 is the system domain radius.
    # RDM_LOCAL_CURVATURE_SIGMA remains a backward-compatible alias.
    local_curvature_sigma: float = float(
        os.environ.get(
            "RDM_LOCAL_CURVATURE_SIGMA_NORM",
            os.environ.get("RDM_LOCAL_CURVATURE_SIGMA", 1.0),
        )
    )
    local_curvature_diag_eps: float = float(os.environ.get("RDM_LOCAL_CURVATURE_DIAG_EPS", 1e-8))
    local_curvature_basis_scale: float = float(os.environ.get("RDM_LOCAL_CURVATURE_BASIS_SCALE", 0.0))
    local_curvature_init_bias: float = float(os.environ.get("RDM_LOCAL_CURVATURE_INIT_BIAS", -10.0))

    # ------------------------------------------------------------------
    # 학습 설정
    # ------------------------------------------------------------------
    batch_size: int = int(os.environ.get("RDM_BATCH_SIZE", 2048))
    steps_per_epoch: int = int(os.environ.get("RDM_STEPS_PER_EPOCH", 80))
    epochs: int = int(os.environ.get("RDM_EPOCHS", 400))
    val_every: int = int(os.environ.get("RDM_VAL_EVERY", 5))
    log_every: int = int(os.environ.get("RDM_LOG_EVERY", 1))
    early_stopping_patience: int = int(os.environ.get("RDM_PATIENCE", 40))
    compile_train_step: bool = env_flag("RDM_COMPILE_TRAIN_STEP", True)
    active_system_tensor_cache_size: int = int(os.environ.get("RDM_ACTIVE_SYSTEM_TENSOR_CACHE_SIZE", 2))
    gradient_diagnostics: bool = env_flag("RDM_GRADIENT_DIAGNOSTICS", False)
    gradient_diagnostics_every: int = int(os.environ.get("RDM_GRADIENT_DIAGNOSTICS_EVERY", 5))
    gradient_diagnostics_start_epoch: int = int(os.environ.get("RDM_GRADIENT_DIAGNOSTICS_START_EPOCH", 0))
    gradient_diagnostic_mode: str = os.environ.get("RDM_GRADIENT_DIAGNOSTIC_MODE", "fast")
    gradient_diagnostic_stencil_centers: int = int(os.environ.get("RDM_GRADIENT_DIAGNOSTIC_STENCIL_CENTERS", 256))
    eval_stencil_centers: int = int(os.environ.get("RDM_EVAL_STENCIL_CENTERS", 4096))
    eval_full_final: bool = env_flag("RDM_EVAL_FULL_FINAL", True)
    val_eval_system_count: int = int(os.environ.get("RDM_VAL_EVAL_SYSTEM_COUNT", 0))
    final_train_eval_system_count: int = int(os.environ.get("RDM_FINAL_TRAIN_EVAL_SYSTEM_COUNT", 0))
    final_val_eval_system_count: int = int(os.environ.get("RDM_FINAL_VAL_EVAL_SYSTEM_COUNT", 0))
    final_test_eval_system_count: int = int(os.environ.get("RDM_FINAL_TEST_EVAL_SYSTEM_COUNT", 0))
    train_diagonal_points: int = int(os.environ.get("RDM_TRAIN_DIAGONAL_POINTS", 4096))
    train_stencil_centers: int = int(os.environ.get("RDM_TRAIN_STENCIL_CENTERS", 0))
    stencil_feature_cache_max_centers: int = int(os.environ.get("RDM_STENCIL_FEATURE_CACHE_MAX_CENTERS", 200000))
    stencil_prediction_chunk_size: int = int(os.environ.get("RDM_STENCIL_PREDICTION_CHUNK_SIZE", 65536))
    overfit_one_system: bool = env_flag("RDM_OVERFIT_ONE_SYSTEM", False)
    overfit_system_index: int = int(os.environ.get("RDM_OVERFIT_SYSTEM_INDEX", 0))
    overfit_system_id: str = os.environ.get("RDM_OVERFIT_SYSTEM_ID", "")
    physics_target: str = os.environ.get("RDM_PHYSICS_TARGET", "orbital")

    initial_lr: float = env_float_any(("RDM_LEARNING_RATE", "RDM_INITIAL_LR", "RDM_LR"), 3e-4)
    min_lr: float = float(os.environ.get("RDM_MIN_LR", 1e-5))
    lr_decay: float = float(os.environ.get("RDM_LR_DECAY", 0.5))
    lr_patience: int = int(os.environ.get("RDM_LR_PATIENCE", 12))
    lr_min_improvement: float = float(os.environ.get("RDM_LR_MIN_IMPROVEMENT", 1e-5))
    weight_decay: float = float(os.environ.get("RDM_WEIGHT_DECAY", 1e-6))
    point_pretrain_epochs: int = int(os.environ.get("RDM_POINT_PRETRAIN_EPOCHS", 120))
    point_pretrain_steps_per_epoch: int = int(os.environ.get("RDM_POINT_PRETRAIN_STEPS_PER_EPOCH", 80))
    point_pretrain_lr: float = float(os.environ.get("RDM_POINT_PRETRAIN_LR", 3e-4))
    point_pretrain_patience: int = int(os.environ.get("RDM_POINT_PRETRAIN_PATIENCE", 30))
    point_pretrain_val_every: int = int(os.environ.get("RDM_POINT_PRETRAIN_VAL_EVERY", 5))
    point_charged_weight: float = float(os.environ.get("RDM_POINT_CHARGED_WEIGHT", 0.25))
    point_fukui_weight: float = float(os.environ.get("RDM_POINT_FUKUI_WEIGHT", 0.05))
    point_charged_start_epoch: int = int(os.environ.get("RDM_POINT_CHARGED_START_EPOCH", 30))
    point_fukui_start_epoch: int = int(os.environ.get("RDM_POINT_FUKUI_START_EPOCH", 60))
    point_fukui_ramp_epochs: int = int(os.environ.get("RDM_POINT_FUKUI_RAMP_EPOCHS", 40))
    point_density_scale_floor: float = float(os.environ.get("RDM_POINT_DENSITY_SCALE_FLOOR", 1e-3))
    point_density_mse_weight: float = float(os.environ.get("RDM_POINT_DENSITY_MSE_WEIGHT", 1.0))
    point_density_rel_l1_weight: float = float(os.environ.get("RDM_POINT_DENSITY_REL_L1_WEIGHT", 0.25))
    point_density_log_weight: float = float(os.environ.get("RDM_POINT_DENSITY_LOG_WEIGHT", 0.02))
    point_density_log_eps: float = float(os.environ.get("RDM_POINT_DENSITY_LOG_EPS", 1e-8))
    point_delta_weight: float = float(os.environ.get("RDM_POINT_DELTA_WEIGHT", 0.0))
    point_delta_huber: float = float(os.environ.get("RDM_POINT_DELTA_HUBER", 1.0))
    point_delta_eps: float = float(os.environ.get("RDM_POINT_DELTA_EPS", 1e-12))
    point_pretrain_lr_decay: float = float(os.environ.get("RDM_POINT_PRETRAIN_LR_DECAY", 0.5))
    point_pretrain_lr_patience: int = int(os.environ.get("RDM_POINT_PRETRAIN_LR_PATIENCE", 15))
    point_pretrain_min_lr: float = float(os.environ.get("RDM_POINT_PRETRAIN_MIN_LR", 1e-5))
    freeze_point_after_pretrain: bool = env_flag("RDM_FREEZE_POINT_AFTER_PRETRAIN", True)

    # ------------------------------------------------------------------
    # loss weight
    # ------------------------------------------------------------------
    loss_preset: str = os.environ.get("RDM_LOSS_PRESET", "core5")
    # "all" / "custom" : individual RDM_USE_*_LOSS switches decide active terms
    # "core5"          : gamma + rho + kernel + trace + mode only
    # "staged-physics" : core5 + RMS-normalized Huber derivative/tau terms

    use_gamma_loss: bool = env_flag("RDM_USE_GAMMA_LOSS", True)
    use_rho_loss: bool = env_flag("RDM_USE_RHO_LOSS", True)
    use_kernel_loss: bool = env_flag("RDM_USE_KERNEL_LOSS", True)
    use_deriv_loss: bool = env_flag("RDM_USE_DERIV_LOSS", True)
    use_tau_loss: bool = env_flag("RDM_USE_TAU_LOSS", True)
    use_trace_loss: bool = env_flag("RDM_USE_TRACE_LOSS", True)
    use_occ_loss: bool = env_flag("RDM_USE_OCC_LOSS", True)
    use_mode_loss: bool = env_flag("RDM_USE_MODE_LOSS", True)
    use_kinetic_loss: bool = env_flag("RDM_USE_KINETIC_LOSS", False)

    lambda_gamma: float = float(os.environ.get("RDM_LAMBDA_GAMMA", 1.0))
    lambda_rho: float = float(os.environ.get("RDM_LAMBDA_RHO", 8.0))
    lambda_kernel: float = float(os.environ.get("RDM_LAMBDA_KERNEL", 1.0))
    lambda_deriv: float = float(os.environ.get("RDM_LAMBDA_DERIV", 2.0))
    lambda_tau: float = float(os.environ.get("RDM_LAMBDA_TAU", 0.5))
    lambda_trace: float = float(os.environ.get("RDM_LAMBDA_TRACE", 3.0))
    lambda_occ: float = float(os.environ.get("RDM_LAMBDA_OCC", 0.25))
    lambda_mode: float = float(os.environ.get("RDM_LAMBDA_MODE", 1e-4))
    lambda_kinetic: float = float(os.environ.get("RDM_LAMBDA_KINETIC", 1.0))

    # ------------------------------------------------------------------
    # loss schedule
    # ------------------------------------------------------------------
    # These keep the base loss definition intact, but allow expensive or
    # delicate targets to enter only after the density/gamma fit has stabilized.
    deriv_start_epoch: int = int(os.environ.get("RDM_DERIV_START_EPOCH", 30))
    deriv_ramp_epochs: int = int(os.environ.get("RDM_DERIV_RAMP_EPOCHS", 40))
    tau_start_epoch: int = int(os.environ.get("RDM_TAU_START_EPOCH", 30))
    tau_ramp_epochs: int = int(os.environ.get("RDM_TAU_RAMP_EPOCHS", 40))
    kinetic_start_epoch: int = int(os.environ.get("RDM_KINETIC_START_EPOCH", 0))
    kinetic_ramp_epochs: int = int(os.environ.get("RDM_KINETIC_RAMP_EPOCHS", 0))
    kinetic_control_variate: bool = env_flag("RDM_KINETIC_CONTROL_VARIATE", True)
    physics_huber_delta: float = float(os.environ.get("RDM_PHYSICS_HUBER_DELTA", 1.0))
    deriv_scale_floor: float = float(os.environ.get("RDM_DERIV_SCALE_FLOOR", 1e-6))
    tau_scale_floor: float = float(os.environ.get("RDM_TAU_SCALE_FLOOR", 1e-6))

    # ------------------------------------------------------------------
    # 출력 설정
    # ------------------------------------------------------------------
    output_dir: str = os.environ.get("RDM_OUTPUT_DIR", "transferable_outputs")
    run_name: str = os.environ.get("RDM_RUN_NAME", "transferable_ks_like")
    auto_run_dir: bool = env_flag("RDM_AUTO_RUN_DIR", False)
    rotate_output_dir: bool = env_flag("RDM_ROTATE_OUTPUT_DIR", False)
    output_rotation_depth: int = int(os.environ.get("RDM_OUTPUT_ROTATION_DEPTH", 2))

    @property
    def step(self) -> float:
        """균일 grid spacing."""
        return 2.0 * self.domain_radius / max(self.axis_points - 1, 1)

    @property
    def cell_volume(self) -> float:
        """uniform real-space cell volume."""
        return self.step**3

    @property
    def n_points(self) -> int:
        return self.axis_points**3

    def output_path(self, filename: str) -> Path:
        out_dir = Path(self.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / filename
