from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no", "n"}


def configure_tensorflow_environment_preimport() -> None:
    """Apply device-selection env vars before TensorFlow is imported."""
    requested_device = os.environ.get("RDM_DEVICE", "auto").strip().lower()
    gpu_ids = os.environ.get("RDM_GPU_IDS", "").strip()

    if requested_device in {"cpu", "none"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif requested_device in {"gpu", "cuda"}:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    elif requested_device == "auto":
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    else:
        raise ValueError("RDM_DEVICE must be one of: auto, cpu, gpu.")

    if env_flag("RDM_GPU_MEMORY_GROWTH", True):
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


configure_tensorflow_environment_preimport()

from transferable_rdm.config import ExperimentConfig
from transferable_rdm.data import DatasetSplit, build_pair_features, compact_system_ids, split_systems
from transferable_rdm.density_features import (
    DENSITY_BASELINE_MODES,
    DENSITY_SOURCES,
    PAIR_DENSITY_FEATURE_MODES,
    density_baseline_mode,
    density_source_mode,
    pair_density_feature_dim,
    pair_density_feature_mode,
)
from transferable_rdm.model import build_models, initialize_point_model_density_bias
from transferable_rdm.plotting import plot_point_pretrain_summary, plot_training_summary
from transferable_rdm.systems import build_system_corpus
from transferable_rdm.training import pretrain_point_model, train_models
from transferable_rdm.utils import print_block, save_json, set_global_seed


CSV_METRIC_KEYS = [
    "objective",
    "pair_loss",
    "pair_mae",
    "rho_loss",
    "density_mae",
    "rho_point_mae",
    "rho_cation_mae",
    "rho_anion_mae",
    "fukui_plus_mae",
    "fukui_minus_mae",
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
    "kinetic_loss",
    "kinetic_mse_loss",
    "kinetic_pred",
    "kinetic_training_ref",
    "kinetic_ref_error",
    "kinetic_abs_error",
    "kinetic_abs_error_p90",
    "kinetic_sq_error",
    "kinetic_rmse",
    "kinetic_rel_abs_error",
    "kinetic_rel_abs_error_p90",
    "kinetic_rel_sq_error",
    "kinetic_rel_rmse",
    "kinetic_stencil_diag_error",
    "kinetic_stencil_offdiag_error",
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
    "energy_total_rmse",
    "energy_grid_total_abs_error",
    "energy_grid_total_rmse",
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

CSV_SYSTEM_KEYS = [
    "split",
    "system_id",
    "family",
    "toy_dimension",
    "particle_mass",
    "formula",
    "axis_points",
    "n_points",
    "grid_spacing_bohr",
    "electron_count",
    "gamma_fd_target_source",
    "kinetic_evaluation_mode",
] + CSV_METRIC_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transferable 3D 1-RDM surrogate.")
    parser.add_argument("--dataset-mode", choices=["ks_like", "toy", "npz", "mixed"], default=None)
    parser.add_argument("--phase", choices=["none", "phase1a", "phase1b"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-systems", type=int, default=None)
    parser.add_argument("--train-system-count", type=int, default=None)
    parser.add_argument("--val-system-count", type=int, default=None)
    parser.add_argument("--test-system-count", type=int, default=None)
    parser.add_argument("--axis-points", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-pair-count", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--point-model-depth", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--loss-preset", type=str, default=None)
    parser.add_argument("--lambda-kinetic", type=float, default=None)
    parser.add_argument("--lambda-tau-mse", type=float, default=None)
    parser.add_argument("--lambda-kinetic-mse", type=float, default=None)
    parser.add_argument("--deriv-start-epoch", type=int, default=None)
    parser.add_argument("--deriv-ramp-epochs", type=int, default=None)
    parser.add_argument("--tau-start-epoch", type=int, default=None)
    parser.add_argument("--tau-ramp-epochs", type=int, default=None)
    parser.add_argument("--kinetic-start-epoch", type=int, default=None)
    parser.add_argument("--kinetic-ramp-epochs", type=int, default=None)
    parser.add_argument("--train-stencil-centers", type=int, default=None)
    parser.add_argument("--stencil-feature-cache-max-centers", type=int, default=None)
    parser.add_argument("--stencil-prediction-chunk-size", type=int, default=None)
    parser.add_argument("--diagonal-prediction-chunk-size", type=int, default=None)
    parser.add_argument(
        "--use-kinetic-loss",
        dest="use_kinetic_loss",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-use-kinetic-loss",
        dest="use_kinetic_loss",
        action="store_false",
    )
    parser.add_argument(
        "--use-tau-mse-loss",
        dest="use_tau_mse_loss",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-use-tau-mse-loss",
        dest="use_tau_mse_loss",
        action="store_false",
    )
    parser.add_argument(
        "--use-kinetic-mse-loss",
        dest="use_kinetic_mse_loss",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-use-kinetic-mse-loss",
        dest="use_kinetic_mse_loss",
        action="store_false",
    )
    parser.add_argument("--point-pretrain-epochs", type=int, default=None)
    parser.add_argument("--point-pretrain-steps-per-epoch", type=int, default=None)
    parser.add_argument("--point-pretrain-lr", type=float, default=None)
    parser.add_argument("--point-charged-weight", type=float, default=None)
    parser.add_argument("--point-fukui-weight", type=float, default=None)
    parser.add_argument("--point-charged-start-epoch", type=int, default=None)
    parser.add_argument("--point-fukui-start-epoch", type=int, default=None)
    parser.add_argument("--point-fukui-ramp-epochs", type=int, default=None)
    parser.add_argument("--point-density-scale-floor", type=float, default=None)
    parser.add_argument("--point-density-mse-weight", type=float, default=None)
    parser.add_argument("--point-density-rel-l1-weight", type=float, default=None)
    parser.add_argument("--point-density-log-weight", type=float, default=None)
    parser.add_argument("--point-density-log-eps", type=float, default=None)
    parser.add_argument("--density-source", choices=DENSITY_SOURCES, default=None)
    parser.add_argument("--pair-density-feature-mode", choices=PAIR_DENSITY_FEATURE_MODES, default=None)
    parser.add_argument("--pair-density-symmetric", dest="pair_density_symmetric", action="store_true", default=None)
    parser.add_argument("--no-pair-density-symmetric", dest="pair_density_symmetric", action="store_false")
    parser.add_argument("--pair-density-hessian", dest="pair_density_hessian", action="store_true", default=None)
    parser.add_argument("--no-pair-density-hessian", dest="pair_density_hessian", action="store_false")
    parser.add_argument("--pair-density-hessian-clip", type=float, default=None)
    parser.add_argument(
        "--use-potential-laplacian-feature",
        dest="use_potential_laplacian_feature",
        action="store_true",
        default=None,
    )
    parser.add_argument("--no-potential-laplacian-feature", dest="use_potential_laplacian_feature", action="store_false")
    parser.add_argument("--potential-laplacian-clip", type=float, default=None)
    parser.add_argument("--density-baseline-mode", choices=DENSITY_BASELINE_MODES, default=None)
    parser.add_argument("--sad-density-floor", type=float, default=None)
    parser.add_argument("--sad-residual-clip", type=float, default=None)
    parser.add_argument("--symmetrize-kernel-output", dest="symmetrize_kernel_output", action="store_true", default=None)
    parser.add_argument("--no-symmetrize-kernel-output", dest="symmetrize_kernel_output", action="store_false")
    parser.add_argument("--local-curvature-form", choices=["quadratic", "legacy"], default=None)
    parser.add_argument("--local-curvature-init-bias", type=float, default=None)
    parser.add_argument(
        "--physics-target",
        choices=["orbital", "ao", "fd"],
        default=None,
        help="'orbital' is the preferred name; legacy 'ao' is an exact alias.",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--resume-summary-json",
        type=Path,
        default=None,
        help="Restore saved config and model weights from a previous run summary.",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=Path,
        default=None,
        help="Path prefix before _point.weights.h5. Defaults to the resume summary run name.",
    )
    parser.add_argument("--npz-glob", type=str, default=None)
    parser.add_argument(
        "--toy-dimensions",
        type=str,
        default=None,
        help="Comma-separated active dimensions for synthetic toy systems, e.g. 1,2,3.",
    )
    parser.add_argument(
        "--toy-particle-mass",
        type=float,
        default=None,
        help="Particle mass for toy Schrodinger systems in electron-mass units.",
    )
    parser.add_argument("--auto-run-dir", dest="auto_run_dir", action="store_true", default=None)
    parser.add_argument("--no-auto-run-dir", dest="auto_run_dir", action="store_false")
    parser.add_argument("--overfit-one-system", dest="overfit_one_system", action="store_true", default=None)
    parser.add_argument("--no-overfit-one-system", dest="overfit_one_system", action="store_false")
    parser.add_argument("--overfit-system-index", type=int, default=None)
    parser.add_argument("--overfit-system-id", type=str, default=None)
    return parser.parse_args()


def apply_phase_preset(config: ExperimentConfig) -> ExperimentConfig:
    """Phase preset for the first QMugs-NPZ experiments."""
    if config.phase == "none":
        return config
    if config.phase == "phase1a":
        updates = {
            "dataset_mode": "npz",
            "axis_points": 7,
            "train_system_count": 300,
            "val_system_count": 100,
            "test_system_count": 0,
        }
        if config.run_name == "transferable_ks_like":
            updates["run_name"] = "qmugs_phase1a"
        return replace(config, **updates)
    if config.phase == "phase1b":
        updates = {
            "dataset_mode": "npz",
            "axis_points": 7,
            "train_system_count": 800,
            "val_system_count": 100,
            "test_system_count": 100,
        }
        if config.run_name == "transferable_ks_like":
            updates["run_name"] = "qmugs_phase1b"
        return replace(config, **updates)
    raise ValueError(f"Unknown RDM phase: {config.phase}")


def apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    updates = {}
    for field_name, arg_name in [
        ("dataset_mode", "dataset_mode"),
        ("phase", "phase"),
        ("epochs", "epochs"),
        ("seed", "seed"),
        ("num_systems", "num_systems"),
        ("train_system_count", "train_system_count"),
        ("val_system_count", "val_system_count"),
        ("test_system_count", "test_system_count"),
        ("axis_points", "axis_points"),
        ("batch_size", "batch_size"),
        ("eval_pair_count", "eval_pair_count"),
        ("model_width", "width"),
        ("point_model_depth", "point_model_depth"),
        ("learned_rank", "rank"),
        ("steps_per_epoch", "steps_per_epoch"),
        ("val_every", "val_every"),
        ("log_every", "log_every"),
        ("initial_lr", "learning_rate"),
        ("min_lr", "min_learning_rate"),
        ("loss_preset", "loss_preset"),
        ("lambda_kinetic", "lambda_kinetic"),
        ("lambda_tau_mse", "lambda_tau_mse"),
        ("lambda_kinetic_mse", "lambda_kinetic_mse"),
        ("deriv_start_epoch", "deriv_start_epoch"),
        ("deriv_ramp_epochs", "deriv_ramp_epochs"),
        ("tau_start_epoch", "tau_start_epoch"),
        ("tau_ramp_epochs", "tau_ramp_epochs"),
        ("kinetic_start_epoch", "kinetic_start_epoch"),
        ("kinetic_ramp_epochs", "kinetic_ramp_epochs"),
        ("train_stencil_centers", "train_stencil_centers"),
        ("stencil_feature_cache_max_centers", "stencil_feature_cache_max_centers"),
        ("stencil_prediction_chunk_size", "stencil_prediction_chunk_size"),
        ("diagonal_prediction_chunk_size", "diagonal_prediction_chunk_size"),
        ("use_kinetic_loss", "use_kinetic_loss"),
        ("use_tau_mse_loss", "use_tau_mse_loss"),
        ("use_kinetic_mse_loss", "use_kinetic_mse_loss"),
        ("point_pretrain_epochs", "point_pretrain_epochs"),
        ("point_pretrain_steps_per_epoch", "point_pretrain_steps_per_epoch"),
        ("point_pretrain_lr", "point_pretrain_lr"),
        ("point_charged_weight", "point_charged_weight"),
        ("point_fukui_weight", "point_fukui_weight"),
        ("point_charged_start_epoch", "point_charged_start_epoch"),
        ("point_fukui_start_epoch", "point_fukui_start_epoch"),
        ("point_fukui_ramp_epochs", "point_fukui_ramp_epochs"),
        ("point_density_scale_floor", "point_density_scale_floor"),
        ("point_density_mse_weight", "point_density_mse_weight"),
        ("point_density_rel_l1_weight", "point_density_rel_l1_weight"),
        ("point_density_log_weight", "point_density_log_weight"),
        ("point_density_log_eps", "point_density_log_eps"),
        ("density_source", "density_source"),
        ("pair_density_feature_mode", "pair_density_feature_mode"),
        ("pair_density_symmetric", "pair_density_symmetric"),
        ("pair_density_hessian", "pair_density_hessian"),
        ("pair_density_hessian_clip", "pair_density_hessian_clip"),
        ("use_potential_laplacian_feature", "use_potential_laplacian_feature"),
        ("potential_laplacian_clip", "potential_laplacian_clip"),
        ("density_baseline_mode", "density_baseline_mode"),
        ("sad_density_floor", "sad_density_floor"),
        ("sad_residual_clip", "sad_residual_clip"),
        ("symmetrize_kernel_output", "symmetrize_kernel_output"),
        ("local_curvature_form", "local_curvature_form"),
        ("local_curvature_init_bias", "local_curvature_init_bias"),
        ("physics_target", "physics_target"),
        ("run_name", "run_name"),
        ("output_dir", "output_dir"),
        ("npz_glob", "npz_glob"),
        ("toy_dimensions", "toy_dimensions"),
        ("toy_particle_mass", "toy_particle_mass"),
        ("auto_run_dir", "auto_run_dir"),
        ("overfit_one_system", "overfit_one_system"),
        ("overfit_system_index", "overfit_system_index"),
        ("overfit_system_id", "overfit_system_id"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    return replace(config, **updates) if updates else config


def config_from_resume_summary(path: Path) -> tuple[ExperimentConfig, dict[str, object]]:
    summary_path = path.resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    valid_fields = {item.name for item in fields(ExperimentConfig)}
    saved = {
        key: value
        for key, value in payload.get("config", {}).items()
        if key in valid_fields
    }
    if not saved:
        raise ValueError(f"Resume summary has no saved ExperimentConfig: {summary_path}")
    return replace(ExperimentConfig(), **saved), payload


def resume_checkpoint_prefix(
    summary_path: Path,
    payload: dict[str, object],
    requested: Path | None,
) -> Path:
    if requested is not None:
        return requested.resolve()
    run_name = str(payload.get("config", {}).get("run_name", "")).strip()
    if not run_name:
        raise ValueError("Resume summary config does not contain run_name.")
    return summary_path.resolve().parent / run_name


def load_model_weights(models, prefix: Path) -> dict[str, Path]:
    paths = {
        "point": Path(f"{prefix}_point.weights.h5"),
        "mode": Path(f"{prefix}_mode.weights.h5"),
        "pair": Path(f"{prefix}_pair.weights.h5"),
        "context": Path(f"{prefix}_context.weights.h5"),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing resume checkpoint file(s): " + ", ".join(missing))
    models.point_model.load_weights(paths["point"])
    models.mode_model.load_weights(paths["mode"])
    models.pair_model.load_weights(paths["pair"])
    models.context_model.load_weights(paths["context"])
    return paths


def apply_auto_run_dir(config: ExperimentConfig) -> ExperimentConfig:
    if not config.auto_run_dir:
        return config
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run_name = config.run_name.strip() or "run"
    output_dir = Path(config.output_dir) / f"{safe_run_name}_{stamp}"
    return replace(config, output_dir=str(output_dir))


def configure_system_feature_metadata(systems: list, config: ExperimentConfig) -> None:
    for system in systems:
        system.metadata["use_potential_laplacian_feature"] = bool(config.use_potential_laplacian_feature)
        system.metadata["potential_laplacian_clip"] = float(config.potential_laplacian_clip)


def make_overfit_split(systems: list, config: ExperimentConfig) -> DatasetSplit:
    """Use one system for train/val/test to test representational capacity."""
    if not systems:
        raise RuntimeError("No systems are available for one-system overfit mode.")
    selected = None
    requested_id = config.overfit_system_id.strip()
    if requested_id:
        exact_matches = [system for system in systems if system.system_id == requested_id]
        partial_matches = [system for system in systems if requested_id in system.system_id]
        matches = exact_matches or partial_matches
        if not matches:
            raise RuntimeError(f"No system matched --overfit-system-id={requested_id!r}.")
        if len(matches) > 1 and not exact_matches:
            raise RuntimeError(
                f"--overfit-system-id={requested_id!r} matched multiple systems: "
                f"{compact_system_ids(matches)}"
            )
        selected = matches[0]
    else:
        index = int(config.overfit_system_index)
        if index < 0:
            index += len(systems)
        if index < 0 or index >= len(systems):
            raise RuntimeError(
                f"--overfit-system-index={config.overfit_system_index} is out of range for {len(systems)} systems."
            )
        selected = systems[index]
    split = DatasetSplit(train_systems=[selected], val_systems=[selected], test_systems=[selected])
    print_block(
        "One-system overfit split",
        [
            ("system", selected.system_id),
            ("formula", selected.metadata.get("formula", "")),
            ("axis points", len(selected.axis)),
            ("n_points", len(selected.points)),
            ("train/val/test", "same system"),
        ],
    )
    return split


def rotated_output_path(path: Path, generation: int) -> Path:
    prefix = "_".join(["old"] * generation)
    return path.with_name(f"{prefix}_{path.name}")


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def rotate_output_dir_if_requested(config: ExperimentConfig) -> None:
    """Rotate an existing fixed output directory before a new run.

    For RDM_OUTPUT_DIR=result and RDM_OUTPUT_ROTATION_DEPTH=2:
        old_old_result is deleted
        old_result -> old_old_result
        result -> old_result
    """
    if not config.rotate_output_dir:
        return
    output_dir = Path(config.output_dir).resolve()
    depth = max(int(config.output_rotation_depth), 0)
    if depth <= 0 or not output_dir.exists():
        return

    oldest = rotated_output_path(output_dir, depth)
    if oldest.exists():
        remove_path(oldest)

    for generation in range(depth - 1, 0, -1):
        src = rotated_output_path(output_dir, generation)
        dst = rotated_output_path(output_dir, generation + 1)
        if src.exists():
            if dst.exists():
                remove_path(dst)
            src.rename(dst)

    first_old = rotated_output_path(output_dir, 1)
    if first_old.exists():
        remove_path(first_old)
    output_dir.rename(first_old)
    print_block(
        "Output rotation",
        [
            ("current -> previous", f"{output_dir} -> {first_old}"),
            ("rotation depth", depth),
        ],
    )


def configure_tensorflow_runtime() -> None:
    """Report visible TensorFlow devices and enable GPU memory growth when possible."""
    import tensorflow as tf

    requested_device = os.environ.get("RDM_DEVICE", "auto").strip().lower()
    gpus = tf.config.list_physical_devices("GPU")
    if gpus and env_flag("RDM_GPU_MEMORY_GROWTH", True):
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

    rows = [
        ("requested device", requested_device),
        ("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")),
        ("visible GPUs", [gpu.name for gpu in gpus] if gpus else "none"),
    ]
    if requested_device in {"gpu", "cuda"} and not gpus:
        rows.append(("warning", "RDM_DEVICE=gpu was requested, but TensorFlow sees no GPU."))
    print_block("TensorFlow runtime", rows)


def summarize_for_json(summary: dict[str, object]) -> dict[str, object]:
    train_avg = {key: value for key, value in summary["train"].items() if isinstance(value, (int, float))}
    val_avg = {key: value for key, value in summary["val"].items() if isinstance(value, (int, float))}
    payload = {
        "train_average": train_avg,
        "val_average": val_avg,
        "evaluation_metadata": summary.get("evaluation_metadata", {}),
        "train_evaluation": {
            "evaluated_system_count": summary["train"].get("evaluated_system_count"),
            "available_system_count": summary["train"].get("available_system_count"),
            "gamma_fd_target_sources": summary["train"].get("gamma_fd_target_sources", []),
            "kinetic_evaluation_modes": summary["train"].get("kinetic_evaluation_modes", []),
        },
        "val_evaluation": {
            "evaluated_system_count": summary["val"].get("evaluated_system_count"),
            "available_system_count": summary["val"].get("available_system_count"),
            "gamma_fd_target_sources": summary["val"].get("gamma_fd_target_sources", []),
            "kinetic_evaluation_modes": summary["val"].get("kinetic_evaluation_modes", []),
        },
    }
    if "test" in summary:
        payload["test_average"] = {
            key: value for key, value in summary["test"].items() if isinstance(value, (int, float))
        }
        payload["test_evaluation"] = {
            "evaluated_system_count": summary["test"].get("evaluated_system_count"),
            "available_system_count": summary["test"].get("available_system_count"),
            "gamma_fd_target_sources": summary["test"].get("gamma_fd_target_sources", []),
            "kinetic_evaluation_modes": summary["test"].get("kinetic_evaluation_modes", []),
        }

    representative = summary["val"]["per_system"][0]
    representative_compact = {
        "system_id": representative["system_id"],
        "pair_loss": representative["pair_loss"],
        "density_mae": representative["density_mae"],
        "tau_mae": representative["tau_mae"],
        "tau_fd_ao_mae": representative["tau_fd_ao_mae"],
        "tau_fd_ao_rms_ratio": representative["tau_fd_ao_rms_ratio"],
        "tau_pred_fd_mae": representative["tau_pred_fd_mae"],
        "gamma_fd_target_source": representative["gamma_fd_target_source"],
        "kinetic_evaluation_mode": representative["kinetic_evaluation_mode"],
        "stencil_eval_centers": representative["stencil_eval_centers"],
        "stencil_eval_total_centers": representative["stencil_eval_total_centers"],
        "stencil_eval_sampled": representative["stencil_eval_sampled"],
        "kinetic_loss": representative["kinetic_loss"],
        "kinetic_pred": representative["kinetic_pred"],
        "kinetic_training_ref": representative["kinetic_training_ref"],
        "kinetic_ref_error": representative["kinetic_ref_error"],
        "kernel_diag_error": representative["kernel_diag_error"],
        "symmetry_mae": representative["symmetry_mae"],
        "trace_true": representative["trace_true"],
        "trace_pred": representative["trace_pred"],
        "tau_true_integral": representative["tau_true_integral"],
        "tau_true_fd_integral": representative["tau_true_fd_integral"],
        "tau_fd_ao_integral_error": representative["tau_fd_ao_integral_error"],
        "tau_pred_integral": representative["tau_pred_integral"],
        "kinetic_energy_ref": representative["kinetic_energy_ref"],
        "kinetic_energy_ref_error": representative["kinetic_energy_ref_error"],
        "energy_total_ref": representative["energy_total_ref"],
        "energy_total_grid_ref": representative["energy_total_grid_ref"],
        "energy_total_pred": representative["energy_total_pred"],
        "energy_total_ref_minus_pred": representative["energy_total_ref_minus_pred"],
        "energy_total_grid_ref_minus_pred": representative["energy_total_grid_ref_minus_pred"],
        "energy_stored_minus_grid_ref": representative["energy_stored_minus_grid_ref"],
        "energy_stored_total_available": representative["energy_stored_total_available"],
        "energy_kinetic_ref_minus_pred": representative["energy_kinetic_ref_minus_pred"],
        "energy_external_ref_minus_pred": representative["energy_external_ref_minus_pred"],
        "energy_hartree_ref_minus_pred": representative["energy_hartree_ref_minus_pred"],
        "energy_xc_lda_ref_minus_pred": representative["energy_xc_lda_ref_minus_pred"],
        "top_mo_occ_true": representative.get("top_mo_occ_true", np.array([])).tolist(),
        "top_subset_eigs_true": representative.get("top_subset_eigs_true", representative["top_occ_true"]).tolist(),
        "top_subset_eigs_pred": representative.get("top_subset_eigs_pred", representative["top_occ_pred"]).tolist(),
    }
    payload["representative_val_system"] = representative_compact
    return payload


def summarize_point_pretrain_for_json(summary: dict[str, object]) -> dict[str, object]:
    payload = {}
    for split_name in split_names(summary):
        split_summary = summary[split_name]
        payload[f"{split_name}_average"] = {
            key: value for key, value in split_summary.items() if isinstance(value, (int, float))
        }
    representative = summary["val"]["per_system"][0]
    payload["representative_val_system"] = {
        key: value
        for key, value in representative.items()
        if isinstance(value, (str, int, float))
    }
    return payload


def csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        return ";".join(f"{float(item):.8g}" for item in flat)
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value


def save_history_csv(path: Path, history) -> None:
    loss_weight_keys = list(history.loss_weights)
    train_component_keys = list(history.train_components)
    val_component_keys = list(history.val_components)
    fieldnames = (
        ["epoch", "validation_ran", "train_objective", "val_objective", "learning_rate", "kinetic_weight"]
        + [f"w_{key}" for key in loss_weight_keys]
        + [f"train_{key}" for key in train_component_keys]
        + [f"val_{key}" for key in val_component_keys]
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(len(history.train_objective)):
            row = {
                "epoch": epoch,
                "validation_ran": history.validation_ran[epoch],
                "train_objective": history.train_objective[epoch],
                "val_objective": history.val_objective[epoch],
                "learning_rate": history.learning_rate[epoch],
                "kinetic_weight": history.kinetic_weight[epoch],
            }
            row.update({f"w_{key}": history.loss_weights[key][epoch] for key in loss_weight_keys})
            row.update({f"train_{key}": history.train_components[key][epoch] for key in train_component_keys})
            row.update({f"val_{key}": history.val_components[key][epoch] for key in val_component_keys})
            writer.writerow(row)


def save_point_pretrain_history_csv(path: Path, history) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "learning_rate"])
        writer.writeheader()
        for epoch in range(len(history.train_loss)):
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": history.train_loss[epoch],
                    "val_loss": history.val_loss[epoch],
                    "learning_rate": history.learning_rate[epoch],
                }
            )


def split_names(summary: dict[str, object]) -> list[str]:
    names = ["train", "val"]
    if "test" in summary:
        names.append("test")
    return names


def save_split_metrics_csv(path: Path, summary: dict[str, object]) -> None:
    fieldnames = ["split", "system_count"] + CSV_METRIC_KEYS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in split_names(summary):
            split_summary = summary[split_name]
            row = {
                "split": split_name,
                "system_count": len(split_summary.get("per_system", [])),
            }
            row.update({key: csv_value(split_summary.get(key, "")) for key in CSV_METRIC_KEYS})
            writer.writerow(row)


def save_per_system_metrics_csv(path: Path, summary: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_SYSTEM_KEYS)
        writer.writeheader()
        for split_name in split_names(summary):
            for entry in summary[split_name].get("per_system", []):
                row = {"split": split_name}
                row.update({key: csv_value(entry.get(key, "")) for key in CSV_SYSTEM_KEYS if key != "split"})
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    resume_payload = None
    resume_paths = None
    if args.resume_summary_json is not None:
        base_config, resume_payload = config_from_resume_summary(args.resume_summary_json)
    else:
        base_config = ExperimentConfig()
    if args.phase is not None:
        base_config = replace(base_config, phase=args.phase)
    config = apply_phase_preset(base_config)
    config = apply_overrides(config, args)
    config = apply_auto_run_dir(config)
    rotate_output_dir_if_requested(config)
    configure_tensorflow_runtime()
    set_global_seed(config.seed)

    systems = build_system_corpus(config)
    configure_system_feature_metadata(systems, config)
    split = make_overfit_split(systems, config) if config.overfit_one_system else split_systems(systems, config)
    if density_source_mode(config) == "predicted" and density_baseline_mode(config) == "sad-multiplicative":
        missing_sad = [system.system_id for system in systems if system.rho_sad is None]
        if missing_sad:
            raise ValueError(
                "density_baseline_mode='sad-multiplicative' requires SAD density for every system. "
                f"Missing {len(missing_sad)} system(s), including {missing_sad[:3]}. "
                "Patch NPZ files with scripts/patch_npz_features.py."
            )

    point_dim = systems[0].local_features.shape[1]
    base_pair_dim = build_pair_features(
        systems[0],
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
    ).shape[1]
    pair_dim = base_pair_dim
    global_dim = len(systems[0].global_context)
    print_block(
        "Density feature configuration",
        [
            ("mode", config.pair_density_feature_mode),
            ("density source", density_source_mode(config)),
            ("base pair dim", base_pair_dim),
            ("density descriptor dim", pair_density_feature_dim(config)),
            ("symmetric pair descriptors", config.pair_density_symmetric),
            ("density Hessian descriptors", config.pair_density_hessian),
            ("potential Laplacian feature", config.use_potential_laplacian_feature),
            ("density baseline mode", density_baseline_mode(config)),
            ("SAD floor/residual clip", f"{config.sad_density_floor:g} / {config.sad_residual_clip:g}"),
        ],
    )
    density_cache_floats_per_point = {
        "off": 1,
        "rho-derivatives": 4,
        "fukui": 12,
    }[pair_density_feature_mode(config)]
    gamma_cache_gib = float(os.environ.get("RDM_GAMMA_CACHE_GB", "1.0"))
    psi_cache_gib = float(os.environ.get("RDM_PSI_OCC_CACHE_GB", "2.0"))
    lazy_psi_occ = env_flag("RDM_LAZY_PSI_OCC", True)
    mmap_cache_enabled = env_flag("RDM_NPZ_MMAP_CACHE", False)
    mmap_cache_dir = os.environ.get("RDM_NPZ_MMAP_CACHE_DIR", "dataset-local .rdm_mmap_cache")
    expanded_gamma_gib = sum(len(system.points) ** 2 * 4 for system in systems) / (1024**3)
    frozen_density_cache_mib = (
        sum(len(system.points) for system in systems) * density_cache_floats_per_point * 4 / (1024**2)
    )
    print_block(
        "Runtime cache estimate",
        [
            ("gamma LRU limit (CPU RAM)", f"{gamma_cache_gib:.2f} GiB"),
            ("psi_occ lazy load", lazy_psi_occ),
            ("psi_occ LRU limit (CPU RAM)", f"{psi_cache_gib:.2f} GiB"),
            ("NPZ mmap cache", mmap_cache_enabled),
            ("NPZ mmap cache dir", mmap_cache_dir if mmap_cache_enabled else "disabled"),
            ("full corpus gamma expanded", f"{expanded_gamma_gib:.2f} GiB"),
            ("frozen density cache (GPU, approx)", f"{frozen_density_cache_mib:.1f} MiB"),
        ],
    )
    models = build_models(config, point_dim, pair_dim, global_dim)
    if args.resume_summary_json is not None:
        assert resume_payload is not None
        prefix = resume_checkpoint_prefix(
            args.resume_summary_json,
            resume_payload,
            args.checkpoint_prefix,
        )
        resume_paths = load_model_weights(models, prefix)
        print_block(
            "Resume checkpoint",
            [
                ("summary", args.resume_summary_json.resolve()),
                ("prefix", prefix),
                ("learned rank", config.learned_rank),
                ("fine-tune epochs", config.epochs),
                ("fine-tune learning rate", f"{config.initial_lr:.6e}"),
            ],
        )
    if density_source_mode(config) == "predicted" and resume_paths is None:
        rho_mean = float(np.mean(np.concatenate([system.rho_diag for system in split.train_systems], axis=0)))
        initialize_point_model_density_bias(
            models.point_model,
            rho_mean,
            residual_baseline=density_baseline_mode(config) == "sad-multiplicative",
        )

    point_config = replace(config, point_pretrain_epochs=0) if resume_paths is not None else config
    point_history, point_summary = pretrain_point_model(point_config, split, models)
    history, summary = train_models(
        config,
        split,
        models,
        initialize_best_from_current=resume_paths is not None,
    )

    out_dir = Path(config.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{config.run_name}.png"
    point_png_path = out_dir / f"{config.run_name}_point_pretrain.png"
    summary_path = out_dir / f"{config.run_name}_summary.json"
    history_csv_path = out_dir / f"{config.run_name}_history.csv"
    point_history_csv_path = out_dir / f"{config.run_name}_point_pretrain_history.csv"
    split_csv_path = out_dir / f"{config.run_name}_split_metrics.csv"
    per_system_csv_path = out_dir / f"{config.run_name}_per_system_metrics.csv"
    point_ckpt = out_dir / f"{config.run_name}_point.weights.h5"
    mode_ckpt = out_dir / f"{config.run_name}_mode.weights.h5"
    pair_ckpt = out_dir / f"{config.run_name}_pair.weights.h5"
    context_ckpt = out_dir / f"{config.run_name}_context.weights.h5"

    models.point_model.save_weights(point_ckpt)
    models.mode_model.save_weights(mode_ckpt)
    models.pair_model.save_weights(pair_ckpt)
    models.context_model.save_weights(context_ckpt)

    payload = {
        "config": asdict(config),
        "resume": (
            {
                "source_summary_json": str(args.resume_summary_json.resolve()),
                "checkpoint": {name: str(path) for name, path in resume_paths.items()},
            }
            if resume_paths is not None
            else None
        ),
        "history": {
            "train_objective": history.train_objective,
            "val_objective": history.val_objective,
            "learning_rate": history.learning_rate,
            "kinetic_weight": history.kinetic_weight,
            "validation_ran": history.validation_ran,
            "loss_weights": history.loss_weights,
            "train_components": history.train_components,
            "val_components": history.val_components,
        },
        "point_pretrain": {
            "history": {
                "train_loss": point_history.train_loss,
                "val_loss": point_history.val_loss,
                "learning_rate": point_history.learning_rate,
            },
            "summary": summarize_point_pretrain_for_json(point_summary),
        },
        "summary": summarize_for_json(summary),
    }
    save_json(summary_path, payload)
    save_history_csv(history_csv_path, history)
    save_point_pretrain_history_csv(point_history_csv_path, point_history)
    save_split_metrics_csv(split_csv_path, summary)
    save_per_system_metrics_csv(per_system_csv_path, summary)

    plot_training_summary(
        history=history,
        summary=summary,
        axis_points=config.axis_points,
        output_png=png_path,
    )
    plot_point_pretrain_summary(point_summary, point_png_path)

    print_block(
        "Saved artifacts",
        [
            ("figure", png_path),
            ("point pretrain figure", point_png_path),
            ("summary json", summary_path),
            ("history csv", history_csv_path),
            ("point pretrain history csv", point_history_csv_path),
            ("split metrics csv", split_csv_path),
            ("per-system metrics csv", per_system_csv_path),
            ("point weights", point_ckpt),
            ("mode weights", mode_ckpt),
            ("pair weights", pair_ckpt),
            ("context weights", context_ckpt),
        ],
    )


if __name__ == "__main__":
    main()
