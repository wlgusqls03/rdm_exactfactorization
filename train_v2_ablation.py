from __future__ import annotations

import argparse
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no", "n"}


def configure_tensorflow_environment_preimport() -> None:
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

import tensorflow as tf

from transferable_rdm.config import ExperimentConfig
from transferable_rdm.data import DatasetSplit, split_systems
from transferable_rdm.systems import build_system_corpus
from transferable_rdm.utils import print_block, set_global_seed
from transferable_rdm.v2_ablation import EXPERIMENTS, V2Config, train_v2, v2_pair_feature_dim


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def parse_four_floats(value: str | None, *, name: str) -> tuple[float, float, float, float] | None:
    if value is None or value.strip() == "":
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"{name} must contain four comma-separated values.")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain numeric values.") from exc
    if any(value < 0.0 for value in values) or sum(values) <= 0.0:
        raise argparse.ArgumentTypeError(f"{name} must be non-negative and have positive sum.")
    return values  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation-first V2 experiments on existing 1-RDM NPZ data."
    )
    parser.add_argument("--experiment", choices=EXPERIMENTS, default=os.environ.get("RDM_V2_EXPERIMENT", "baseline"))

    parser.add_argument("--dataset-mode", choices=["ks_like", "npz", "mixed"], default=None)
    parser.add_argument("--npz-glob", type=str, default=None)
    parser.add_argument("--num-systems", type=int, default=None)
    parser.add_argument("--train-system-count", type=int, default=None)
    parser.add_argument("--val-system-count", type=int, default=None)
    parser.add_argument("--test-system-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--overfit-one-system",
        action="store_true",
        default=env_flag("RDM_V2_OVERFIT_ONE_SYSTEM", False),
        help="Use the same single molecule as train, validation, and test for capacity debugging.",
    )
    parser.add_argument(
        "--overfit-system-index",
        type=int,
        default=env_int("RDM_V2_OVERFIT_SYSTEM_INDEX", 0),
        help="System index to use with --overfit-one-system when --overfit-system-id is not set.",
    )
    parser.add_argument(
        "--overfit-system-id",
        type=str,
        default=os.environ.get("RDM_V2_OVERFIT_SYSTEM_ID", ""),
        help="Exact or substring system id to use with --overfit-one-system.",
    )

    parser.add_argument("--output-dir", type=str, default=os.environ.get("RDM_V2_OUTPUT_DIR", "v2_outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--auto-run-dir", dest="auto_run_dir", action="store_true", default=None)
    parser.add_argument("--no-auto-run-dir", dest="auto_run_dir", action="store_false")

    parser.add_argument("--width", type=int, default=env_int("RDM_V2_WIDTH", 128))
    parser.add_argument("--depth", type=int, default=env_int("RDM_V2_DEPTH", 2))
    parser.add_argument("--rank", type=int, default=env_int("RDM_V2_RANK", 8))
    parser.add_argument("--rff-features", type=int, default=env_int("RDM_V2_RFF_FEATURES", 16))
    parser.add_argument("--rff-scale", type=float, default=env_float("RDM_V2_RFF_SCALE", 2.0))
    parser.add_argument(
        "--context-rff",
        dest="context_rff",
        action="store_true",
        default=env_flag("RDM_V2_CONTEXT_RFF", False),
        help="Use random Fourier features in the context model. Default is off.",
    )
    parser.add_argument("--no-context-rff", dest="context_rff", action="store_false")
    parser.add_argument("--residual-scale", type=float, default=env_float("RDM_V2_RESIDUAL_SCALE", 0.25))

    parser.add_argument("--epochs", type=int, default=env_int("RDM_V2_EPOCHS", 120))
    parser.add_argument("--steps-per-epoch", type=int, default=env_int("RDM_V2_STEPS_PER_EPOCH", 40))
    parser.add_argument("--batch-size", type=int, default=env_int("RDM_V2_BATCH_SIZE", 1024))
    parser.add_argument("--val-every", type=int, default=env_int("RDM_V2_VAL_EVERY", 10))
    parser.add_argument("--log-every", type=int, default=env_int("RDM_V2_LOG_EVERY", 1))
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=env_float("RDM_V2_LR", env_float("RDM_LEARNING_RATE", env_float("RDM_LR", 3e-4))),
    )
    parser.add_argument("--weight-decay", type=float, default=env_float("RDM_V2_WEIGHT_DECAY", 0.0))
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=env_float("RDM_V2_LR_DECAY_FACTOR", 1.0),
        help="Multiply learning rate by this factor after validation plateau. Use <1 to enable.",
    )
    parser.add_argument(
        "--lr-decay-patience",
        type=int,
        default=env_int("RDM_V2_LR_DECAY_PATIENCE", 0),
        help="Epochs without validation improvement before decaying learning rate. 0 disables.",
    )
    parser.add_argument(
        "--lr-decay-min",
        type=float,
        default=env_float("RDM_V2_LR_DECAY_MIN", 1e-6),
        help="Minimum learning rate for plateau decay.",
    )
    parser.add_argument(
        "--lr-decay-min-delta",
        type=float,
        default=env_float("RDM_V2_LR_DECAY_MIN_DELTA", 0.0),
        help="Minimum validation improvement counted by LR plateau decay.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=env_int("RDM_V2_EARLY_STOPPING_PATIENCE", 0),
        help="Epochs without validation improvement before stopping. 0 disables.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=env_float("RDM_V2_EARLY_STOPPING_MIN_DELTA", 0.0),
        help="Minimum validation improvement counted by early stopping.",
    )
    parser.add_argument(
        "--restore-best-weights",
        dest="restore_best_weights",
        action="store_true",
        default=env_flag("RDM_V2_RESTORE_BEST_WEIGHTS", False),
        help="Restore the best validation checkpoint before final train/val/test evaluation.",
    )
    parser.add_argument("--no-restore-best-weights", dest="restore_best_weights", action="store_false")
    parser.add_argument("--eval-pair-count", type=int, default=env_int("RDM_V2_EVAL_PAIR_COUNT", 8192))
    parser.add_argument(
        "--cache-eval-batches",
        dest="cache_eval_batches",
        action="store_true",
        default=env_flag("RDM_V2_CACHE_EVAL_BATCHES", True),
        help="Reuse fixed validation/test pair batches to reduce CPU sampling and gamma I/O.",
    )
    parser.add_argument(
        "--no-cache-eval-batches",
        dest="cache_eval_batches",
        action="store_false",
        help="Resample validation/test pair batches every evaluation.",
    )
    parser.add_argument(
        "--steps-per-system",
        type=int,
        default=env_int("RDM_V2_STEPS_PER_SYSTEM", 1),
        help="Keep the same sampled system for this many optimizer steps to amortize gamma loading.",
    )
    parser.add_argument(
        "--pair-features-on-device",
        dest="pair_features_on_device",
        action="store_true",
        default=env_flag("RDM_V2_PAIR_FEATURES_ON_DEVICE", True),
        help="Build base pair features with TensorFlow tensors instead of NumPy when no true-rho oracle features are needed.",
    )
    parser.add_argument(
        "--no-pair-features-on-device",
        dest="pair_features_on_device",
        action="store_false",
        help="Build all pair features eagerly in NumPy.",
    )

    parser.add_argument("--lambda-gamma", type=float, default=env_float("RDM_V2_LAMBDA_GAMMA", 1.0))
    parser.add_argument("--lambda-rho", type=float, default=env_float("RDM_V2_LAMBDA_RHO", 1.0))
    parser.add_argument("--lambda-trace", type=float, default=env_float("RDM_V2_LAMBDA_TRACE", 1.0))
    parser.add_argument("--lambda-kernel", type=float, default=env_float("RDM_V2_LAMBDA_KERNEL", 1.0))
    parser.add_argument("--lambda-k-highrho", type=float, default=env_float("RDM_V2_LAMBDA_K_HIGHRHO", 0.0))

    parser.add_argument("--baseline-fit-batches", type=int, default=env_int("RDM_V2_BASELINE_FIT_BATCHES", 24))
    parser.add_argument("--baseline-alpha-min", type=float, default=env_float("RDM_V2_BASELINE_ALPHA_MIN", 1e-3))
    parser.add_argument("--baseline-alpha-max", type=float, default=env_float("RDM_V2_BASELINE_ALPHA_MAX", 3.0))
    parser.add_argument("--baseline-alpha-count", type=int, default=env_int("RDM_V2_BASELINE_ALPHA_COUNT", 36))
    parser.add_argument(
        "--baseline-density-power",
        choices=["sqrt", "product"],
        default=os.environ.get("RDM_V2_BASELINE_DENSITY_POWER", "sqrt"),
    )

    parser.add_argument("--normalize-rho", dest="normalize_rho", action="store_true", default=None)
    parser.add_argument("--no-normalize-rho", dest="normalize_rho", action="store_false")
    parser.add_argument("--kernel-rho-floor", type=float, default=env_float("RDM_V2_KERNEL_RHO_FLOOR", 1e-8))
    parser.add_argument("--kernel-target-clip", type=float, default=env_float("RDM_V2_KERNEL_TARGET_CLIP", 20.0))
    parser.add_argument("--kernel-base-alpha", type=float, default=env_float("RDM_V2_KERNEL_BASE_ALPHA", 0.0))
    parser.add_argument("--k-highrho-cut", type=float, default=env_float("RDM_V2_K_HIGHRHO_CUT", 1e-6))
    parser.add_argument("--k-highrho-eps", type=float, default=env_float("RDM_V2_K_HIGHRHO_EPS", 1e-6))
    parser.add_argument(
        "--pair-rho-features",
        choices=["off", "all", "true"],
        default=os.environ.get("RDM_V2_PAIR_RHO_FEATURES", "off").strip().lower(),
        help="Convenience switch for rho pair features. 'all' or 'true' enables all three rho feature toggles.",
    )
    parser.add_argument(
        "--pair-rho-source",
        choices=["auto", "true", "pred"],
        default=os.environ.get("RDM_V2_PAIR_RHO_SOURCE", "auto").strip().lower(),
        help="Source for rho pair features. auto uses true rho for k-only and predicted rho otherwise.",
    )
    parser.add_argument(
        "--pair-rho-stop-gradient",
        dest="pair_rho_stop_gradient",
        action="store_true",
        default=env_flag("RDM_V2_PAIR_RHO_STOP_GRADIENT", True),
        help="For predicted-rho pair features, prevent pair loss from backpropagating through the rho feature path.",
    )
    parser.add_argument(
        "--no-pair-rho-stop-gradient",
        dest="pair_rho_stop_gradient",
        action="store_false",
        help="Allow gradients through predicted-rho pair features.",
    )
    parser.add_argument(
        "--pair-rho-log-mean",
        action="store_true",
        default=env_flag("RDM_V2_PAIR_RHO_LOG_MEAN", False),
        help="Append centered log geometric mean of rho_i and rho_j to pair features.",
    )
    parser.add_argument(
        "--pair-rho-log-diff",
        action="store_true",
        default=env_flag("RDM_V2_PAIR_RHO_LOG_DIFF", False),
        help="Append absolute log-rho difference to pair features.",
    )
    parser.add_argument(
        "--pair-rho-scaled-product",
        action="store_true",
        default=env_flag("RDM_V2_PAIR_RHO_SCALED_PRODUCT", False),
        help="Append sqrt(rho_i rho_j) divided by the system mean rho to pair features.",
    )
    parser.add_argument("--pair-rho-eps", type=float, default=env_float("RDM_V2_PAIR_RHO_EPS", 1e-14))
    parser.add_argument("--pair-rho-log-scale", type=float, default=env_float("RDM_V2_PAIR_RHO_LOG_SCALE", 8.0))
    parser.add_argument("--pair-rho-log-clip", type=float, default=env_float("RDM_V2_PAIR_RHO_LOG_CLIP", 4.0))
    parser.add_argument(
        "--pair-rho-scaled-clip",
        type=float,
        default=env_float("RDM_V2_PAIR_RHO_SCALED_CLIP", 20.0),
        help="Upper clip for the scaled rho product feature. Use <=0 to disable clipping.",
    )
    parser.add_argument(
        "--pair-rho-product-transform",
        choices=["log1p", "linear"],
        default=os.environ.get("RDM_V2_PAIR_RHO_PRODUCT_TRANSFORM", "log1p").strip().lower(),
        help="Normalize the sqrt(rho_i rho_j)/mean(rho) feature after clipping.",
    )
    parser.add_argument("--sep-factor-scale", type=float, default=env_float("RDM_V2_SEP_FACTOR_SCALE", 0.05))
    parser.add_argument(
        "--pair-sampling-probs",
        type=lambda text: parse_four_floats(text, name="--pair-sampling-probs"),
        default=parse_four_floats(os.environ.get("RDM_V2_PAIR_SAMPLING_PROBS"), name="RDM_V2_PAIR_SAMPLING_PROBS"),
        help="Optional fixed diag,near,mid,far sampling probabilities. Default uses the curriculum.",
    )
    parser.add_argument(
        "--pair-category-weights",
        type=lambda text: parse_four_floats(text, name="--pair-category-weights"),
        default=parse_four_floats(
            os.environ.get("RDM_V2_PAIR_CATEGORY_WEIGHTS", "20,8,4,1"),
            name="RDM_V2_PAIR_CATEGORY_WEIGHTS",
        ),
        help="diag,near,mid,far loss weights. Values are normalized to mean 1 per batch.",
    )
    parser.add_argument("--no-save-weights", dest="save_weights", action="store_false", default=True)
    return parser.parse_args()


def apply_data_overrides(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    updates = {}
    for field_name, arg_name in [
        ("dataset_mode", "dataset_mode"),
        ("npz_glob", "npz_glob"),
        ("num_systems", "num_systems"),
        ("train_system_count", "train_system_count"),
        ("val_system_count", "val_system_count"),
        ("test_system_count", "test_system_count"),
        ("seed", "seed"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    return replace(config, **updates) if updates else config


def make_v2_config(args: argparse.Namespace) -> V2Config:
    seed = args.seed if args.seed is not None else env_int("RDM_SEED", 0)
    run_name = args.run_name or f"v2_{args.experiment}"
    output_dir = args.output_dir
    auto_run_dir = env_flag("RDM_V2_AUTO_RUN_DIR", False) if args.auto_run_dir is None else args.auto_run_dir
    if auto_run_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(Path(output_dir) / f"{run_name}_{stamp}")

    normalize_rho = (
        env_flag("RDM_NORMALIZE_RHO", True)
        if args.normalize_rho is None
        else bool(args.normalize_rho)
    )
    pair_rho_all = args.pair_rho_features in {"all", "true"}

    return V2Config(
        experiment=args.experiment,
        output_dir=output_dir,
        run_name=run_name,
        seed=seed,
        width=args.width,
        depth=args.depth,
        rank=args.rank,
        rff_features=args.rff_features,
        rff_scale=args.rff_scale,
        context_rff=args.context_rff,
        residual_scale=args.residual_scale,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        val_every=args.val_every,
        log_every=args.log_every,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_patience=max(int(args.lr_decay_patience), 0),
        lr_decay_min=args.lr_decay_min,
        lr_decay_min_delta=max(float(args.lr_decay_min_delta), 0.0),
        early_stopping_patience=max(int(args.early_stopping_patience), 0),
        early_stopping_min_delta=max(float(args.early_stopping_min_delta), 0.0),
        restore_best_weights=args.restore_best_weights,
        eval_pair_count=args.eval_pair_count,
        cache_eval_batches=args.cache_eval_batches,
        steps_per_system=max(int(args.steps_per_system), 1),
        pair_features_on_device=args.pair_features_on_device,
        baseline_fit_batches=args.baseline_fit_batches,
        baseline_alpha_min=args.baseline_alpha_min,
        baseline_alpha_max=args.baseline_alpha_max,
        baseline_alpha_count=args.baseline_alpha_count,
        baseline_density_power=args.baseline_density_power,
        normalize_rho=normalize_rho,
        kernel_rho_floor=args.kernel_rho_floor,
        kernel_target_clip=args.kernel_target_clip,
        kernel_base_alpha=args.kernel_base_alpha,
        sep_factor_scale=args.sep_factor_scale,
        pair_sampling_probs=args.pair_sampling_probs,
        pair_category_weights=args.pair_category_weights,
        lambda_gamma=args.lambda_gamma,
        lambda_rho=args.lambda_rho,
        lambda_trace=args.lambda_trace,
        lambda_kernel=args.lambda_kernel,
        lambda_k_highrho=args.lambda_k_highrho,
        k_highrho_cut=args.k_highrho_cut,
        k_highrho_eps=args.k_highrho_eps,
        pair_rho_log_mean=pair_rho_all or args.pair_rho_log_mean,
        pair_rho_log_diff=pair_rho_all or args.pair_rho_log_diff,
        pair_rho_scaled_product=pair_rho_all or args.pair_rho_scaled_product,
        pair_rho_source=args.pair_rho_source,
        pair_rho_stop_gradient=args.pair_rho_stop_gradient,
        pair_rho_eps=args.pair_rho_eps,
        pair_rho_log_scale=args.pair_rho_log_scale,
        pair_rho_log_clip=args.pair_rho_log_clip,
        pair_rho_scaled_clip=args.pair_rho_scaled_clip,
        pair_rho_product_transform=args.pair_rho_product_transform,
        save_weights=args.save_weights,
    )


def make_overfit_split(systems, args: argparse.Namespace) -> DatasetSplit:
    if not systems:
        raise RuntimeError("Cannot build one-molecule overfit split from an empty corpus.")

    selected = None
    if args.overfit_system_id:
        exact = [system for system in systems if system.system_id == args.overfit_system_id]
        contains = [system for system in systems if args.overfit_system_id in system.system_id]
        matches = exact or contains
        if not matches:
            raise RuntimeError(f"No system id matched --overfit-system-id={args.overfit_system_id!r}.")
        selected = matches[0]
    else:
        index = int(args.overfit_system_index)
        if index < 0:
            index += len(systems)
        if index < 0 or index >= len(systems):
            raise RuntimeError(
                f"--overfit-system-index={args.overfit_system_index} is out of range for {len(systems)} systems."
            )
        selected = systems[index]

    print_block(
        "V2 one-molecule overfit split",
        [
            ("system id", selected.system_id),
            ("formula", selected.metadata.get("formula", "")),
            ("axis points", len(selected.axis)),
            ("n points", len(selected.points)),
            ("electron count", selected.electron_count),
            ("note", "same system is used for train/val/test"),
        ],
    )
    return DatasetSplit(train_systems=[selected], val_systems=[selected], test_systems=[selected])


def main() -> None:
    args = parse_args()
    data_config = apply_data_overrides(ExperimentConfig(), args)
    v2_config = make_v2_config(args)
    set_global_seed(v2_config.seed)

    print_block(
        "V2 runtime",
        [
            ("requested device", os.environ.get("RDM_DEVICE", "auto")),
            ("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")),
            ("visible GPUs", len(tf.config.list_physical_devices("GPU"))),
            ("output dir", Path(v2_config.output_dir).resolve()),
        ],
    )

    systems = build_system_corpus(data_config)
    split = make_overfit_split(systems, args) if args.overfit_one_system else split_systems(systems, data_config)
    point_dim = systems[0].local_features.shape[1]
    pair_dim = v2_pair_feature_dim(systems[0], v2_config)
    global_dim = len(systems[0].global_context)

    summary = train_v2(v2_config, split, point_dim, pair_dim, global_dim)
    print_block(
        "V2 final summary",
        [
            ("experiment", v2_config.experiment),
            ("train pair loss", f"{summary.get('train', {}).get('pair_loss', float('nan')):.6e}"),
            ("val pair loss", f"{summary.get('val', {}).get('pair_loss', float('nan')):.6e}"),
            ("test pair loss", f"{summary.get('test', {}).get('pair_loss', float('nan')):.6e}"),
            ("val rho MAE", f"{summary.get('val', {}).get('rho_mae', float('nan')):.6e}"),
            ("val K loss", f"{summary.get('val', {}).get('kernel_loss', float('nan')):.6e}"),
            ("best epoch", summary.get("best", {}).get("epoch", "n/a")),
            (
                "best val pair",
                f"{summary.get('best', {}).get('val', {}).get('pair_loss', float('nan')):.6e}",
            ),
            ("stopped epoch", summary.get("stopped_epoch", "n/a")),
        ],
    )


if __name__ == "__main__":
    main()
