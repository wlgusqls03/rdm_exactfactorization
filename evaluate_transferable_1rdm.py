from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from train_transferable_1rdm import (
    configure_system_feature_metadata,
    configure_tensorflow_runtime,
    save_per_system_metrics_csv,
    save_split_metrics_csv,
    summarize_for_json,
)
from transferable_rdm.config import ExperimentConfig
from transferable_rdm.data import build_pair_features, split_systems
from transferable_rdm.model import build_models
from transferable_rdm.plotting import plot_training_summary
from transferable_rdm.systems import build_system_corpus
from transferable_rdm.training import (
    TrainingHistory,
    clear_gpu_evaluation_caches,
    evaluate_systems,
    select_evaluation_systems,
)
from transferable_rdm.utils import print_block, save_json, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved transferable 1-RDM checkpoint on full tau/T grids."
    )
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-prefix",
        type=Path,
        default=None,
        help="Path prefix before _point.weights.h5. Defaults to summary config run_name.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--splits", default="val,test", help="Comma-separated train,val,test.")
    parser.add_argument("--train-system-count", type=int, default=0)
    parser.add_argument("--val-system-count", type=int, default=3)
    parser.add_argument("--test-system-count", type=int, default=3)
    parser.add_argument("--npz-glob", default=None)
    parser.add_argument("--mmap-cache-dir", type=Path, default=None)
    parser.add_argument("--eval-pair-count", type=int, default=None)
    parser.add_argument("--stencil-prediction-chunk-size", type=int, default=2048)
    parser.add_argument("--diagonal-prediction-chunk-size", type=int, default=16384)
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def load_config(payload: dict[str, object], args: argparse.Namespace) -> ExperimentConfig:
    valid_fields = {item.name for item in fields(ExperimentConfig)}
    saved = {
        key: value
        for key, value in payload.get("config", {}).items()
        if key in valid_fields
    }
    config = replace(ExperimentConfig(), **saved)
    updates: dict[str, object] = {
        "eval_full_final": True,
        "eval_stencil_centers": 0,
        "stencil_prediction_chunk_size": max(args.stencil_prediction_chunk_size, 1),
        "diagonal_prediction_chunk_size": max(args.diagonal_prediction_chunk_size, 1),
        "auto_run_dir": False,
        "rotate_output_dir": False,
    }
    if args.npz_glob is not None:
        updates["npz_glob"] = args.npz_glob
    if args.eval_pair_count is not None:
        updates["eval_pair_count"] = max(args.eval_pair_count, 1)
    return replace(config, **updates)


def restore_history(payload: dict[str, object]) -> TrainingHistory:
    saved = payload.get("history", {})
    history = TrainingHistory()
    for name in (
        "train_objective",
        "val_objective",
        "learning_rate",
        "kinetic_weight",
        "validation_ran",
        "loss_weights",
        "train_components",
        "val_components",
    ):
        if name in saved:
            setattr(history, name, saved[name])
    return history


def checkpoint_prefix(summary_path: Path, payload: dict[str, object], requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    run_name = str(payload.get("config", {}).get("run_name", "")).strip()
    if not run_name:
        stem = summary_path.stem
        run_name = stem[: -len("_summary")] if stem.endswith("_summary") else stem
    return summary_path.resolve().parent / run_name


def load_weights(models, prefix: Path) -> dict[str, Path]:
    paths = {
        "point": Path(f"{prefix}_point.weights.h5"),
        "mode": Path(f"{prefix}_mode.weights.h5"),
        "pair": Path(f"{prefix}_pair.weights.h5"),
        "context": Path(f"{prefix}_context.weights.h5"),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint file(s): " + ", ".join(missing))
    models.point_model.load_weights(paths["point"])
    models.mode_model.load_weights(paths["mode"])
    models.pair_model.load_weights(paths["pair"])
    models.context_model.load_weights(paths["context"])
    return paths


def empty_split_summary(
    payload: dict[str, object],
    split_name: str,
    available_count: int,
) -> dict[str, object]:
    saved_summary = payload.get("summary", {})
    averages = saved_summary.get(f"{split_name}_average", {})
    return {
        **averages,
        "per_system": [],
        "evaluated_system_count": 0,
        "available_system_count": available_count,
    }


def main() -> None:
    args = parse_args()
    summary_path = args.summary_json.resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if args.mmap_cache_dir is not None:
        os.environ["RDM_NPZ_MMAP_CACHE"] = "1"
        os.environ["RDM_NPZ_MMAP_CACHE_DIR"] = str(args.mmap_cache_dir.resolve())

    config = load_config(payload, args)
    configure_tensorflow_runtime()
    set_global_seed(config.seed)

    systems = build_system_corpus(config)
    configure_system_feature_metadata(systems, config)
    split = split_systems(systems, config)

    point_dim = systems[0].local_features.shape[1]
    pair_dim = build_pair_features(
        systems[0],
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
    ).shape[1]
    global_dim = len(systems[0].global_context)
    models = build_models(config, point_dim, pair_dim, global_dim)
    prefix = checkpoint_prefix(summary_path, payload, args.checkpoint_prefix)
    weight_paths = load_weights(models, prefix)

    requested_splits = {
        value.strip().lower()
        for value in args.splits.split(",")
        if value.strip()
    }
    invalid = requested_splits.difference({"train", "val", "test"})
    if invalid:
        raise ValueError(f"Unknown split(s): {sorted(invalid)}")

    split_systems_map = {
        "train": split.train_systems,
        "val": split.val_systems,
        "test": split.test_systems,
    }
    requested_counts = {
        "train": args.train_system_count,
        "val": args.val_system_count,
        "test": args.test_system_count,
    }
    summary: dict[str, object] = {}
    final_epoch = max(int(config.epochs) - 1, 0)
    for split_name in ("train", "val", "test"):
        available = split_systems_map[split_name]
        if split_name not in requested_splits or not available:
            summary[split_name] = empty_split_summary(payload, split_name, len(available))
            continue
        selected = select_evaluation_systems(available, requested_counts[split_name])
        clear_gpu_evaluation_caches()
        metrics = evaluate_systems(
            selected,
            models,
            config,
            epoch=final_epoch,
            progress_label=f"Full-grid {split_name}",
        )
        metrics["evaluated_system_count"] = len(selected)
        metrics["available_system_count"] = len(available)
        summary[split_name] = metrics
    clear_gpu_evaluation_caches()

    if not summary["val"].get("per_system"):
        raise ValueError("The val split must be evaluated because it supplies the representative figure panels.")
    summary["evaluation_metadata"] = {
        "density_source": "true oracle" if config.density_source == "true" else config.density_source,
        "kinetic_integral_active": bool(config.use_kinetic_loss),
        "eval_full_final": True,
        "checkpoint_summary": str(summary_path),
    }

    run_name = args.run_name or f"{config.run_name}_full_grid"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else summary_path.parent / f"{run_name}_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"{run_name}_summary.json"
    split_csv = output_dir / f"{run_name}_split_metrics.csv"
    per_system_csv = output_dir / f"{run_name}_per_system_metrics.csv"
    figure_path = output_dir / f"{run_name}.png"

    output_payload = {
        "config": {
            **payload.get("config", {}),
            "eval_full_final": True,
            "eval_stencil_centers": 0,
            "stencil_prediction_chunk_size": config.stencil_prediction_chunk_size,
            "diagonal_prediction_chunk_size": config.diagonal_prediction_chunk_size,
        },
        "checkpoint": {name: str(path) for name, path in weight_paths.items()},
        "source_summary_json": str(summary_path),
        "summary": summarize_for_json(summary),
    }
    save_json(output_json, output_payload)
    save_split_metrics_csv(split_csv, summary)
    save_per_system_metrics_csv(per_system_csv, summary)
    if not args.no_figure:
        plot_training_summary(
            history=restore_history(payload),
            summary=summary,
            axis_points=config.axis_points,
            output_png=figure_path,
        )

    rows = []
    for split_name in ("val", "test"):
        metrics = summary.get(split_name, {})
        if metrics.get("per_system"):
            rows.extend(
                [
                    (f"{split_name} systems", f"{metrics['evaluated_system_count']} / {metrics['available_system_count']}"),
                    (f"{split_name} tau MAE", f"{metrics['tau_mae']:.6e}"),
                    (f"{split_name} kinetic MAE", f"{metrics['kinetic_abs_error']:.6e} Ha"),
                    (f"{split_name} stencil eval", f"{metrics['stencil_eval_centers']:.0f} / {metrics['stencil_eval_total_centers']:.0f}"),
                ]
            )
    rows.extend(
        [
            ("summary json", output_json),
            ("split metrics", split_csv),
            ("per-system metrics", per_system_csv),
            ("figure", figure_path if not args.no_figure else "disabled"),
        ]
    )
    print_block("Full-grid checkpoint evaluation", rows)


if __name__ == "__main__":
    main()
