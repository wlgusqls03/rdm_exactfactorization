from __future__ import annotations

import argparse
import csv
from datetime import datetime
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from transferable_rdm.config import ExperimentConfig
from transferable_rdm.data import build_pair_features, split_systems
from transferable_rdm.model import build_models, initialize_point_model_density_bias
from transferable_rdm.plotting import plot_training_summary
from transferable_rdm.systems import build_system_corpus
from transferable_rdm.training import train_models
from transferable_rdm.utils import print_block, save_json, set_global_seed


CSV_METRIC_KEYS = [
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

CSV_SYSTEM_KEYS = [
    "split",
    "system_id",
    "formula",
    "axis_points",
    "n_points",
    "grid_spacing_bohr",
    "electron_count",
] + CSV_METRIC_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transferable 3D 1-RDM surrogate.")
    parser.add_argument("--dataset-mode", choices=["ks_like", "npz", "mixed"], default=None)
    parser.add_argument("--phase", choices=["none", "phase1a", "phase1b"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-systems", type=int, default=None)
    parser.add_argument("--train-system-count", type=int, default=None)
    parser.add_argument("--val-system-count", type=int, default=None)
    parser.add_argument("--test-system-count", type=int, default=None)
    parser.add_argument("--axis-points", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--npz-glob", type=str, default=None)
    parser.add_argument("--auto-run-dir", dest="auto_run_dir", action="store_true", default=None)
    parser.add_argument("--no-auto-run-dir", dest="auto_run_dir", action="store_false")
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
        ("num_systems", "num_systems"),
        ("train_system_count", "train_system_count"),
        ("val_system_count", "val_system_count"),
        ("test_system_count", "test_system_count"),
        ("axis_points", "axis_points"),
        ("batch_size", "batch_size"),
        ("run_name", "run_name"),
        ("npz_glob", "npz_glob"),
        ("auto_run_dir", "auto_run_dir"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    return replace(config, **updates) if updates else config


def apply_auto_run_dir(config: ExperimentConfig) -> ExperimentConfig:
    if not config.auto_run_dir:
        return config
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run_name = config.run_name.strip() or "run"
    output_dir = Path(config.output_dir) / f"{safe_run_name}_{stamp}"
    return replace(config, output_dir=str(output_dir))


def summarize_for_json(summary: dict[str, object]) -> dict[str, object]:
    train_avg = {key: value for key, value in summary["train"].items() if isinstance(value, (int, float))}
    val_avg = {key: value for key, value in summary["val"].items() if isinstance(value, (int, float))}
    payload = {
        "train_average": train_avg,
        "val_average": val_avg,
    }
    if "test" in summary:
        payload["test_average"] = {
            key: value for key, value in summary["test"].items() if isinstance(value, (int, float))
        }

    representative = summary["val"]["per_system"][0]
    representative_compact = {
        "system_id": representative["system_id"],
        "pair_loss": representative["pair_loss"],
        "density_mae": representative["density_mae"],
        "tau_mae": representative["tau_mae"],
        "kinetic_loss": representative["kinetic_loss"],
        "kinetic_pred": representative["kinetic_pred"],
        "kinetic_training_ref": representative["kinetic_training_ref"],
        "kinetic_ref_error": representative["kinetic_ref_error"],
        "kp_loss": representative["kp_loss"],
        "kp_mae": representative["kp_mae"],
        "kernel_diag_error": representative["kernel_diag_error"],
        "symmetry_mae": representative["symmetry_mae"],
        "trace_true": representative["trace_true"],
        "trace_pred": representative["trace_pred"],
        "tau_true_integral": representative["tau_true_integral"],
        "tau_pred_integral": representative["tau_pred_integral"],
        "kinetic_energy_ref": representative["kinetic_energy_ref"],
        "kinetic_energy_ref_error": representative["kinetic_energy_ref_error"],
        "top_mo_occ_true": representative.get("top_mo_occ_true", np.array([])).tolist(),
        "top_subset_eigs_true": representative.get("top_subset_eigs_true", representative["top_occ_true"]).tolist(),
        "top_subset_eigs_pred": representative.get("top_subset_eigs_pred", representative["top_occ_pred"]).tolist(),
    }
    payload["representative_val_system"] = representative_compact
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
        ["epoch", "validation_ran", "train_objective", "val_objective", "learning_rate", "kinetic_weight", "kp_weight"]
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
                "kp_weight": history.kp_weight[epoch],
            }
            row.update({f"w_{key}": history.loss_weights[key][epoch] for key in loss_weight_keys})
            row.update({f"train_{key}": history.train_components[key][epoch] for key in train_component_keys})
            row.update({f"val_{key}": history.val_components[key][epoch] for key in val_component_keys})
            writer.writerow(row)


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
    base_config = ExperimentConfig()
    if args.phase is not None:
        base_config = replace(base_config, phase=args.phase)
    config = apply_phase_preset(base_config)
    config = apply_overrides(config, args)
    config = apply_auto_run_dir(config)
    set_global_seed(config.seed)

    systems = build_system_corpus(config)
    split = split_systems(systems, config)

    point_dim = systems[0].local_features.shape[1]
    pair_dim = build_pair_features(
        systems[0],
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
    ).shape[1]
    global_dim = len(systems[0].global_context)
    models = build_models(config, point_dim, pair_dim, global_dim)
    rho_mean = float(np.mean(np.concatenate([system.rho_diag for system in split.train_systems], axis=0)))
    initialize_point_model_density_bias(models.point_model, rho_mean)

    history, summary = train_models(config, split, models)

    out_dir = Path(config.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{config.run_name}.png"
    summary_path = out_dir / f"{config.run_name}_summary.json"
    history_csv_path = out_dir / f"{config.run_name}_history.csv"
    split_csv_path = out_dir / f"{config.run_name}_split_metrics.csv"
    per_system_csv_path = out_dir / f"{config.run_name}_per_system_metrics.csv"
    point_ckpt = out_dir / f"{config.run_name}_point.weights.h5"
    pair_ckpt = out_dir / f"{config.run_name}_pair.weights.h5"
    context_ckpt = out_dir / f"{config.run_name}_context.weights.h5"

    plot_training_summary(
        history=history,
        summary=summary,
        axis_points=config.axis_points,
        output_png=png_path,
    )

    models.point_model.save_weights(point_ckpt)
    models.pair_model.save_weights(pair_ckpt)
    models.context_model.save_weights(context_ckpt)

    payload = {
        "config": asdict(config),
        "history": {
            "train_objective": history.train_objective,
            "val_objective": history.val_objective,
            "learning_rate": history.learning_rate,
            "kinetic_weight": history.kinetic_weight,
            "kp_weight": history.kp_weight,
            "validation_ran": history.validation_ran,
            "loss_weights": history.loss_weights,
            "train_components": history.train_components,
            "val_components": history.val_components,
        },
        "summary": summarize_for_json(summary),
    }
    save_json(summary_path, payload)
    save_history_csv(history_csv_path, history)
    save_split_metrics_csv(split_csv_path, summary)
    save_per_system_metrics_csv(per_system_csv_path, summary)

    print_block(
        "Saved artifacts",
        [
            ("figure", png_path),
            ("summary json", summary_path),
            ("history csv", history_csv_path),
            ("split metrics csv", split_csv_path),
            ("per-system metrics csv", per_system_csv_path),
            ("point weights", point_ckpt),
            ("pair weights", pair_ckpt),
            ("context weights", context_ckpt),
        ],
    )


if __name__ == "__main__":
    main()
