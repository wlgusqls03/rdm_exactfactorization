from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from .training import TrainingHistory


TITLE_SIZE = 12
LABEL_SIZE = 10
TICK_SIZE = 9


def reshape_center_slice(values: np.ndarray, axis_points: int) -> np.ndarray:
    """(n_points,1) -> 중앙 z-plane 2D slice."""
    cube = values.reshape(axis_points, axis_points, axis_points)
    mid = axis_points // 2
    return cube[:, :, mid]


def reshape_tau_slice(tau_values: np.ndarray, axis_points: int) -> np.ndarray:
    """interior tau -> 중앙 interior plane."""
    n_values = int(np.asarray(tau_values).reshape(-1).shape[0])
    n_interior_axis = int(round(n_values ** (1.0 / 3.0)))
    if n_interior_axis**3 != n_values:
        n_interior_axis = axis_points - 2
    cube = tau_values.reshape(n_interior_axis, n_interior_axis, n_interior_axis)
    mid = n_interior_axis // 2
    return cube[:, :, mid]


def infer_axis_points(values: np.ndarray) -> int:
    """Infer cubic grid axis count from a flattened point array."""
    n_points = int(np.asarray(values).reshape(-1).shape[0])
    axis_points = int(round(n_points ** (1.0 / 3.0)))
    if axis_points**3 != n_points:
        raise ValueError(f"Cannot infer cubic axis size from {n_points} points.")
    return axis_points


def shared_limits(
    *images: np.ndarray,
    symmetric: bool = False,
    include_zero: bool = False,
) -> tuple[float, float]:
    values = np.concatenate([np.asarray(image, dtype=np.float64).reshape(-1) for image in images])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    if symmetric:
        limit = float(np.max(np.abs(values)))
        limit = max(limit, 1e-12)
        return -limit, limit
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if include_zero:
        vmin = min(vmin, 0.0)
        vmax = max(vmax, 0.0)
    if abs(vmax - vmin) < 1e-12:
        if include_zero and vmin >= 0.0:
            vmin = 0.0
            vmax = max(vmax, 1e-12)
        else:
            pad = max(abs(vmax), 1.0) * 1e-6
            vmin -= pad
            vmax += pad
    return vmin, vmax


def add_imshow(
    ax,
    image: np.ndarray,
    title: str,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "coolwarm",
) -> None:
    im = ax.imshow(image.T, origin="lower", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    colorbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    colorbar.ax.tick_params(labelsize=TICK_SIZE)


def add_parity_plot(
    ax,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    title: str,
) -> None:
    true_flat = np.asarray(true_values, dtype=np.float64).reshape(-1)
    pred_flat = np.asarray(pred_values, dtype=np.float64).reshape(-1)
    if true_flat.shape != pred_flat.shape:
        raise ValueError(
            f"Parity arrays must have matching shapes, got {true_flat.shape} and {pred_flat.shape}."
        )
    finite = np.isfinite(true_flat) & np.isfinite(pred_flat)
    true_flat = true_flat[finite]
    pred_flat = pred_flat[finite]
    if true_flat.size == 0:
        raise ValueError(f"{title} has no finite values.")
    ax.scatter(true_flat, pred_flat, s=7, alpha=0.3, edgecolors="none", rasterized=True)
    lower = float(min(np.min(true_flat), np.min(pred_flat)))
    upper = float(max(np.max(true_flat), np.max(pred_flat)))
    ax.plot([lower, upper], [lower, upper], "k--", linewidth=1.0)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlabel("True", fontsize=LABEL_SIZE)
    ax.set_ylabel("Pred", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(alpha=0.18)


def add_absolute_error_histogram(
    ax,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    title: str,
    *,
    color: str,
) -> None:
    errors = np.abs(
        np.asarray(pred_values, dtype=np.float64).reshape(-1)
        - np.asarray(true_values, dtype=np.float64).reshape(-1)
    )
    errors = errors[np.isfinite(errors)]
    ax.hist(errors, bins=40, color=color, alpha=0.82)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlabel("Absolute error", fontsize=LABEL_SIZE)
    ax.set_ylabel("Count", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(axis="y", alpha=0.18)


def add_sorted_comparison(
    ax,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    title: str,
    *,
    max_points: int = 2048,
) -> None:
    true_flat = np.asarray(true_values, dtype=np.float64).reshape(-1)
    pred_flat = np.asarray(pred_values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(true_flat) & np.isfinite(pred_flat)
    true_flat = true_flat[finite]
    pred_flat = pred_flat[finite]
    order = np.argsort(true_flat)
    if order.size > max_points:
        order = order[np.linspace(0, order.size - 1, max_points, dtype=np.int64)]
    ax.plot(true_flat[order], label="true", linewidth=1.5)
    ax.plot(pred_flat[order], label="pred", linewidth=1.1, alpha=0.85)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlabel("Samples sorted by true value", fontsize=LABEL_SIZE)
    ax.set_ylabel("Value", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(alpha=0.18)
    ax.legend(fontsize=9)


def add_kinetic_energy_plot(ax, representative: dict[str, object]) -> None:
    kinetic_true = float(
        representative.get(
            "kinetic_training_ref",
            representative.get("tau_true_integral", np.nan),
        )
    )
    kinetic_pred = float(
        representative.get(
            "kinetic_pred",
            representative.get("tau_pred_integral", np.nan),
        )
    )
    delta = kinetic_pred - kinetic_true
    abs_error = abs(delta)
    relative_error = 100.0 * delta / kinetic_true if abs(kinetic_true) > 1e-12 else np.nan

    bars = ax.bar(
        ["Reference", "Prediction"],
        [kinetic_true, kinetic_pred],
        color=["#4C78A8", "#F58518"],
        width=0.62,
    )
    value_scale = max(abs(kinetic_true), abs(kinetic_pred), 1.0)
    for bar, value in zip(bars, (kinetic_true, kinetic_pred)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025 * value_scale,
            f"{value:.6f} Ha",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(
        min(0.0, kinetic_true, kinetic_pred),
        max(kinetic_true, kinetic_pred, 0.0) + 0.14 * value_scale,
    )
    ax.set_title(
        "Kinetic Energy Comparison\n"
        rf"$\Delta T={delta:+.3e}$ Ha, "
        rf"$|\Delta T|={abs_error:.3e}$ Ha, "
        f"rel.={relative_error:+.2f}%",
        fontsize=TITLE_SIZE,
        linespacing=1.35,
    )
    ax.set_ylabel("Kinetic energy [Ha]", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(axis="y", alpha=0.2)


def build_metrics_text(summary: dict[str, object], representative: dict[str, object]) -> str:
    val_avg = summary["val"]
    metadata = summary.get("evaluation_metadata", {})
    run_config = metadata.get("run_config", {})
    loss_weights = metadata.get("final_scheduled_loss_weights", metadata.get("loss_weights", {}))
    sampled_stencil = bool(val_avg.get("stencil_eval_sampled", 0.0))
    active_loss_parts = [
        f"{name}={float(weight):g}"
        for name, weight in loss_weights.items()
        if isinstance(weight, (int, float)) and float(weight) != 0.0
    ]
    if not active_loss_parts:
        active_loss_lines = ["active losses   : none"]
    else:
        active_loss_lines = []
        current = "active losses   : "
        continuation = "                  "
        for part in active_loss_parts:
            candidate = f"{current}{part}" if current.endswith(": ") else f"{current}, {part}"
            if len(candidate) > 64 and current.strip() != "active losses":
                active_loss_lines.append(current)
                current = continuation + part
            else:
                current = candidate
        active_loss_lines.append(current)
    lines = [
        "RUN SETTINGS",
        f"loss preset     : {run_config.get('loss_preset', 'unknown')}",
        (
            "rank/width/depth: "
            f"{run_config.get('rank', 'n/a')} / "
            f"{run_config.get('width', 'n/a')} / "
            f"pair{run_config.get('pair_model_depth', 'n/a')}"
        ),
        f"pair rho feat   : {run_config.get('pair_density_feature_mode', 'unknown')}",
        *active_loss_lines,
        (
            "batch/stencil   : "
            f"{run_config.get('batch_size', 'n/a')} / "
            f"{run_config.get('train_stencil_centers', 'n/a')}"
        ),
        "",
        "HELD-OUT AVERAGE",
        f"gamma loss      : {val_avg['pair_loss']:10.3e}",
        f"gamma MAE/RMSE  : {val_avg.get('pair_mae', np.nan):10.3e} / {val_avg.get('pair_rmse', np.nan):10.3e}",
        (
            "gamma d/n/m/f  : "
            f"{val_avg.get('diag_pair_mae', np.nan):.2e} / "
            f"{val_avg.get('near_diag_mae', np.nan):.2e} / "
            f"{val_avg.get('mid_pair_mae', np.nan):.2e} / "
            f"{val_avg.get('far_offdiag_mae', np.nan):.2e}"
        ),
        (
            "stencil gamma  : "
            f"{val_avg.get('stencil_gamma_mae', np.nan):.2e} / "
            f"{val_avg.get('stencil_gamma_rmse', np.nan):.2e}"
        ),
        f"density MAE     : {val_avg['density_mae']:10.3e}",
        f"tau MAE/RMSE    : {val_avg['tau_mae']:10.3e} / {val_avg.get('tau_rmse', np.nan):10.3e}",
        f"KINETIC ABS [Ha]: {val_avg.get('kinetic_abs_error', np.nan):10.3e}",
        f"KINETIC RMSE/P90: {val_avg.get('kinetic_rmse', np.nan):10.3e} / {val_avg.get('kinetic_abs_error_p90', np.nan):10.3e}",
        f"rho-vW T [Ha]  : {val_avg.get('rho_vw_kinetic', np.nan):10.3e}",
        f"rho-vW - gammaT: {val_avg.get('rho_vw_minus_gamma_kinetic', np.nan):+10.3e}",
        (
            "T stencil d/off: "
            f"{val_avg.get('kinetic_stencil_diag_error', np.nan):+.2e} / "
            f"{val_avg.get('kinetic_stencil_offdiag_error', np.nan):+.2e}"
        ),
        f"GRID E REF-PRED : {val_avg.get('energy_total_grid_ref_minus_pred', np.nan):+10.3e}",
        f"grid E MAE [Ha] : {val_avg.get('energy_grid_total_abs_error', np.nan):10.3e}",
        (
            "final eval systems = "
            f"{val_avg.get('evaluated_system_count', len(val_avg.get('per_system', [])))} / "
            f"{val_avg.get('available_system_count', len(val_avg.get('per_system', [])))}"
        ),
        (
            "stencil eval centers = "
            f"{val_avg.get('stencil_eval_centers', np.nan):.0f} / "
            f"{val_avg.get('stencil_eval_total_centers', np.nan):.0f}"
        ),
        f"sampled stencil eval = {sampled_stencil}",
        f"kinetic integral active = {bool(metadata.get('kinetic_integral_active', False))}",
        f"density source = {metadata.get('density_source', 'unknown')}",
        "kinetic estimate = " + ",".join(val_avg.get("kinetic_evaluation_modes", [])),
        "gamma-FD source = " + ",".join(val_avg.get("gamma_fd_target_sources", [])),
        "",
        "REPRESENTATIVE SYSTEM",
        f"id              : {representative['system_id']}",
        f"near-diag MAE   : {representative['near_diag_mae']:10.3e}",
        f"mid-pair MAE    : {representative.get('mid_pair_mae', np.nan):10.3e}",
        f"far-offdiag MAE : {representative['far_offdiag_mae']:10.3e}",
        (
            "stencil g MAE/RMSE: "
            f"{representative.get('stencil_gamma_mae', np.nan):.2e} / "
            f"{representative.get('stencil_gamma_rmse', np.nan):.2e}"
        ),
        f"gamma samples   : {np.asarray(representative['gamma_true_sample']).size:10d}",
        f"|K(r,r)-1|      : {representative['kernel_diag_error']:10.3e}",
        f"tau MAE         : {representative['tau_mae']:10.3e}",
        f"T true [Ha]     : {representative.get('kinetic_training_ref', np.nan):10.3e}",
        f"T pred [Ha]     : {representative.get('kinetic_pred', np.nan):10.3e}",
        f"rho-vW T [Ha]   : {representative.get('rho_vw_kinetic', np.nan):10.3e}",
        (
            "anchor xyz idx  : "
            + np.array2string(
                np.asarray(representative.get("gamma_anchor_xyz_index", [])),
                separator=",",
            )
        ),
    ]
    if val_avg.get("energy_stored_total_available", 0.0) >= 1.0:
        lines.insert(
            7,
            (
                "stored E - grid pred (diagnostic) = "
                f"{val_avg.get('energy_total_ref_minus_pred', np.nan):+10.3e}"
            ),
        )
    return "\n".join(lines)


def plot_training_summary(
    *,
    history: TrainingHistory,
    summary: dict[str, object],
    axis_points: int,
    output_png: Path,
) -> None:
    """Save a dense, publication-ready training and held-out summary."""
    representative = summary["val"]["per_system"][0]
    representative_axis_points = infer_axis_points(representative["rho_true_diag"])

    fig = plt.figure(figsize=(22, 19), constrained_layout=False)
    grid = fig.add_gridspec(
        4,
        4,
        left=0.045,
        right=0.985,
        bottom=0.045,
        top=0.925,
        hspace=0.44,
        wspace=0.32,
    )
    fig.suptitle(
        f"1-RDM Transferable Model Summary | {representative['system_id']}",
        x=0.5,
        y=0.978,
        fontsize=15,
        fontweight="semibold",
    )

    ax = fig.add_subplot(grid[0, 0])
    ax.plot(history.train_objective, label="train obj")
    ax.plot(history.val_objective, label="held-out val obj")
    ax.set_title("Training Curve", fontsize=TITLE_SIZE)
    ax.set_xlabel("Epoch", fontsize=LABEL_SIZE)
    ax.set_ylabel("Objective", fontsize=LABEL_SIZE)
    ax.set_yscale("log")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)

    true_pairs = np.asarray(representative["gamma_true_sample"]).reshape(-1)
    pred_pairs = np.asarray(representative["gamma_pred_sample"]).reshape(-1)
    add_parity_plot(
        fig.add_subplot(grid[0, 1]),
        true_pairs,
        pred_pairs,
        r"Sampled $\gamma$ Parity",
    )
    add_absolute_error_histogram(
        fig.add_subplot(grid[0, 2]),
        true_pairs,
        pred_pairs,
        r"Sampled $|\Delta\gamma|$",
        color="#4C78A8",
    )
    add_sorted_comparison(
        fig.add_subplot(grid[0, 3]),
        true_pairs,
        pred_pairs,
        r"Sampled $\gamma$ Values",
    )

    tau_target = np.asarray(representative.get("tau_target_eval", representative["tau_true"]))
    tau_pred = np.asarray(representative["tau_pred"])
    sampled_tau = bool(representative.get("stencil_eval_sampled", 0.0))
    gamma_true_slice = np.asarray(representative["gamma_anchor_true_slice"])
    gamma_pred_slice = np.asarray(representative["gamma_anchor_pred_slice"])
    gamma_vmin, gamma_vmax = shared_limits(gamma_true_slice, gamma_pred_slice, symmetric=True)
    add_imshow(
        fig.add_subplot(grid[1, 0]),
        gamma_true_slice,
        r"True $\gamma(r,r_0)$ Slice",
        vmin=gamma_vmin,
        vmax=gamma_vmax,
    )
    add_imshow(
        fig.add_subplot(grid[1, 1]),
        gamma_pred_slice,
        r"Pred $\gamma(r,r_0)$ Slice",
        vmin=gamma_vmin,
        vmax=gamma_vmax,
    )
    gamma_error_slice = np.abs(gamma_pred_slice - gamma_true_slice)
    gamma_err_vmin, gamma_err_vmax = shared_limits(gamma_error_slice, include_zero=True)
    add_imshow(
        fig.add_subplot(grid[1, 2]),
        gamma_error_slice,
        r"$|\Delta\gamma(r,r_0)|$ Slice",
        vmin=gamma_err_vmin,
        vmax=gamma_err_vmax,
        cmap="Reds",
    )
    add_parity_plot(
        fig.add_subplot(grid[1, 3]),
        tau_target,
        tau_pred,
        "Sampled Tau Parity" if sampled_tau else "Tau Parity",
    )

    rho_true_slice = reshape_center_slice(representative["rho_true_diag"], representative_axis_points)
    rho_pred_slice = reshape_center_slice(representative["rho_pred_diag"], representative_axis_points)
    rho_vmin, rho_vmax = shared_limits(rho_true_slice, rho_pred_slice, include_zero=True)
    add_imshow(
        fig.add_subplot(grid[2, 0]),
        rho_true_slice,
        r"True $\rho$ Slice",
        vmin=rho_vmin,
        vmax=rho_vmax,
        cmap="viridis",
    )
    add_imshow(
        fig.add_subplot(grid[2, 1]),
        rho_pred_slice,
        r"Pred $\rho$ Slice",
        vmin=rho_vmin,
        vmax=rho_vmax,
        cmap="viridis",
    )
    rho_err_slice = np.abs(rho_pred_slice - rho_true_slice)
    rho_err_vmin, rho_err_vmax = shared_limits(rho_err_slice, include_zero=True)
    add_imshow(
        fig.add_subplot(grid[2, 2]),
        rho_err_slice,
        r"$|\Delta\rho|$ Slice",
        vmin=rho_err_vmin,
        vmax=rho_err_vmax,
        cmap="magma",
    )
    add_sorted_comparison(
        fig.add_subplot(grid[2, 3]),
        tau_target,
        tau_pred,
        "Sampled Tau Values" if sampled_tau else "Tau Values",
    )

    add_kinetic_energy_plot(fig.add_subplot(grid[3, 0]), representative)

    ax = fig.add_subplot(grid[3, 1])
    per_system = summary["val"]["per_system"]
    kinetic_errors = np.asarray(
        [item.get("kinetic_ref_error", np.nan) for item in per_system],
        dtype=np.float64,
    )
    kinetic_errors = kinetic_errors[np.isfinite(kinetic_errors)]
    ax.hist(kinetic_errors, bins=min(20, max(6, kinetic_errors.size)), color="#72B7B2", alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(r"Held-out $\Delta T$ Distribution", fontsize=TITLE_SIZE)
    ax.set_xlabel(r"$T_{pred}-T_{true}$ [Ha]", fontsize=LABEL_SIZE)
    ax.set_ylabel("Systems", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(axis="y", alpha=0.2)

    ax = fig.add_subplot(grid[3, 2])
    top_true = representative.get("top_subset_eigs_true", representative["top_occ_true"])
    top_pred = representative.get("top_subset_eigs_pred", representative["top_occ_pred"])
    x = np.arange(max(len(top_true), len(top_pred)))
    ax.plot(x[: len(top_true)], top_true, "o-", label="true")
    ax.plot(x[: len(top_pred)], top_pred, "s--", label="pred")
    ax.set_title("Coarse Subset Eigenvalues", fontsize=TITLE_SIZE)
    ax.set_xlabel("Mode index", fontsize=LABEL_SIZE)
    ax.set_ylabel("Eigenvalue", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)

    ax = fig.add_subplot(grid[3, 3])
    ax.axis("off")
    ax.text(
        0.0,
        0.98,
        build_metrics_text(summary, representative),
        va="top",
        ha="left",
        family="monospace",
        fontsize=7.3,
        linespacing=1.08,
        transform=ax.transAxes,
    )

    fig.savefig(
        output_png,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.22,
        facecolor="white",
    )
    print(f"Saved figure to: {output_png}")

    if plt.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close(fig)


def plot_point_pretrain_summary(summary: dict[str, object], output_png: Path) -> None:
    """Save held-out density-head diagnostics before pair/context training."""
    representative = summary["val"]["per_system"][0]
    rows = [("rho_neutral", r"$\rho_N$")]
    if "fukui_plus_true" in representative:
        rows.extend([("fukui_plus", r"$f^+$"), ("fukui_minus", r"$f^-$")])
    axis_points = infer_axis_points(representative[f"{rows[0][0]}_true"])
    fig, axes = plt.subplots(len(rows), 3, figsize=(14, 4.2 * len(rows)), squeeze=False)
    for row_idx, (name, label) in enumerate(rows):
        true_slice = reshape_center_slice(representative[f"{name}_true"], axis_points)
        pred_slice = reshape_center_slice(representative[f"{name}_pred"], axis_points)
        vmin, vmax = shared_limits(true_slice, pred_slice, symmetric=name.startswith("fukui"), include_zero=True)
        add_imshow(axes[row_idx, 0], true_slice, f"True {label}", vmin=vmin, vmax=vmax)
        add_imshow(axes[row_idx, 1], pred_slice, f"Pred {label}", vmin=vmin, vmax=vmax)
        err = np.abs(pred_slice - true_slice)
        err_vmin, err_vmax = shared_limits(err, include_zero=True)
        add_imshow(axes[row_idx, 2], err, f"Absolute error: {label}", vmin=err_vmin, vmax=err_vmax, cmap="Reds")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved point pretrain figure to: {output_png}")
