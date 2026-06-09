from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .training import TrainingHistory


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


def center_grid_index(axis_points: int) -> int:
    mid = axis_points // 2
    return mid * axis_points * axis_points + mid * axis_points + mid


def reshape_gamma_anchor_slice(gamma_matrix: np.ndarray, axis_points: int) -> np.ndarray:
    """gamma(r, r0) -> 중앙 anchor r0에 대한 spatial z-plane."""
    anchor = center_grid_index(axis_points)
    gamma_column = gamma_matrix[:, anchor]
    return reshape_center_slice(gamma_column, axis_points)


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
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def add_parity_plot(
    ax,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    title: str,
) -> None:
    true_flat = np.asarray(true_values).reshape(-1)
    pred_flat = np.asarray(pred_values).reshape(-1)
    if true_flat.shape != pred_flat.shape:
        raise ValueError(
            f"Parity arrays must have matching shapes, got {true_flat.shape} and {pred_flat.shape}."
        )
    ax.scatter(true_flat, pred_flat, s=4, alpha=0.35)
    lower = float(min(np.min(true_flat), np.min(pred_flat)))
    upper = float(max(np.max(true_flat), np.max(pred_flat)))
    ax.plot([lower, upper], [lower, upper], "k--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("True")
    ax.set_ylabel("Pred")


def add_integral_ratio_plot(ax, representative: dict[str, object]) -> None:
    trace_true = float(representative["trace_true"])
    trace_pred = float(representative["trace_pred"])
    tau_true = float(representative["tau_true_integral"])
    tau_pred = float(representative["tau_pred_integral"])
    kinetic_ref = float(representative.get("kinetic_energy_ref", np.nan))

    labels = [r"$\int\rho/N$", r"$T_\tau/T_\tau^{true}$"]
    pred_ratios = [
        trace_pred / trace_true if abs(trace_true) > 1e-12 else np.nan,
        tau_pred / tau_true if abs(tau_true) > 1e-12 else np.nan,
    ]
    if np.isfinite(kinetic_ref) and abs(kinetic_ref) > 1e-12:
        labels.append(r"$T_\tau/T_s^{DFT}$")
        pred_ratios.append(tau_pred / kinetic_ref)
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, [1.0] * len(labels), width, label="reference")
    ax.bar(x + width / 2, pred_ratios, width, label="pred")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelrotation=12)
    ax.set_ylabel("Ratio")
    ax.set_title("Kinetic Energy Ratios")
    ax.legend()


def build_metrics_text(summary: dict[str, object], representative: dict[str, object]) -> str:
    val_avg = summary["val"]
    lines = [
        "Held-out Average",
        f"gamma_pair loss: {val_avg['pair_loss']:.3e}",
        f"density MAE : {val_avg['density_mae']:.3e}",
        f"tau MAE     : {val_avg['tau_mae']:.3e}",
        f"T loss      : {val_avg.get('kinetic_loss', np.nan):.3e}",
        f"T abs err   : {val_avg.get('kinetic_abs_error', np.nan):.3e}",
        f"trace loss  : {val_avg['trace_loss']:.3e}",
        f"occ penalty : {val_avg['occ_penalty']:.3e}",
        f"symmetry    : {val_avg['symmetry_mae']:.3e}",
        "",
        f"Representative: {representative['system_id']}",
        f"near diag MAE : {representative['near_diag_mae']:.3e}",
        f"far offdiag   : {representative['far_offdiag_mae']:.3e}",
        f"gamma samples : {np.asarray(representative['gamma_true_sample']).size}",
        f"|K(r,r)-1|    : {representative['kernel_diag_error']:.3e}",
        f"T_tau true/pred: {representative['tau_true_integral']:.3e} / {representative['tau_pred_integral']:.3e}",
        f"T_s DFT ref   : {representative.get('kinetic_energy_ref', np.nan):.3e}",
        "",
        "DFT MO occupations",
        np.array2string(representative.get("top_mo_occ_true", np.array([])), precision=3),
        "Subset eigs (true)",
        np.array2string(representative.get("top_subset_eigs_true", representative["top_occ_true"]), precision=3),
        "Subset eigs (pred)",
        np.array2string(representative.get("top_subset_eigs_pred", representative["top_occ_pred"]), precision=3),
    ]
    return "\n".join(lines)


def plot_training_summary(
    *,
    history: TrainingHistory,
    summary: dict[str, object],
    axis_points: int,
    output_png: Path,
) -> None:
    """학습 곡선 + held-out system 시각화."""
    representative = summary["val"]["per_system"][0]
    representative_axis_points = infer_axis_points(representative["rho_true_diag"])

    fig, axes = plt.subplots(4, 4, figsize=(21, 17))
    axes = axes.reshape(4, 4)

    ax = axes[0, 0]
    ax.plot(history.train_objective, label="train obj")
    ax.plot(history.val_objective, label="held-out val obj")
    ax.set_title("Training Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Objective")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[0, 1]
    true_source = representative.get("gamma_true_matrix", representative.get("gamma_true_sample"))
    pred_source = representative.get("gamma_pred_matrix", representative.get("gamma_pred_sample"))
    true_pairs = true_source.reshape(-1)
    pred_pairs = pred_source.reshape(-1)
    ax.scatter(true_pairs, pred_pairs, s=4, alpha=0.35)
    lims = [
        float(min(np.min(true_pairs), np.min(pred_pairs))),
        float(max(np.max(true_pairs), np.max(pred_pairs))),
    ]
    ax.plot(lims, lims, "k--", linewidth=1.0)
    ax.set_title("Held-out Gamma Parity")
    ax.set_xlabel(r"$\gamma_{\rm true}$")
    ax.set_ylabel(r"$\gamma_{\rm pred}$")

    rho_true_slice = reshape_center_slice(representative["rho_true_diag"], representative_axis_points)
    rho_pred_slice = reshape_center_slice(representative["rho_pred_diag"], representative_axis_points)
    rho_vmin, rho_vmax = shared_limits(rho_true_slice, rho_pred_slice, include_zero=True)
    add_imshow(
        axes[0, 2],
        rho_true_slice,
        r"True $\rho$ Slice",
        vmin=rho_vmin,
        vmax=rho_vmax,
    )
    add_imshow(
        axes[0, 3],
        rho_pred_slice,
        r"Pred $\rho$ Slice",
        vmin=rho_vmin,
        vmax=rho_vmax,
    )
    tau_target = np.asarray(representative.get("tau_target_eval", representative["tau_true"]))
    tau_pred = np.asarray(representative["tau_pred"])
    sampled_tau = bool(representative.get("stencil_eval_sampled", 0.0))
    if not sampled_tau and tau_target.size == tau_pred.size:
        tau_true_slice = reshape_tau_slice(tau_target, representative_axis_points)
        tau_pred_slice = reshape_tau_slice(tau_pred, representative_axis_points)
        tau_vmin, tau_vmax = shared_limits(tau_true_slice, tau_pred_slice, include_zero=True)
        add_imshow(
            axes[1, 0],
            tau_true_slice,
            r"True $\tau$ Slice",
            vmin=tau_vmin,
            vmax=tau_vmax,
        )
        add_imshow(
            axes[1, 1],
            tau_pred_slice,
            r"Pred $\tau$ Slice",
            vmin=tau_vmin,
            vmax=tau_vmax,
        )
    else:
        add_parity_plot(axes[1, 0], tau_target, tau_pred, r"Sampled $\tau$ Parity")
        tau_abs_error = np.abs(tau_pred.reshape(-1) - tau_target.reshape(-1))
        axes[1, 1].hist(tau_abs_error, bins=40, color="tab:red", alpha=0.8)
        axes[1, 1].set_title(r"Sampled $|\Delta\tau|$")
        axes[1, 1].set_xlabel("Absolute error")
        axes[1, 1].set_ylabel("Count")

    has_full_gamma = "gamma_true_matrix" in representative and "gamma_pred_matrix" in representative
    if has_full_gamma:
        gamma_true_slice = reshape_gamma_anchor_slice(
            representative["gamma_true_matrix"],
            representative_axis_points,
        )
        gamma_pred_slice = reshape_gamma_anchor_slice(
            representative["gamma_pred_matrix"],
            representative_axis_points,
        )
        gamma_vmin, gamma_vmax = shared_limits(gamma_true_slice, gamma_pred_slice, symmetric=True)
        add_imshow(
            axes[1, 2],
            gamma_true_slice,
            r"True $\gamma(r,r_0)$ Slice",
            vmin=gamma_vmin,
            vmax=gamma_vmax,
        )
        add_imshow(
            axes[1, 3],
            gamma_pred_slice,
            r"Pred $\gamma(r,r_0)$ Slice",
            vmin=gamma_vmin,
            vmax=gamma_vmax,
        )
        gamma_err_vmin, gamma_err_vmax = shared_limits(np.abs(gamma_pred_slice - gamma_true_slice), include_zero=True)
        add_imshow(
            axes[2, 0],
            np.abs(gamma_pred_slice - gamma_true_slice),
            r"$|\Delta\gamma(r,r_0)|$ Slice",
            vmin=gamma_err_vmin,
            vmax=gamma_err_vmax,
            cmap="Reds",
        )
    else:
        ax = axes[1, 2]
        abs_err = np.abs(pred_pairs - true_pairs)
        ax.hist(abs_err, bins=40, color="tab:blue", alpha=0.8)
        ax.set_title(r"Sampled $|\Delta\gamma|$")
        ax.set_xlabel("Absolute error")
        ax.set_ylabel("Count")

        axes[1, 3].axis("off")
        axes[1, 3].text(
            0.0,
            0.5,
            "Full gamma slice skipped\n(sample parity only)",
            va="center",
            ha="left",
            family="monospace",
        )

        axes[2, 0].axis("off")

    rho_err_slice = np.abs(rho_pred_slice - rho_true_slice)
    rho_err_vmin, rho_err_vmax = shared_limits(rho_err_slice, include_zero=True)
    add_imshow(axes[2, 1], rho_err_slice, r"$|\Delta\rho|$ Slice", vmin=rho_err_vmin, vmax=rho_err_vmax, cmap="Reds")
    if not sampled_tau and tau_target.size == tau_pred.size:
        tau_err_slice = np.abs(tau_pred_slice - tau_true_slice)
        tau_err_vmin, tau_err_vmax = shared_limits(tau_err_slice, include_zero=True)
        add_imshow(
            axes[2, 2],
            tau_err_slice,
            r"$|\Delta\tau|$ Slice",
            vmin=tau_err_vmin,
            vmax=tau_err_vmax,
            cmap="Reds",
        )
    else:
        axes[2, 2].axis("off")
        axes[2, 2].text(
            0.0,
            0.5,
            "Spatial tau slice skipped\n(sampled stencil centers)",
            va="center",
            ha="left",
            family="monospace",
        )
    axes[2, 3].axis("off")

    add_integral_ratio_plot(axes[3, 0], representative)

    ax = axes[3, 1]
    top_true = representative.get("top_subset_eigs_true", representative["top_occ_true"])
    top_pred = representative.get("top_subset_eigs_pred", representative["top_occ_pred"])
    x = np.arange(max(len(top_true), len(top_pred)))
    ax.plot(x[: len(top_true)], top_true, "o-", label="true")
    ax.plot(x[: len(top_pred)], top_pred, "s--", label="pred")
    ax.set_title("Coarse Subset Eigenvalues")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Eigenvalue")
    ax.legend()

    ax = axes[3, 2]
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        build_metrics_text(summary, representative),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    axes[3, 3].axis("off")

    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
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
