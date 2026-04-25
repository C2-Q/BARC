# v2 (2026-04-25): layout fixes, family-coded control markers, raw QFT peak-window traces, consolidated gap-case legend
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_posthoc_analysis import (
    _display_case_id,
    _display_family,
    _display_policy,
    _reconstruct_gap_case,
    _select_representative_gap_cases,
)
from src.plots import (
    FAMILY_COLORS,
    PAPER_LABEL_SIZE,
    PAPER_LEGEND_SIZE,
    PAPER_TICK_SIZE,
    _apply_paper_axis_style,
)
from src.predictive_analysis import (
    _binary_metric_eval,
    _fit_linear_model,
    _fit_logistic_auc_model,
    _subset_regression_metric,
)
from src.real_trace.analysis import load_real_trace_from_csv
from src.real_trace.adder_trace import extract_t_demand_trace, to_clifford_t
from src.real_trace.qft_trace import generate_qft_circuit
from src.utils import OUTPUT_ROOT, TABLE_DIR, ensure_output_dirs


FINAL_PAPER_REVISED_DIR = OUTPUT_ROOT / "figures" / "final_paper_revised"
PREDICTOR_ORDER = ["T_depth", "slack_ratio", "delta_max"]
PREDICTOR_LABELS = {
    "T_depth": "T-depth",
    "slack_ratio": "Slack ratio",
    "delta_max": r"$\Delta_{\max}$",
}
PREDICTOR_COLORS = {
    "T_depth": "#7a7a7a",
    "slack_ratio": "#1f78b4",
    "delta_max": "#d95f02",
}
FAMILY_ORDER = ["high_compressibility", "medium_compressibility", "low_compressibility"]
FAMILY_LABELS = {
    "high_compressibility": "High",
    "medium_compressibility": "Medium",
    "low_compressibility": "Low",
}


def main() -> None:
    ensure_output_dirs()
    FINAL_PAPER_REVISED_DIR.mkdir(parents=True, exist_ok=True)

    predictive_df = pd.read_csv(TABLE_DIR / "predictive_static_dataset.csv")
    lower_bound_df = pd.read_csv(TABLE_DIR / "lower_bound_validation.csv")
    qft_reduced_df = pd.read_csv(TABLE_DIR / "qft_approximation_reduced_grid.csv")
    grid_df = pd.read_csv(TABLE_DIR / "stage1_grid_scan.csv")

    outputs = [
        plot_predictor_comparison_revised(predictive_df),
        plot_lower_bound_vs_actual_revised(lower_bound_df),
        plot_qft_approximation_reduced_grid_revised(qft_reduced_df),
        plot_lower_bound_gap_cases_revised(lower_bound_df, grid_df),
    ]
    for path in outputs:
        print(path)


def plot_predictor_comparison_revised(predictive_df: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), constrained_layout=True)
    task_specs = [
        ("stall", "Classification AUC", "Stall"),
        ("inversion", "Classification AUC", "Inversion"),
        ("slowdown_ratio", r"Regression $|\rho|$", "Slowdown"),
    ]

    for axis, (task, ylabel, title) in zip(axes, task_specs, strict=True):
        slice_df = _build_predictor_slice_distribution_overall(predictive_df, task)
        family_overlay_df = _build_predictor_family_overlay_distribution(predictive_df, task)
        _draw_predictor_boxplot_panel(axis, slice_df, family_overlay_df, "score", ylabel, title, task)

    predictor_handles = [
        Patch(facecolor=PREDICTOR_COLORS[predictor], edgecolor="#444444", label=PREDICTOR_LABELS[predictor], alpha=0.72)
        for predictor in PREDICTOR_ORDER
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            markersize=5.5,
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#555555",
            label="Low-compressibility slice (control regime)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=5.0,
            linestyle="none",
            markerfacecolor="#555555",
            markeredgecolor="#555555",
            label="High / medium compressibility slice",
        ),
    ]
    figure.legend(
        handles=predictor_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        ncol=1,
        frameon=False,
        fontsize=12,
        borderaxespad=0.0,
        handletextpad=0.5,
    )
    figure.legend(
        handles=family_handles,
        loc="lower center",
        bbox_to_anchor=(0.44, -0.08),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    return _save_revised_figure(figure, "predictor_comparison_final_v2")


def plot_incremental_predictive_gain_revised(
    predictive_df: pd.DataFrame,
    incremental_df: pd.DataFrame,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.2))
    gain_df = _build_incremental_gain_distribution(predictive_df)

    panel_specs = [
        ("slowdown_ratio", r"$\Delta R^2$", "Slowdown"),
        ("stall", r"$\Delta$AUC", "Stall"),
    ]
    for axis, (task, ylabel, title) in zip(axes, panel_specs, strict=True):
        subset = gain_df[gain_df["task"] == task].copy()
        _draw_incremental_gain_panel(axis, subset, ylabel, title)

    figure.suptitle("Incremental gains across bounded-delivery slices", fontsize=9.2, y=0.99)
    figure.subplots_adjust(left=0.09, right=0.995, bottom=0.25, top=0.8, wspace=0.24)
    return _save_revised_figure(figure, "incremental_predictive_gain_final")


def plot_lower_bound_vs_actual_revised(lower_bound_df: pd.DataFrame) -> Path:
    subset = lower_bound_df[np.isfinite(lower_bound_df["T_exe"])].copy()
    gaps = subset["gap"].astype(float).to_numpy()
    sorted_gaps = np.sort(gaps)
    cdf = np.arange(1, len(sorted_gaps) + 1) / max(1, len(sorted_gaps))
    within_one = float((gaps <= 1.0 + 1e-9).mean()) if len(gaps) else 0.0

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.5, 3.2),
        constrained_layout=True,
        gridspec_kw={"wspace": 0.35},
    )

    scatter = axes[0].scatter(
        subset["predicted_lower_bound"],
        subset["T_exe"],
        s=9,
        c=subset["gap"],
        cmap="cividis",
        alpha=0.62,
        edgecolors="none",
        rasterized=True,
    )
    if not subset.empty:
        lower = float(min(subset["predicted_lower_bound"].min(), subset["T_exe"].min()))
        upper = float(max(subset["predicted_lower_bound"].max(), subset["T_exe"].max()))
        axes[0].plot([lower, upper], [lower, upper], color="#444444", linewidth=1.0, zorder=2)
    axes[0].set_title("Lower bound vs observed makespan", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Predicted lower bound", fontsize=PAPER_LABEL_SIZE)
    axes[0].set_ylabel(r"Observed $T_{\mathrm{exe}}$", fontsize=PAPER_LABEL_SIZE)
    _apply_paper_axis_style(axes[0])
    colorbar = figure.colorbar(scatter, ax=axes[0], fraction=0.05, pad=0.025)
    colorbar.set_label(r"Gap = $T_{\mathrm{exe}} - \mathrm{bound}$ (cycles)", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)

    axes[1].step(sorted_gaps, cdf, where="post", color="#1f78b4", linewidth=2.0)
    axes[1].fill_between(sorted_gaps, cdf, step="post", alpha=0.15, color="#1f78b4")
    axes[1].axvline(1.0, color="#b22222", linestyle="--", linewidth=1.0)
    axes[1].axhline(within_one, color="#b22222", linestyle="--", linewidth=1.0)
    axes[1].scatter([1.0], [within_one], color="#b22222", s=18, zorder=4)
    axes[1].annotate(
        f"{within_one * 100:.1f}% within 1 cycle",
        xy=(1.0, within_one),
        xytext=(1.5, min(0.98, within_one + 0.05)),
        textcoords="data",
        fontsize=8,
        color="#8b1e1e",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
    )
    axes[1].set_title("Distribution of gap to bound", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Gap to lower bound (cycles)", fontsize=PAPER_LABEL_SIZE)
    axes[1].set_ylabel("Cumulative fraction of instances", fontsize=PAPER_LABEL_SIZE)
    axes[1].set_xlim(left=0.0)
    axes[1].set_ylim(0.0, 1.02)
    _apply_paper_axis_style(axes[1])

    return _save_revised_figure(figure, "lower_bound_vs_actual_final_v2")


def plot_qft_approximation_reduced_grid_revised(qft_reduced_df: pd.DataFrame) -> Path:
    exact_trace = load_real_trace_from_csv(str(PROJECT_ROOT / "data" / "real_traces" / "qft_n12.csv"))
    approx_trace = extract_t_demand_trace(
        to_clifford_t(generate_qft_circuit(12, do_swaps=False, approximation_degree=4))
    )
    representative_C = 2
    representative_B = 8
    summary_df = pd.read_csv(TABLE_DIR / "qft_approximation_reduced_grid_summary.csv")
    exact_color = "#2b6cb0"
    approx_color = "#2f855a"
    supply_color = "#444444"
    excess_color = "#e53e3e"

    figure = plt.figure(figsize=(7.8, 4.6), constrained_layout=True)
    grid_spec = figure.add_gridspec(2, 2, width_ratios=[1.0, 1.1])
    ax_exact_step = figure.add_subplot(grid_spec[0, 0])
    ax_approx_step = figure.add_subplot(grid_spec[1, 0], sharex=ax_exact_step, sharey=ax_exact_step)
    ax_cum = figure.add_subplot(grid_spec[:, 1])

    exact_offsets, exact_window, exact_local_peak = _extract_peak_window(np.array(exact_trace, dtype=int), half_window=1000)
    approx_offsets, approx_window, approx_local_peak = _extract_peak_window(np.array(approx_trace, dtype=int), half_window=1000)

    _plot_qft_step_panel(
        ax_exact_step,
        exact_offsets,
        exact_window,
        representative_C,
        exact_color,
        "Exact QFT (n=12)",
        exact_local_peak,
    )
    _plot_qft_step_panel(
        ax_approx_step,
        approx_offsets,
        approx_window,
        representative_C,
        approx_color,
        "Degree-4 approximate QFT",
        approx_local_peak,
    )
    ax_exact_step.tick_params(labelbottom=False)
    ax_exact_step.set_ylabel("T-gates per step", fontsize=9)
    ax_approx_step.set_ylabel("T-gates per step", fontsize=9)
    ax_approx_step.set_xlabel("Logical time-step offset (window around peak)", fontsize=9)
    ax_exact_step.set_ylim(0, 11)
    ax_approx_step.set_ylim(0, 11)

    x_full = np.arange(1, len(exact_trace) + 1)
    exact_cumulative = np.cumsum(np.array(exact_trace, dtype=int))
    approx_cumulative = np.cumsum(np.array(approx_trace, dtype=int))
    supply = representative_C * x_full + representative_B
    x_plot, exact_plot = _downsample_series(x_full, exact_cumulative, max_points=2200)
    _, approx_plot = _downsample_series(x_full, approx_cumulative, max_points=2200)
    _, supply_plot = _downsample_series(x_full, supply, max_points=2200)
    ax_cum.plot(x_plot, exact_plot, color=exact_color, linewidth=1.35, label="Exact cumulative demand")
    ax_cum.plot(x_plot, approx_plot, color=approx_color, linewidth=1.35, label="Approx. cumulative demand")
    ax_cum.plot(
        x_plot,
        supply_plot,
        color=supply_color,
        linewidth=1.1,
        linestyle="--",
        label=rf"Supply envelope ($C={representative_C}, B={representative_B}$)",
    )
    ax_cum.set_xlabel("Logical cycle", fontsize=9)
    ax_cum.set_ylabel("Cumulative T demand", fontsize=9)
    ax_cum.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.85)
    _apply_paper_axis_style(ax_cum)

    exact_overall = summary_df[(summary_df["variant"] == "exact") & (summary_df["C"].isna())].iloc[0]
    approx_overall = summary_df[(summary_df["variant"] == "approx_deg_4") & (summary_df["C"].isna())].iloc[0]
    summary_text = "\n".join(
        [
            f"DAG depth: {len(exact_trace):,} (unchanged)",
            f"T-count:   {int(sum(exact_trace))/1e6:.2f}M \u2192 {int(sum(approx_trace))/1e6:.2f}M",
            f"Peak:      {int(np.max(exact_trace))} \u2192 {int(np.max(approx_trace))}",
            f"Mean \u0394max: {int(round(exact_overall['mean_delta_max'])):,} \u2192 {int(round(approx_overall['mean_delta_max'])):,}",
        ]
    )
    ax_cum.text(
        0.98,
        0.04,
        summary_text,
        transform=ax_cum.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.25"},
    )
    return _save_revised_figure(figure, "qft_approximation_reduced_grid_final_v2")


def plot_lower_bound_gap_cases_revised(lower_bound_df: pd.DataFrame, grid_df: pd.DataFrame) -> Path:
    enriched = lower_bound_df.merge(
        grid_df[["family", "seed", "policy", "C", "B", "slack_ratio", "T_static", "peak_demand", "mean_demand"]].drop_duplicates(),
        on=["family", "seed", "policy", "C", "B", "T_static"],
        how="left",
    )
    enriched = enriched[
        (enriched["feasible"] == 1)
        & np.isfinite(enriched["T_exe"])
        & np.isfinite(enriched["predicted_lower_bound"])
        & np.isfinite(enriched["gap"])
        & (enriched["gap"] > 0.0)
    ].copy()
    selected = _select_representative_gap_cases(enriched)
    selected = _reorder_gap_cases_for_diversity(selected, enriched)

    figure = plt.figure(figsize=(8.3, 6.7), constrained_layout=True)
    outer_grid = figure.add_gridspec(2, 2, top=0.92)
    case_axes: list[tuple[plt.Axes, plt.Axes]] = []
    for index in range(min(4, len(selected))):
        row_index, column_index = divmod(index, 2)
        sub_grid = outer_grid[row_index, column_index].subgridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.35)
        case_axes.append(
            (
                figure.add_subplot(sub_grid[0]),
                figure.add_subplot(sub_grid[1]),
            )
        )

    for index, row in enumerate(selected.head(4).itertuples(index=False), start=1):
        case_id = f"case_{index}"
        reconstruction = _reconstruct_gap_case(row)
        _plot_gap_case_row_revised(case_axes[index - 1], reconstruction, case_id)

    legend_handles = [
        Line2D([0], [0], color="#2b6cb0", linewidth=1.25, label="Attempted demand"),
        Line2D([0], [0], color="#2f855a", linewidth=1.1, linestyle="--", label="Served demand"),
        Line2D([0], [0], color="#dd6b20", linewidth=1.25, label="Buffer stock"),
        Patch(facecolor="#e53e3e", edgecolor="none", alpha=0.18, label="Stall cycle"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.04),
        fontsize=7.8,
        frameon=False,
    )
    return _save_revised_figure(figure, "lower_bound_gap_cases_final_v2")


def _build_predictor_slice_distribution_overall(predictive_df: pd.DataFrame, task: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    working_df = predictive_df.copy()
    if task == "slowdown_ratio":
        working_df = working_df[np.isfinite(working_df["slowdown_ratio"])].copy()
    for (C, B), subset in working_df.groupby(["C", "B"], sort=True):
        if len(subset) < 6:
            continue
        for predictor in PREDICTOR_ORDER:
            x = subset[predictor].astype(float).to_numpy()
            if np.allclose(x, x[0]):
                continue
            if task == "slowdown_ratio":
                score_info = _subset_regression_metric(subset, predictor, "slowdown_ratio")
            else:
                labels = subset[task].astype(int).to_numpy()
                if len(np.unique(labels)) < 2:
                    continue
                score_info = _binary_metric_eval(x, labels)
                score_info = {"score": score_info["roc_auc"]}
            rows.append(
                {
                    "C": int(C),
                    "B": int(B),
                    "predictor": predictor,
                    "score": float(score_info["score"]),
                }
            )
    return pd.DataFrame(rows)


def _draw_predictor_boxplot_panel(
    axis: plt.Axes,
    dataframe: pd.DataFrame,
    overlay_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    task: str,
) -> None:
    if dataframe.empty:
        axis.text(0.5, 0.5, "No informative slices", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return

    positions = np.arange(len(PREDICTOR_ORDER))
    data = [dataframe[dataframe["predictor"] == predictor][value_col].astype(float).tolist() for predictor in PREDICTOR_ORDER]
    boxplot = axis.boxplot(
        data,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
        whiskerprops={"color": "#666666", "linewidth": 0.9},
        capprops={"color": "#666666", "linewidth": 0.9},
        boxprops={"linewidth": 0.9, "edgecolor": "#555555"},
    )
    for patch, predictor in zip(boxplot["boxes"], PREDICTOR_ORDER, strict=True):
        patch.set_facecolor(PREDICTOR_COLORS[predictor])
        patch.set_alpha(0.72)
    for position, predictor in zip(positions, PREDICTOR_ORDER, strict=True):
        values = dataframe[dataframe["predictor"] == predictor][value_col].astype(float).tolist()
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
        if task == "slowdown_ratio":
            axis.scatter(
                np.full(len(values), position) + jitter,
                values,
                s=10,
                color=PREDICTOR_COLORS[predictor],
                alpha=0.18,
                edgecolors="none",
                zorder=2,
            )
        overlay_subset = overlay_df[overlay_df["predictor"] == predictor].copy()
        if task == "slowdown_ratio" or overlay_subset.empty:
            continue
        overlay_subset = overlay_subset.sort_values(["family", "C", "B"]).reset_index(drop=True)
        jitter = np.linspace(-0.1, 0.1, len(overlay_subset)) if len(overlay_subset) > 1 else np.array([0.0])
        for offset, row in zip(jitter, overlay_subset.itertuples(index=False), strict=True):
            if row.family == "low_compressibility":
                axis.scatter(
                    position + offset,
                    row.score,
                    s=55,
                    marker="^",
                    facecolors="white",
                    edgecolors=PREDICTOR_COLORS[predictor],
                    alpha=1.0,
                    linewidths=1.4,
                    zorder=4,
                )
            else:
                axis.scatter(
                    position + offset,
                    row.score,
                    s=18,
                    marker="o",
                    facecolors=PREDICTOR_COLORS[predictor],
                    edgecolors="white",
                    alpha=0.8,
                    linewidths=0.25,
                    zorder=4,
                )

    axis.set_xticks(positions, [PREDICTOR_LABELS[predictor] for predictor in PREDICTOR_ORDER])
    axis.set_ylabel(ylabel, fontsize=12)
    axis.set_title(title, fontsize=14, fontweight="bold")
    axis.tick_params(axis="both", labelsize=11)
    _apply_paper_axis_style(axis, horizontal_grid=True)
    axis.tick_params(axis="both", labelsize=11)
    if "AUC" in ylabel:
        axis.set_ylim(0.45, 1.03)
    else:
        axis.set_ylim(-0.02, 1.03)


def _build_incremental_gain_distribution(predictive_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    linear_specs = [
        ("T_depth_only", ["T_depth"]),
        ("T_depth_plus_slack", ["T_depth", "slack_ratio"]),
        ("T_depth_slack_delta", ["T_depth", "slack_ratio", "delta_max"]),
    ]
    logistic_specs = [
        ("T_depth_only", ["T_depth"]),
        ("T_depth_plus_slack", ["T_depth", "slack_ratio"]),
        ("T_depth_slack_delta", ["T_depth", "slack_ratio", "delta_max"]),
    ]
    for (C, B), subset in predictive_df.groupby(["C", "B"], sort=True):
        finite_subset = subset[np.isfinite(subset["slowdown_ratio"])].copy()
        if len(finite_subset) >= 5:
            previous = None
            for model_name, features in linear_specs:
                result = _fit_linear_model(finite_subset, features, "slowdown_ratio")
                if previous is not None:
                    rows.append(
                        {
                            "task": "slowdown_ratio",
                            "step": "+ slack ratio" if model_name == "T_depth_plus_slack" else r"+ $\Delta_{\max}$",
                            "gain": float(result["metric_value"] - previous),
                            "C": int(C),
                            "B": int(B),
                        }
                    )
                previous = float(result["metric_value"])
        if len(subset) >= 5 and subset["stall"].nunique() >= 2:
            previous = None
            for model_name, features in logistic_specs:
                result = _fit_logistic_auc_model(subset, features, "stall")
                if previous is not None:
                    rows.append(
                        {
                            "task": "stall",
                            "step": "+ slack ratio" if model_name == "T_depth_plus_slack" else r"+ $\Delta_{\max}$",
                            "gain": float(result["metric_value"] - previous),
                            "C": int(C),
                            "B": int(B),
                        }
                    )
                previous = float(result["metric_value"])
    return pd.DataFrame(rows)


def _build_predictor_family_overlay_distribution(predictive_df: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "slowdown_ratio":
        return pd.DataFrame(columns=["family", "C", "B", "predictor", "score"])

    rows: list[dict[str, object]] = []
    for (family, C, B), subset in predictive_df.groupby(["family", "C", "B"], sort=True):
        if len(subset) < 4:
            continue
        labels = subset[task].astype(int).to_numpy()
        for predictor in PREDICTOR_ORDER:
            x = subset[predictor].astype(float).to_numpy()
            if len(np.unique(labels)) < 2 or np.allclose(x, x[0]):
                score_value = 0.5
            else:
                score_info = _binary_metric_eval(x, labels)
                score_value = float(score_info["roc_auc"])
            rows.append(
                {
                    "family": family,
                    "C": int(C),
                    "B": int(B),
                    "predictor": predictor,
                    "score": score_value,
                }
            )
    return pd.DataFrame(rows)


def _draw_incremental_gain_panel(axis: plt.Axes, dataframe: pd.DataFrame, ylabel: str, title: str) -> None:
    step_order = ["+ slack ratio", r"+ $\Delta_{\max}$"]
    colors = ["#1f78b4", "#d95f02"]
    data = [dataframe[dataframe["step"] == step]["gain"].astype(float).tolist() for step in step_order]
    positions = np.arange(len(step_order))
    boxplot = axis.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.0},
        whiskerprops={"color": "#666666", "linewidth": 0.9},
        capprops={"color": "#666666", "linewidth": 0.9},
        boxprops={"linewidth": 0.9, "edgecolor": "#555555"},
    )
    for patch, color in zip(boxplot["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for position, values, color in zip(positions, data, colors, strict=True):
        jitter = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else np.array([0.0])
        axis.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=12,
            color=color,
            alpha=0.35,
            edgecolors="none",
            zorder=3,
        )
    axis.axhline(0.0, color="#777777", linewidth=0.9, linestyle="--")
    axis.set_xticks(positions, step_order)
    axis.set_ylabel(ylabel, fontsize=PAPER_LABEL_SIZE)
    axis.set_title(title, fontsize=PAPER_LABEL_SIZE)
    _apply_paper_axis_style(axis, horizontal_grid=True)


def _window_mean(values: np.ndarray, target_points: int) -> np.ndarray:
    if len(values) <= target_points:
        return values.astype(float)
    bin_size = int(math.ceil(len(values) / target_points))
    trimmed_length = int(math.ceil(len(values) / bin_size) * bin_size)
    padded = np.pad(values.astype(float), (0, trimmed_length - len(values)), mode="constant", constant_values=np.nan)
    reshaped = padded.reshape(-1, bin_size)
    return np.nanmean(reshaped, axis=1)


def _representative_prefix_end(trace: list[int], C: int, B: int) -> int:
    cumulative = np.cumsum(np.array(trace, dtype=int))
    supply = C * np.arange(1, len(trace) + 1) + B
    deficit = cumulative - supply
    max_deficit = float(np.max(deficit)) if len(deficit) else 0.0
    if max_deficit <= 0.0:
        return min(len(trace), 100000)
    threshold = 0.65 * max_deficit
    index = int(np.argmax(deficit >= threshold))
    return min(len(trace), max(index + 1, 80000))


def _downsample_series(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    indices = np.linspace(0, len(x) - 1, max_points).astype(int)
    return x[indices], y[indices]


def _extract_peak_window(trace: np.ndarray, half_window: int) -> tuple[np.ndarray, np.ndarray, int]:
    peak_index = int(np.argmax(trace)) if len(trace) else 0
    start = max(0, peak_index - half_window)
    end = min(len(trace), peak_index + half_window + 1)
    values = trace[start:end]
    offsets = np.arange(start, end) - peak_index
    local_peak = int(np.argmax(values)) if len(values) else 0
    return offsets.astype(int), values.astype(int), local_peak


def _plot_qft_step_panel(
    axis: plt.Axes,
    x_offsets: np.ndarray,
    demand_values: np.ndarray,
    capacity: int,
    color: str,
    label: str,
    local_peak_index: int,
) -> None:
    axis.step(x_offsets, demand_values, where="mid", color=color, linewidth=0.9)
    axis.fill_between(x_offsets, 0, demand_values, step="mid", color=color, alpha=0.5)
    axis.axhline(capacity, color="#777777", linestyle="--", linewidth=1.0)
    axis.fill_between(
        x_offsets,
        capacity,
        demand_values,
        where=demand_values > capacity,
        step="mid",
        color="#e53e3e",
        alpha=0.35,
    )
    peak_x = int(x_offsets[local_peak_index]) if len(x_offsets) else 0
    peak_y = int(demand_values[local_peak_index]) if len(demand_values) else 0
    axis.annotate(
        f"peak = {peak_y}",
        xy=(peak_x, peak_y),
        xytext=(8, 2),
        textcoords="offset points",
        fontsize=8,
        color=color,
        fontweight="bold",
    )
    axis.text(
        0.02,
        0.92,
        label,
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        fontweight="bold",
        color=color,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5},
    )
    _apply_paper_axis_style(axis)


def _reorder_gap_cases_for_diversity(selected: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected
    selected = selected.copy().reset_index(drop=True)
    medium_rows = selected[selected["family"] == "medium_compressibility"]
    if not medium_rows.empty:
        medium_row = medium_rows.iloc[[0]]
        remainder = selected.drop(index=medium_rows.index[0]).reset_index(drop=True)
        return pd.concat([remainder, medium_row], ignore_index=True)

    medium_pool = enriched[enriched["family"] == "medium_compressibility"].copy()
    if medium_pool.empty:
        return selected
    replacement = medium_pool.assign(distance=(medium_pool["gap"] - selected["gap"].median()).abs()).sort_values(
        ["distance", "gap", "seed", "policy", "C", "B"]
    ).iloc[0]
    selected.loc[selected.index[-1]] = replacement
    return selected


def _plot_gap_case_row_revised(axes_row: tuple[plt.Axes, plt.Axes], reconstruction: dict[str, object], case_id: str) -> None:
    left_axis, right_axis = axes_row
    demand = np.array(reconstruction["demand"], dtype=int)
    cumulative_demand = np.array(reconstruction["cumulative_demand"], dtype=int)
    supply_envelope = np.array(reconstruction["supply_envelope"], dtype=int)
    backlog = np.array(reconstruction["backlog"], dtype=int)
    peak_index = int(reconstruction["peak_index"])
    active_segments = list(reconstruction["active_segments"])
    row = reconstruction["row"]
    sim_result = reconstruction["sim_result"]

    logical_x = np.arange(len(demand))
    left_axis.plot(logical_x, cumulative_demand, color="#1f77b4", linewidth=1.5, label="Cumulative demand")
    left_axis.plot(logical_x, supply_envelope, color="#444444", linewidth=1.1, linestyle="--", label="Supply envelope")
    left_axis.fill_between(logical_x, supply_envelope, cumulative_demand, where=backlog > 0, color="#e45756", alpha=0.18)
    left_axis.axvline(peak_index, color="#777777", linewidth=0.9, linestyle=":")
    left_axis.set_ylabel("Cumulative T", fontsize=8)
    left_axis.set_xlabel("Logical cycle", fontsize=8)
    _apply_paper_axis_style(left_axis)
    left_axis.set_title(
        f"{_display_case_id(case_id)}: {_display_family(row.family)}, {_display_policy(row.policy)}, "
        f"gap={float(row.gap):.1f}, Δ={int(row.delta_max)}",
        fontsize=8,
        loc="left",
        pad=2.0,
    )

    real_x = np.arange(1, sim_result.T_exe + 1)
    attempted_demand = np.array([demand[idx] for idx in sim_result.logical_cycle_history], dtype=int)
    served_demand = np.array(sim_result.served_history, dtype=int)
    buffer_stock = np.array(sim_result.buffer_history, dtype=int)
    stall_mask = (attempted_demand > 0) & (served_demand == 0)

    right_axis.step(real_x, attempted_demand, where="post", color="#2b6cb0", linewidth=1.25, label="Attempted demand")
    right_axis.step(real_x, served_demand, where="post", color="#2f855a", linewidth=1.1, linestyle="--", label="Served demand")
    for cycle in real_x[stall_mask]:
        right_axis.axvspan(cycle - 0.5, cycle + 0.5, color="#e45756", alpha=0.12)
    right_axis.set_ylabel("T demand", fontsize=8, color="#222222")
    right_axis.set_xlabel("Real cycle", fontsize=8)
    _apply_paper_axis_style(right_axis)

    buffer_axis = right_axis.twinx()
    buffer_axis.step(real_x, buffer_stock, where="post", color="#dd6b20", linewidth=1.25, label="Buffer stock")
    buffer_axis.axhline(int(row.B), color="#dd6b20", linewidth=0.9, linestyle=":")
    buffer_axis.set_ylim(0, max(1, int(row.B)))
    buffer_axis.set_ylabel("Buffer stock", fontsize=8, color="#dd6b20")
    buffer_axis.tick_params(axis="y", labelsize=8, colors="#dd6b20")
    for spine in buffer_axis.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")


def _save_revised_figure(figure: plt.Figure, stem: str) -> Path:
    FINAL_PAPER_REVISED_DIR.mkdir(parents=True, exist_ok=True)
    path = FINAL_PAPER_REVISED_DIR / f"{stem}.png"
    figure.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    figure.savefig(FINAL_PAPER_REVISED_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)
    return path


if __name__ == "__main__":
    main()
