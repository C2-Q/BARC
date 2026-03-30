from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.paper_config import COMPRESSIBILITY_SCAN_B, COMPRESSIBILITY_SCAN_C
from src.utils import APPENDIX_DIR, FIGURE_DIR, FINAL_PAPER_DIR, TABLE_DIR, ensure_output_dirs


POLICY_COLORS = {
    "static_min": "#1f77b4",
    "capacity_aware_static": "#ff7f0e",
    "smoothed": "#d62728",
}

FAMILY_COLORS = {
    "high_compressibility": "#1b9e77",
    "medium_compressibility": "#d95f02",
    "low_compressibility": "#7570b3",
}

FAMILY_MARKERS = {
    "high_compressibility": "o",
    "medium_compressibility": "s",
    "low_compressibility": "^",
}

PAPER_LABEL_SIZE = 10.5
PAPER_TICK_SIZE = 9
PAPER_TITLE_SIZE = 10
PAPER_LEGEND_SIZE = 8
PAPER_BAR_COLORS = {
    "T_depth": "#7a7a7a",
    "slack_ratio": "#1f78b4",
    "delta_max": "#d95f02",
}


def generate_all_plots(results: dict[str, pd.DataFrame]) -> list[Path]:
    ensure_output_dirs()
    paths = [
        plot_policy_comparison(results["compare"]),
        plot_buffer_transition(results["buffer"]),
        plot_capacity_scan(results["capacity"]),
        plot_compressibility_vs_improvement(results["grid_pairs"]),
        plot_qtv_tradeoff(results["grid"]),
        plot_family_boxplot(results["compressibility_summary"]),
        plot_predictor_comparison(
            results["predictive_classification"],
            results["predictive_regression"],
        ),
        plot_incremental_predictive_gain(results["incremental_predictive_models"]),
        plot_predictor_stability_heatmap(results["predictor_stability_summary"]),
        plot_slack_vs_slowdown(results["predictive_dataset"]),
        plot_t_depth_vs_slowdown(results["predictive_dataset"]),
        plot_delta_max_vs_slowdown(results["predictive_dataset"]),
        plot_slack_vs_stall(results["predictive_dataset"]),
        plot_structure_to_execution_chain(results["causal_chain_summary"]),
        plot_structure_to_execution_chain_empirical(results["causal_chain_summary"]),
        plot_lower_bound_vs_actual(results["lower_bound_validation"]),
    ]
    return paths


def generate_final_paper_figures_from_tables() -> list[Path]:
    ensure_output_dirs()
    classification_df = pd.read_csv(TABLE_DIR / "predictive_classification_summary.csv")
    regression_df = pd.read_csv(TABLE_DIR / "predictive_regression_summary.csv")
    causal_chain_df = pd.read_csv(TABLE_DIR / "causal_chain_summary.csv")
    predictive_df = pd.read_csv(TABLE_DIR / "predictive_static_dataset.csv")
    lower_bound_df = pd.read_csv(TABLE_DIR / "lower_bound_validation.csv")
    real_trace_scaling_df = pd.read_csv(TABLE_DIR / "real_trace_scaling_summary.csv")
    incremental_df = pd.read_csv(TABLE_DIR / "incremental_predictive_models.csv")
    stability_df = pd.read_csv(TABLE_DIR / "predictor_stability_summary.csv")
    return [
        plot_predictor_comparison_final(classification_df, regression_df),
        plot_incremental_predictive_gain_final(incremental_df),
        plot_predictor_stability_heatmap_final(stability_df),
        plot_structure_to_execution_chain_empirical_final(causal_chain_df),
        plot_delta_max_vs_slowdown_final(predictive_df),
        plot_lower_bound_vs_actual_final(lower_bound_df),
        plot_real_trace_scaling_final(real_trace_scaling_df),
    ]


def plot_predictor_comparison(
    classification_df: pd.DataFrame,
    regression_df: pd.DataFrame,
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), constrained_layout=True)
    metrics = [("T_depth", "T_depth"), ("slack_ratio", "slack_ratio"), ("Delta_max", "delta_max")]
    colors = ["#7f7f7f", "#1f77b4", "#d62728"]
    task_specs = [
        ("stall", "mean_roc_auc", "AUC", "Stall Classification"),
        ("inversion", "mean_roc_auc", "AUC", "Inversion Classification"),
        ("slowdown_ratio", "mean_abs_spearman", "|Spearman|", "Slowdown Regression"),
    ]
    for axis, (task, value_col, ylabel, title) in zip(axes, task_specs, strict=True):
        source = regression_df if task == "slowdown_ratio" else classification_df
        subset = source[source["task"] == task].set_index("metric")
        values = [float(subset.loc[metric_key, value_col]) for metric_key, _ in metrics]
        axis.bar([label for _, label in metrics], values, color=colors, width=0.62)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
    primary_path = FIGURE_DIR / "predictor_comparison.png"
    figure.savefig(primary_path, dpi=180)
    plt.close(figure)
    return primary_path


def plot_incremental_predictive_gain(incremental_df: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 4.1), constrained_layout=True)
    _plot_incremental_panel(
        axes[0],
        incremental_df[incremental_df["task"] == "slowdown_ratio"],
        "r_squared",
        r"$R^2$",
        "Slowdown Incremental Fit",
    )
    _plot_incremental_panel(
        axes[1],
        incremental_df[incremental_df["task"] == "stall"],
        "auc",
        "AUC",
        "Stall Incremental Fit",
    )
    path = FIGURE_DIR / "incremental_predictive_gain.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_predictor_stability_heatmap(stability_df: pd.DataFrame) -> Path:
    stall_path = plot_predictor_stability_stall(stability_df)
    plot_predictor_stability_slowdown(stability_df)
    return stall_path


def plot_policy_comparison(compare_df: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    policies = compare_df["policy"].tolist()
    x_positions = np.arange(len(policies))
    width = 0.32
    axis.bar(
        x_positions - width / 2,
        compare_df["T_static"],
        width=width,
        color="#7f7f7f",
        label="T_static",
    )
    axis.bar(
        x_positions + width / 2,
        compare_df["T_exe"],
        width=width,
        color="#2ca02c",
        label="T_exe",
    )
    axis.set_xticks(x_positions, policies)
    axis.set_ylabel("Cycles")
    axis.set_title("Representative Same-DAG Comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = APPENDIX_DIR / "policy_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_buffer_transition(buffer_df: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(2, 1, figsize=(8, 8.4), sharex=True)
    _plot_buffer_metric(buffer_df, axes[0], metric="stall_cycles", ylabel="stall_cycles")
    axes[0].set_title("Buffer Transition: Stall Cycles")
    _plot_buffer_metric(
        buffer_df,
        axes[1],
        metric="normalized_makespan",
        ylabel="T_exe / T_static",
    )
    axes[1].set_title("Buffer Transition: Normalized Makespan")
    figure.tight_layout()
    path = FIGURE_DIR / "buffer_transition.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_capacity_scan(capacity_df: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for policy in ("static_min", "capacity_aware_static", "smoothed"):
        subset = capacity_df[(capacity_df["policy"] == policy) & np.isfinite(capacity_df["normalized_makespan"])].sort_values("C")
        axis.plot(
            subset["C"],
            subset["normalized_makespan"],
            marker="o",
            color=POLICY_COLORS[policy],
            label=policy,
        )
    axis.set_xlabel("Delivery capacity C")
    axis.set_ylabel("T_exe / T_static")
    axis.set_title("Capacity Scan")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = FIGURE_DIR / "capacity_scan.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_compressibility_vs_improvement(grid_pair_df: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    scatter = None
    comparison_styles = {
        "static_min_vs_smoothed": ("o", "relative_T_exe_improvement_static_min_vs_smoothed"),
        "capacity_aware_vs_smoothed": ("s", "relative_T_exe_improvement_capacity_aware_vs_smoothed"),
    }
    color_values = grid_pair_df["compressibility_mean_t_slack"].clip(lower=0.0)
    for family in FAMILY_COLORS:
        family_subset = grid_pair_df[grid_pair_df["family"] == family]
        for comparison_name, (marker, metric_column) in comparison_styles.items():
            subset = family_subset[family_subset[f"both_feasible_{comparison_name}"] == 1]
            if subset.empty:
                continue
            label = f"{family} {comparison_name}"
            axis.scatter(
                subset["compressibility_slack_ratio"],
                subset[metric_column],
                s=58,
                marker=marker,
                c=color_values.loc[subset.index],
                cmap="viridis",
                edgecolors=FAMILY_COLORS[family],
                linewidths=0.9,
                alpha=0.8,
                label=label,
            )
            if scatter is None:
                scatter = axis.collections[-1]
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axis.set_xlabel("compressibility_slack_ratio")
    axis.set_ylabel("relative improvement in T_exe")
    axis.set_title("Slack-Based Compressibility vs. Improvement")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)
    if scatter is not None:
        colorbar = figure.colorbar(scatter, ax=axis)
        colorbar.set_label("mean T-node slack")
    figure.tight_layout()
    path = FIGURE_DIR / "compressibility_vs_improvement.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_qtv_tradeoff(grid_df: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    feasible_df = grid_df[np.isfinite(grid_df["normalized_makespan"]) & np.isfinite(grid_df["QTV_proxy"])].copy()
    for family in FAMILY_MARKERS:
        for policy in POLICY_COLORS:
            subset = feasible_df[(feasible_df["family"] == family) & (feasible_df["policy"] == policy)]
            axis.scatter(
                subset["normalized_makespan"],
                subset["QTV_proxy"],
                s=28,
                marker=FAMILY_MARKERS[family],
                color=POLICY_COLORS[policy],
                alpha=0.55,
            )

    policy_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=POLICY_COLORS[policy], label=policy)
        for policy in POLICY_COLORS
    ]
    family_handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[family],
            linestyle="",
            color="#444444",
            label=family,
        )
        for family in FAMILY_MARKERS
    ]
    first_legend = axis.legend(handles=policy_handles, title="policy", loc="upper left")
    axis.add_artist(first_legend)
    axis.legend(handles=family_handles, title="family", loc="upper right")
    axis.set_xlabel("T_exe / T_static")
    axis.set_ylabel("QTV_proxy")
    axis.set_title("Policy Pareto Tradeoff")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = APPENDIX_DIR / "qtv_tradeoff.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_family_boxplot(compressibility_summary_df: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    seed_level = compressibility_summary_df[compressibility_summary_df["summary_level"] == "family_seed"].copy()
    families = ["high_compressibility", "medium_compressibility", "low_compressibility"]
    static_data = [
        seed_level[seed_level["family"] == family]["mean_relative_improvement_static_min_vs_smoothed"].tolist()
        for family in families
    ]
    strong_data = [
        seed_level[seed_level["family"] == family]["mean_relative_improvement_capacity_aware_vs_smoothed"].tolist()
        for family in families
    ]
    positions_static = np.arange(1, len(families) + 1) - 0.18
    positions_strong = np.arange(1, len(families) + 1) + 0.18
    box_static = axis.boxplot(static_data, positions=positions_static, widths=0.28, patch_artist=True)
    box_strong = axis.boxplot(strong_data, positions=positions_strong, widths=0.28, patch_artist=True)
    for patch, family in zip(box_static["boxes"], families):
        patch.set_facecolor(FAMILY_COLORS[family])
        patch.set_alpha(0.55)
    for patch in box_strong["boxes"]:
        patch.set_facecolor("#cccccc")
        patch.set_alpha(0.75)
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axis.set_xticks(np.arange(1, len(families) + 1), families)
    axis.set_ylabel("mean relative improvement in T_exe")
    axis.set_title("Family-Level Gain Distribution Across Seeds")
    axis.grid(axis="y", alpha=0.25)
    legend_handles = [
        Line2D([0], [0], color="#666666", linewidth=8, label="static_min vs smoothed"),
        Line2D([0], [0], color="#cccccc", linewidth=8, label="capacity_aware_static vs smoothed"),
    ]
    axis.legend(handles=legend_handles)
    figure.tight_layout()
    path = APPENDIX_DIR / "family_boxplot.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_slack_vs_slowdown(predictive_df: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    subset = _representative_predictive_slice(predictive_df, finite_only=True)
    _scatter_by_family(axis, subset, x_col="slack_ratio", y_col="slowdown_ratio")
    _annotate_correlation(axis, subset["slack_ratio"], subset["slowdown_ratio"])
    axis.set_xlabel("slack_ratio")
    axis.set_ylabel("slowdown_ratio")
    axis.set_title(f"Slack vs Slowdown (C={COMPRESSIBILITY_SCAN_C}, B={COMPRESSIBILITY_SCAN_B})")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = FIGURE_DIR / "slack_vs_slowdown.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_t_depth_vs_slowdown(predictive_df: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    subset = _representative_predictive_slice(predictive_df, finite_only=True)
    _scatter_by_family(axis, subset, x_col="T_depth", y_col="slowdown_ratio")
    _annotate_correlation(axis, subset["T_depth"], subset["slowdown_ratio"])
    axis.set_xlabel("T_depth")
    axis.set_ylabel("slowdown_ratio")
    axis.set_title(f"T-depth vs Slowdown (C={COMPRESSIBILITY_SCAN_C}, B={COMPRESSIBILITY_SCAN_B})")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = FIGURE_DIR / "t_depth_vs_slowdown.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_delta_max_vs_slowdown(predictive_df: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    subset = _representative_predictive_slice(predictive_df, finite_only=True)
    _scatter_by_family(axis, subset, x_col="Delta_max", y_col="slowdown_ratio")
    _annotate_correlation(axis, subset["Delta_max"], subset["slowdown_ratio"])
    axis.set_xlabel("Delta_max")
    axis.set_ylabel("slowdown_ratio")
    axis.set_title(f"Delta_max vs Slowdown (C={COMPRESSIBILITY_SCAN_C}, B={COMPRESSIBILITY_SCAN_B})")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    path = FIGURE_DIR / "delta_max_vs_slowdown.png"
    figure.savefig(path, dpi=180)
    figure.savefig(FIGURE_DIR / "delta_vs_slowdown.png", dpi=180)
    plt.close(figure)
    return path


def plot_slack_vs_stall(predictive_df: pd.DataFrame) -> Path:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    subset = _representative_predictive_slice(predictive_df, finite_only=False).copy()
    jitter = np.linspace(-0.03, 0.03, max(1, len(subset)))
    for family in FAMILY_COLORS:
        family_subset = subset[subset["family"] == family].copy()
        if family_subset.empty:
            continue
        axis.scatter(
            family_subset["slack_ratio"],
            family_subset["stall"] + jitter[: len(family_subset)],
            marker=FAMILY_MARKERS[family],
            s=55,
            color=FAMILY_COLORS[family],
            alpha=0.75,
            label=family,
        )
    bin_summary = subset.groupby("slack_ratio", dropna=False)["stall"].mean().reset_index().sort_values("slack_ratio")
    axis.plot(
        bin_summary["slack_ratio"],
        bin_summary["stall"],
        color="#222222",
        linewidth=1.4,
        linestyle="--",
        label="stall rate",
    )
    axis.set_xlabel("slack_ratio")
    axis.set_ylabel("stall (0/1)")
    axis.set_title(f"Slack vs Stall (C={COMPRESSIBILITY_SCAN_C}, B={COMPRESSIBILITY_SCAN_B})")
    axis.set_yticks([0, 1])
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = FIGURE_DIR / "slack_vs_stall.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def plot_structure_to_execution_chain(causal_chain_df: pd.DataFrame) -> Path:
    subset = causal_chain_df[
        (causal_chain_df["policy"] == "static_min")
        & (causal_chain_df["C"] == COMPRESSIBILITY_SCAN_C)
        & (causal_chain_df["B"] == COMPRESSIBILITY_SCAN_B)
        & np.isfinite(causal_chain_df["slowdown_ratio"])
    ].copy()

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    panels = [
        ("slack_ratio", "Delta_max", "slack_ratio", "Delta_max", "Structure to Burst"),
        ("Delta_max", "stall_observed", "Delta_max", "stall (0/1)", "Burst to Stall"),
        ("stall_observed", "slowdown_ratio", "stall (0/1)", "slowdown_ratio", "Stall to Slowdown"),
    ]
    for axis, (x_col, y_col, xlabel, ylabel, title) in zip(axes, panels, strict=True):
        if y_col == "stall_observed":
            family_jitter = {
                "high_compressibility": -0.035,
                "medium_compressibility": 0.0,
                "low_compressibility": 0.035,
            }
        else:
            family_jitter = {family: 0.0 for family in FAMILY_COLORS}
        for family in FAMILY_COLORS:
            family_subset = subset[subset["family"] == family].copy()
            if family_subset.empty:
                continue
            y_values = family_subset[y_col].astype(float).to_numpy() + family_jitter.get(family, 0.0)
            axis.scatter(
                family_subset[x_col],
                y_values,
                s=55,
                marker=FAMILY_MARKERS[family],
                color=FAMILY_COLORS[family],
                alpha=0.8,
                label=family,
            )
        _annotate_correlation(axis, subset[x_col], subset[y_col])
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if y_col == "stall_observed":
            axis.set_yticks([0, 1])
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
    path = FIGURE_DIR / "structure_to_execution_chain.png"
    figure.savefig(path, dpi=180)
    figure.savefig(FIGURE_DIR / "causal_chain.png", dpi=180)
    plt.close(figure)
    return path


def plot_structure_to_execution_chain_empirical(causal_chain_df: pd.DataFrame) -> Path:
    delta_column = "delta_max" if "delta_max" in causal_chain_df.columns else "Delta_max"
    subset = causal_chain_df[
        (causal_chain_df["policy"] == "static_min")
        & (causal_chain_df["C"] == COMPRESSIBILITY_SCAN_C)
        & (causal_chain_df["B"] == COMPRESSIBILITY_SCAN_B)
        & np.isfinite(causal_chain_df["slowdown_ratio"])
    ].copy()

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    panel_specs = [
        ("slack_ratio", delta_column, "Slack ratio", r"$\Delta_{\max}$"),
        (delta_column, "slowdown_ratio", r"$\Delta_{\max}$", "Slowdown ratio"),
    ]

    delta_upper = float(subset[delta_column].max()) if not subset.empty else 1.0
    slowdown_upper = float(subset["slowdown_ratio"].max()) if not subset.empty else 1.2

    for axis, (x_col, y_col, xlabel, ylabel) in zip(axes, panel_specs, strict=True):
        for family in FAMILY_COLORS:
            family_subset = subset[subset["family"] == family]
            if family_subset.empty:
                continue
            axis.scatter(
                family_subset[x_col],
                family_subset[y_col],
                s=28,
                marker=FAMILY_MARKERS[family],
                color=FAMILY_COLORS[family],
                alpha=0.82,
                edgecolors="none",
                label=family,
            )
        _add_trend_line(axis, subset[x_col], subset[y_col])
        _annotate_correlation_compact(axis, subset[x_col], subset[y_col])
        axis.set_xlabel(xlabel, fontsize=13)
        axis.set_ylabel(ylabel, fontsize=13)
        axis.tick_params(labelsize=10)
        axis.grid(alpha=0.15, linewidth=0.8)

    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_ylim(-0.3, delta_upper * 1.08 if delta_upper > 0 else 1.0)
    axes[1].set_xlim(-0.3, delta_upper * 1.08 if delta_upper > 0 else 1.0)
    axes[1].set_ylim(0.95, slowdown_upper * 1.06 if slowdown_upper > 1.0 else 1.1)

    handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[family],
            linestyle="",
            color=FAMILY_COLORS[family],
            label=family,
            markersize=5.2,
        )
        for family in FAMILY_COLORS
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.4,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.subplots_adjust(bottom=0.28, left=0.09, right=0.98, top=0.98, wspace=0.22)

    path = FIGURE_DIR / "structure_to_execution_chain_empirical.png"
    figure.savefig(path, dpi=300)
    figure.savefig(FIGURE_DIR / "structure_to_execution_chain_empirical.pdf")
    plt.close(figure)
    return path


def plot_lower_bound_vs_actual(lower_bound_df: pd.DataFrame) -> Path:
    subset = lower_bound_df[np.isfinite(lower_bound_df["T_exe"])].copy()
    figure, axis = plt.subplots(figsize=(8, 5.2))
    for family in FAMILY_COLORS:
        family_subset = subset[subset["family"] == family]
        if family_subset.empty:
            continue
        axis.scatter(
            family_subset["predicted_lower_bound"],
            family_subset["T_exe"],
            s=34,
            marker=FAMILY_MARKERS[family],
            color=FAMILY_COLORS[family],
            alpha=0.7,
            label=family,
        )
    if not subset.empty:
        lower = float(min(subset["predicted_lower_bound"].min(), subset["T_exe"].min()))
        upper = float(max(subset["predicted_lower_bound"].max(), subset["T_exe"].max()))
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="#222222", linewidth=1.2, label="y = x")
        coverage = float((subset["T_exe"] + 1e-9 >= subset["predicted_lower_bound"]).mean())
        mean_gap = float((subset["T_exe"] - subset["predicted_lower_bound"]).mean())
        axis.text(
            0.03,
            0.97,
            f"coverage={coverage:.3f}\nmean gap={mean_gap:.2f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
        )
    axis.set_xlabel("Predicted lower bound")
    axis.set_ylabel("Actual T_exe")
    axis.set_title("Lower Bound Tightness")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    path = FIGURE_DIR / "lower_bound_vs_actual.png"
    figure.savefig(path, dpi=180)
    figure.savefig(FIGURE_DIR / "lower_bound.png", dpi=180)
    plt.close(figure)
    return path


def plot_predictor_comparison_final(
    classification_df: pd.DataFrame,
    regression_df: pd.DataFrame,
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.7))
    metrics = [("T_depth", "T-depth"), ("slack_ratio", "Slack ratio"), ("delta_max", r"$\Delta_{\max}$")]
    task_specs = [
        ("stall", "mean_roc_auc", "AUC", "Stall Classification"),
        ("inversion", "mean_roc_auc", "AUC", "Inversion Classification"),
        ("slowdown_ratio", "mean_abs_spearman", r"$|\rho|$", "Slowdown Regression"),
    ]
    for axis, (task, value_col, ylabel, title) in zip(axes, task_specs, strict=True):
        source = regression_df if task == "slowdown_ratio" else classification_df
        subset = source[source["task"] == task].set_index("metric")
        values = [float(subset.loc[metric_key, value_col]) for metric_key, _ in metrics]
        labels = [label for _, label in metrics]
        colors = [PAPER_BAR_COLORS[metric_key] for metric_key, _ in metrics]
        bars = axis.bar(labels, values, color=colors, width=0.62, edgecolor="#4d4d4d", linewidth=0.4)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )
        axis.set_title(title, fontsize=PAPER_TITLE_SIZE, pad=5)
        axis.set_ylabel(ylabel, fontsize=PAPER_LABEL_SIZE)
        axis.set_ylim(0.0, _predictor_panel_upper_bound(values))
        _apply_paper_axis_style(axis, horizontal_grid=True)
        axis.tick_params(axis="x", labelsize=PAPER_TICK_SIZE, rotation=0)
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.86, wspace=0.32)
    return _save_final_figure_bundle(figure, "predictor_comparison_final")


def plot_incremental_predictive_gain_final(incremental_df: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.85))
    _plot_incremental_panel(
        axes[0],
        incremental_df[incremental_df["task"] == "slowdown_ratio"],
        "r_squared",
        r"$R^2$",
        "",
        paper_style=True,
    )
    _plot_incremental_panel(
        axes[1],
        incremental_df[incremental_df["task"] == "stall"],
        "auc",
        "AUC",
        "",
        paper_style=True,
    )
    axes[0].set_title("Slowdown", fontsize=PAPER_TITLE_SIZE, pad=4)
    axes[1].set_title("Stall", fontsize=PAPER_TITLE_SIZE, pad=4)
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.24, top=0.88, wspace=0.28)
    return _save_final_figure_bundle(figure, "incremental_predictive_gain_final")


def plot_predictor_stability_heatmap_final(stability_df: pd.DataFrame) -> Path:
    stall_path = plot_predictor_stability_stall_final(stability_df)
    plot_predictor_stability_slowdown_final(stability_df)
    return stall_path


def plot_structure_to_execution_chain_empirical_final(causal_chain_df: pd.DataFrame) -> Path:
    delta_column = "delta_max" if "delta_max" in causal_chain_df.columns else "Delta_max"
    subset = causal_chain_df[
        (causal_chain_df["policy"] == "static_min")
        & (causal_chain_df["C"] == COMPRESSIBILITY_SCAN_C)
        & (causal_chain_df["B"] == COMPRESSIBILITY_SCAN_B)
        & np.isfinite(causal_chain_df["slowdown_ratio"])
    ].copy()

    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    panel_specs = [
        ("slack_ratio", delta_column, "Slack ratio", r"$\Delta_{\max}$"),
        (delta_column, "slowdown_ratio", r"$\Delta_{\max}$", "Slowdown ratio"),
    ]
    delta_upper = float(subset[delta_column].max()) if not subset.empty else 1.0
    slowdown_upper = float(subset["slowdown_ratio"].max()) if not subset.empty else 1.1

    for axis, (x_col, y_col, xlabel, ylabel) in zip(axes, panel_specs, strict=True):
        for family in FAMILY_COLORS:
            family_subset = subset[subset["family"] == family]
            if family_subset.empty:
                continue
            axis.scatter(
                family_subset[x_col],
                family_subset[y_col],
                s=22,
                marker=FAMILY_MARKERS[family],
                color=FAMILY_COLORS[family],
                alpha=0.85,
                edgecolors="white",
                linewidths=0.25,
                zorder=3,
            )
        _add_trend_line(axis, subset[x_col], subset[y_col])
        _annotate_correlation_compact(axis, subset[x_col], subset[y_col])
        axis.set_xlabel(xlabel, fontsize=PAPER_LABEL_SIZE)
        axis.set_ylabel(ylabel, fontsize=PAPER_LABEL_SIZE)
        _apply_paper_axis_style(axis)

    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_ylim(-0.15, delta_upper * 1.08 if delta_upper > 0 else 1.0)
    axes[1].set_xlim(-0.2, delta_upper * 1.08 if delta_upper > 0 else 1.0)
    axes[1].set_ylim(0.97, slowdown_upper * 1.05 if slowdown_upper > 1.0 else 1.08)

    handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[family],
            linestyle="",
            color=FAMILY_COLORS[family],
            markerfacecolor=FAMILY_COLORS[family],
            label=_family_display_name(family),
            markersize=5.0,
        )
        for family in FAMILY_COLORS
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=PAPER_LEGEND_SIZE,
        handletextpad=0.4,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.subplots_adjust(left=0.1, right=0.99, bottom=0.28, top=0.96, wspace=0.25)
    return _save_final_figure_bundle(figure, "structure_to_execution_chain_empirical_final")


def plot_delta_max_vs_slowdown_final(predictive_df: pd.DataFrame) -> Path:
    subset = _representative_predictive_slice(predictive_df, finite_only=True).copy()
    delta_column = "delta_max" if "delta_max" in subset.columns else "Delta_max"
    figure, axis = plt.subplots(figsize=(3.45, 2.8))
    for family in FAMILY_COLORS:
        family_subset = subset[subset["family"] == family]
        if family_subset.empty:
            continue
        axis.scatter(
            family_subset[delta_column],
            family_subset["slowdown_ratio"],
            s=23,
            marker=FAMILY_MARKERS[family],
            color=FAMILY_COLORS[family],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
            label=_family_display_name(family),
        )
    _add_trend_line(axis, subset[delta_column], subset["slowdown_ratio"])
    _annotate_correlation_compact(axis, subset[delta_column], subset["slowdown_ratio"])
    axis.set_xlabel(r"$\Delta_{\max}$", fontsize=PAPER_LABEL_SIZE)
    axis.set_ylabel("Slowdown ratio", fontsize=PAPER_LABEL_SIZE)
    _apply_paper_axis_style(axis)
    axis.legend(loc="lower right", frameon=False, fontsize=PAPER_LEGEND_SIZE, handletextpad=0.35)
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.97)
    return _save_final_figure_bundle(figure, "delta_max_vs_slowdown_final")


def plot_lower_bound_vs_actual_final(lower_bound_df: pd.DataFrame) -> Path:
    subset = lower_bound_df[np.isfinite(lower_bound_df["T_exe"])].copy()
    figure, axis = plt.subplots(figsize=(3.45, 2.8))
    scatter = axis.scatter(
        subset["predicted_lower_bound"],
        subset["T_exe"],
        s=18,
        c=subset["gap"],
        cmap="cividis",
        alpha=0.72,
        edgecolors="none",
        rasterized=True,
    )
    if not subset.empty:
        lower = float(min(subset["predicted_lower_bound"].min(), subset["T_exe"].min()))
        upper = float(max(subset["predicted_lower_bound"].max(), subset["T_exe"].max()))
        axis.plot([lower, upper], [lower, upper], linestyle="-", color="#444444", linewidth=1.1, zorder=2)
        coverage = float((subset["T_exe"] + 1e-9 >= subset["predicted_lower_bound"]).mean())
        mean_gap = float((subset["T_exe"] - subset["predicted_lower_bound"]).mean())
        axis.text(
            0.04,
            0.96,
            f"Coverage={coverage:.3f}\nMean gap={mean_gap:.2f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=7.2,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.18"},
        )
    axis.set_xlabel("Predicted lower bound", fontsize=PAPER_LABEL_SIZE)
    axis.set_ylabel(r"Observed $T_{\mathrm{exe}}$", fontsize=PAPER_LABEL_SIZE)
    _apply_paper_axis_style(axis)
    figure.subplots_adjust(left=0.18, right=0.98, bottom=0.2, top=0.97)
    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.055, pad=0.03)
    colorbar.set_label("Gap (cycles)", fontsize=PAPER_LABEL_SIZE - 0.5)
    colorbar.ax.tick_params(labelsize=PAPER_TICK_SIZE - 0.5)
    return _save_final_figure_bundle(figure, "lower_bound_vs_actual_final")


def plot_real_trace_scaling_final(real_trace_scaling_df: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    trace_styles = {
        "adder": {"color": "#4c78a8", "marker": "o", "label": "Adder"},
        "multiplier": {"color": "#e45756", "marker": "s", "label": "Multiplier"},
    }
    for trace_family, style in trace_styles.items():
        subset = real_trace_scaling_df[real_trace_scaling_df["trace_family"] == trace_family].sort_values("n_bits")
        if subset.empty:
            continue
        axes[0].plot(
            subset["n_bits"],
            subset["slack_ratio"],
            marker=style["marker"],
            markersize=4.5,
            color=style["color"],
            linewidth=1.4,
            label=style["label"],
        )
        axes[1].plot(
            subset["n_bits"],
            subset["delta_max_mean"],
            marker=style["marker"],
            markersize=4.5,
            color=style["color"],
            linewidth=1.4,
            label=style["label"],
        )
    axes[0].set_xlabel("Bit width", fontsize=PAPER_LABEL_SIZE)
    axes[0].set_ylabel("Slack ratio", fontsize=PAPER_LABEL_SIZE)
    axes[1].set_xlabel("Bit width", fontsize=PAPER_LABEL_SIZE)
    axes[1].set_ylabel(r"Mean $\Delta_{\max}$", fontsize=PAPER_LABEL_SIZE)
    for axis in axes:
        _apply_paper_axis_style(axis)
    axes[0].legend(loc="lower right", frameon=False, fontsize=PAPER_LEGEND_SIZE, handletextpad=0.35)
    figure.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.96, wspace=0.3)
    return _save_final_figure_bundle(figure, "real_trace_scaling_final")


def _build_predictor_stability_heatmap(stability_df: pd.DataFrame, paper_style: bool) -> plt.Figure:
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.6) if paper_style else (9.2, 7.0), sharex=False)
    _draw_predictor_stability_panel(axes[0], stability_df, "stall_auc", "Stall Prediction (AUC)", "stall", paper_style)
    _draw_predictor_stability_panel(
        axes[1],
        stability_df,
        "slowdown_abs_spearman",
        r"Slowdown Prediction (Spearman $|\rho|$)",
        "slowdown_ratio",
        paper_style,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=PAPER_LEGEND_SIZE if paper_style else 9)
    if paper_style:
        figure.subplots_adjust(left=0.24, right=0.98, bottom=0.08, top=0.85, hspace=0.42)
    else:
        figure.subplots_adjust(left=0.26, right=0.98, bottom=0.07, top=0.87, hspace=0.42)
    return figure


def plot_predictor_stability_stall(stability_df: pd.DataFrame) -> Path:
    figure = _build_predictor_stability_table(
        stability_df,
        "stall_auc",
        "Stall Stability (Classification AUC)",
        "stall",
        False,
    )
    path = FIGURE_DIR / "predictor_stability_stall.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    figure.savefig(FIGURE_DIR / "predictor_stability_stall.pdf", bbox_inches="tight")
    plt.close(figure)
    return path


def plot_predictor_stability_slowdown(stability_df: pd.DataFrame) -> Path:
    figure = _build_predictor_stability_table(
        stability_df,
        "slowdown_abs_spearman",
        r"Slowdown Stability (Regression Spearman $|\rho|$)",
        "slowdown_ratio",
        False,
    )
    path = FIGURE_DIR / "predictor_stability_slowdown.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    figure.savefig(FIGURE_DIR / "predictor_stability_slowdown.pdf", bbox_inches="tight")
    plt.close(figure)
    return path


def plot_predictor_stability_stall_final(stability_df: pd.DataFrame) -> Path:
    figure = _build_predictor_stability_table(
        stability_df,
        "stall_auc",
        "Stall Stability (Classification AUC)",
        "stall",
        True,
    )
    return _save_final_figure_bundle(figure, "predictor_stability_stall_final")


def plot_predictor_stability_slowdown_final(stability_df: pd.DataFrame) -> Path:
    figure = _build_predictor_stability_table(
        stability_df,
        "slowdown_abs_spearman",
        r"Slowdown Stability (Regression Spearman $|\rho|$)",
        "slowdown_ratio",
        True,
    )
    return _save_final_figure_bundle(figure, "predictor_stability_slowdown_final")


def _build_predictor_stability_table(
    stability_df: pd.DataFrame,
    metric_col: str,
    title: str,
    task_name: str,
    paper_style: bool,
) -> plt.Figure:
    figure, axis = plt.subplots(1, 1, figsize=(7.2, 3.5) if paper_style else (9.0, 4.4))
    _draw_predictor_stability_table(axis, stability_df, metric_col, title, task_name, paper_style)
    if paper_style:
        figure.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.88)
    else:
        figure.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.9)
    return figure


def _draw_predictor_stability_table(
    axis: plt.Axes,
    stability_df: pd.DataFrame,
    metric_col: str,
    title: str,
    task_name: str,
    paper_style: bool,
) -> None:
    predictors = ["T_depth", "slack_ratio", "delta_max"]
    significance_df = _load_predictor_significance()
    row_specs, section_ranges = _build_stability_row_specs(stability_df, metric_col, significance_df, task_name)
    axis.axis("off")
    axis.set_title(title, fontsize=PAPER_TITLE_SIZE + 0.5 if paper_style else 12.5, pad=8)

    display_rows: list[list[str]] = []
    row_kinds: list[str] = []
    raw_values: list[dict[str, float] | None] = []
    for section_name, (start, end) in section_ranges.items():
        display_rows.append([section_name, "", "", ""])
        row_kinds.append("section")
        raw_values.append(None)
        for row_spec in row_specs[start : end + 1]:
            values = row_spec["values"]
            display_rows.append(
                [
                    str(row_spec["label"]),
                    f"{float(values['T_depth']):.2f}",
                    f"{float(values['slack_ratio']):.2f}",
                    f"{float(values['delta_max']):.2f}",
                ]
            )
            row_kinds.append("data")
            raw_values.append({predictor: float(values[predictor]) for predictor in predictors})

    table = axis.table(
        cellText=display_rows,
        colLabels=["Subset", "T-depth", "Slack ratio", r"$\Delta_{\max}$"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.42, 0.18, 0.2, 0.2],
        bbox=[0.0, 0.1, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(PAPER_TICK_SIZE if paper_style else 9)
    table.scale(1.0, 1.12 if paper_style else 1.18)

    header_colors = ["#efefef", PAPER_BAR_COLORS["T_depth"], PAPER_BAR_COLORS["slack_ratio"], PAPER_BAR_COLORS["delta_max"]]
    for col_index in range(4):
        cell = table[(0, col_index)]
        cell.set_facecolor("#f0f0f0")
        cell.set_edgecolor("#666666")
        cell.set_linewidth(0.8)
        cell.get_text().set_fontweight("bold")
        if col_index > 0:
            cell.get_text().set_color(header_colors[col_index])

    table_row_index = 1
    for row_kind, row_values in zip(row_kinds, raw_values, strict=True):
        if row_kind == "section":
            for col_index in range(4):
                cell = table[(table_row_index, col_index)]
                cell.set_facecolor("#e8e8e8")
                cell.set_edgecolor("#bbbbbb")
                cell.set_linewidth(0.9)
                if col_index == 0:
                    cell.get_text().set_fontweight("bold")
                    cell.get_text().set_color("#3f3f3f")
                    cell.get_text().set_ha("left")
                    cell.PAD = 0.02
                else:
                    cell.get_text().set_text("")
        else:
            assert row_values is not None
            best_value = max(row_values.values())
            for col_index in range(4):
                cell = table[(table_row_index, col_index)]
                cell.set_edgecolor("#d0d0d0")
                cell.set_linewidth(0.55)
                if col_index == 0:
                    cell.get_text().set_ha("left")
                    cell.PAD = 0.02
                    cell.set_facecolor("white")
                else:
                    predictor = predictors[col_index - 1]
                    value = row_values[predictor]
                    if np.isclose(value, best_value, atol=1e-9):
                        cell.set_facecolor("#fff3db")
                        cell.get_text().set_fontweight("bold")
                    elif value == 0.0 and task_name == "slowdown_ratio":
                        cell.set_facecolor("#f7f7f7")
                    else:
                        cell.set_facecolor("white")
        table_row_index += 1

    if any(row_spec["has_sig_marker"] for row_spec in row_specs):
        axis.text(
            0.0,
            0.035,
            "* Aggregate only: slack ratio > T-depth (95% bootstrap CI excludes 0).",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=(PAPER_TICK_SIZE - 0.6) if paper_style else 8.0,
            color="#555555",
        )
    if task_name == "slowdown_ratio":
        axis.text(
            1.0,
            0.035,
            "0.00 denotes observed zero within-slice correlation; not missing data.",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=(PAPER_TICK_SIZE - 0.6) if paper_style else 8.0,
            color="#555555",
        )


def _load_predictor_significance() -> pd.DataFrame:
    bootstrap_path = TABLE_DIR / "bootstrap_slack_vs_tdepth.csv"
    if not bootstrap_path.exists():
        return pd.DataFrame()
    return pd.read_csv(bootstrap_path)


def _build_stability_row_specs(
    stability_df: pd.DataFrame,
    metric_col: str,
    significance_df: pd.DataFrame,
    task_name: str,
) -> tuple[list[dict[str, object]], dict[str, tuple[int, int]]]:
    predictors = ["T_depth", "slack_ratio", "delta_max"]
    group_sections = [
        (
            "Aggregate",
            [("overall", "ALL", "All")],
        ),
        (
            "Structural Family",
            [
                ("family", "high_compressibility", "High family"),
                ("family", "medium_compressibility", "Medium family"),
                ("family", "low_compressibility", "Low family"),
            ],
        ),
        (
            "System Capacity",
            [
                ("capacity_band", "low_C", "Low C"),
                ("capacity_band", "mid_C", "Mid C"),
                ("capacity_band", "high_C", "High C"),
            ],
        ),
        (
            "Buffer Budget",
            [
                ("buffer_band", "low_B", "Low B"),
                ("buffer_band", "mid_B", "Mid B"),
                ("buffer_band", "high_B", "High B"),
            ],
        ),
    ]
    row_specs: list[dict[str, object]] = []
    section_ranges: dict[str, tuple[int, int]] = {}

    for section_name, group_specs in group_sections:
        section_start = len(row_specs)
        pending: list[dict[str, object]] = []
        for group_type, group_name, display_name in group_specs:
            subset = stability_df[
                (stability_df["group_type"] == group_type) & (stability_df["group_name"] == group_name)
            ].copy()
            if subset.empty:
                continue
            subset["predictor"] = subset["predictor"].replace({"Delta_max": "delta_max"})
            subset = subset.set_index("predictor")
            values = {predictor: float(subset.loc[predictor, metric_col]) for predictor in predictors}
            sig_marker = _significance_marker(significance_df, group_name, task_name)
            pending.append(
                {
                    "group_type": group_type,
                    "group_name": group_name,
                    "label": display_name + ("*" if sig_marker else ""),
                    "values": values,
                    "has_sig_marker": sig_marker,
                }
            )

        collapsed = _collapse_redundant_stability_rows(pending)
        row_specs.extend(collapsed)
        if len(row_specs) > section_start:
            section_ranges[section_name] = (section_start, len(row_specs) - 1)

    return row_specs, section_ranges


def _collapse_redundant_stability_rows(row_specs: list[dict[str, object]]) -> list[dict[str, object]]:
    if not row_specs:
        return []

    collapsed: list[dict[str, object]] = []
    current = row_specs[0].copy()
    labels = [str(current["label"])]

    for candidate in row_specs[1:]:
        current_values = current["values"]
        candidate_values = candidate["values"]
        same_profile = all(
            np.isclose(float(current_values[predictor]), float(candidate_values[predictor]), atol=1e-9)
            for predictor in ("T_depth", "slack_ratio", "delta_max")
        )
        same_sig = bool(current["has_sig_marker"]) == bool(candidate["has_sig_marker"])
        if same_profile and same_sig:
            labels.append(str(candidate["label"]))
            continue

        current["label"] = " / ".join(labels)
        collapsed.append(current)
        current = candidate.copy()
        labels = [str(current["label"])]

    current["label"] = " / ".join(labels)
    collapsed.append(current)
    return collapsed


def _significance_marker(significance_df: pd.DataFrame, group_name: str, task_name: str) -> bool:
    if significance_df.empty:
        return False
    target = significance_df[(significance_df["group"] == group_name) & (significance_df["task"] == task_name)]
    if target.empty:
        return False
    row = target.iloc[0]
    return bool(row["ci_excludes_zero"]) and float(row["mean_diff"]) > 0.0


def _stability_y_positions(num_rows: int, section_ranges: dict[str, tuple[int, int]]) -> np.ndarray:
    positions = np.arange(num_rows, dtype=float)
    for section_index, (_, (start, _)) in enumerate(section_ranges.items()):
        if section_index == 0:
            continue
        positions[start:] += 1.2
    return positions


def _plot_incremental_panel(
    axis: plt.Axes,
    dataframe: pd.DataFrame,
    metric_name: str,
    ylabel: str,
    title: str,
    paper_style: bool = False,
) -> None:
    model_order = ["T_depth_only", "T_depth_plus_slack", "T_depth_slack_delta"]
    label_map = {
        "T_depth_only": "T-depth",
        "T_depth_plus_slack": "+ Slack ratio",
        "T_depth_slack_delta": "+ " + r"$\Delta_{\max}$",
    }
    subset = dataframe[dataframe["metric_name"] == metric_name].copy()
    subset["model"] = pd.Categorical(subset["model"], categories=model_order, ordered=True)
    subset = subset.sort_values("model")
    x_positions = np.arange(len(subset))
    values = subset["metric_value"].astype(float).to_numpy()
    colors = [PAPER_BAR_COLORS["T_depth"], PAPER_BAR_COLORS["slack_ratio"], PAPER_BAR_COLORS["delta_max"]][: len(subset)]
    bars = axis.bar(x_positions, values, color=colors, width=0.62, edgecolor="#4d4d4d", linewidth=0.4)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(0.01, 0.02 * max(values, default=1.0)),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.2 if paper_style else 8.2,
            color="#333333",
        )
    axis.set_xticks(x_positions, [label_map[model] for model in subset["model"].astype(str)])
    axis.set_ylabel(ylabel, fontsize=PAPER_LABEL_SIZE if paper_style else 10)
    if title:
        axis.set_title(title)
    upper = max(values) if len(values) else 1.0
    axis.set_ylim(0.0, min(1.1, max(0.15, upper + 0.12)))
    if paper_style:
        _apply_paper_axis_style(axis, horizontal_grid=True)
        axis.tick_params(axis="x", labelsize=PAPER_TICK_SIZE, rotation=0)
    else:
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(axis="x", rotation=0)


def _plot_buffer_metric(buffer_df: pd.DataFrame, axis: plt.Axes, metric: str, ylabel: str) -> None:
    for policy in ("static_min", "capacity_aware_static", "smoothed"):
        subset = buffer_df[(buffer_df["policy"] == policy) & np.isfinite(buffer_df[metric])].sort_values("B")
        axis.plot(
            subset["B"],
            subset[metric],
            marker="o",
            color=POLICY_COLORS[policy],
            label=policy,
        )

    delta_static = int(buffer_df["delta_reference_static_min"].iloc[0])
    delta_capacity = int(buffer_df["delta_reference_capacity_aware_static"].iloc[0])
    delta_smoothed = int(buffer_df["delta_reference_smoothed"].iloc[0])
    axis.axvline(
        delta_static,
        color=POLICY_COLORS["static_min"],
        linestyle="--",
        linewidth=1.2,
        label=f"Delta_max static_min = {delta_static}",
    )
    axis.axvline(
        delta_capacity,
        color=POLICY_COLORS["capacity_aware_static"],
        linestyle="--",
        linewidth=1.2,
        label=f"Delta_max capacity_aware_static = {delta_capacity}",
    )
    axis.axvline(
        delta_smoothed,
        color=POLICY_COLORS["smoothed"],
        linestyle="--",
        linewidth=1.2,
        label=f"Delta_max smoothed = {delta_smoothed}",
    )
    axis.set_xlabel("Buffer capacity B")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend()


def _representative_predictive_slice(predictive_df: pd.DataFrame, finite_only: bool) -> pd.DataFrame:
    subset = predictive_df[(predictive_df["C"] == COMPRESSIBILITY_SCAN_C) & (predictive_df["B"] == COMPRESSIBILITY_SCAN_B)].copy()
    if finite_only:
        subset = subset[np.isfinite(subset["slowdown_ratio"])].copy()
    return subset


def _apply_paper_axis_style(axis: plt.Axes, horizontal_grid: bool = False) -> None:
    axis.tick_params(axis="both", labelsize=PAPER_TICK_SIZE)
    if horizontal_grid:
        axis.grid(axis="y", alpha=0.18, linewidth=0.7)
    else:
        axis.grid(alpha=0.16, linewidth=0.7)
    axis.set_facecolor("white")
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")


def _save_final_figure_bundle(figure: plt.Figure, stem: str) -> Path:
    ensure_output_dirs()
    path = FINAL_PAPER_DIR / f"{stem}.png"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(FINAL_PAPER_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)
    return path


def _family_display_name(name: str) -> str:
    return name.replace("_compressibility", "").replace("_", " ")


def _predictor_panel_upper_bound(values: list[float]) -> float:
    upper = max(values) if values else 1.0
    return min(1.1, max(0.15, upper + 0.09))


def _scatter_by_family(axis: plt.Axes, dataframe: pd.DataFrame, x_col: str, y_col: str) -> None:
    for family in FAMILY_COLORS:
        subset = dataframe[dataframe["family"] == family]
        if subset.empty:
            continue
        axis.scatter(
            subset[x_col],
            subset[y_col],
            s=55,
            marker=FAMILY_MARKERS[family],
            color=FAMILY_COLORS[family],
            alpha=0.8,
            label=family,
        )


def _annotate_correlation(axis: plt.Axes, x_values: pd.Series, y_values: pd.Series) -> None:
    if len(x_values) < 2 or x_values.nunique() < 2 or y_values.nunique() < 2:
        return
    pearson = float(np.corrcoef(x_values.to_numpy(), y_values.to_numpy())[0, 1])
    x_rank = x_values.rank(method="average").to_numpy()
    y_rank = y_values.rank(method="average").to_numpy()
    spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
    axis.text(
        0.03,
        0.97,
        f"Pearson={pearson:.2f}\nSpearman={spearman:.2f}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )


def _annotate_correlation_compact(axis: plt.Axes, x_values: pd.Series, y_values: pd.Series) -> None:
    if len(x_values) < 2 or x_values.nunique() < 2 or y_values.nunique() < 2:
        return
    pearson = float(np.corrcoef(x_values.to_numpy(), y_values.to_numpy())[0, 1])
    x_rank = x_values.rank(method="average").to_numpy()
    y_rank = y_values.rank(method="average").to_numpy()
    spearman = float(np.corrcoef(x_rank, y_rank)[0, 1])
    axis.text(
        0.04,
        0.96,
        f"r={pearson:.2f}\nρ={spearman:.2f}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "#c7c7c7", "boxstyle": "round,pad=0.18"},
    )


def _add_trend_line(axis: plt.Axes, x_values: pd.Series, y_values: pd.Series) -> None:
    if len(x_values) < 2 or x_values.nunique() < 2:
        return
    coefficients = np.polyfit(x_values.to_numpy(dtype=float), y_values.to_numpy(dtype=float), deg=1)
    x_line = np.linspace(float(x_values.min()), float(x_values.max()), 100)
    y_line = coefficients[0] * x_line + coefficients[1]
    axis.plot(x_line, y_line, color="#333333", linewidth=1.1, linestyle="-", alpha=0.8, zorder=1)
