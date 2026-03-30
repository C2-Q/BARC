from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.metrics import compute_trace_statistics
from src.paper_config import B_VALUES, C_VALUES
from src.real_trace.adder_trace import generate_and_save_adder_trace
from src.real_trace.multiplier_trace import generate_and_save_multiplier_trace
from src.real_trace.qft_trace import generate_and_save_qft_trace
from src.real_trace.plot_utils import (
    ANNOTATION_SIZE,
    CAPACITY_AWARE_COLOR,
    SMOOTHED_COLOR,
    STATIC_COLOR,
    apply_axis_style,
    configure_publication_style,
    save_figure_bundle,
)
from src.real_trace.stats_utils import (
    NORMALIZED_TARGET,
    compute_burst_lengths,
    compute_threshold_rows,
    compute_workload_statistics,
    find_critical_buffer,
    find_critical_capacity,
    find_representative_buffer,
    find_representative_capacity,
    find_stall_saturation_buffer,
    infer_trace_label,
    select_zoom_window,
)
from src.simulator import check_trace_feasibility, simulate_trace
from src.utils import OUTPUT_ROOT, TABLE_DIR, ensure_output_dirs


REAL_TRACE_OUTPUT_DIR = OUTPUT_ROOT / "real_trace"
REAL_TRACE_POLICY_COLORS = {
    "static_min": STATIC_COLOR,
    "capacity_aware_static": CAPACITY_AWARE_COLOR,
    "smoothed": SMOOTHED_COLOR,
}


def load_real_trace_from_csv(path: str) -> list[int]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    dataframe = pd.read_csv(csv_path)
    if "demand" in dataframe.columns:
        values = dataframe["demand"].astype(int).tolist()
    else:
        values = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row in reader:
                for cell in row:
                    cell = cell.strip()
                    if not cell or cell == "timestep":
                        continue
                    values.append(int(cell))
    _validate_trace(values)
    return values


def load_real_trace_from_json(path: str) -> list[int]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        values = [int(item) for item in payload]
    elif isinstance(payload, dict) and isinstance(payload.get("demand"), list):
        values = [int(item) for item in payload["demand"]]
    else:
        raise ValueError("JSON real trace must be a list or contain a 'demand' list")
    _validate_trace(values)
    return values


def load_real_trace(path: str) -> list[int]:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_real_trace_from_csv(path)
    if suffix == ".json":
        return load_real_trace_from_json(path)
    raise ValueError("real trace path must end with .csv or .json")


def ensure_real_trace_exists(path: str) -> Path:
    trace_path = Path(path)
    if trace_path.exists():
        return trace_path
    adder_match = re.fullmatch(r"adder_n(\d+)\.csv", trace_path.name)
    multiplier_match = re.fullmatch(r"multiplier_n(\d+)\.csv", trace_path.name)
    qft_match = re.fullmatch(r"qft_n(\d+)\.csv", trace_path.name)
    if adder_match:
        n_bits = int(adder_match.group(1))
        generate_and_save_adder_trace(n_bits=n_bits, path=trace_path)
    elif multiplier_match:
        n_bits = int(multiplier_match.group(1))
        generate_and_save_multiplier_trace(n_bits=n_bits, path=trace_path)
    elif qft_match:
        n_bits = int(qft_match.group(1))
        generate_and_save_qft_trace(n_bits=n_bits, path=trace_path)
    else:
        raise FileNotFoundError(trace_path)
    return trace_path


def estimate_compressibility(trace: list[int]) -> float:
    mean = float(np.mean(trace)) if trace else 0.0
    if mean <= 0.0:
        return 0.0
    return float(np.std(trace) / mean)


def smooth_trace(trace: list[int]) -> list[int]:
    if not trace:
        return []
    total_t = int(sum(trace))
    if total_t == 0:
        return list(trace)
    compressibility = estimate_compressibility(trace)
    extra_horizon = max(1, int(np.ceil(compressibility)))
    horizon = len(trace) + extra_horizon
    target = max(1, int(np.ceil(total_t / horizon)))

    backlog = 0
    smoothed: list[int] = []
    for value in trace:
        backlog += int(value)
        served = min(backlog, target)
        smoothed.append(served)
        backlog -= served
    while backlog > 0:
        served = min(backlog, target)
        smoothed.append(served)
        backlog -= served
    return smoothed


def capacity_aware_trace(trace: list[int], capacity_limit: int) -> list[int]:
    """Apply a local per-cycle quota and push excess demand forward."""
    if capacity_limit <= 0:
        return list(trace)
    shaped: list[int] = []
    backlog = 0
    for value in trace:
        backlog += int(value)
        served = min(backlog, capacity_limit)
        shaped.append(served)
        backlog -= served
    while backlog > 0:
        served = min(backlog, capacity_limit)
        shaped.append(served)
        backlog -= served
    return shaped


def evaluate_real_trace_policies(
    trace: list[int],
    trace_name: str,
    logical_qubit_budget: int,
    C_values: list[int] | None = None,
    B_values: list[int] | None = None,
) -> pd.DataFrame:
    ensure_output_dirs()
    REAL_TRACE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    C_scan = C_VALUES if C_values is None else C_values
    B_scan = B_VALUES if B_values is None else B_values

    compressibility = estimate_compressibility(trace)
    rows: list[dict[str, Any]] = []
    smoothed_trace = smooth_trace(trace)
    for C in C_scan:
        policies = {
            "static_min": list(trace),
            "capacity_aware_static": capacity_aware_trace(trace, capacity_limit=C),
            "smoothed": list(smoothed_trace),
        }
        for policy, demand in policies.items():
            _, Delta_max, Gamma, peak_demand, mean_demand, total_T = compute_trace_statistics(demand, C=C)
            for B in B_scan:
                feasible, feasible_reason = check_trace_feasibility(demand, C=C, B=B)
                if feasible:
                    sim_result = simulate_trace(demand, C=C, B=B)
                    T_exe = float(sim_result.T_exe)
                    stall_cycles = float(sim_result.stall_cycles)
                    stall_ratio = float(sim_result.stall_ratio)
                    QTV_proxy = float(logical_qubit_budget * sim_result.T_exe)
                    normalized_makespan = T_exe / max(1, len(demand))
                else:
                    T_exe = float("inf")
                    stall_cycles = float("inf")
                    stall_ratio = 1.0
                    QTV_proxy = float("inf")
                    normalized_makespan = float("inf")
                rows.append(
                    {
                        "trace_name": trace_name,
                        "policy": policy,
                        "trace_length": len(demand),
                        "logical_qubit_budget": logical_qubit_budget,
                        "real_trace_compressibility": compressibility,
                        "C": C,
                        "B": B,
                        "feasible": int(feasible),
                        "feasible_reason": feasible_reason,
                        "T_static": len(demand),
                        "T_exe": T_exe,
                        "stall_cycles": stall_cycles,
                        "stall_ratio": stall_ratio,
                        "Delta_max": Delta_max,
                        "Gamma": Gamma,
                        "peak_demand": peak_demand,
                        "mean_demand": mean_demand,
                        "total_T": total_T,
                        "QTV_proxy": QTV_proxy,
                        "normalized_makespan": normalized_makespan,
                    }
                )
    return pd.DataFrame(rows)


def summarize_real_trace(results_df: pd.DataFrame) -> pd.DataFrame:
    pivot = results_df.pivot_table(
        index=["trace_name", "C", "B", "real_trace_compressibility"],
        columns="policy",
        values=["feasible", "T_static", "T_exe", "stall_cycles", "Delta_max", "Gamma", "QTV_proxy"],
        aggfunc="first",
    )
    pivot.columns = [f"{value}_{policy}" for value, policy in pivot.columns]
    summary = pivot.reset_index()
    summary["both_feasible"] = (
        (summary["feasible_static_min"] == 1) & (summary["feasible_smoothed"] == 1)
    ).astype(int)
    summary["inversion"] = (
        (summary["both_feasible"] == 1)
        & (summary["T_static_static_min"] < summary["T_static_smoothed"])
        & (summary["T_exe_static_min"] > summary["T_exe_smoothed"])
    ).astype(int)
    summary["T_exe_improvement"] = 0.0
    summary["stall_reduction"] = 0.0
    summary["relative_change_in_QTV"] = 0.0
    valid = summary["both_feasible"] == 1
    valid_rows = summary.loc[valid]
    summary.loc[valid, "T_exe_improvement"] = (
        valid_rows["T_exe_static_min"] - valid_rows["T_exe_smoothed"]
    ) / valid_rows["T_exe_static_min"]
    summary.loc[valid, "stall_reduction"] = (
        valid_rows["stall_cycles_static_min"] - valid_rows["stall_cycles_smoothed"]
    )
    summary.loc[valid, "relative_change_in_QTV"] = (
        valid_rows["QTV_proxy_smoothed"] - valid_rows["QTV_proxy_static_min"]
    ) / valid_rows["QTV_proxy_static_min"]
    return summary


def export_real_trace_workload_statistics(
    trace: list[int],
    trace_name: str,
    output_path: str | Path,
) -> pd.DataFrame:
    dataframe = pd.DataFrame([compute_workload_statistics(trace, trace_name)])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return dataframe


def export_real_trace_threshold_summary(
    results_df: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    summary_df = summarize_real_trace(results_df)
    threshold_df = compute_threshold_rows(results_df, summary_df)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(output_path, index=False)
    return threshold_df


def plot_real_trace_demand(trace: list[int], output_path: str | Path) -> Path:
    configure_publication_style()
    trace_name = Path(output_path).stem.replace("_trace_plot", "")
    title = f"{infer_trace_label(trace_name)}: Workload Characterization"
    stats = compute_workload_statistics(trace, trace_name)
    demand_levels = sorted(set(trace))
    burst_lengths = compute_burst_lengths(trace, stats["burst_threshold"])
    zoom_start, zoom_end = select_zoom_window(trace, stats["burst_threshold"])

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))

    x_values = np.arange(len(trace))
    axes[0].step(x_values, trace, where="post", color=STATIC_COLOR)
    axes[0].set_title(title)
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("T-demand")
    apply_axis_style(axes[0])
    stats_text = (
        f"mean={stats['mean_demand']:.2f}\n"
        f"max={stats['max_demand']}\n"
        f"burst >= {stats['burst_threshold']}"
    )
    axes[0].text(
        0.02,
        0.96,
        stats_text,
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=ANNOTATION_SIZE,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.25"},
    )
    inset = inset_axes(axes[0], width="38%", height="38%", loc="upper right")
    inset.step(
        np.arange(zoom_start, zoom_end),
        trace[zoom_start:zoom_end],
        where="post",
        color=STATIC_COLOR,
    )
    inset.set_title(f"Zoom [{zoom_start}, {zoom_end})", fontsize=10)
    inset.tick_params(axis="both", labelsize=9)
    inset.grid(True, alpha=0.15)

    level_counts = [trace.count(level) / max(1, len(trace)) for level in demand_levels]
    axes[1].bar(demand_levels, level_counts, color="#7fb3d5", edgecolor=STATIC_COLOR, linewidth=0.8)
    axes[1].set_title("Demand Frequency Distribution")
    axes[1].set_xlabel("Demand level")
    axes[1].set_ylabel("Proportion")
    apply_axis_style(axes[1])

    if burst_lengths:
        burst_levels = sorted(set(burst_lengths))
        burst_counts = [burst_lengths.count(level) for level in burst_levels]
        axes[2].bar(burst_levels, burst_counts, color="#f4a6a6", edgecolor=SMOOTHED_COLOR, linewidth=0.8)
    else:
        axes[2].bar([0], [1], color="#f4a6a6", edgecolor=SMOOTHED_COLOR, linewidth=0.8)
        burst_levels = [0]
    axes[2].set_title("Burst Length Distribution")
    axes[2].set_xlabel("Consecutive burst length")
    axes[2].set_ylabel("Count")
    apply_axis_style(axes[2])

    return save_figure_bundle(figure, output_path)


def plot_real_trace_capacity_scan(results_df: pd.DataFrame, output_path: str | Path) -> Path:
    configure_publication_style()
    summary_df = summarize_real_trace(results_df)
    representative_B = find_representative_buffer(summary_df)
    trace_name = str(results_df["trace_name"].iloc[0])
    title = f"{infer_trace_label(trace_name)}: Capacity Sensitivity"
    display_max = _display_upper_bound(
        results_df[(results_df["B"] == representative_B)]["normalized_makespan"].tolist()
    )

    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    axis.axhline(NORMALIZED_TARGET, color="#666666", linestyle=":", linewidth=1.2)

    policy_order = [policy for policy in ("static_min", "capacity_aware_static", "smoothed") if policy in set(results_df["policy"])]
    for policy in policy_order:
        color = REAL_TRACE_POLICY_COLORS[policy]
        subset = results_df[
            (results_df["policy"] == policy)
            & (results_df["B"] == representative_B)
        ].sort_values("C")
        y_values = [_display_value(value, display_max) for value in subset["normalized_makespan"]]
        axis.plot(subset["C"], y_values, marker="o", color=color, label=policy)
        infeasible_subset = subset[~np.isfinite(subset["normalized_makespan"])]
        if not infeasible_subset.empty:
            axis.scatter(
                infeasible_subset["C"],
                [_display_value(float("inf"), display_max)] * len(infeasible_subset),
                marker="x",
                s=34,
                color=color,
            )
        critical_capacity = find_critical_capacity(results_df, policy, representative_B)
        if critical_capacity is not None:
            axis.axvline(critical_capacity, color=color, linestyle="--", linewidth=1.2, alpha=0.9)
            axis.annotate(
                f"{policy}: C={critical_capacity}",
                xy=(critical_capacity, NORMALIZED_TARGET),
                xytext=(critical_capacity + 0.1, NORMALIZED_TARGET + 0.08),
                fontsize=ANNOTATION_SIZE,
                color=color,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 1.0},
            )
    axis.set_xlabel("Capacity C")
    axis.set_ylabel("Normalized execution time")
    axis.set_title(f"{title} (B* = {representative_B})")
    axis.set_ylim(bottom=0.95, top=max(1.05, display_max * 1.02))
    apply_axis_style(axis)
    axis.legend(frameon=False)
    return save_figure_bundle(figure, output_path)


def plot_real_trace_buffer_transition(results_df: pd.DataFrame, output_path: str | Path) -> Path:
    configure_publication_style()
    summary_df = summarize_real_trace(results_df)
    trace_name = str(results_df["trace_name"].iloc[0])
    representative_C = find_representative_capacity(summary_df)
    title = f"{infer_trace_label(trace_name)}: Buffer Sensitivity"
    display_max_stall = _display_upper_bound(
        results_df[(results_df["C"] == representative_C)]["stall_cycles"].tolist()
    )
    display_max_norm = _display_upper_bound(
        results_df[(results_df["C"] == representative_C)]["normalized_makespan"].tolist()
    )

    figure, axes = plt.subplots(2, 1, figsize=(8.8, 8.2), sharex=True)
    policy_order = [policy for policy in ("static_min", "capacity_aware_static", "smoothed") if policy in set(results_df["policy"])]
    for policy in policy_order:
        color = REAL_TRACE_POLICY_COLORS[policy]
        subset = results_df[(results_df["policy"] == policy) & (results_df["C"] == representative_C)].sort_values("B")
        stall_values = [_display_value(value, display_max_stall) for value in subset["stall_cycles"]]
        norm_values = [_display_value(value, display_max_norm) for value in subset["normalized_makespan"]]
        axes[0].plot(subset["B"], stall_values, marker="o", color=color, label=policy)
        axes[1].plot(subset["B"], norm_values, marker="o", color=color, label=policy)

        critical_buffer = find_critical_buffer(results_df, policy, representative_C)
        saturation_buffer = find_stall_saturation_buffer(results_df, policy, representative_C)
        if critical_buffer is not None:
            axes[0].axvline(critical_buffer, color=color, linestyle="--", linewidth=1.2, alpha=0.9)
            axes[1].axvline(critical_buffer, color=color, linestyle="--", linewidth=1.2, alpha=0.9)
            axes[1].annotate(
                f"{policy}: B={critical_buffer}",
                xy=(critical_buffer, NORMALIZED_TARGET),
                xytext=(critical_buffer + 0.15, NORMALIZED_TARGET + 0.08),
                fontsize=ANNOTATION_SIZE,
                color=color,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 1.0},
            )
        if saturation_buffer is not None:
            saturation_row = subset[subset["B"] == saturation_buffer]
            if not saturation_row.empty:
                axes[0].scatter(
                    [saturation_buffer],
                    [float(saturation_row["stall_cycles"].iloc[0])],
                    color=color,
                    marker="D",
                    s=28,
                    zorder=5,
                )
                axes[0].annotate(
                    f"sat={saturation_buffer}",
                    xy=(saturation_buffer, float(saturation_row["stall_cycles"].iloc[0])),
                    xytext=(saturation_buffer + 0.2, float(saturation_row["stall_cycles"].iloc[0]) + 0.4),
                    fontsize=ANNOTATION_SIZE,
                    color=color,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 1.0},
                )

    axes[0].set_title(f"{title} (C* = {representative_C})")
    axes[0].set_ylabel("Stall cycles")
    axes[0].set_ylim(bottom=0.0, top=display_max_stall * 1.03)
    apply_axis_style(axes[0])
    axes[0].legend(frameon=False)

    axes[1].axhline(NORMALIZED_TARGET, color="#666666", linestyle=":", linewidth=1.2)
    axes[1].set_xlabel("Buffer B")
    axes[1].set_ylabel("Normalized execution time")
    axes[1].set_ylim(bottom=0.95, top=max(1.05, display_max_norm * 1.02))
    apply_axis_style(axes[1])
    return save_figure_bundle(figure, output_path)


def _validate_trace(values: list[int]) -> None:
    if any(value < 0 for value in values):
        raise ValueError("real trace entries must be non-negative")


def generate_real_trace_comparison(
    adder_summary_df: pd.DataFrame,
    multiplier_summary_df: pd.DataFrame,
    adder_results_df: pd.DataFrame,
    multiplier_results_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        _build_real_trace_comparison_row(adder_summary_df, adder_results_df),
        _build_real_trace_comparison_row(multiplier_summary_df, multiplier_results_df),
    ]
    return pd.DataFrame(rows)


def plot_adder_vs_multiplier_trace(
    adder_trace: list[int],
    multiplier_trace: list[int],
    output_path: str | Path,
) -> Path:
    configure_publication_style()
    adder_stats = compute_workload_statistics(adder_trace, "adder")
    multiplier_stats = compute_workload_statistics(multiplier_trace, "multiplier")

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for axis, trace, stats, color, title in (
        (axes[0], adder_trace, adder_stats, STATIC_COLOR, "Adder N4"),
        (axes[1], multiplier_trace, multiplier_stats, SMOOTHED_COLOR, "Multiplier N4"),
    ):
        axis.step(np.arange(len(trace)), trace, where="post", color=color)
        axis.set_title(title)
        axis.set_xlabel("Timestep")
        axis.set_ylabel("T-demand")
        apply_axis_style(axis)
        axis.text(
            0.03,
            0.96,
            f"max={stats['max_demand']}\nspikes={stats['spike_count']}\nmean={stats['mean_demand']:.2f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=ANNOTATION_SIZE,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.25"},
        )
    figure.suptitle("Cross-Workload Trace Characterization", fontsize=18)
    return save_figure_bundle(figure, output_path)


def plot_adder_vs_multiplier_capacity_scan(
    adder_results_df: pd.DataFrame,
    multiplier_results_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    configure_publication_style()
    adder_summary = summarize_real_trace(adder_results_df)
    multiplier_summary = summarize_real_trace(multiplier_results_df)
    adder_B = find_representative_buffer(adder_summary)
    multiplier_B = find_representative_buffer(multiplier_summary)

    figure, axis = plt.subplots(figsize=(9.4, 5.1))
    styles = {
        ("adder", "static_min"): ("#7fb3d5", "-"),
        ("adder", "smoothed"): ("#7fb3d5", "--"),
        ("multiplier", "static_min"): ("#ef8a62", "-"),
        ("multiplier", "smoothed"): ("#ef8a62", "--"),
    }
    axis.axhline(NORMALIZED_TARGET, color="#666666", linestyle=":", linewidth=1.2)
    for label, results_df, chosen_B in (
        ("adder", adder_results_df, adder_B),
        ("multiplier", multiplier_results_df, multiplier_B),
    ):
        display_max = _display_upper_bound(
            results_df[(results_df["B"] == chosen_B)]["normalized_makespan"].tolist()
        )
        for policy in ("static_min", "smoothed"):
            subset = results_df[(results_df["B"] == chosen_B) & (results_df["policy"] == policy)].sort_values("C")
            y_values = [_display_value(value, display_max) for value in subset["normalized_makespan"]]
            color, linestyle = styles[(label, policy)]
            axis.plot(
                subset["C"],
                y_values,
                marker="o",
                color=color,
                linestyle=linestyle,
                label=f"{label} {policy} (B={chosen_B})",
            )
        for policy in ("static_min", "smoothed"):
            critical_c = find_critical_capacity(results_df, policy, chosen_B)
            if critical_c is not None:
                color, _ = styles[(label, policy)]
                axis.axvline(critical_c, color=color, linestyle="--", linewidth=1.0, alpha=0.4)
    axis.set_xlabel("Capacity C")
    axis.set_ylabel("Normalized execution time")
    axis.set_title("Cross-Workload Capacity Comparison")
    apply_axis_style(axis)
    axis.legend(frameon=False, ncol=2)
    return save_figure_bundle(figure, output_path)


def plot_cross_workload_critical_capacity(
    threshold_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    configure_publication_style()
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    traces = sorted(threshold_df["trace_name"].unique())
    positions = np.arange(len(traces))
    width = 0.34
    static_values = [
        float(threshold_df[(threshold_df["trace_name"] == trace) & (threshold_df["policy"] == "static_min")]["critical_capacity"].iloc[0])
        for trace in traces
    ]
    smoothed_values = [
        float(threshold_df[(threshold_df["trace_name"] == trace) & (threshold_df["policy"] == "smoothed")]["critical_capacity"].iloc[0])
        for trace in traces
    ]
    axis.bar(positions - width / 2, static_values, width=width, color=STATIC_COLOR, label="static_min")
    axis.bar(positions + width / 2, smoothed_values, width=width, color=SMOOTHED_COLOR, label="smoothed")
    axis.set_xticks(positions, [infer_trace_label(trace) for trace in traces])
    axis.set_ylabel("Critical capacity C")
    axis.set_title("Cross-Workload Comparison of Critical Capacity")
    apply_axis_style(axis)
    axis.legend(frameon=False)
    return save_figure_bundle(figure, output_path)


def plot_cross_workload_critical_buffer(
    threshold_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    configure_publication_style()
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    traces = sorted(threshold_df["trace_name"].unique())
    positions = np.arange(len(traces))
    width = 0.34

    for axis, column, title in (
        (axes[0], "critical_buffer", "Critical Buffer for Normalized Time"),
        (axes[1], "stall_saturation_buffer", "Buffer Where Stall Saturates"),
    ):
        static_values = [
            float(threshold_df[(threshold_df["trace_name"] == trace) & (threshold_df["policy"] == "static_min")][column].iloc[0])
            for trace in traces
        ]
        smoothed_values = [
            float(threshold_df[(threshold_df["trace_name"] == trace) & (threshold_df["policy"] == "smoothed")][column].iloc[0])
            for trace in traces
        ]
        axis.bar(positions - width / 2, static_values, width=width, color=STATIC_COLOR, label="static_min")
        axis.bar(positions + width / 2, smoothed_values, width=width, color=SMOOTHED_COLOR, label="smoothed")
        axis.set_xticks(positions, [infer_trace_label(trace) for trace in traces])
        axis.set_title(title)
        axis.set_ylabel("Buffer B")
        apply_axis_style(axis)
    axes[0].legend(frameon=False)
    figure.suptitle("Cross-Workload Comparison of Critical Buffer", fontsize=18)
    return save_figure_bundle(figure, output_path)


def plot_cross_workload_improvement_gap(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    configure_publication_style()
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    traces = [infer_trace_label(value) for value in comparison_df["trace_name"]]
    positions = np.arange(len(traces))
    width = 0.24
    metrics = [
        ("capacity_reduction", "#4c78a8", "Capacity reduction"),
        ("buffer_reduction", "#f58518", "Buffer reduction"),
        ("best_stall_reduction", "#54a24b", "Best stall reduction"),
        ("best_normalized_improvement", "#b279a2", "Best normalized improvement"),
    ]
    for index, (column, color, label) in enumerate(metrics):
        axis.bar(
            positions + (index - 1.5) * width,
            comparison_df[column],
            width=width,
            color=color,
            label=label,
        )
    axis.set_xticks(positions, traces)
    axis.set_ylabel("Improvement magnitude")
    axis.set_title("Cross-Workload Improvement Gap")
    apply_axis_style(axis)
    axis.legend(frameon=False, ncol=2)
    return save_figure_bundle(figure, output_path)


def _display_upper_bound(values: list[float]) -> float:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    if not finite_values:
        return 1.0
    return max(finite_values) * 1.15 + 0.1


def _display_value(value: float, display_max: float) -> float:
    return float(value) if np.isfinite(value) else display_max


def _as_float_or_nan(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    return float(value)


def _threshold_reduction(static_value: Any, smoothed_value: Any) -> float:
    static_float = _as_float_or_nan(static_value)
    smoothed_float = _as_float_or_nan(smoothed_value)
    if static_float < 0 or smoothed_float < 0 or np.isnan(static_float) or np.isnan(smoothed_float):
        return -1.0
    return static_float - smoothed_float


def _build_real_trace_comparison_row(summary_df: pd.DataFrame, results_df: pd.DataFrame) -> dict[str, Any]:
    trace_name = str(results_df["trace_name"].iloc[0])
    representative = _select_representative_summary_row(summary_df)
    static_results = results_df[results_df["policy"] == "static_min"]
    threshold_df = compute_threshold_rows(results_df, summary_df)
    static_thresholds = threshold_df[threshold_df["policy"] == "static_min"].iloc[0]
    smoothed_thresholds = threshold_df[threshold_df["policy"] == "smoothed"].iloc[0]
    return {
        "trace_name": trace_name,
        "n_bits": _infer_n_bits_from_trace_name(trace_name),
        "total_T": int(static_results["total_T"].iloc[0]),
        "peak_demand": int(static_results["peak_demand"].iloc[0]),
        "mean_demand": float(static_results["mean_demand"].iloc[0]),
        "compressibility_proxy": float(static_results["real_trace_compressibility"].iloc[0]),
        "Delta_max_static_min": int(representative["Delta_max_static_min"]),
        "Delta_max_smoothed": int(representative["Delta_max_smoothed"]),
        "T_exe_improvement": float(representative["T_exe_improvement"]),
        "inversion_observed": int((summary_df["inversion"] == 1).any()),
        "critical_capacity_static_min": _as_float_or_nan(static_thresholds["critical_capacity"]),
        "critical_capacity_smoothed": _as_float_or_nan(smoothed_thresholds["critical_capacity"]),
        "critical_buffer_static_min": _as_float_or_nan(static_thresholds["critical_buffer"]),
        "critical_buffer_smoothed": _as_float_or_nan(smoothed_thresholds["critical_buffer"]),
        "stall_saturation_buffer_static_min": _as_float_or_nan(static_thresholds["stall_saturation_buffer"]),
        "stall_saturation_buffer_smoothed": _as_float_or_nan(smoothed_thresholds["stall_saturation_buffer"]),
        "capacity_reduction": _threshold_reduction(static_thresholds["critical_capacity"], smoothed_thresholds["critical_capacity"]),
        "buffer_reduction": _threshold_reduction(static_thresholds["critical_buffer"], smoothed_thresholds["critical_buffer"]),
        "best_stall_reduction": float(summary_df["stall_reduction"].fillna(0.0).max()),
        "best_normalized_improvement": float(summary_df["T_exe_improvement"].fillna(0.0).max()),
    }


def _select_representative_summary_row(summary_df: pd.DataFrame) -> pd.Series:
    both_feasible = summary_df[summary_df["both_feasible"] == 1].copy()
    if not both_feasible.empty:
        positive = both_feasible[both_feasible["T_exe_improvement"] > 0]
        target = positive if not positive.empty else both_feasible
        sort_columns = ["T_exe_improvement", "stall_reduction", "C", "B"]
        return target.sort_values(sort_columns, ascending=[False, False, True, True]).iloc[0]
    return summary_df.sort_values(["feasible_static_min", "feasible_smoothed", "C", "B"], ascending=[False, False, True, True]).iloc[0]


def _infer_n_bits_from_trace_name(trace_name: str) -> int:
    match = re.search(r"_n(\d+)$", trace_name)
    return int(match.group(1)) if match else 0


def _title_from_trace_name(trace_name: str) -> str:
    return infer_trace_label(trace_name)


def _title_from_trace_path(stem: str) -> str:
    return infer_trace_label(stem)
