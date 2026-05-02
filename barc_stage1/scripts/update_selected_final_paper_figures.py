# v2 (2026-04-25): compact lower-bound figure and clearer QFT approximation figure for IEEE two-column layout
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plots import PAPER_LABEL_SIZE, PAPER_LEGEND_SIZE, PAPER_TICK_SIZE, _apply_paper_axis_style
from src.real_trace.analysis import load_real_trace_from_csv
from src.real_trace.adder_trace import extract_t_demand_trace, to_clifford_t
from src.real_trace.qft_trace import generate_qft_circuit
from src.utils import FINAL_PAPER_DIR, TABLE_DIR, ensure_output_dirs


def main() -> None:
    ensure_output_dirs()
    lower_bound_df = pd.read_csv(TABLE_DIR / "lower_bound_validation.csv")
    update_lower_bound_vs_actual(lower_bound_df)
    update_qft_approximation_figure()


def update_lower_bound_vs_actual(lower_bound_df: pd.DataFrame) -> Path:
    subset = lower_bound_df[np.isfinite(lower_bound_df["T_exe"])].copy()
    gaps = subset["gap"].astype(float).to_numpy()
    sorted_gaps = np.sort(gaps)
    cdf = np.arange(1, len(sorted_gaps) + 1) / max(1, len(sorted_gaps))
    within_one = float((gaps <= 1.0 + 1e-9).mean()) if len(gaps) else 0.0

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(6.9, 2.75),
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )

    scatter = axes[0].scatter(
        subset["predicted_lower_bound"],
        subset["T_exe"],
        s=8,
        c=subset["gap"],
        cmap="cividis",
        alpha=0.64,
        edgecolors="none",
        rasterized=True,
    )

    lower = float(min(subset["predicted_lower_bound"].min(), subset["T_exe"].min()))
    upper = float(max(subset["predicted_lower_bound"].max(), subset["T_exe"].max()))
    axes[0].plot([lower, upper], [lower, upper], color="#444444", linewidth=1.0, zorder=2)

    axes[0].set_xlabel("Predicted lower bound", fontsize=9)
    axes[0].set_ylabel(r"Observed $T_{\mathrm{exe}}$", fontsize=9)
    _apply_paper_axis_style(axes[0])
    axes[0].tick_params(axis="both", labelsize=8)

    colorbar = figure.colorbar(scatter, ax=axes[0], fraction=0.045, pad=0.012)
    colorbar.set_label(r"Gap = $T_{\mathrm{exe}} -$ bound (cycles)", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)

    axes[1].step(sorted_gaps, cdf, where="post", color="#1f78b4", linewidth=1.8)
    axes[1].fill_between(sorted_gaps, cdf, step="post", alpha=0.10, color="#1f78b4")

    axes[1].axvline(1.0, color="#b22222", linestyle="--", linewidth=1.0)
    axes[1].axhline(within_one, color="#b22222", linestyle="--", linewidth=1.0)
    axes[1].scatter([1.0], [within_one], color="#b22222", s=18, zorder=4)

    # Place the annotation away from the CDF curve and avoid a white box.
    axes[1].text(
        2.0,
        0.94,
        rf"{within_one * 100:.1f}% $\leq$ 1 cycle",
        fontsize=7.2,
        color="#8b1e1e",
        ha="left",
        va="center",
    )

    axes[1].set_xlabel("Gap to lower bound (cycles)", fontsize=9)
    axes[1].set_ylabel("Cumulative fraction of instances", fontsize=9)
    axes[1].set_xlim(0.0, max(15.0, float(sorted_gaps.max())))
    axes[1].set_ylim(0.0, 1.02)
    _apply_paper_axis_style(axes[1])
    axes[1].tick_params(axis="both", labelsize=8)

    # Manual layout is more stable than constrained_layout when colorbars are used.
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.96, wspace=0.42)

    return _save_final_figure(figure, "lower_bound_vs_actual_final")


def update_qft_approximation_figure() -> Path:
    exact_trace = np.array(
        load_real_trace_from_csv(str(PROJECT_ROOT / "data" / "real_traces" / "qft_n12.csv")),
        dtype=int,
    )
    approx_trace = np.array(
        extract_t_demand_trace(to_clifford_t(generate_qft_circuit(12, do_swaps=False, approximation_degree=4))),
        dtype=int,
    )
    summary_df = pd.read_csv(TABLE_DIR / "qft_approximation_reduced_grid_summary.csv")

    exact_peak = int(exact_trace.max())
    approx_peak = int(approx_trace.max())
    assert exact_peak == 10
    assert approx_peak == 6
    assert len(exact_trace) == 932784
    assert len(approx_trace) == 932784

    exact_summary = summary_df[(summary_df["variant"] == "exact") & (summary_df["C"].isna())].iloc[0]
    approx_summary = summary_df[(summary_df["variant"] == "approx_deg_4") & (summary_df["C"].isna())].iloc[0]
    assert int(round(exact_summary["mean_delta_max"])) == 567516
    assert int(round(approx_summary["mean_delta_max"])) == 309644

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.6),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )

    representative_C = 2
    representative_B = 8

    exact_color = "#2b6cb0"
    approx_color = "#2f855a"
    supply_color = "#555555"

    # Binned peak-window view: avoids the unreadable raw spike forest.
    offsets, exact_binned = _binned_peak_window(
        exact_trace,
        peak_index=int(np.argmax(exact_trace)),
        half_window=200,
        bin_width=20,
    )
    _, approx_binned = _binned_peak_window(
        approx_trace,
        peak_index=int(np.argmax(approx_trace)),
        half_window=200,
        bin_width=20,
    )

    axes[0].step(
        offsets,
        exact_binned,
        where="mid",
        color=exact_color,
        linewidth=1.7,
        label="Exact QFT",
    )
    axes[0].step(
        offsets,
        approx_binned,
        where="mid",
        color=approx_color,
        linewidth=1.7,
        label="Degree-4 approx. QFT",
    )
    axes[0].axhline(
        representative_C,
        color=supply_color,
        linestyle="--",
        linewidth=1.0,
        label=r"Delivery capacity $C=2$",
    )

    exact_peak_bin_x = int(offsets[np.argmax(exact_binned)])
    approx_peak_bin_x = int(offsets[np.argmax(approx_binned)])

    axes[0].annotate(
        "peak = 10",
        xy=(exact_peak_bin_x, exact_binned.max()),
        xytext=(8, 4),
        textcoords="offset points",
        fontsize=8.5,
        color=exact_color,
        fontweight="bold",
    )
    axes[0].annotate(
        "peak = 6",
        xy=(approx_peak_bin_x, approx_binned.max()),
        xytext=(6, 4),
        textcoords="offset points",
        fontsize=8.5,
        color=approx_color,
        fontweight="bold",
    )

    axes[0].set_xlabel("Logical time-step offset around peak", fontsize=9)
    axes[0].set_ylabel("Max T-gates per bin", fontsize=9)
    axes[0].set_xlim(offsets.min() - 10, offsets.max() + 10)
    axes[0].set_ylim(0, 10.8)

    # Place legend below the axes (stacked) to avoid overlapping the spike curves.
    axes[0].legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.24),
        frameon=False,
        fontsize=7.5,
        ncol=1,
        handlelength=1.8,
        handletextpad=0.45,
        borderaxespad=0.0,
        labelspacing=0.35,
    )

    _apply_paper_axis_style(axes[0])
    axes[0].tick_params(axis="both", labelsize=8)

    # Cumulative demand view.
    x_full = np.arange(1, len(exact_trace) + 1)
    exact_cumulative = np.cumsum(exact_trace)
    approx_cumulative = np.cumsum(approx_trace)
    supply = representative_C * x_full + representative_B

    x_plot, exact_plot = _downsample_series(x_full, exact_cumulative, max_points=1800)
    _, approx_plot = _downsample_series(x_full, approx_cumulative, max_points=1800)
    _, supply_plot = _downsample_series(x_full, supply, max_points=1800)

    axes[1].plot(
        x_plot,
        exact_plot,
        color=exact_color,
        linewidth=1.5,
        label="Exact cumulative demand",
    )
    axes[1].plot(
        x_plot,
        approx_plot,
        color=approx_color,
        linewidth=1.5,
        label="Approx. cumulative demand",
    )
    axes[1].plot(
        x_plot,
        supply_plot,
        color=supply_color,
        linewidth=1.1,
        linestyle="--",
        label=r"Supply envelope $(C=2,B=8)$",
    )

    axes[1].set_xlabel("Logical cycle", fontsize=9)
    axes[1].set_ylabel("Cumulative T demand", fontsize=9)

    axes[1].legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.24),
        frameon=False,
        fontsize=7.5,
        ncol=1,
        handlelength=1.8,
        handletextpad=0.45,
        borderaxespad=0.0,
        labelspacing=0.35,
    )

    # Compact summary box; keep only the key mechanism values.
    axes[1].text(
        0.98,
        0.055,
        "Depth unchanged: 932,784\n"
        "Peak: 10 \u2192 6\n"
        r"Mean $\Delta_{\max}$: 567,516 $\rightarrow$ 309,644",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        bbox={
            "facecolor": "white",
            "alpha": 0.78,
            "edgecolor": "#d0d0d0",
            "linewidth": 0.5,
            "boxstyle": "round,pad=0.18",
        },
    )

    _apply_paper_axis_style(axes[1])
    axes[1].tick_params(axis="both", labelsize=8)

    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.36, top=0.96, wspace=0.28)

    return _save_final_figure(figure, "qft_approximation_reduced_grid_final")


def _binned_peak_window(trace: np.ndarray, peak_index: int, half_window: int, bin_width: int) -> tuple[np.ndarray, np.ndarray]:
    start = max(0, peak_index - half_window)
    end = min(len(trace), peak_index + half_window + 1)
    window_values = trace[start:end]
    window_offsets = np.arange(start, end) - peak_index
    bin_offsets: list[int] = []
    bin_values: list[int] = []
    for bin_start in range(0, len(window_values), bin_width):
        bin_end = min(len(window_values), bin_start + bin_width)
        bin_slice = window_values[bin_start:bin_end]
        offset_slice = window_offsets[bin_start:bin_end]
        bin_offsets.append(int(np.round(offset_slice.mean())))
        bin_values.append(int(bin_slice.max()))
    return np.array(bin_offsets, dtype=int), np.array(bin_values, dtype=int)


def _downsample_series(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    indices = np.linspace(0, len(x) - 1, max_points).astype(int)
    return x[indices], y[indices]


def _save_final_figure(figure: plt.Figure, stem: str) -> Path:
    FINAL_PAPER_DIR.mkdir(parents=True, exist_ok=True)
    path = FINAL_PAPER_DIR / f"{stem}.png"
    figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(FINAL_PAPER_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    return path


if __name__ == "__main__":
    main()
