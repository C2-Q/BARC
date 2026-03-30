from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import run_all_experiments, summarize_results
from src.paper_config import B_VALUES, C_VALUES
from src.plots import generate_all_plots, generate_final_paper_figures_from_tables
from src.real_trace import (
    build_real_trace_note,
    ensure_real_trace_exists,
    evaluate_real_trace_policies,
    export_real_trace_threshold_summary,
    export_real_trace_workload_statistics,
    generate_real_trace_comparison,
    load_real_trace,
    plot_adder_vs_multiplier_capacity_scan,
    plot_adder_vs_multiplier_trace,
    plot_cross_workload_critical_buffer,
    plot_cross_workload_critical_capacity,
    plot_cross_workload_improvement_gap,
    plot_real_trace_buffer_transition,
    plot_real_trace_capacity_scan,
    plot_real_trace_demand,
    summarize_real_trace,
)
from src.utils import OUTPUT_ROOT, TABLE_DIR, ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen QCE paper experiment package.")
    parser.add_argument(
        "--real-trace",
        type=str,
        default="",
        help="Optional path to a CSV or JSON demand trace. If an adder_n*.csv or multiplier_n*.csv path is missing, it will be generated.",
    )
    return parser.parse_args()


def _infer_logical_qubit_budget_from_trace_name(trace_path: Path) -> int:
    if trace_path.stem.startswith("adder_n"):
        try:
            n_bits = int(trace_path.stem.split("adder_n", maxsplit=1)[1])
            return 2 * n_bits + 2
        except ValueError:
            pass
    if trace_path.stem.startswith("multiplier_n"):
        try:
            n_bits = int(trace_path.stem.split("multiplier_n", maxsplit=1)[1])
            return 4 * n_bits
        except ValueError:
            pass
    if trace_path.stem.startswith("qft_n"):
        try:
            n_bits = int(trace_path.stem.split("qft_n", maxsplit=1)[1])
            return n_bits
        except ValueError:
            pass
    return max(1, len(trace_path.stem))


def _infer_n_bits_from_trace_name(trace_path: Path) -> int:
    stem = trace_path.stem
    for prefix in ("adder_n", "multiplier_n", "qft_n"):
        if stem.startswith(prefix):
            try:
                return int(stem.split(prefix, maxsplit=1)[1])
            except ValueError:
                return 0
    return 0


def _generate_real_trace_suite(trace_path: Path) -> tuple[list[int], pd.DataFrame, pd.DataFrame]:
    trace = load_real_trace(str(trace_path))
    real_trace_dir = OUTPUT_ROOT / "real_trace"
    real_trace_dir.mkdir(parents=True, exist_ok=True)
    logical_qubit_budget = _infer_logical_qubit_budget_from_trace_name(trace_path)
    trace_results = evaluate_real_trace_policies(
        trace=trace,
        trace_name=trace_path.stem,
        logical_qubit_budget=logical_qubit_budget,
        C_values=C_VALUES,
        B_values=B_VALUES,
    )
    summary_df = summarize_real_trace(trace_results)

    trace_prefix = trace_path.stem
    results_path = real_trace_dir / f"{trace_prefix}_results.csv"
    summary_path = real_trace_dir / f"{trace_prefix}_summary.csv"
    workload_stats_path = TABLE_DIR / f"{trace_prefix}_workload_stats.csv"
    threshold_path = TABLE_DIR / f"{trace_prefix}_threshold_summary.csv"

    trace_results.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    export_real_trace_workload_statistics(trace, trace_prefix, workload_stats_path)
    export_real_trace_threshold_summary(trace_results, threshold_path)

    trace_plot_path = plot_real_trace_demand(trace, real_trace_dir / f"{trace_prefix}_trace_plot.png")
    capacity_plot_path = plot_real_trace_capacity_scan(
        trace_results,
        real_trace_dir / f"{trace_prefix}_capacity_scan.png",
    )
    buffer_plot_path = plot_real_trace_buffer_transition(
        trace_results,
        real_trace_dir / f"{trace_prefix}_buffer_transition.png",
    )

    print(f"real_trace_compressibility: {summary_df['real_trace_compressibility'].iloc[0]:.3f}")
    print(f"{trace_prefix}_results: {len(trace_results)} rows")
    print(summary_df.head().to_string(index=False))
    print(f"saved: {results_path}")
    print(f"saved: {summary_path}")
    print(f"saved: {workload_stats_path}")
    print(f"saved: {threshold_path}")
    print(f"saved: {trace_plot_path}")
    print(f"saved: {capacity_plot_path}")
    print(f"saved: {buffer_plot_path}")
    return trace, trace_results, summary_df


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    results = run_all_experiments()
    summary = summarize_results(results)
    figures = generate_all_plots(results)
    final_figures = generate_final_paper_figures_from_tables()

    print("QCE paper experiment summary")
    print("=" * 32)
    for name in (
        "family_summary",
        "shallower_slower_summary",
        "buffer_threshold_summary",
    ):
        dataframe = results[name]
        print(f"{name}: {len(dataframe)} rows")
        print(dataframe.to_string(index=False))
        print("-" * 32)

    print("Summary")
    print("-" * 32)
    print(
        "representative relative improvement:",
        f"{summary['representative_relative_improvement']:.3f}",
    )
    print("compressibility family means:")
    for family, value in summary["compressibility_family_means"].items():
        print(f" - {family}: {value:.3f}")
    print("stall-regime family means:")
    for family, value in summary["stall_regime_family_means"].items():
        print(f" - {family}: {value:.3f}")
    print("inversion cases:", summary["inversion_count"])
    predictive_summary_path = TABLE_DIR / "predictive_analysis_summary.txt"
    reframing_summary_path = TABLE_DIR / "paper1_reframing_summary.txt"
    print(f"predictive summary: {predictive_summary_path}")
    print(f"paper reframing summary: {reframing_summary_path}")
    print("-" * 32)

    if args.real_trace:
        trace_path = ensure_real_trace_exists(args.real_trace)
        trace, trace_results, summary_df = _generate_real_trace_suite(trace_path)
        summary_df.to_csv(TABLE_DIR / "real_trace_summary.csv", index=False)
        note_path = build_real_trace_note(OUTPUT_ROOT / "real_trace" / "real_trace_figure_note.md")
        print(f"saved: {TABLE_DIR / 'real_trace_summary.csv'}")
        print(f"saved: {note_path}")

        if trace_path.stem.startswith("multiplier_n"):
            n_bits = _infer_n_bits_from_trace_name(trace_path)
            adder_trace_path = ensure_real_trace_exists(
                str(PROJECT_ROOT / "data" / "real_traces" / f"adder_n{max(4, n_bits)}.csv")
            )
            adder_trace, adder_results, adder_summary = _generate_real_trace_suite(adder_trace_path)
            comparison_df = generate_real_trace_comparison(
                adder_summary_df=adder_summary,
                multiplier_summary_df=summary_df,
                adder_results_df=adder_results,
                multiplier_results_df=trace_results,
            )
            comparison_path = TABLE_DIR / "real_trace_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            comparison_trace_plot = plot_adder_vs_multiplier_trace(
                adder_trace,
                trace,
                OUTPUT_ROOT / "figures" / "adder_vs_multiplier_trace.png",
            )
            comparison_capacity_plot = plot_adder_vs_multiplier_capacity_scan(
                adder_results,
                trace_results,
                OUTPUT_ROOT / "figures" / "adder_vs_multiplier_capacity_scan.png",
            )
            adder_thresholds = export_real_trace_threshold_summary(adder_results, TABLE_DIR / "adder_comparison_threshold_summary.csv")
            multiplier_thresholds = export_real_trace_threshold_summary(trace_results, TABLE_DIR / "multiplier_comparison_threshold_summary.csv")
            threshold_df = pd.concat([adder_thresholds, multiplier_thresholds], ignore_index=True)
            threshold_summary_path = TABLE_DIR / "real_trace_threshold_summary.csv"
            threshold_df.to_csv(threshold_summary_path, index=False)
            critical_capacity_plot = plot_cross_workload_critical_capacity(
                threshold_df,
                OUTPUT_ROOT / "figures" / "cross_workload_critical_capacity.png",
            )
            critical_buffer_plot = plot_cross_workload_critical_buffer(
                threshold_df,
                OUTPUT_ROOT / "figures" / "cross_workload_critical_buffer.png",
            )
            improvement_gap_plot = plot_cross_workload_improvement_gap(
                comparison_df,
                OUTPUT_ROOT / "figures" / "cross_workload_improvement_gap.png",
            )
            print(f"saved: {comparison_path}")
            print(f"saved: {threshold_summary_path}")
            print(f"saved: {comparison_trace_plot}")
            print(f"saved: {comparison_capacity_plot}")
            print(f"saved: {critical_capacity_plot}")
            print(f"saved: {critical_buffer_plot}")
            print(f"saved: {improvement_gap_plot}")
        print("-" * 32)

    print("Generated figures:")
    for figure_path in figures:
        print(f" - {figure_path}")
    print("Generated final-paper figures:")
    for figure_path in final_figures:
        print(f" - {figure_path}")


if __name__ == "__main__":
    main()
