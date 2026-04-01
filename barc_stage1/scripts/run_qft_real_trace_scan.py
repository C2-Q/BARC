from __future__ import annotations

import math
import shutil
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_config import B_VALUES, C_VALUES
from src.real_trace import (
    evaluate_real_trace_policies,
    export_real_trace_threshold_summary,
    export_real_trace_workload_statistics,
    plot_real_trace_capacity_scan,
    plot_real_trace_demand,
    summarize_real_trace,
)
from src.real_trace.adder_trace import extract_t_demand_trace, save_trace, to_clifford_t
from src.real_trace.circuit_slack import CircuitSlackMetrics, compute_circuit_slack_metrics
from src.real_trace.qft_trace import generate_qft_circuit
from src.utils import FIGURE_DIR, OUTPUT_ROOT, TABLE_DIR, ensure_output_dirs


QFT_SIZES = [8, 12, 16]
TRACE_LENGTH_LIMIT = 600_000
T_COUNT_LIMIT = 1_000_000
GENERATION_TIMEOUT_SECONDS = 180
QFT_COLOR = "#6a3d9a"
ADDER_COLOR = "#4c78a8"
MULTIPLIER_COLOR = "#e45756"


class QftGenerationTimeoutError(TimeoutError):
    """Raised when exact QFT generation exceeds the configured timeout."""


@dataclass(frozen=True)
class QftTraceArtifacts:
    n_bits: int
    trace: list[int]
    trace_length: int
    peak_demand: int
    t_count: int
    clifford_t_circuit: object
    slack_metrics: CircuitSlackMetrics


def main() -> None:
    ensure_output_dirs()
    family_summary_df = pd.read_csv(TABLE_DIR / "family_summary.csv")
    scaling_summary_df = pd.read_csv(TABLE_DIR / "real_trace_scaling_summary.csv")

    rows: list[dict[str, object]] = []
    representative_trace_path: Path | None = None
    representative_capacity_path: Path | None = None
    representative_trace_name: str | None = None

    for n_bits in QFT_SIZES:
        trace_name = f"qft_n{n_bits}"
        print(f"processing {trace_name}")
        try:
            artifacts = _generate_qft_artifacts_with_timeout(n_bits)
        except QftGenerationTimeoutError:
            rows.append(
                {
                    "trace_name": trace_name,
                    "n_bits": n_bits,
                    "status": "generation_timeout",
                    "reason": "exact_qft_generation_timeout",
                    "T_count": math.nan,
                    "T_depth": math.nan,
                    "trace_length": math.nan,
                    "peak_demand": math.nan,
                    "slack_ratio": math.nan,
                    "mean_t_slack": math.nan,
                    "delta_max_mean": math.nan,
                    "delta_max_max": math.nan,
                    "frac_CB_with_slowdown": math.nan,
                    "frac_CB_with_stall": math.nan,
                    "inversion_observed": 0,
                    "closest_synthetic_family": "unknown",
                }
            )
            continue

        trace_path = PROJECT_ROOT / "data" / "real_traces" / f"{trace_name}.csv"
        save_trace(artifacts.trace, trace_path)

        row = {
            "trace_name": trace_name,
            "n_bits": n_bits,
            "T_count": artifacts.t_count,
            "T_depth": artifacts.trace_length,
            "trace_length": artifacts.trace_length,
            "peak_demand": artifacts.peak_demand,
            "slack_ratio": float(artifacts.slack_metrics.fraction_t_slack_positive),
            "mean_t_slack": float(artifacts.slack_metrics.mean_t_slack),
            "closest_synthetic_family": _closest_synthetic_family(
                float(artifacts.slack_metrics.fraction_t_slack_positive),
                family_summary_df,
            ),
        }

        if artifacts.trace_length <= TRACE_LENGTH_LIMIT and artifacts.t_count <= T_COUNT_LIMIT:
            results_df = evaluate_real_trace_policies(
                trace=artifacts.trace,
                trace_name=trace_name,
                logical_qubit_budget=n_bits,
                C_values=C_VALUES,
                B_values=B_VALUES,
            )
            summary_df = summarize_real_trace(results_df)
            results_path = OUTPUT_ROOT / "real_trace" / f"{trace_name}_results.csv"
            summary_path = OUTPUT_ROOT / "real_trace" / f"{trace_name}_summary.csv"
            results_df.to_csv(results_path, index=False)
            summary_df.to_csv(summary_path, index=False)
            export_real_trace_workload_statistics(artifacts.trace, trace_name, TABLE_DIR / f"{trace_name}_workload_stats.csv")
            export_real_trace_threshold_summary(results_df, TABLE_DIR / f"{trace_name}_threshold_summary.csv")

            static_rows = results_df[results_df["policy"] == "static_min"].copy()
            static_unique = static_rows.drop_duplicates(subset=["C"])
            slowdown_fraction = float(
                ((static_rows["feasible"] == 1) & (static_rows["normalized_makespan"] > 1.05)).mean()
            )
            stall_fraction = float(
                ((static_rows["feasible"] == 0) | (static_rows["stall_cycles"] > 0)).mean()
            )

            row.update(
                {
                    "status": "full_eval",
                    "reason": "within_trace_budget",
                    "delta_max_mean": float(static_unique["Delta_max"].mean()),
                    "delta_max_max": float(static_unique["Delta_max"].max()),
                    "frac_CB_with_slowdown": slowdown_fraction,
                    "frac_CB_with_stall": stall_fraction,
                    "inversion_observed": int((summary_df["inversion"] == 1).any()),
                }
            )

            if representative_trace_name is None:
                representative_trace_name = trace_name
                representative_trace_path = plot_real_trace_demand(
                    artifacts.trace,
                    OUTPUT_ROOT / "real_trace" / f"{trace_name}_trace_plot.png",
                )
                representative_capacity_path = plot_real_trace_capacity_scan(
                    results_df,
                    OUTPUT_ROOT / "real_trace" / f"{trace_name}_capacity_scan.png",
                )
        else:
            row.update(
                {
                    "status": "trace_only",
                    "reason": "trace_too_large_for_full_grid",
                    "delta_max_mean": math.nan,
                    "delta_max_max": math.nan,
                    "frac_CB_with_slowdown": math.nan,
                    "frac_CB_with_stall": math.nan,
                    "inversion_observed": 0,
                }
            )
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("n_bits").reset_index(drop=True)
    summary_path = TABLE_DIR / "qft_real_trace_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if representative_trace_path is not None:
        shutil.copy2(representative_trace_path, FIGURE_DIR / "qft_trace_characterization.png")
        shutil.copy2(representative_trace_path.with_suffix(".pdf"), FIGURE_DIR / "qft_trace_characterization.pdf")
    if representative_capacity_path is not None:
        shutil.copy2(representative_capacity_path, FIGURE_DIR / "qft_capacity_scan.png")
        shutil.copy2(representative_capacity_path.with_suffix(".pdf"), FIGURE_DIR / "qft_capacity_scan.pdf")

    comparison_path = plot_qft_vs_other_real_traces(summary_df, scaling_summary_df)
    interpretation = build_qft_interpretation(summary_df)
    interpretation_path = TABLE_DIR / "qft_real_trace_interpretation.txt"
    interpretation_path.write_text(interpretation, encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"saved: {summary_path}")
    print(f"saved: {comparison_path}")
    print(f"saved: {interpretation_path}")


def _generate_qft_artifacts_with_timeout(n_bits: int) -> QftTraceArtifacts:
    previous = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise QftGenerationTimeoutError(f"QFT generation timed out for n={n_bits}")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(GENERATION_TIMEOUT_SECONDS)
    try:
        circuit = generate_qft_circuit(n_bits=n_bits, do_swaps=False)
        clifford_t = to_clifford_t(circuit)
        trace = extract_t_demand_trace(clifford_t)
        slack_metrics = compute_circuit_slack_metrics(clifford_t)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

    return QftTraceArtifacts(
        n_bits=n_bits,
        trace=trace,
        trace_length=len(trace),
        peak_demand=max(trace, default=0),
        t_count=int(sum(trace)),
        clifford_t_circuit=clifford_t,
        slack_metrics=slack_metrics,
    )


def _closest_synthetic_family(slack_ratio: float, family_summary_df: pd.DataFrame) -> str:
    static_summary = family_summary_df[family_summary_df["policy"] == "static_min"][["family", "mean_slack_ratio"]].copy()
    distances = (static_summary["mean_slack_ratio"] - slack_ratio).abs()
    return str(static_summary.iloc[int(distances.argmin())]["family"])


def plot_qft_vs_other_real_traces(qft_summary_df: pd.DataFrame, scaling_summary_df: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    qft_full = qft_summary_df[qft_summary_df["status"] == "full_eval"].copy()
    if qft_full.empty:
        qft_row = qft_summary_df.iloc[0]
    else:
        qft_row = qft_full.sort_values("n_bits").iloc[-1]
    qft_n = int(qft_row["n_bits"])

    adder_row = scaling_summary_df[(scaling_summary_df["trace_family"] == "adder") & (scaling_summary_df["n_bits"] == qft_n)]
    multiplier_row = scaling_summary_df[(scaling_summary_df["trace_family"] == "multiplier") & (scaling_summary_df["n_bits"] == qft_n)]
    if adder_row.empty:
        adder_row = scaling_summary_df[scaling_summary_df["trace_family"] == "adder"].sort_values("n_bits").iloc[[-1]]
    if multiplier_row.empty:
        multiplier_row = scaling_summary_df[scaling_summary_df["trace_family"] == "multiplier"].sort_values("n_bits").iloc[[-1]]

    labels = [
        f"Adder n={int(adder_row.iloc[0]['n_bits'])}",
        f"Multiplier n={int(multiplier_row.iloc[0]['n_bits'])}",
        f"QFT n={qft_n}",
    ]
    slack_values = [
        float(adder_row.iloc[0]["slack_ratio"]),
        float(multiplier_row.iloc[0]["slack_ratio"]),
        float(qft_row["slack_ratio"]) if pd.notna(qft_row["slack_ratio"]) else 0.0,
    ]
    delta_values = [
        float(adder_row.iloc[0]["delta_max_mean"]),
        float(multiplier_row.iloc[0]["delta_max_mean"]),
        float(qft_row["delta_max_mean"]) if pd.notna(qft_row["delta_max_mean"]) else 0.0,
    ]
    colors = [ADDER_COLOR, MULTIPLIER_COLOR, QFT_COLOR]

    axes[0].bar(labels, slack_values, color=colors, width=0.65)
    for index, value in enumerate(slack_values):
        axes[0].text(index, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0.0, max(slack_values, default=1.0) + 0.08)
    axes[0].set_ylabel("Slack ratio")
    axes[0].set_title("Structural flexibility")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.18)

    axes[1].bar(labels, delta_values, color=colors, width=0.65)
    for index, value in enumerate(delta_values):
        label_y = value * 1.08 + max(0.2, 0.02 * max(delta_values, default=1.0))
        axes[1].text(index, label_y, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_ylim(0.0, max(max(delta_values, default=1.0) * 2.1, 10.0))
    axes[1].set_ylabel(r"Mean $\Delta_{\max}$")
    axes[1].set_title("Delivery pressure")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.18)

    figure.tight_layout()
    path = FIGURE_DIR / "qft_vs_other_real_traces.png"
    figure.savefig(path, dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return path


def build_qft_interpretation(summary_df: pd.DataFrame) -> str:
    rows = summary_df.sort_values("n_bits")
    full_eval = rows[rows["status"] == "full_eval"]
    trace_only = rows[rows["status"] == "trace_only"]
    timeout = rows[rows["status"] == "generation_timeout"]

    lines: list[str] = []
    if not full_eval.empty:
        largest = full_eval.iloc[-1]
        inversion_text = "observed" if int(largest["inversion_observed"]) == 1 else "not observed"
        lines.append(
            f"The largest exact QFT instance that completed the full bounded-delivery scan is n={int(largest['n_bits'])}. "
            f"It has slack_ratio={float(largest['slack_ratio']):.3f}, mean_t_slack={float(largest['mean_t_slack']):.3f}, "
            f"delta_max_mean={float(largest['delta_max_mean']):.3f}, and the closest synthetic family is {largest['closest_synthetic_family']}. "
            f"Inversion was {inversion_text} in the scanned (C,B) grid."
        )
    if not trace_only.empty:
        n_bits = ", ".join(str(int(value)) for value in trace_only["n_bits"])
        lines.append(
            f"Exact QFT sizes {n_bits} exceeded the full-grid evaluation budget after trace extraction and were recorded as trace_only. "
            f"This indicates that exact Clifford+T expansion, rather than schedule inversion alone, becomes the dominant practical limitation at larger QFT sizes."
        )
    if not full_eval.empty:
        full_eval_frac = ", ".join(
            f"n={int(row.n_bits)}: slowdown fraction={float(row.frac_CB_with_slowdown):.3f}"
            for row in full_eval.itertuples(index=False)
        )
        lines.append(
            f"Across the fully evaluated QFT sizes, bounded-delivery pressure appears as follows: {full_eval_frac}."
        )
    if not timeout.empty:
        n_bits = ", ".join(str(int(value)) for value in timeout["n_bits"])
        lines.append(f"Generation timed out for exact QFT sizes {n_bits} under the current fixed transpilation settings.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
