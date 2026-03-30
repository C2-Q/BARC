from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_config import B_VALUES, C_VALUES
from src.real_trace.adder_trace import (
    extract_t_demand_trace,
    generate_adder_circuit,
    save_trace,
    to_clifford_t,
)
from src.real_trace.analysis import evaluate_real_trace_policies, summarize_real_trace
from src.real_trace.circuit_slack import compute_circuit_slack_metrics
from src.real_trace.multiplier_trace import generate_multiplier_circuit
from src.utils import FIGURE_DIR, TABLE_DIR, ensure_output_dirs


ADDER_SIZES = [4, 6, 8, 12, 16]
MULTIPLIER_SIZES = [4, 5, 6, 7, 8, 12, 16]
STATIC_COLOR = "#1f77b4"
MULTIPLIER_COLOR = "#d62728"


def main() -> None:
    ensure_output_dirs()
    family_summary_df = pd.read_csv(TABLE_DIR / "family_summary.csv")

    rows: list[dict[str, object]] = []
    for trace_family, sizes, circuit_builder in (
        ("adder", ADDER_SIZES, generate_adder_circuit),
        ("multiplier", MULTIPLIER_SIZES, generate_multiplier_circuit),
    ):
        for n_bits in sizes:
            trace_name = f"{trace_family}_n{n_bits}"
            print(f"processing {trace_name}")
            circuit = circuit_builder(n_bits=n_bits)
            clifford_t = to_clifford_t(circuit)
            trace = extract_t_demand_trace(clifford_t)
            save_trace(trace, PROJECT_ROOT / "data" / "real_traces" / f"{trace_name}.csv")

            slack_metrics = compute_circuit_slack_metrics(clifford_t)
            results_df = evaluate_real_trace_policies(
                trace=trace,
                trace_name=trace_name,
                logical_qubit_budget=_infer_logical_qubit_budget(trace_family, n_bits),
                C_values=C_VALUES,
                B_values=B_VALUES,
            )
            summary_df = summarize_real_trace(results_df)
            static_rows = results_df[results_df["policy"] == "static_min"].copy()
            static_unique = static_rows.drop_duplicates(subset=["C"])
            slowdown_fraction = float(
                ((static_rows["feasible"] == 1) & (static_rows["normalized_makespan"] > 1.05)).mean()
            )
            stall_fraction = float(
                ((static_rows["feasible"] == 0) | (static_rows["stall_cycles"] > 0)).mean()
            )
            rows.append(
                {
                    "trace_name": trace_name,
                    "trace_family": trace_family,
                    "n_bits": n_bits,
                    "T_count": int(sum(trace)),
                    "T_depth": int(len(trace)),
                    "slack_ratio": float(slack_metrics.fraction_t_slack_positive),
                    "mean_t_slack": float(slack_metrics.mean_t_slack),
                    "peak_demand": int(max(trace, default=0)),
                    "delta_max_mean": float(static_unique["Delta_max"].mean()),
                    "delta_max_max": float(static_unique["Delta_max"].max()),
                    "frac_CB_with_slowdown": slowdown_fraction,
                    "frac_CB_with_stall": stall_fraction,
                    "closest_synthetic_family": _closest_synthetic_family(
                        slack_metrics.fraction_t_slack_positive,
                        family_summary_df,
                    ),
                }
            )

    summary_path = TABLE_DIR / "real_trace_scaling_summary.csv"
    summary_df = pd.DataFrame(rows).sort_values(by=["trace_family", "n_bits"]).reset_index(drop=True)
    summary_df.to_csv(summary_path, index=False)

    figure_path = plot_scaling(summary_df)
    interpretation_path = TABLE_DIR / "real_trace_scaling_interpretation.txt"
    interpretation_path.write_text(build_interpretation(summary_df), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"saved: {summary_path}")
    print(f"saved: {figure_path}")
    print(f"saved: {interpretation_path}")


def plot_scaling(summary_df: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    for trace_family, color in (("adder", STATIC_COLOR), ("multiplier", MULTIPLIER_COLOR)):
        subset = summary_df[summary_df["trace_family"] == trace_family].sort_values("n_bits")
        axes[0].plot(
            subset["n_bits"],
            subset["slack_ratio"],
            marker="o",
            color=color,
            linewidth=1.6,
            label=trace_family,
        )
        axes[1].plot(
            subset["n_bits"],
            subset["delta_max_mean"],
            marker="o",
            color=color,
            linewidth=1.6,
            label=trace_family,
        )

    axes[0].set_xlabel("n_bits")
    axes[0].set_ylabel("slack_ratio")
    axes[0].set_title("Real Trace Scaling: Slack")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_xlabel("n_bits")
    axes[1].set_ylabel("mean delta_max")
    axes[1].set_title("Real Trace Scaling: Delivery Pressure")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    path = FIGURE_DIR / "real_trace_scaling.png"
    figure.savefig(path, dpi=300)
    figure.savefig(FIGURE_DIR / "real_scaling.png", dpi=300)
    plt.close(figure)
    return path


def build_interpretation(summary_df: pd.DataFrame) -> str:
    adder = summary_df[summary_df["trace_family"] == "adder"].sort_values("n_bits")
    multiplier = summary_df[summary_df["trace_family"] == "multiplier"].sort_values("n_bits")
    adder_start = adder.iloc[0]
    adder_end = adder.iloc[-1]
    multiplier_start = multiplier.iloc[0]
    multiplier_end = multiplier.iloc[-1]
    return (
        f"Across the scanned adder sizes, slack_ratio changes from {adder_start['slack_ratio']:.3f} at n={int(adder_start['n_bits'])} "
        f"to {adder_end['slack_ratio']:.3f} at n={int(adder_end['n_bits'])}, while mean delta_max changes from "
        f"{adder_start['delta_max_mean']:.3f} to {adder_end['delta_max_mean']:.3f}. The fraction of scanned (C,B) settings with stall changes from "
        f"{adder_start['frac_CB_with_stall']:.3f} to {adder_end['frac_CB_with_stall']:.3f}. The observed adder traces remain in the "
        f"{adder_end['closest_synthetic_family']} range under the current slack-based comparison.\n"
        f"Across the scanned multiplier sizes, slack_ratio changes from {multiplier_start['slack_ratio']:.3f} at n={int(multiplier_start['n_bits'])} "
        f"to {multiplier_end['slack_ratio']:.3f} at n={int(multiplier_end['n_bits'])}, while mean delta_max changes from "
        f"{multiplier_start['delta_max_mean']:.3f} to {multiplier_end['delta_max_mean']:.3f}. The fraction of scanned (C,B) settings with stall changes from "
        f"{multiplier_start['frac_CB_with_stall']:.3f} to {multiplier_end['frac_CB_with_stall']:.3f}. The multiplier traces show larger delivery pressure than the adder traces, "
        f"but they remain below the synthetic high-slack regime in this scan.\n"
    )


def _closest_synthetic_family(slack_ratio: float, family_summary_df: pd.DataFrame) -> str:
    static_summary = family_summary_df[family_summary_df["policy"] == "static_min"][["family", "mean_slack_ratio"]].copy()
    distances = (static_summary["mean_slack_ratio"] - slack_ratio).abs()
    return str(static_summary.iloc[int(distances.argmin())]["family"])


def _infer_logical_qubit_budget(trace_family: str, n_bits: int) -> int:
    if trace_family == "adder":
        return 2 * n_bits + 2
    if trace_family == "multiplier":
        return 4 * n_bits
    raise ValueError(f"unsupported trace family: {trace_family}")


if __name__ == "__main__":
    main()
