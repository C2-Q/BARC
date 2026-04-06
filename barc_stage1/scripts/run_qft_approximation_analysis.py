from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from qiskit.converters import circuit_to_dag

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.real_trace import evaluate_real_trace_policies, load_real_trace_from_csv, summarize_real_trace
from src.real_trace.circuit_slack import compute_circuit_slack_metrics
from src.real_trace.qft_trace import generate_qft_circuit
from src.real_trace.adder_trace import extract_t_demand_trace, to_clifford_t
from src.utils import FIGURE_DIR, TABLE_DIR, ensure_output_dirs


EXACT_SIZES = [8, 12, 16]
APPROX_N_BITS = 12
APPROXIMATION_DEGREE = 4
TRACE_LIMIT = 600_000
T_COUNT_LIMIT = 1_000_000
REDUCED_C_VALUES = [1, 2, 3]
REDUCED_B_VALUES = [0, 4, 8, 12]


def main() -> None:
    ensure_output_dirs()
    family_summary_df = pd.read_csv(TABLE_DIR / "family_summary.csv")
    qft_summary_df = pd.read_csv(TABLE_DIR / "qft_real_trace_summary.csv")

    complexity_rows = _build_exact_complexity_rows(qft_summary_df)

    exact_row = _build_comparison_row(
        n_bits=APPROX_N_BITS,
        approximation_degree=0,
        family_summary_df=family_summary_df,
        label="exact",
        precomputed_row=qft_summary_df[qft_summary_df["n_bits"] == APPROX_N_BITS].iloc[0].to_dict(),
    )
    approx_row = _build_comparison_row(
        n_bits=APPROX_N_BITS,
        approximation_degree=APPROXIMATION_DEGREE,
        family_summary_df=family_summary_df,
        label=f"approx_deg_{APPROXIMATION_DEGREE}",
    )

    complexity_df = pd.DataFrame(complexity_rows)
    complexity_path = TABLE_DIR / "qft_complexity_scaling.csv"
    complexity_df.to_csv(complexity_path, index=False)

    comparison_df = pd.DataFrame([exact_row, approx_row])
    comparison_path = TABLE_DIR / "qft_approximation_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    figure_path = plot_qft_approximation_comparison(comparison_df)
    reduced_grid_df, reduced_grid_summary_df = run_reduced_grid_analysis()
    reduced_grid_path = TABLE_DIR / "qft_approximation_reduced_grid.csv"
    reduced_grid_summary_path = TABLE_DIR / "qft_approximation_reduced_grid_summary.csv"
    reduced_grid_df.to_csv(reduced_grid_path, index=False)
    reduced_grid_summary_df.to_csv(reduced_grid_summary_path, index=False)
    reduced_grid_figure_path = plot_qft_reduced_grid_summary(reduced_grid_summary_df)

    print(complexity_df.to_string(index=False))
    print(comparison_df.to_string(index=False))
    print(reduced_grid_summary_df.to_string(index=False))
    print(f"saved: {complexity_path}")
    print(f"saved: {comparison_path}")
    print(f"saved: {figure_path}")
    print(f"saved: {reduced_grid_path}")
    print(f"saved: {reduced_grid_summary_path}")
    print(f"saved: {reduced_grid_figure_path}")


def _build_complexity_row(n_bits: int, approximation_degree: int) -> dict[str, object]:
    circuit = generate_qft_circuit(n_bits=n_bits, do_swaps=False, approximation_degree=approximation_degree)
    clifford_t = to_clifford_t(circuit)
    trace = extract_t_demand_trace(clifford_t)
    slack = compute_circuit_slack_metrics(clifford_t)
    return {
        "trace_name": f"qft_n{n_bits}",
        "n_bits": n_bits,
        "approximation_degree": approximation_degree,
        "T_count": int(sum(trace)),
        "dag_depth": int(circuit_to_dag(clifford_t).depth()),
        "trace_length": int(len(trace)),
        "peak_demand": int(max(trace, default=0)),
        "slack_ratio": float(slack.fraction_t_slack_positive),
        "mean_t_slack": float(slack.mean_t_slack),
    }


def _build_exact_complexity_rows(qft_summary_df: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_bits in EXACT_SIZES:
        base_row = qft_summary_df[qft_summary_df["n_bits"] == n_bits].iloc[0]
        rows.append(
            {
                "trace_name": str(base_row["trace_name"]),
                "n_bits": int(base_row["n_bits"]),
                "approximation_degree": 0,
                "T_count": int(base_row["T_count"]),
                "dag_depth": int(base_row["T_depth"]),
                "trace_length": int(base_row["trace_length"]),
                "peak_demand": int(base_row["peak_demand"]),
                "slack_ratio": float(base_row["slack_ratio"]),
                "mean_t_slack": float(base_row["mean_t_slack"]),
            }
        )
    return rows


def _build_comparison_row(
    n_bits: int,
    approximation_degree: int,
    family_summary_df: pd.DataFrame,
    label: str,
    precomputed_row: dict[str, object] | None = None,
) -> dict[str, object]:
    if precomputed_row is None:
        complexity = _build_complexity_row(n_bits=n_bits, approximation_degree=approximation_degree)
    else:
        complexity = {
            "trace_name": str(precomputed_row["trace_name"]),
            "n_bits": int(precomputed_row["n_bits"]),
            "approximation_degree": int(approximation_degree),
            "T_count": int(precomputed_row["T_count"]),
            "dag_depth": int(precomputed_row["T_depth"]),
            "trace_length": int(precomputed_row["trace_length"]),
            "peak_demand": int(precomputed_row["peak_demand"]),
            "slack_ratio": float(precomputed_row["slack_ratio"]),
            "mean_t_slack": float(precomputed_row["mean_t_slack"]),
        }
    row = {
        "variant": label,
        **complexity,
        "status": "trace_only",
        "delta_max_mean": float("nan"),
        "delta_max_max": float("nan"),
        "frac_CB_with_slowdown": float("nan"),
        "frac_CB_with_stall": float("nan"),
        "inversion_observed": 0,
        "closest_synthetic_family": _closest_synthetic_family(float(complexity["slack_ratio"]), family_summary_df),
    }
    if int(complexity["trace_length"]) <= TRACE_LIMIT and int(complexity["T_count"]) <= T_COUNT_LIMIT:
        circuit = generate_qft_circuit(n_bits=n_bits, do_swaps=False, approximation_degree=approximation_degree)
        clifford_t = to_clifford_t(circuit)
        trace = extract_t_demand_trace(clifford_t)
        results_df = evaluate_real_trace_policies(
            trace=trace,
            trace_name=f"qft_n{n_bits}_{label}",
            logical_qubit_budget=n_bits,
            C_values=C_VALUES,
            B_values=B_VALUES,
        )
        summary_df = summarize_real_trace(results_df)
        static_rows = results_df[results_df["policy"] == "static_min"].copy()
        static_unique = static_rows.drop_duplicates(subset=["C"])
        row.update(
            {
                "status": "full_eval",
                "delta_max_mean": float(static_unique["Delta_max"].mean()),
                "delta_max_max": float(static_unique["Delta_max"].max()),
                "frac_CB_with_slowdown": float(
                    ((static_rows["feasible"] == 1) & (static_rows["normalized_makespan"] > 1.05)).mean()
                ),
                "frac_CB_with_stall": float(
                    ((static_rows["feasible"] == 0) | (static_rows["stall_cycles"] > 0)).mean()
                ),
                "inversion_observed": int((summary_df["inversion"] == 1).any()),
            }
        )
    return row


def _closest_synthetic_family(slack_ratio: float, family_summary_df: pd.DataFrame) -> str:
    static_summary = family_summary_df[family_summary_df["policy"] == "static_min"][["family", "mean_slack_ratio"]].copy()
    distances = (static_summary["mean_slack_ratio"] - slack_ratio).abs()
    return str(static_summary.iloc[int(distances.argmin())]["family"])


def run_reduced_grid_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    exact_trace = load_real_trace_from_csv(str(PROJECT_ROOT / "data" / "real_traces" / "qft_n12.csv"))
    approx_trace = extract_t_demand_trace(
        to_clifford_t(generate_qft_circuit(APPROX_N_BITS, do_swaps=False, approximation_degree=APPROXIMATION_DEGREE))
    )
    traces = {
        "exact": exact_trace,
        f"approx_deg_{APPROXIMATION_DEGREE}": approx_trace,
    }

    rows: list[pd.DataFrame] = []
    for variant, trace in traces.items():
        results_df = evaluate_real_trace_policies(
            trace=trace,
            trace_name=f"qft_n{APPROX_N_BITS}_{variant}",
            logical_qubit_budget=APPROX_N_BITS,
            C_values=REDUCED_C_VALUES,
            B_values=REDUCED_B_VALUES,
        ).copy()
        results_df["variant"] = variant
        rows.append(results_df)
    reduced_grid_df = pd.concat(rows, ignore_index=True)

    static_df = reduced_grid_df[reduced_grid_df["policy"] == "static_min"].copy()
    summary_rows: list[dict[str, object]] = []
    for variant in sorted(static_df["variant"].unique()):
        subset = static_df[static_df["variant"] == variant]
        summary_rows.append(
            {
                "variant": variant,
                "mean_delta_max": float(subset["Delta_max"].mean()),
                "max_delta_max": float(subset["Delta_max"].max()),
                "mean_slowdown_ratio": float(subset.loc[subset["feasible"] == 1, "normalized_makespan"].mean()),
                "frac_with_slowdown": float(
                    ((subset["feasible"] == 1) & (subset["normalized_makespan"] > 1.05)).mean()
                ),
                "frac_with_stall": float(
                    ((subset["feasible"] == 0) | (subset["stall_cycles"] > 0)).mean()
                ),
            }
        )
        for C in REDUCED_C_VALUES:
            c_subset = subset[subset["C"] == C]
            summary_rows.append(
                {
                    "variant": variant,
                    "mean_delta_max": float(c_subset["Delta_max"].mean()),
                    "max_delta_max": float(c_subset["Delta_max"].max()),
                    "mean_slowdown_ratio": float(c_subset.loc[c_subset["feasible"] == 1, "normalized_makespan"].mean()),
                    "frac_with_slowdown": float(
                        ((c_subset["feasible"] == 1) & (c_subset["normalized_makespan"] > 1.05)).mean()
                    ),
                    "frac_with_stall": float(
                        ((c_subset["feasible"] == 0) | (c_subset["stall_cycles"] > 0)).mean()
                    ),
                    "C": C,
                }
            )
    reduced_grid_summary_df = pd.DataFrame(summary_rows)
    return reduced_grid_df, reduced_grid_summary_df


def plot_qft_approximation_comparison(comparison_df: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))
    variant_order = comparison_df["variant"].tolist()
    x_positions = range(len(variant_order))
    colors = ["#4c78a8", "#f58518"]

    metric_specs = [
        ("slack_ratio", "Slack ratio"),
        ("T_count", "T-count"),
        ("dag_depth", "DAG depth"),
    ]
    for axis, (column, ylabel) in zip(axes, metric_specs, strict=True):
        values = comparison_df[column].tolist()
        axis.bar(x_positions, values, color=colors[: len(values)], width=0.62)
        for index, value in enumerate(values):
            if value >= 1000:
                text = f"{int(value):,}"
            else:
                text = f"{value:.2f}" if isinstance(value, float) else str(value)
            axis.text(index, value * 1.01 + (0.02 if value < 5 else 0), text, ha="center", va="bottom", fontsize=8)
        axis.set_xticks(list(x_positions), variant_order)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.18)
    figure.tight_layout()
    path = FIGURE_DIR / "qft_approximation_comparison.png"
    figure.savefig(path, dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return path


def plot_qft_reduced_grid_summary(summary_df: pd.DataFrame) -> Path:
    overall = summary_df[summary_df.get("C").isna()].copy() if "C" in summary_df.columns else summary_df.copy()
    per_c = summary_df[summary_df.get("C").notna()].copy() if "C" in summary_df.columns else pd.DataFrame()

    figure, axes = plt.subplots(1, 3, figsize=(10.6, 3.4))
    variant_label_map = {
        "approx_deg_4": "Degree-4 approx.",
        "exact": "Exact",
    }
    variants = [variant_label_map.get(variant, str(variant)) for variant in overall["variant"].tolist()]
    colors = ["#4c78a8", "#f58518"]

    metric_specs = [
        ("mean_delta_max", r"Mean $\Delta_{\max}$"),
        ("frac_with_slowdown", "Fraction with >5% slowdown"),
        ("frac_with_stall", "Fraction with stall"),
    ]
    for axis, (column, ylabel) in zip(axes, metric_specs, strict=True):
        values = overall[column].tolist()
        axis.bar(range(len(variants)), values, color=colors[: len(values)], width=0.62)
        for index, value in enumerate(values):
            if column == "mean_delta_max":
                label_offset = max(0.02 * max(values, default=1.0), 5000.0)
                axis.text(index, value + label_offset, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
                axis.set_ylim(0.0, max(values, default=1.0) * 1.18)
            else:
                label_offset = 0.025 if value < 0.95 else 0.015
                axis.text(index, value + label_offset, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
                axis.set_ylim(0.0, 1.08)
        axis.set_xticks(range(len(variants)), variants)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.18)
    figure.tight_layout()
    path = FIGURE_DIR / "qft_approximation_reduced_grid.png"
    figure.savefig(path, dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return path


if __name__ == "__main__":
    main()
