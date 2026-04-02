from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import compute_trace_statistics
from src.robustness_workloads import build_real_workload_variants, build_synthetic_workload_variants
from src.simulator import check_trace_feasibility, simulate_trace
from src.utils import FIGURE_DIR, TABLE_DIR, ensure_output_dirs


ROUTING_ALPHAS = (1.0, 1.25, 1.5)
C_VALUES = (3, 5, 7)
B_VALUES = (0, 8)


def main() -> None:
    ensure_output_dirs()
    workloads = build_synthetic_workload_variants() + build_real_workload_variants()
    rows: list[dict[str, object]] = []

    for workload in workloads:
        for C in C_VALUES:
            delta_max_nominal = compute_trace_statistics(workload.demand, C=C)[1]
            for B in B_VALUES:
                for alpha in ROUTING_ALPHAS:
                    effective_capacity = max(1, int(C // alpha))
                    feasible, feasible_reason = check_trace_feasibility(workload.demand, C=effective_capacity, B=B)
                    if feasible:
                        sim_result = simulate_trace(workload.demand, C=effective_capacity, B=B)
                        t_exe = float(sim_result.T_exe)
                        slowdown = t_exe / max(1, len(workload.demand))
                        stall_cycles = float(sim_result.stall_cycles)
                    else:
                        t_exe = float("inf")
                        slowdown = float("inf")
                        stall_cycles = float("inf")
                    delta_max_proxy = compute_trace_statistics(workload.demand, C=effective_capacity)[1]
                    rows.append(
                        {
                            "workload_scope": workload.workload_scope,
                            "workload_name": workload.workload_name,
                            "policy": workload.policy,
                            "variant_id": workload.variant_id,
                            "trace_length": workload.trace_length,
                            "C_nominal": C,
                            "B": B,
                            "routing_alpha": alpha,
                            "C_effective": effective_capacity,
                            "feasible": int(feasible),
                            "feasible_reason": feasible_reason,
                            "delta_max_nominal": delta_max_nominal,
                            "delta_max_proxy": delta_max_proxy,
                            "T_exe_proxy": t_exe,
                            "slowdown_proxy": slowdown,
                            "stall_cycles_proxy": stall_cycles,
                        }
                    )

    results_df = pd.DataFrame(rows).sort_values(
        ["workload_scope", "workload_name", "policy", "C_nominal", "B", "routing_alpha"]
    )
    ranking_df = _build_proxy_summary(results_df)
    results_path = TABLE_DIR / "routing_proxy_sensitivity.csv"
    ranking_path = TABLE_DIR / "routing_proxy_ranking_summary.csv"
    results_df.to_csv(results_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)

    figure_path = _plot_proxy_summary(ranking_df)

    print(results_df.head(12).to_string(index=False))
    print(f"saved: {results_path}")
    print(f"saved: {ranking_path}")
    print(f"saved: {figure_path}")


def _build_proxy_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for alpha, subset in results_df.groupby("routing_alpha", dropna=False):
        valid = subset[subset["feasible"] == 1].copy()
        nominal_spearman = valid["delta_max_nominal"].corr(valid["slowdown_proxy"], method="spearman")
        proxy_spearman = valid["delta_max_proxy"].corr(valid["slowdown_proxy"], method="spearman")
        rows.append(
            {
                "routing_alpha": alpha,
                "row_count": int(len(valid)),
                "spearman_nominal_delta_max_vs_proxy_slowdown": float(nominal_spearman) if pd.notna(nominal_spearman) else 0.0,
                "spearman_proxy_delta_max_vs_proxy_slowdown": float(proxy_spearman) if pd.notna(proxy_spearman) else 0.0,
                "policy_ordering_agreement_rate": _policy_ordering_agreement_rate(valid),
            }
        )
    return pd.DataFrame(rows).sort_values("routing_alpha")


def _policy_ordering_agreement_rate(results_df: pd.DataFrame) -> float:
    static_df = results_df[results_df["policy"] == "static_min"].copy()
    smoothed_df = results_df[results_df["policy"] == "smoothed"].copy()
    merge_cols = ["workload_scope", "workload_name", "C_nominal", "B", "routing_alpha"]
    merged = static_df.merge(
        smoothed_df,
        on=merge_cols,
        suffixes=("_static", "_smoothed"),
        how="inner",
    )
    if merged.empty:
        return 0.0
    nominal_order = (merged["delta_max_nominal_static"] - merged["delta_max_nominal_smoothed"]).apply(_sign)
    proxy_order = (merged["T_exe_proxy_static"] - merged["T_exe_proxy_smoothed"]).apply(_sign)
    return float((nominal_order == proxy_order).mean())


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _plot_proxy_summary(ranking_df: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    axes[0].plot(
        ranking_df["routing_alpha"],
        ranking_df["spearman_nominal_delta_max_vs_proxy_slowdown"],
        marker="o",
        label="Nominal Delta_max",
        color="#4c78a8",
    )
    axes[0].plot(
        ranking_df["routing_alpha"],
        ranking_df["spearman_proxy_delta_max_vs_proxy_slowdown"],
        marker="o",
        label="Proxy Delta_max",
        color="#f58518",
    )
    axes[0].set_title("Correlation under routing proxy")
    axes[0].set_xlabel("Routing alpha")
    axes[0].set_ylabel("Spearman correlation")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        ranking_df["routing_alpha"],
        ranking_df["policy_ordering_agreement_rate"],
        marker="o",
        color="#e45756",
    )
    axes[1].set_title("Policy ordering agreement")
    axes[1].set_xlabel("Routing alpha")
    axes[1].set_ylabel("Agreement rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    path = FIGURE_DIR / "routing_proxy_sensitivity_summary.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    main()
