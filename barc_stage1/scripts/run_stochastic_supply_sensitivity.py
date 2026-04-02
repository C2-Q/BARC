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
from src.simulator import check_trace_feasibility
from src.stochastic_supply import run_stochastic_supply_trials
from src.utils import FIGURE_DIR, TABLE_DIR, ensure_output_dirs


BASE_SEED = 314159
SMALL_TRACE_TRIALS = 100
LARGE_TRACE_TRIALS = 20
SMALL_TRACE_LIMIT = 50_000
SMALL_TRACE_P_ACC_VALUES = (1.0, 0.999, 0.995, 0.99, 0.95)
LARGE_TRACE_P_ACC_VALUES = (1.0, 0.999, 0.995, 0.99, 0.95)
DEFAULT_C_VALUES = (1, 3, 6)
DEFAULT_B_VALUES = (0, 8)
LARGE_TRACE_C_VALUES = (1, 3)
LARGE_TRACE_B_VALUES = (0, 8)


def main() -> None:
    ensure_output_dirs()
    trial_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    workloads = build_synthetic_workload_variants() + build_real_workload_variants()
    workloads = [variant for variant in workloads if not (variant.workload_name == "qft_n8" and variant.policy == "smoothed")]

    for workload_index, workload in enumerate(workloads):
        is_large = workload.trace_length > SMALL_TRACE_LIMIT
        p_acc_values = LARGE_TRACE_P_ACC_VALUES if is_large else SMALL_TRACE_P_ACC_VALUES
        c_values = LARGE_TRACE_C_VALUES if is_large else DEFAULT_C_VALUES
        b_values = LARGE_TRACE_B_VALUES if is_large else DEFAULT_B_VALUES
        trials = LARGE_TRACE_TRIALS if is_large else SMALL_TRACE_TRIALS
        max_cycles = max(20 * workload.trace_length, workload.trace_length + 1_000)

        for C in c_values:
            _, delta_max_det, _, peak_demand, mean_demand, total_t = compute_trace_statistics(workload.demand, C=C)
            for B in b_values:
                feasible, feasible_reason = check_trace_feasibility(workload.demand, C=C, B=B)
                for p_index, p_acc in enumerate(p_acc_values):
                    delta_max_expected_service = _compute_expected_service_delta_max(
                        demand=workload.demand,
                        C=C,
                        p_acc=p_acc,
                    )
                    if not feasible:
                        summary_rows.append(
                            {
                                "workload_scope": workload.workload_scope,
                                "workload_name": workload.workload_name,
                                "policy": workload.policy,
                                "variant_id": workload.variant_id,
                                "trace_length": workload.trace_length,
                                "total_t": workload.total_t,
                                "peak_demand": peak_demand,
                                "mean_demand": mean_demand,
                                "delta_max_det": delta_max_det,
                                "delta_max_expected_service": delta_max_expected_service,
                                "C": C,
                                "B": B,
                                "p_acc": p_acc,
                                "trials": 0,
                                "feasible": 0,
                                "feasible_reason": feasible_reason,
                                "mean_T_exe_stochastic": float("inf"),
                                "std_T_exe_stochastic": float("inf"),
                                "mean_slowdown_stochastic": float("inf"),
                                "mean_stall_cycles_stochastic": float("inf"),
                                "frac_runs_with_stall": 1.0,
                                "frac_runs_hitting_cycle_cap": 1.0,
                                "mean_service": 0.0,
                            }
                        )
                        continue
                    trial_df = run_stochastic_supply_trials(
                        demand=workload.demand,
                        C=C,
                        B=B,
                        p_acc=p_acc,
                        trials=trials,
                        seed=BASE_SEED + workload_index * 10_000 + p_index * 1_000 + C * 10 + B,
                        max_cycles=max_cycles,
                    )
                    for row in trial_df.to_dict("records"):
                        trial_rows.append(
                            {
                                "workload_scope": workload.workload_scope,
                                "workload_name": workload.workload_name,
                                "policy": workload.policy,
                                "variant_id": workload.variant_id,
                                "trace_length": workload.trace_length,
                                "total_t": workload.total_t,
                                "peak_demand": workload.peak_demand,
                                "C": C,
                                "B": B,
                                "p_acc": p_acc,
                                **row,
                            }
                        )
                    summary_rows.append(
                        {
                            "workload_scope": workload.workload_scope,
                            "workload_name": workload.workload_name,
                            "policy": workload.policy,
                            "variant_id": workload.variant_id,
                            "trace_length": workload.trace_length,
                            "total_t": workload.total_t,
                            "peak_demand": peak_demand,
                            "mean_demand": mean_demand,
                            "delta_max_det": delta_max_det,
                            "delta_max_expected_service": delta_max_expected_service,
                            "C": C,
                            "B": B,
                            "p_acc": p_acc,
                            "trials": trials,
                            "feasible": 1,
                            "feasible_reason": feasible_reason,
                            "mean_T_exe_stochastic": float(trial_df["T_exe"].mean()),
                            "std_T_exe_stochastic": float(trial_df["T_exe"].std(ddof=0)),
                            "mean_slowdown_stochastic": float((trial_df["T_exe"] / trial_df["T_static"]).mean()),
                            "mean_stall_cycles_stochastic": float(trial_df["stall_cycles"].mean()),
                            "frac_runs_with_stall": float((trial_df["stall_cycles"] > 0).mean()),
                            "frac_runs_hitting_cycle_cap": float((trial_df["max_cycle_hit"] == 1).mean()),
                            "mean_service": float(trial_df["mean_service"].mean()),
                        }
                    )

    trials_df = pd.DataFrame(trial_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["workload_scope", "workload_name", "policy", "C", "B", "p_acc"]
    )
    ranking_df = _build_ranking_summary(summary_df)

    trials_path = TABLE_DIR / "stochastic_supply_trials.csv"
    summary_path = TABLE_DIR / "stochastic_supply_summary.csv"
    ranking_path = TABLE_DIR / "stochastic_supply_ranking_summary.csv"
    trials_df.to_csv(trials_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)

    figure_path = _plot_stochastic_summary(summary_df)

    print(summary_df.head(12).to_string(index=False))
    print(f"saved: {trials_path}")
    print(f"saved: {summary_path}")
    print(f"saved: {ranking_path}")
    print(f"saved: {figure_path}")


def _build_ranking_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for p_acc, subset in summary_df.groupby("p_acc", dropna=False):
        valid_subset = subset[(subset["feasible"] == 1) & (subset["mean_slowdown_stochastic"].map(pd.notna))]
        spearman = valid_subset["delta_max_det"].corr(valid_subset["mean_slowdown_stochastic"], method="spearman")
        expected_service_spearman = valid_subset["delta_max_expected_service"].corr(
            valid_subset["mean_slowdown_stochastic"],
            method="spearman",
        )
        rows.append(
            {
                "p_acc": p_acc,
                "row_count": int(len(valid_subset)),
                "spearman_delta_max_vs_mean_slowdown": float(spearman) if pd.notna(spearman) else 0.0,
                "spearman_expected_service_delta_max_vs_mean_slowdown": float(expected_service_spearman)
                if pd.notna(expected_service_spearman)
                else 0.0,
                "policy_ordering_agreement_rate": _policy_ordering_agreement_rate(valid_subset),
            }
        )
    return pd.DataFrame(rows).sort_values("p_acc", ascending=False)


def _policy_ordering_agreement_rate(summary_df: pd.DataFrame) -> float:
    static_df = summary_df[summary_df["policy"] == "static_min"].copy()
    smoothed_df = summary_df[summary_df["policy"] == "smoothed"].copy()
    merge_cols = ["workload_scope", "workload_name", "C", "B", "p_acc"]
    merged = static_df.merge(
        smoothed_df,
        on=merge_cols,
        suffixes=("_static", "_smoothed"),
        how="inner",
    )
    if merged.empty:
        return 0.0
    deterministic_order = (merged["delta_max_det_static"] - merged["delta_max_det_smoothed"]).apply(_sign)
    stochastic_order = (
        merged["mean_T_exe_stochastic_static"] - merged["mean_T_exe_stochastic_smoothed"]
    ).apply(_sign)
    return float((deterministic_order == stochastic_order).mean())


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _plot_stochastic_summary(summary_df: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    ranking_df = _build_ranking_summary(summary_df)
    axes[0].plot(
        ranking_df["p_acc"],
        ranking_df["spearman_delta_max_vs_mean_slowdown"],
        marker="o",
        label="Nominal Delta_max",
        color="#1f77b4",
    )
    axes[0].plot(
        ranking_df["p_acc"],
        ranking_df["spearman_expected_service_delta_max_vs_mean_slowdown"],
        marker="o",
        label="Expected-service Delta_max",
        color="#f58518",
    )
    axes[0].set_title("Delta_max vs stochastic slowdown")
    axes[0].set_xlabel("Acceptance probability")
    axes[0].set_ylabel("Spearman correlation")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        ranking_df["p_acc"],
        ranking_df["policy_ordering_agreement_rate"],
        marker="o",
        color="#e45756",
    )
    axes[1].set_title("Policy ordering agreement")
    axes[1].set_xlabel("Acceptance probability")
    axes[1].set_ylabel("Agreement rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    path = FIGURE_DIR / "stochastic_supply_sensitivity_summary.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return path


def _compute_expected_service_delta_max(demand: list[int], C: int, p_acc: float) -> float:
    cumulative = 0.0
    delta_max = 0.0
    expected_service_rate = C * p_acc
    for index, value in enumerate(demand, start=1):
        cumulative += value
        delta_max = max(delta_max, cumulative - expected_service_rate * index)
    return float(max(0.0, delta_max))


if __name__ == "__main__":
    main()
