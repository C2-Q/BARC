from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dag import generate_dag
from src.metrics import compute_trace_statistics
from src.schedule import build_schedule
from src.simulator import check_trace_feasibility, simulate_trace
from src.trace import schedule_to_trace_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

FAMILIES = ("high_compressibility", "medium_compressibility", "low_compressibility")
SEEDS = (1, 2, 3)
C_VALUES = (1, 2, 3)
B_VALUES = (0, 2, 4)
FAMILY_SPECS = {
    "high_compressibility": {"num_layers": 8, "width": 6, "t_ratio": 0.45},
    "medium_compressibility": {"num_layers": 8, "width": 6, "t_ratio": 0.45},
    "low_compressibility": {"num_layers": 8, "width": 6, "t_ratio": 0.45},
}
POLICIES = ("capacity_aware_static", "delivery_aware_slack")


def _run_single_policy(family: str, seed: int, C: int, B: int, policy: str) -> dict[str, float | int | str]:
    dag = generate_dag(family=family, seed=seed, **FAMILY_SPECS[family])
    schedule = build_schedule(dag, policy=policy, capacity_limit=C)
    trace_bundle = schedule_to_trace_bundle(dag, schedule)
    feasible, reason = check_trace_feasibility(trace_bundle.demand, C=C, B=B)
    delta_max = compute_trace_statistics(trace_bundle.demand, C=C)[1]
    if feasible:
        sim_result = simulate_trace(trace_bundle.demand, C=C, B=B)
        t_exe = sim_result.T_exe
        stall_cycles = sim_result.stall_cycles
    else:
        t_exe = -1
        stall_cycles = -1

    return {
        "family": family,
        "seed": seed,
        "C": C,
        "B": B,
        "policy": policy,
        "feasible": int(feasible),
        "feasible_reason": reason,
        "T_static": schedule.static_depth,
        "T_exe": t_exe,
        "stall_cycles": stall_cycles,
        "Delta_max": delta_max,
    }


def main() -> None:
    rows: list[dict[str, float | int | str]] = []
    for family in FAMILIES:
        for seed in SEEDS:
            for C in C_VALUES:
                for B in B_VALUES:
                    for policy in POLICIES:
                        rows.append(_run_single_policy(family, seed, C, B, policy))

    detail_df = pd.DataFrame(rows)
    pivot_df = detail_df.pivot_table(
        index=["family", "seed", "C", "B"],
        columns="policy",
        values=["feasible", "T_static", "T_exe", "stall_cycles", "Delta_max"],
        aggfunc="first",
    )
    pivot_df.columns = [f"{metric}_{policy}" for metric, policy in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    feasible_mask = (pivot_df["feasible_capacity_aware_static"] == 1) & (pivot_df["feasible_delivery_aware_slack"] == 1)
    pivot_df["static_depth_improvement"] = pivot_df["T_static_capacity_aware_static"] - pivot_df["T_static_delivery_aware_slack"]
    pivot_df["exe_improvement"] = pd.NA
    pivot_df.loc[feasible_mask, "exe_improvement"] = (
        pivot_df.loc[feasible_mask, "T_exe_capacity_aware_static"] - pivot_df.loc[feasible_mask, "T_exe_delivery_aware_slack"]
    )
    pivot_df["delta_max_reduction"] = pivot_df["Delta_max_capacity_aware_static"] - pivot_df["Delta_max_delivery_aware_slack"]

    summary_df = (
        pivot_df.groupby("family", dropna=False)
        .agg(
            instances=("family", "size"),
            feasible_pairs=("exe_improvement", lambda s: int(s.notna().sum())),
            mean_static_depth_improvement=("static_depth_improvement", "mean"),
            max_static_depth_improvement=("static_depth_improvement", "max"),
            mean_exe_improvement=("exe_improvement", "mean"),
            positive_exe_improvement_rate=(
                "exe_improvement",
                lambda s: float((s.dropna() > 0).mean()) if not s.dropna().empty else 0.0,
            ),
            mean_delta_max_reduction=("delta_max_reduction", "mean"),
        )
        .reset_index()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(OUTPUT_DIR / "delivery_aware_pass_probe_detail.csv", index=False)
    pivot_df.to_csv(OUTPUT_DIR / "delivery_aware_pass_probe_pairs.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "delivery_aware_pass_probe_summary.csv", index=False)

    summary_lines = [
        "Minimal delivery-aware pass probe",
        "",
    ]
    for row in summary_df.itertuples(index=False):
        summary_lines.append(
            f"{row.family}: mean static-depth improvement={row.mean_static_depth_improvement:.3f}, "
            f"max static-depth improvement={row.max_static_depth_improvement:.0f}, "
            f"mean executed-depth improvement={row.mean_exe_improvement:.3f}, "
            f"positive-rate={row.positive_exe_improvement_rate:.3f}, "
            f"mean Delta_max reduction={row.mean_delta_max_reduction:.3f}"
        )

    (OUTPUT_DIR / "delivery_aware_pass_probe_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
