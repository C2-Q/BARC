from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.metrics import compute_trace_statistics
from src.real_trace import (
    generate_adder_circuit,
    generate_multiplier_circuit,
    quantum_circuit_to_internal_dag,
    to_clifford_t,
)
from src.schedule import build_schedule
from src.simulator import check_trace_feasibility, simulate_trace
from src.trace import schedule_to_trace_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
POLICIES = ("capacity_aware_static", "delivery_aware_slack")

WORKLOADS = [
    ("adder", 4, lambda n: generate_adder_circuit(n_bits=n)),
    ("multiplier", 4, lambda n: generate_multiplier_circuit(n_bits=n)),
    ("multiplier", 5, lambda n: generate_multiplier_circuit(n_bits=n)),
    ("multiplier", 6, lambda n: generate_multiplier_circuit(n_bits=n)),
]

C_VALUES = (1, 2)
B_VALUES = (0, 4, 8)


def _evaluate_policy(workload_name: str, dag, policy: str, C: int, B: int) -> dict[str, float | int | str]:
    start = time.perf_counter()
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

    eval_seconds = time.perf_counter() - start

    return {
        "workload": workload_name,
        "policy": policy,
        "C": C,
        "B": B,
        "feasible": int(feasible),
        "feasible_reason": reason,
        "T_static": schedule.static_depth,
        "T_exe": t_exe,
        "stall_cycles": stall_cycles,
        "Delta_max": delta_max,
        "t_count": dag.total_t,
        "node_count": dag.total_nodes,
        "eval_seconds": eval_seconds,
    }


def main() -> None:
    rows: list[dict[str, float | int | str]] = []

    for family, size, builder in WORKLOADS:
        workload_name = f"{family}_n{size}"
        build_start = time.perf_counter()
        circuit = to_clifford_t(builder(size))
        dag = quantum_circuit_to_internal_dag(circuit, family=workload_name)
        build_seconds = time.perf_counter() - build_start
        for C in C_VALUES:
            for B in B_VALUES:
                for policy in POLICIES:
                    row = _evaluate_policy(workload_name, dag, policy, C, B)
                    row["build_seconds"] = build_seconds
                    rows.append(row)

    detail_df = pd.DataFrame(rows)
    pivot_df = detail_df.pivot_table(
        index=["workload", "C", "B", "t_count", "node_count"],
        columns="policy",
        values=["feasible", "T_static", "T_exe", "stall_cycles", "Delta_max", "build_seconds", "eval_seconds"],
        aggfunc="first",
    )
    pivot_df.columns = [f"{metric}_{policy}" for metric, policy in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    feasible_mask = (pivot_df["feasible_capacity_aware_static"] == 1) & (pivot_df["feasible_delivery_aware_slack"] == 1)
    pivot_df["static_depth_improvement"] = (
        pivot_df["T_static_capacity_aware_static"] - pivot_df["T_static_delivery_aware_slack"]
    )
    pivot_df["exe_improvement"] = pd.NA
    pivot_df.loc[feasible_mask, "exe_improvement"] = (
        pivot_df.loc[feasible_mask, "T_exe_capacity_aware_static"] - pivot_df.loc[feasible_mask, "T_exe_delivery_aware_slack"]
    )
    pivot_df["delta_max_reduction"] = (
        pivot_df["Delta_max_capacity_aware_static"] - pivot_df["Delta_max_delivery_aware_slack"]
    )

    summary_df = (
        pivot_df.groupby("workload", dropna=False)
        .agg(
            instances=("workload", "size"),
            feasible_pairs=("exe_improvement", lambda s: int(s.notna().sum())),
            mean_static_depth_improvement=("static_depth_improvement", "mean"),
            max_static_depth_improvement=("static_depth_improvement", "max"),
            mean_exe_improvement=("exe_improvement", "mean"),
            positive_exe_improvement_rate=(
                "exe_improvement",
                lambda s: float((s.dropna() > 0).mean()) if not s.dropna().empty else 0.0,
            ),
            mean_delta_max_reduction=("delta_max_reduction", "mean"),
            mean_build_seconds=("build_seconds_capacity_aware_static", "mean"),
            mean_eval_seconds=("eval_seconds_delivery_aware_slack", "mean"),
        )
        .reset_index()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(OUTPUT_DIR / "delivery_aware_pass_probe_round2_detail.csv", index=False)
    pivot_df.to_csv(OUTPUT_DIR / "delivery_aware_pass_probe_round2_pairs.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "delivery_aware_pass_probe_round2_summary.csv", index=False)

    lines = ["Delivery-aware pass round-2 probe", ""]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"{row.workload}: mean static-depth improvement={row.mean_static_depth_improvement:.3f}, "
            f"max static-depth improvement={row.max_static_depth_improvement:.0f}, "
            f"mean executed-depth improvement={0.0 if pd.isna(row.mean_exe_improvement) else row.mean_exe_improvement:.3f}, "
            f"positive-rate={row.positive_exe_improvement_rate:.3f}, "
            f"mean Delta_max reduction={row.mean_delta_max_reduction:.3f}, "
            f"build-seconds={row.mean_build_seconds:.2f}, "
            f"pass-eval-seconds={row.mean_eval_seconds:.2f}"
        )

    summary_text = "\n".join(lines)
    (OUTPUT_DIR / "delivery_aware_pass_probe_round2_summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
