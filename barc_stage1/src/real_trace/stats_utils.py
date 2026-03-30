from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NORMALIZED_TARGET = 1.0
NORMALIZED_TOLERANCE = 1e-9
MISSING_THRESHOLD = -1


def infer_trace_label(trace_name: str) -> str:
    if trace_name.startswith("qft_n"):
        return f"QFT n={trace_name.split('qft_n', maxsplit=1)[1]}"
    if trace_name.startswith("adder_n"):
        return f"Adder n={trace_name.split('adder_n', maxsplit=1)[1]}"
    if trace_name.startswith("multiplier_n"):
        return f"Multiplier n={trace_name.split('multiplier_n', maxsplit=1)[1]}"
    return trace_name.replace("_", " ").title()


def infer_burst_threshold(trace: list[int]) -> int:
    if not trace:
        return 1
    mean = float(np.mean(trace))
    std = float(np.std(trace))
    return max(1, int(math.ceil(mean + std)))


def compute_burst_lengths(trace: list[int], threshold: int | None = None) -> list[int]:
    burst_threshold = infer_burst_threshold(trace) if threshold is None else threshold
    burst_lengths: list[int] = []
    current_length = 0
    for demand in trace:
        if demand >= burst_threshold:
            current_length += 1
        elif current_length > 0:
            burst_lengths.append(current_length)
            current_length = 0
    if current_length > 0:
        burst_lengths.append(current_length)
    return burst_lengths


def select_zoom_window(trace: list[int], threshold: int | None = None, window_size: int | None = None) -> tuple[int, int]:
    if not trace:
        return 0, 0
    burst_threshold = infer_burst_threshold(trace) if threshold is None else threshold
    size = window_size or min(max(24, len(trace) // 12), 80, len(trace))
    if size >= len(trace):
        return 0, len(trace)

    best_start = 0
    best_score = -1
    for start in range(0, len(trace) - size + 1):
        window = trace[start : start + size]
        score = sum(window) + 2 * sum(1 for value in window if value >= burst_threshold)
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start + size


def compute_demand_level_proportions(trace: list[int]) -> dict[str, float]:
    if not trace:
        return {"0": 1.0}
    total = float(len(trace))
    levels, counts = np.unique(trace, return_counts=True)
    return {str(int(level)): float(count / total) for level, count in zip(levels, counts, strict=True)}


def compute_workload_statistics(trace: list[int], trace_name: str) -> dict[str, Any]:
    burst_threshold = infer_burst_threshold(trace)
    burst_lengths = compute_burst_lengths(trace, threshold=burst_threshold)
    zoom_start, zoom_end = select_zoom_window(trace, threshold=burst_threshold)
    level_proportions = compute_demand_level_proportions(trace)
    return {
        "trace_name": trace_name,
        "timesteps": len(trace),
        "total_T": int(sum(trace)),
        "mean_demand": float(np.mean(trace)) if trace else 0.0,
        "max_demand": int(max(trace)) if trace else 0,
        "compressibility_proxy": float(np.std(trace) / np.mean(trace)) if trace and np.mean(trace) > 0 else 0.0,
        "burst_threshold": burst_threshold,
        "spike_count": int(sum(1 for value in trace if value >= burst_threshold)),
        "average_burst_length": float(np.mean(burst_lengths)) if burst_lengths else 0.0,
        "max_burst_length": int(max(burst_lengths)) if burst_lengths else 0,
        "burst_count": int(len(burst_lengths)),
        "demand_level_proportions": json.dumps(level_proportions, sort_keys=True),
        "zoom_start": int(zoom_start),
        "zoom_end": int(zoom_end),
    }


def find_representative_buffer(summary_df: pd.DataFrame) -> int:
    summary = summary_df.copy()
    summary["difference_score"] = 0.0
    infeasible_gap = (
        (summary["feasible_static_min"] != summary["feasible_smoothed"]).astype(int) * 1000.0
    )
    feasible_rows = summary["both_feasible"] == 1
    summary.loc[feasible_rows, "difference_score"] = (
        summary.loc[feasible_rows, "T_exe_improvement"].abs() + infeasible_gap.loc[feasible_rows]
    )
    summary.loc[~feasible_rows, "difference_score"] = infeasible_gap.loc[~feasible_rows]
    return int(summary.sort_values(["difference_score", "B"], ascending=[False, True]).iloc[0]["B"])


def find_representative_capacity(summary_df: pd.DataFrame) -> int:
    summary = summary_df.copy()
    summary["difference_score"] = 0.0
    infeasible_gap = (
        (summary["feasible_static_min"] != summary["feasible_smoothed"]).astype(int) * 1000.0
    )
    feasible_rows = summary["both_feasible"] == 1
    summary.loc[feasible_rows, "difference_score"] = (
        summary.loc[feasible_rows, "stall_reduction"].abs() + infeasible_gap.loc[feasible_rows]
    )
    summary.loc[~feasible_rows, "difference_score"] = infeasible_gap.loc[~feasible_rows]
    return int(summary.sort_values(["difference_score", "C"], ascending=[False, True]).iloc[0]["C"])


def find_critical_capacity(
    results_df: pd.DataFrame,
    policy: str,
    representative_B: int,
    normalized_target: float = NORMALIZED_TARGET,
    tolerance: float = NORMALIZED_TOLERANCE,
) -> int | None:
    subset = results_df[(results_df["policy"] == policy) & (results_df["B"] == representative_B)].sort_values("C")
    for _, row in subset.iterrows():
        if int(row["feasible"]) == 1 and float(row["normalized_makespan"]) <= normalized_target + tolerance:
            return int(row["C"])
    return None


def find_critical_buffer(
    results_df: pd.DataFrame,
    policy: str,
    representative_C: int,
    normalized_target: float = NORMALIZED_TARGET,
    tolerance: float = NORMALIZED_TOLERANCE,
) -> int | None:
    subset = results_df[(results_df["policy"] == policy) & (results_df["C"] == representative_C)].sort_values("B")
    for _, row in subset.iterrows():
        if int(row["feasible"]) == 1 and float(row["normalized_makespan"]) <= normalized_target + tolerance:
            return int(row["B"])
    return None


def find_stall_saturation_buffer(
    results_df: pd.DataFrame,
    policy: str,
    representative_C: int,
    epsilon: float = 0.0,
) -> int | None:
    subset = results_df[(results_df["policy"] == policy) & (results_df["C"] == representative_C)].copy()
    subset = subset[np.isfinite(subset["stall_cycles"])].sort_values("B")
    if subset.empty:
        return None
    minimum_stall = float(subset["stall_cycles"].min())
    saturated = subset[subset["stall_cycles"] <= minimum_stall + epsilon]
    if saturated.empty:
        return None
    return int(saturated.iloc[0]["B"])


def compute_threshold_rows(results_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    trace_name = str(results_df["trace_name"].iloc[0])
    representative_B = find_representative_buffer(summary_df)
    representative_C = find_representative_capacity(summary_df)
    rows: list[dict[str, Any]] = []
    for policy in ("static_min", "smoothed"):
        policy_results = results_df[results_df["policy"] == policy]
        critical_capacity = find_critical_capacity(results_df, policy, representative_B)
        critical_buffer = find_critical_buffer(results_df, policy, representative_C)
        saturation_buffer = find_stall_saturation_buffer(results_df, policy, representative_C)
        rows.append(
            {
                "trace_name": trace_name,
                "policy": policy,
                "representative_B": representative_B,
                "representative_C": representative_C,
                "critical_capacity": critical_capacity if critical_capacity is not None else MISSING_THRESHOLD,
                "critical_buffer": critical_buffer if critical_buffer is not None else MISSING_THRESHOLD,
                "stall_saturation_buffer": saturation_buffer if saturation_buffer is not None else MISSING_THRESHOLD,
                "Delta_max": int(policy_results["Delta_max"].max()),
                "peak_demand": int(policy_results["peak_demand"].iloc[0]),
                "mean_demand": float(policy_results["mean_demand"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def build_real_trace_note(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = """# Real-Trace Figure Note

- Plotting was refactored into shared publication-style utilities for trace characterization, capacity sensitivity, buffer sensitivity, and cross-workload summary views.
- A burst is defined deterministically as a timestep whose demand is at least `ceil(mean(trace) + std(trace))`, clipped to be at least 1.
- Burst length is the number of consecutive timesteps that stay above that threshold.
- Critical capacity is the smallest scanned capacity `C` at the representative buffer `B*` where normalized execution time is at most 1.0.
- Critical buffer is the smallest scanned buffer `B` at the representative capacity `C*` where normalized execution time is at most 1.0.
- Stall saturation buffer is the smallest scanned `B` at `C*` where stall cycles reach their minimum finite value over the scanned range.
- `B*` and `C*` are chosen deterministically as the scan slices with the largest policy separation, prioritizing feasibility differences.
- A value of `-1` in a threshold summary means the threshold was not reached within the frozen scan range.
"""
    path.write_text(contents, encoding="utf-8")
    return path
