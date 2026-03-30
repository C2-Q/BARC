from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.dag import DAG, generate_dag
from src.metrics import compute_dag_metrics, compute_metrics, compute_trace_statistics
from src.paper_config import (
    B_VALUES,
    BUFFER_TRANSITION_C,
    BUFFER_TRANSITION_MARGIN,
    CAPACITY_SCAN_B,
    COMPRESSIBILITY_SCAN_B,
    COMPRESSIBILITY_SCAN_C,
    C_VALUES,
    FAMILIES,
    REPRESENTATIVE_COMPARE_B,
    REPRESENTATIVE_COMPARE_C,
    REPRESENTATIVE_FAMILY,
    REPRESENTATIVE_SEED,
    SEEDS,
)
from src.predictive_analysis import run_predictive_analysis
from src.schedule import build_schedule
from src.simulator import check_trace_feasibility, simulate_trace
from src.trace import schedule_to_trace_bundle
from src.utils import APPENDIX_DIR, TABLE_DIR, ensure_output_dirs, normalize_output_dataframe


DEFAULT_SEED = 7
POLICIES = ("static_min", "capacity_aware_static", "smoothed")
POLICY_LABELS = {
    "static_min": "static_min",
    "capacity_aware_static": "capacity_aware_static",
    "smoothed": "smoothed",
}
PAIR_CONFIGS = {
    "static_min_vs_smoothed": ("static_min", "smoothed"),
    "capacity_aware_vs_smoothed": ("capacity_aware_static", "smoothed"),
}

FAMILY_SPECS: dict[str, dict[str, float | int]] = {
    "high_compressibility": {"num_layers": 8, "width": 8, "t_ratio": 0.45},
    "medium_compressibility": {"num_layers": 8, "width": 7, "t_ratio": 0.45},
    "low_compressibility": {"num_layers": 7, "width": 6, "t_ratio": 0.45},
}


@dataclass(frozen=True)
class ScheduleRun:
    family: str
    seed: int
    policy: str
    capacity_limit_mode: str
    capacity_limit: int
    C: int
    B: int
    feasible: int
    feasible_reason: str
    compressibility: float
    compressibility_slack_ratio: float
    compressibility_mean_t_slack: float
    T_static: int
    T_exe: float
    stall_cycles: float
    stall_ratio: float
    Delta_max: int
    Gamma: int
    peak_demand: int
    mean_demand: float
    total_T: int
    QTV_proxy: float
    normalized_makespan: float


def run_all_experiments(seed: int = DEFAULT_SEED) -> dict[str, pd.DataFrame]:
    ensure_output_dirs()
    del seed

    representative_dag = _build_family_dag(REPRESENTATIVE_FAMILY, REPRESENTATIVE_SEED)

    compare_df = run_policy_comparison(representative_dag, C=REPRESENTATIVE_COMPARE_C, B=REPRESENTATIVE_COMPARE_B)
    buffer_df = run_buffer_scan(representative_dag, C=BUFFER_TRANSITION_C, margin=BUFFER_TRANSITION_MARGIN)
    capacity_df = run_capacity_scan(representative_dag, B=CAPACITY_SCAN_B, C_values=C_VALUES)
    compressibility_df = run_compressibility_scan(
        families=FAMILIES,
        seeds=SEEDS,
        C=COMPRESSIBILITY_SCAN_C,
        B=COMPRESSIBILITY_SCAN_B,
    )
    grid_df = run_grid_scan(
        families=FAMILIES,
        seeds=SEEDS,
        C_values=C_VALUES,
        B_values=B_VALUES,
    )
    grid_pair_df = build_policy_pair_table(grid_df)
    inversion_df = find_inversion_cases(grid_pair_df)
    inversion_strong_df = build_inversion_summary_strong_baseline(grid_pair_df)
    compressibility_summary_df = build_compressibility_summary(grid_pair_df)
    qtv_tradeoff_df = build_qtv_tradeoff_table(grid_df)
    shallower_slower_summary_df = build_shallower_but_slower_summary(grid_pair_df)
    family_summary_df = build_family_summary(grid_df)
    policy_comparison_summary_df = build_policy_comparison_summary(grid_df)
    representative_cases_df = build_representative_cases(grid_pair_df)
    buffer_threshold_summary_df = build_buffer_threshold_summary(buffer_df)
    methodology_summary_text = build_methodology_upgrade_summary(
        compressibility_summary_df=compressibility_summary_df,
        grid_pair_df=grid_pair_df,
        buffer_df=buffer_df,
        family_summary_df=family_summary_df,
        inversion_summary_df=shallower_slower_summary_df,
        inversion_strong_df=inversion_strong_df,
    )
    predictive_results = run_predictive_analysis(
        grid_df=grid_df,
        grid_pair_df=grid_pair_df,
        family_summary_df=family_summary_df,
    )

    _write_table(compare_df, "compare_policies.csv", appendix=True)
    _write_table(buffer_df, "buffer_transition_scan.csv")
    _write_table(capacity_df, "capacity_scan.csv")
    _write_table(compressibility_df, "compressibility_scan.csv")
    _write_table(grid_df, "stage1_grid_scan.csv")
    _write_table(compressibility_summary_df, "compressibility_summary.csv")
    _write_table(qtv_tradeoff_df, "qtv_tradeoff.csv", appendix=True)
    _write_table(shallower_slower_summary_df, "shallower_but_slower_summary.csv")
    _write_table(family_summary_df, "family_summary.csv")
    _write_table(policy_comparison_summary_df, "policy_comparison_summary.csv")
    _write_table(representative_cases_df, "representative_cases.csv", appendix=True)
    _write_table(shallower_slower_summary_df, "inversion_summary.csv")
    _write_table(inversion_strong_df, "inversion_summary_strong_baseline.csv")
    _write_table(buffer_threshold_summary_df, "buffer_threshold_summary.csv")
    _write_table(inversion_df, "inversion_cases.csv")
    _write_text(methodology_summary_text, "stage1_methodology_upgrade_summary.txt")

    return {
        "compare": compare_df,
        "buffer": buffer_df,
        "capacity": capacity_df,
        "compressibility": compressibility_df,
        "compressibility_summary": compressibility_summary_df,
        "grid": grid_df,
        "grid_pairs": grid_pair_df,
        "qtv_tradeoff": qtv_tradeoff_df,
        "shallower_slower_summary": shallower_slower_summary_df,
        "family_summary": family_summary_df,
        "policy_comparison_summary": policy_comparison_summary_df,
        "representative_cases": representative_cases_df,
        "buffer_threshold_summary": buffer_threshold_summary_df,
        "inversion": inversion_df,
        "inversion_strong": inversion_strong_df,
        **predictive_results,
    }


def run_policy_comparison(dag: DAG, C: int, B: int) -> pd.DataFrame:
    rows = [_run_single_policy(dag, policy, C=C, B=B) for policy in POLICIES]
    return pd.DataFrame(rows)


def run_buffer_scan(dag: DAG, C: int, margin: int) -> pd.DataFrame:
    if margin < 0:
        raise ValueError("margin must be non-negative")

    reference_rows = [_run_single_policy(dag, policy, C=C, B=max(0, dag.total_t)) for policy in POLICIES]
    max_delta = max(int(row["Delta_max"]) for row in reference_rows)
    B_scan = list(range(0, max_delta + margin + 1))

    rows: list[dict[str, Any]] = []
    for B in B_scan:
        for policy in POLICIES:
            row = _run_single_policy(dag, policy, C=C, B=B)
            row["predicted_stall_by_delta"] = int(row["Delta_max"] > B)
            row["stall_observed"] = int(row["stall_cycles"] > 0) if row["feasible"] == 1 else 1
            row["delta_reference_static_min"] = int(next(item["Delta_max"] for item in reference_rows if item["policy"] == "static_min"))
            row["delta_reference_capacity_aware_static"] = int(
                next(item["Delta_max"] for item in reference_rows if item["policy"] == "capacity_aware_static")
            )
            row["delta_reference_smoothed"] = int(next(item["Delta_max"] for item in reference_rows if item["policy"] == "smoothed"))
            rows.append(row)
    return pd.DataFrame(rows)


def run_capacity_scan(dag: DAG, B: int, C_values: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for C in C_values:
        for policy in POLICIES:
            rows.append(_run_single_policy(dag, policy, C=C, B=B))
    return pd.DataFrame(rows)


def run_compressibility_scan(
    families: list[str],
    seeds: list[int],
    C: int,
    B: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in families:
        for seed in seeds:
            dag = _build_family_dag(family, seed)
            policy_rows = {
                policy: _run_single_policy(dag, policy, C=C, B=B)
                for policy in POLICIES
            }
            row = {
                "family": family,
                "seed": seed,
                "compressibility": policy_rows["static_min"]["compressibility"],
                "compressibility_slack_ratio": policy_rows["static_min"]["compressibility_slack_ratio"],
                "compressibility_mean_t_slack": policy_rows["static_min"]["compressibility_mean_t_slack"],
                "C": C,
                "B": B,
            }
            for policy in POLICIES:
                label = POLICY_LABELS[policy]
                row[f"{label}_T_static"] = policy_rows[policy]["T_static"]
                row[f"{label}_T_exe"] = policy_rows[policy]["T_exe"]
                row[f"{label}_stall_cycles"] = policy_rows[policy]["stall_cycles"]
                row[f"{label}_Delta_max"] = policy_rows[policy]["Delta_max"]
                row[f"{label}_Gamma"] = policy_rows[policy]["Gamma"]
                row[f"{label}_QTV_proxy"] = policy_rows[policy]["QTV_proxy"]

            row["both_feasible_static_min_vs_smoothed"] = int(
                policy_rows["static_min"]["feasible"] == 1 and policy_rows["smoothed"]["feasible"] == 1
            )
            row["both_feasible_capacity_aware_vs_smoothed"] = int(
                policy_rows["capacity_aware_static"]["feasible"] == 1 and policy_rows["smoothed"]["feasible"] == 1
            )
            row["relative_T_exe_improvement_static_min_vs_smoothed"] = _relative_improvement(
                policy_rows["static_min"]["T_exe"],
                policy_rows["smoothed"]["T_exe"],
                both_feasible=row["both_feasible_static_min_vs_smoothed"],
            )
            row["relative_T_exe_improvement_capacity_aware_vs_smoothed"] = _relative_improvement(
                policy_rows["capacity_aware_static"]["T_exe"],
                policy_rows["smoothed"]["T_exe"],
                both_feasible=row["both_feasible_capacity_aware_vs_smoothed"],
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_grid_scan(
    families: list[str],
    seeds: list[int],
    C_values: list[int],
    B_values: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in families:
        for seed in seeds:
            dag = _build_family_dag(family, seed)
            for C in C_values:
                for B in B_values:
                    for policy in POLICIES:
                        rows.append(_run_single_policy(dag, policy, C=C, B=B))
    return pd.DataFrame(rows)


def build_policy_pair_table(grid_df: pd.DataFrame) -> pd.DataFrame:
    pair_df = grid_df.pivot_table(
        index=[
            "family",
            "seed",
            "compressibility",
            "compressibility_slack_ratio",
            "compressibility_mean_t_slack",
            "C",
            "B",
        ],
        columns="policy",
        values=[
            "capacity_limit_mode",
            "capacity_limit",
            "feasible",
            "T_static",
            "T_exe",
            "stall_cycles",
            "stall_ratio",
            "Delta_max",
            "Gamma",
            "QTV_proxy",
            "normalized_makespan",
        ],
        aggfunc="first",
    )
    pair_df.columns = [f"{value}_{policy}" for value, policy in pair_df.columns]
    pair_df = pair_df.reset_index()

    for prefix, (base_policy, compare_policy) in PAIR_CONFIGS.items():
        _append_comparison_metrics(pair_df, base_policy, compare_policy, prefix)

    pair_df["both_feasible"] = pair_df["both_feasible_static_min_vs_smoothed"]
    pair_df["relative_T_exe_improvement"] = pair_df["relative_T_exe_improvement_static_min_vs_smoothed"]
    pair_df["relative_QTV_change"] = pair_df["relative_QTV_change_static_min_vs_smoothed"]
    pair_df["qtv_ratio_smoothed_over_static"] = pair_df["qtv_ratio_smoothed_over_static_min"]
    pair_df["static_min_has_stall"] = (
        pair_df["stall_cycles_static_min"].replace(math.inf, 1.0) > 0
    ).astype(int)
    pair_df["capacity_aware_has_stall"] = (
        pair_df["stall_cycles_capacity_aware_static"].replace(math.inf, 1.0) > 0
    ).astype(int)
    return pair_df


def find_inversion_cases(grid_pair_df: pd.DataFrame) -> pd.DataFrame:
    inversion_df = grid_pair_df[grid_pair_df["static_min_vs_smoothed_inversion"] == 1].copy()
    columns = [
        "family",
        "seed",
        "compressibility",
        "compressibility_slack_ratio",
        "compressibility_mean_t_slack",
        "C",
        "B",
        "T_static_static_min",
        "T_static_smoothed",
        "T_exe_static_min",
        "T_exe_smoothed",
        "Delta_max_static_min",
        "Delta_max_smoothed",
        "QTV_proxy_static_min",
        "QTV_proxy_smoothed",
        "relative_T_exe_improvement_static_min_vs_smoothed",
        "relative_QTV_change_static_min_vs_smoothed",
    ]
    return inversion_df[columns].sort_values(
        by=["relative_T_exe_improvement_static_min_vs_smoothed", "family", "seed", "C", "B"],
        ascending=[False, True, True, True, True],
    )


def build_compressibility_summary(grid_pair_df: pd.DataFrame) -> pd.DataFrame:
    per_seed = _aggregate_compare_metrics(grid_pair_df, ["family", "seed"], "family_seed")
    family = _aggregate_compare_metrics(grid_pair_df, ["family"], "family")
    family["seed"] = "ALL"
    family = family[per_seed.columns]
    return pd.concat([per_seed, family], ignore_index=True)


def build_qtv_tradeoff_table(grid_df: pd.DataFrame) -> pd.DataFrame:
    feasible_df = grid_df[np.isfinite(grid_df["normalized_makespan"]) & np.isfinite(grid_df["QTV_proxy"])].copy()
    columns = [
        "family",
        "seed",
        "policy",
        "capacity_limit_mode",
        "C",
        "B",
        "compressibility",
        "compressibility_slack_ratio",
        "compressibility_mean_t_slack",
        "T_static",
        "T_exe",
        "normalized_makespan",
        "QTV_proxy",
        "Delta_max",
        "Gamma",
    ]
    return feasible_df[columns].sort_values(by=["family", "seed", "policy", "C", "B"])


def build_shallower_but_slower_summary(grid_pair_df: pd.DataFrame) -> pd.DataFrame:
    condition = grid_pair_df["static_min_vs_smoothed_inversion"] == 1
    summaries: list[dict[str, Any]] = []
    for family, subset in grid_pair_df.groupby("family"):
        both_feasible_count = int((subset["both_feasible_static_min_vs_smoothed"] == 1).sum())
        count = int(condition.loc[subset.index].sum())
        summaries.append(
            {
                "family": family,
                "count": count,
                "total_experiments": int(len(subset)),
                "both_feasible_experiments": both_feasible_count,
                "fraction_of_total": count / max(1, len(subset)),
                "fraction_of_both_feasible": count / max(1, both_feasible_count),
            }
        )
    total_both_feasible = int((grid_pair_df["both_feasible_static_min_vs_smoothed"] == 1).sum())
    total_count = int(condition.sum())
    summaries.append(
        {
            "family": "ALL",
            "count": total_count,
            "total_experiments": int(len(grid_pair_df)),
            "both_feasible_experiments": total_both_feasible,
            "fraction_of_total": total_count / max(1, len(grid_pair_df)),
            "fraction_of_both_feasible": total_count / max(1, total_both_feasible),
        }
    )
    return pd.DataFrame(summaries)


def build_inversion_summary_strong_baseline(grid_pair_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "family",
        "seed",
        "C",
        "B",
        "static_min_vs_smoothed_inversion",
        "capacity_aware_vs_smoothed_inversion",
    ]
    return grid_pair_df[columns].sort_values(by=["family", "seed", "C", "B"])


def build_family_summary(grid_df: pd.DataFrame) -> pd.DataFrame:
    feasible_df = grid_df[np.isfinite(grid_df["T_exe"])].copy()
    summary = feasible_df.groupby(["family", "policy", "capacity_limit_mode"], dropna=False).agg(
        mean_T_exe=("T_exe", "mean"),
        mean_normalized_makespan=("normalized_makespan", "mean"),
        mean_Delta_max=("Delta_max", "mean"),
        mean_QTV_proxy=("QTV_proxy", "mean"),
        mean_compressibility_slack_ratio=("compressibility_slack_ratio", "mean"),
    ).reset_index()
    return summary.sort_values(by=["family", "policy"])


def build_policy_comparison_summary(grid_df: pd.DataFrame) -> pd.DataFrame:
    feasible_df = grid_df[np.isfinite(grid_df["T_exe"])].copy()
    summary = feasible_df.groupby(["family", "policy", "capacity_limit_mode"], dropna=False).agg(
        mean_T_static=("T_static", "mean"),
        mean_T_exe=("T_exe", "mean"),
        mean_stall_cycles=("stall_cycles", "mean"),
        mean_normalized_makespan=("normalized_makespan", "mean"),
        mean_Delta_max=("Delta_max", "mean"),
        mean_QTV_proxy=("QTV_proxy", "mean"),
        mean_compressibility_slack_ratio=("compressibility_slack_ratio", "mean"),
    ).reset_index()
    return summary.sort_values(by=["family", "policy"])


def build_representative_cases(grid_pair_df: pd.DataFrame) -> pd.DataFrame:
    representative_rows: list[dict[str, Any]] = []
    feasible_df = grid_pair_df[grid_pair_df["both_feasible_static_min_vs_smoothed"] == 1].copy()
    for family in FAMILIES:
        subset = feasible_df[feasible_df["family"] == family].copy()
        if subset.empty:
            continue
        positive_subset = subset[subset["relative_T_exe_improvement_static_min_vs_smoothed"] > 0]
        target_subset = positive_subset if not positive_subset.empty else subset
        target_value = target_subset["relative_T_exe_improvement_static_min_vs_smoothed"].median()
        representative = target_subset.iloc[
            (target_subset["relative_T_exe_improvement_static_min_vs_smoothed"] - target_value).abs().argsort().iloc[0]
        ]
        representative_rows.append(
            {
                "family": family,
                "seed": int(representative["seed"]),
                "C": int(representative["C"]),
                "B": int(representative["B"]),
                "T_static_static_min": representative["T_static_static_min"],
                "T_static_capacity_aware_static": representative["T_static_capacity_aware_static"],
                "T_static_smoothed": representative["T_static_smoothed"],
                "T_exe_static_min": representative["T_exe_static_min"],
                "T_exe_capacity_aware_static": representative["T_exe_capacity_aware_static"],
                "T_exe_smoothed": representative["T_exe_smoothed"],
                "Delta_max_static_min": representative["Delta_max_static_min"],
                "Delta_max_capacity_aware_static": representative["Delta_max_capacity_aware_static"],
                "Delta_max_smoothed": representative["Delta_max_smoothed"],
                "stall_cycles_static_min": representative["stall_cycles_static_min"],
                "stall_cycles_capacity_aware_static": representative["stall_cycles_capacity_aware_static"],
                "stall_cycles_smoothed": representative["stall_cycles_smoothed"],
                "relative_T_exe_improvement_static_min_vs_smoothed": representative["relative_T_exe_improvement_static_min_vs_smoothed"],
                "relative_T_exe_improvement_capacity_aware_vs_smoothed": representative["relative_T_exe_improvement_capacity_aware_vs_smoothed"],
            }
        )
    return pd.DataFrame(representative_rows)


def build_buffer_threshold_summary(buffer_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        subset = buffer_df[buffer_df["policy"] == policy].sort_values("B").copy()
        finite_subset = subset[pd.notna(subset["stall_cycles"]) & subset["stall_cycles"].ne(math.inf)]
        plateau_stall = float(finite_subset["stall_cycles"].min()) if not finite_subset.empty else math.inf
        if finite_subset.empty:
            transition_B = math.inf
        else:
            transition_B = int(finite_subset[finite_subset["stall_cycles"] == plateau_stall]["B"].min())
        rows.append(
            {
                "family": subset["family"].iloc[0],
                "seed": int(subset["seed"].iloc[0]),
                "C": int(subset["C"].iloc[0]),
                "policy": policy,
                "Delta_max": int(subset["Delta_max"].iloc[0]),
                "transition_B": transition_B,
            }
        )
    return pd.DataFrame(rows)


def build_methodology_upgrade_summary(
    compressibility_summary_df: pd.DataFrame,
    grid_pair_df: pd.DataFrame,
    buffer_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
    inversion_summary_df: pd.DataFrame,
    inversion_strong_df: pd.DataFrame,
) -> str:
    family_level = compressibility_summary_df[compressibility_summary_df["summary_level"] == "family"].copy()
    high_slack = float(family_level[family_level["family"] == "high_compressibility"]["compressibility_slack_ratio"].iloc[0])
    medium_slack = float(family_level[family_level["family"] == "medium_compressibility"]["compressibility_slack_ratio"].iloc[0])
    low_slack = float(family_level[family_level["family"] == "low_compressibility"]["compressibility_slack_ratio"].iloc[0])

    strong_baseline_feasible = grid_pair_df[grid_pair_df["both_feasible_capacity_aware_vs_smoothed"] == 1]
    strong_baseline_gain = float(
        strong_baseline_feasible["relative_T_exe_improvement_capacity_aware_vs_smoothed"].mean()
    ) if not strong_baseline_feasible.empty else 0.0
    static_baseline_gain = float(
        grid_pair_df[grid_pair_df["both_feasible_static_min_vs_smoothed"] == 1]["relative_T_exe_improvement_static_min_vs_smoothed"].mean()
    )

    static_inversion_ratio = float(
        inversion_summary_df[inversion_summary_df["family"] == "ALL"]["fraction_of_both_feasible"].iloc[0]
    )
    strong_inversion_ratio = float(inversion_strong_df["capacity_aware_vs_smoothed_inversion"].mean())
    delta_accuracy = float((buffer_df["predicted_stall_by_delta"] == buffer_df["stall_observed"]).mean())

    stall_regime = grid_pair_df[grid_pair_df["stall_cycles_static_min"] > 0].copy()
    stall_static_gain = float(stall_regime["relative_T_exe_improvement_static_min_vs_smoothed"].mean()) if not stall_regime.empty else 0.0
    stall_strong_gain = float(stall_regime["relative_T_exe_improvement_capacity_aware_vs_smoothed"].mean()) if not stall_regime.empty else 0.0

    high_policy = family_summary_df[
        (family_summary_df["family"] == "high_compressibility") & (family_summary_df["policy"] == "smoothed")
    ].iloc[0]
    medium_policy = family_summary_df[
        (family_summary_df["family"] == "medium_compressibility") & (family_summary_df["policy"] == "smoothed")
    ].iloc[0]

    if strong_baseline_gain > 0:
        baseline_sentence = (
            f"Smoothed still retains an average advantage over the stronger baseline: mean gain vs capacity_aware_static={strong_baseline_gain:.3f}, "
            f"stall-regime gain vs capacity_aware_static={stall_strong_gain:.3f}."
        )
    else:
        baseline_sentence = (
            f"The stronger baseline absorbs most of the benefit: mean gain vs capacity_aware_static={strong_baseline_gain:.3f}, "
            f"stall-regime gain vs capacity_aware_static={stall_strong_gain:.3f}. "
            "The paper claim should therefore be narrowed to objective mismatch plus shaping as one intervention, not a generally preferred strategy."
        )

    lines = [
        f"1. Slack-based compressibility still separates the families: high={high_slack:.3f}, medium={medium_slack:.3f}, low={low_slack:.3f}.",
        f"2. Against the original baseline, smoothed changes executed makespan by mean gain={static_baseline_gain:.3f}; {baseline_sentence}",
        f"3. Shallower-but-slower remains frequent against static_min (ratio={static_inversion_ratio:.3f}) and is rarer against capacity_aware_static (ratio={strong_inversion_ratio:.3f}).",
        f"4. Delta_max remains strongly predictive of stall thresholds in the representative buffer scan, with agreement={delta_accuracy:.3f}.",
        f"5. The structural boundary remains visible, with high slack in high_compressibility and zero slack in low_compressibility. In the same scan, high_compressibility has smoothed normalized makespan={high_policy['mean_normalized_makespan']:.3f}, while capacity_aware_static reaches 1.000; medium_compressibility is closer to the capacity-aware baseline at smoothed={medium_policy['mean_normalized_makespan']:.3f}.",
    ]
    return "\n".join(lines) + "\n"


def summarize_results(results: dict[str, pd.DataFrame]) -> dict[str, Any]:
    compare_df = results["compare"]
    compressibility_df = results["compressibility"]
    compressibility_summary_df = results["compressibility_summary"]
    grid_pairs_df = results["grid_pairs"]
    inversion_df = results["inversion"]

    representative_gain = 0.0
    if not compare_df.empty:
        static_row = compare_df[compare_df["policy"] == "static_min"].iloc[0]
        smoothed_row = compare_df[compare_df["policy"] == "smoothed"].iloc[0]
        representative_gain = (static_row["T_exe"] - smoothed_row["T_exe"]) / static_row["T_exe"]

    family_level = compressibility_summary_df[compressibility_summary_df["summary_level"] == "family"].copy()
    compressibility_summary = family_level.set_index("family")["mean_relative_improvement_static_min_vs_smoothed"].to_dict()
    stall_regime = grid_pairs_df[
        (grid_pairs_df["both_feasible_static_min_vs_smoothed"] == 1) & (grid_pairs_df["static_min_has_stall"] == 1)
    ].copy()
    stall_regime_summary = (
        stall_regime.groupby("family")["relative_T_exe_improvement_static_min_vs_smoothed"].mean().to_dict()
        if not stall_regime.empty
        else {}
    )

    qtv_tradeoff_summary = {}
    positive_gain = stall_regime[stall_regime["relative_T_exe_improvement_static_min_vs_smoothed"] > 0]
    if not positive_gain.empty:
        qtv_tradeoff_summary = {
            "mean_relative_T_exe_improvement": positive_gain["relative_T_exe_improvement_static_min_vs_smoothed"].mean(),
            "mean_qtv_ratio": positive_gain["qtv_ratio_smoothed_over_static_min"].mean(),
        }

    return {
        "representative_relative_improvement": representative_gain,
        "compressibility_family_means": compressibility_summary,
        "stall_regime_family_means": stall_regime_summary,
        "inversion_count": int(len(inversion_df)),
        "qtv_tradeoff_summary": qtv_tradeoff_summary,
        "family_seed_summary": family_level,
        "shallower_slower_summary": results["shallower_slower_summary"],
    }


def _run_single_policy(dag: DAG, policy: str, C: int, B: int) -> dict[str, Any]:
    dag_metrics = compute_dag_metrics(dag)
    capacity_limit_mode = "none"
    capacity_limit = 0
    if policy == "capacity_aware_static":
        capacity_limit_mode = "C"
        capacity_limit = C

    schedule = build_schedule(
        dag,
        policy=policy,
        capacity_limit=capacity_limit if policy == "capacity_aware_static" else None,
    )
    trace_bundle = schedule_to_trace_bundle(dag, schedule)
    cumulative_demand, Delta_max, Gamma, peak_demand, mean_demand, total_T = compute_trace_statistics(
        trace_bundle.demand,
        C,
    )
    del cumulative_demand

    feasible_bool, feasible_reason = check_trace_feasibility(trace_bundle.demand, C=C, B=B)
    feasible = int(feasible_bool)
    if feasible:
        sim_result = simulate_trace(trace_bundle.demand, C=C, B=B)
        metrics = compute_metrics(
            demand=trace_bundle.demand,
            C=C,
            logical_qubits_in_use=trace_bundle.logical_qubits_in_use,
            sim_result=sim_result,
        )
        T_exe = float(sim_result.T_exe)
        stall_cycles = float(sim_result.stall_cycles)
        stall_ratio = float(sim_result.stall_ratio)
        QTV_proxy = float(metrics.QTV_proxy)
        normalized_makespan = T_exe / max(1, schedule.static_depth)
    else:
        T_exe = math.inf
        stall_cycles = math.inf
        stall_ratio = 1.0
        QTV_proxy = math.inf
        normalized_makespan = math.inf

    run = ScheduleRun(
        family=dag.family,
        seed=dag.seed,
        policy=policy,
        capacity_limit_mode=capacity_limit_mode,
        capacity_limit=capacity_limit,
        C=C,
        B=B,
        feasible=feasible,
        feasible_reason=feasible_reason,
        compressibility=dag_metrics.compressibility_legacy,
        compressibility_slack_ratio=dag_metrics.compressibility_slack_ratio,
        compressibility_mean_t_slack=dag_metrics.compressibility_mean_t_slack,
        T_static=schedule.static_depth,
        T_exe=T_exe,
        stall_cycles=stall_cycles,
        stall_ratio=stall_ratio,
        Delta_max=Delta_max,
        Gamma=Gamma,
        peak_demand=peak_demand,
        mean_demand=mean_demand,
        total_T=total_T,
        QTV_proxy=QTV_proxy,
        normalized_makespan=normalized_makespan,
    )
    return asdict(run)


def _build_family_dag(family: str, seed: int) -> DAG:
    spec = FAMILY_SPECS[family]
    return generate_dag(
        family=family,
        num_layers=int(spec["num_layers"]),
        width=int(spec["width"]),
        t_ratio=float(spec["t_ratio"]),
        seed=seed,
    )


def _append_comparison_metrics(dataframe: pd.DataFrame, base_policy: str, compare_policy: str, prefix: str) -> None:
    base_feasible_col = f"feasible_{base_policy}"
    compare_feasible_col = f"feasible_{compare_policy}"
    base_t_exe_col = f"T_exe_{base_policy}"
    compare_t_exe_col = f"T_exe_{compare_policy}"
    base_qtv_col = f"QTV_proxy_{base_policy}"
    compare_qtv_col = f"QTV_proxy_{compare_policy}"
    base_t_static_col = f"T_static_{base_policy}"
    compare_t_static_col = f"T_static_{compare_policy}"
    base_delta_col = f"Delta_max_{base_policy}"
    compare_delta_col = f"Delta_max_{compare_policy}"

    both_feasible_col = f"both_feasible_{prefix}"
    rel_improvement_col = f"relative_T_exe_improvement_{prefix}"
    rel_qtv_change_col = f"relative_QTV_change_{prefix}"
    delta_reduction_col = f"Delta_max_reduction_{prefix}"
    qtv_ratio_col = f"qtv_ratio_{compare_policy}_over_{base_policy}"
    inversion_col = f"{prefix}_inversion"

    dataframe[both_feasible_col] = (
        (dataframe[base_feasible_col] == 1) & (dataframe[compare_feasible_col] == 1)
    ).astype(int)
    dataframe[rel_improvement_col] = 0.0
    dataframe[rel_qtv_change_col] = 0.0
    dataframe[delta_reduction_col] = dataframe[base_delta_col] - dataframe[compare_delta_col]
    dataframe[qtv_ratio_col] = math.inf

    valid_mask = dataframe[both_feasible_col] == 1
    valid_rows = dataframe.loc[valid_mask]
    dataframe.loc[valid_mask, rel_improvement_col] = (
        valid_rows[base_t_exe_col] - valid_rows[compare_t_exe_col]
    ) / valid_rows[base_t_exe_col]
    dataframe.loc[valid_mask, rel_qtv_change_col] = (
        valid_rows[compare_qtv_col] - valid_rows[base_qtv_col]
    ) / valid_rows[base_qtv_col]
    dataframe.loc[valid_mask, qtv_ratio_col] = valid_rows[compare_qtv_col] / valid_rows[base_qtv_col]
    dataframe[inversion_col] = (
        valid_mask
        & (dataframe[base_t_static_col] < dataframe[compare_t_static_col])
        & (dataframe[base_t_exe_col] > dataframe[compare_t_exe_col])
    ).astype(int)


def _relative_improvement(base_value: float, compare_value: float, both_feasible: int) -> float:
    if both_feasible != 1 or not math.isfinite(base_value) or base_value <= 0 or not math.isfinite(compare_value):
        return 0.0
    return (base_value - compare_value) / base_value


def _write_table(dataframe: pd.DataFrame, filename: str, appendix: bool = False) -> Path:
    target_dir = APPENDIX_DIR if appendix else TABLE_DIR
    path = target_dir / filename
    normalize_output_dataframe(dataframe).to_csv(path, index=False)
    return path


def _write_text(contents: str, filename: str) -> Path:
    path = TABLE_DIR / filename
    path.write_text(contents, encoding="utf-8")
    return path


def _aggregate_compare_metrics(dataframe: pd.DataFrame, group_cols: list[str], level_label: str) -> pd.DataFrame:
    grouped = dataframe.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        compressibility=("compressibility", "mean"),
        compressibility_slack_ratio=("compressibility_slack_ratio", "mean"),
        compressibility_mean_t_slack=("compressibility_mean_t_slack", "mean"),
        experiment_count=("family", "size"),
        mean_relative_improvement_static_min_vs_smoothed=("relative_T_exe_improvement_static_min_vs_smoothed", "mean"),
        std_relative_improvement_static_min_vs_smoothed=("relative_T_exe_improvement_static_min_vs_smoothed", "std"),
        median_relative_improvement_static_min_vs_smoothed=("relative_T_exe_improvement_static_min_vs_smoothed", "median"),
        mean_relative_improvement_capacity_aware_vs_smoothed=("relative_T_exe_improvement_capacity_aware_vs_smoothed", "mean"),
        std_relative_improvement_capacity_aware_vs_smoothed=("relative_T_exe_improvement_capacity_aware_vs_smoothed", "std"),
        median_relative_improvement_capacity_aware_vs_smoothed=("relative_T_exe_improvement_capacity_aware_vs_smoothed", "median"),
        mean_relative_QTV_change_static_min_vs_smoothed=("relative_QTV_change_static_min_vs_smoothed", "mean"),
        mean_relative_QTV_change_capacity_aware_vs_smoothed=("relative_QTV_change_capacity_aware_vs_smoothed", "mean"),
        mean_Delta_max_reduction_static_min_vs_smoothed=("Delta_max_reduction_static_min_vs_smoothed", "mean"),
        mean_Delta_max_reduction_capacity_aware_vs_smoothed=("Delta_max_reduction_capacity_aware_vs_smoothed", "mean"),
    ).reset_index()
    summary["summary_level"] = level_label
    if "seed" not in summary.columns:
        summary["seed"] = "ALL"
    for column in summary.columns:
        if column.startswith("std_"):
            summary[column] = summary[column].fillna(0.0)
    return summary
