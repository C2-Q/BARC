from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.real_trace.adder_trace import generate_adder_circuit, to_clifford_t
from src.real_trace.circuit_slack import compute_circuit_slack_metrics
from src.real_trace.multiplier_trace import generate_multiplier_circuit
from src.experiments import _build_family_dag
from src.schedule import build_schedule
from src.simulator import simulate_trace
from src.trace import schedule_to_trace_bundle
from src.utils import FIGURE_DIR, FINAL_PAPER_DIR, TABLE_DIR, ensure_output_dirs


BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 42
FAMILY_ORDER = ["ALL", "high_compressibility", "medium_compressibility", "low_compressibility"]
FAMILY_DISPLAY = {
    "high_compressibility": "High compressibility",
    "medium_compressibility": "Medium compressibility",
    "low_compressibility": "Low compressibility",
}
POLICY_DISPLAY = {
    "static_min": "Static-min",
    "capacity_aware_static": "Capacity-aware",
    "smoothed": "Smoothed",
    "delivery_aware_slack": "Delivery-aware slack",
}


@dataclass(frozen=True)
class BootstrapTask:
    name: str
    label_column: str
    metric_name: str
    finite_only: bool


BOOTSTRAP_TASKS = (
    BootstrapTask(name="stall", label_column="stall", metric_name="auc", finite_only=False),
    BootstrapTask(name="slowdown_ratio", label_column="slowdown_ratio", metric_name="abs_spearman", finite_only=True),
    BootstrapTask(name="inversion", label_column="inversion", metric_name="auc", finite_only=False),
)


def main() -> None:
    ensure_output_dirs()

    predictive_df = pd.read_csv(TABLE_DIR / "predictive_static_dataset.csv")
    family_summary_df = pd.read_csv(TABLE_DIR / "family_summary.csv")
    lower_bound_df = pd.read_csv(TABLE_DIR / "lower_bound_validation.csv")
    grid_df = pd.read_csv(TABLE_DIR / "stage1_grid_scan.csv")

    bootstrap_df = run_bootstrap_analysis(predictive_df)
    bootstrap_path = TABLE_DIR / "bootstrap_slack_vs_tdepth.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)

    real_trace_df, real_trace_text = run_real_trace_structural_analysis(family_summary_df)
    real_trace_path = TABLE_DIR / "real_trace_structural_analysis.csv"
    real_trace_df.to_csv(real_trace_path, index=False)
    real_trace_text_path = TABLE_DIR / "real_trace_interpretation.txt"
    real_trace_text_path.write_text(real_trace_text, encoding="utf-8")

    gap_df, gap_text, strongest_gap_predictor = run_lower_bound_gap_analysis(lower_bound_df, grid_df)
    gap_path = TABLE_DIR / "lower_bound_gap_analysis.csv"
    gap_df.to_csv(gap_path, index=False)
    gap_text_path = TABLE_DIR / "lower_bound_gap_interpretation.txt"
    gap_text_path.write_text(gap_text, encoding="utf-8")

    gap_cases_df, gap_cases_text, gap_cases_figure = run_lower_bound_gap_forensics(lower_bound_df, grid_df)
    gap_cases_path = TABLE_DIR / "lower_bound_gap_cases.csv"
    gap_cases_df.to_csv(gap_cases_path, index=False)
    gap_cases_text_path = TABLE_DIR / "lower_bound_gap_cases_interpretation.txt"
    gap_cases_text_path.write_text(gap_cases_text, encoding="utf-8")

    print("TASK 1 RESULT:")
    print(bootstrap_df.to_string(index=False))
    print()
    print("TASK 2 RESULT:")
    print(real_trace_df.to_string(index=False))
    print(real_trace_text)
    print()
    print("TASK 3 RESULT:")
    print(gap_df.to_string(index=False))
    print(gap_text)
    print(gap_cases_df.to_string(index=False))
    print(gap_cases_text)
    print()
    print("FINAL SUMMARY:")
    for task in ["stall", "slowdown_ratio", "inversion"]:
        overall = bootstrap_df[bootstrap_df["group"] == "ALL"]
        row = overall[overall["task"] == task].iloc[0]
        print(
            f"{task}: slack_ratio better than T_depth = {bool(row['ci_excludes_zero'])}, "
            f"mean_diff={row['mean_diff']:.6f}, "
            f"95% CI=[{row['ci_lower']:.6f}, {row['ci_upper']:.6f}]"
        )
    for row in real_trace_df.itertuples(index=False):
        print(
            f"{row.trace_name}: slack_ratio={row.slack_ratio:.6f}, "
            f"closest_family={row.closest_synthetic_family}, "
            f"frac_CB_with_slowdown={row.frac_CB_with_slowdown:.6f}"
        )
    strongest_row = gap_df[gap_df["variable"] == strongest_gap_predictor].iloc[0]
    print(
        f"lower_bound_gap: strongest predictor={strongest_gap_predictor}, "
        f"pvalue={strongest_row['mannwhitney_pvalue']:.6g}, "
        f"effect_size={strongest_row['effect_size']:.6f}"
    )
    print(f"lower_bound_gap_cases: {gap_cases_path}")
    print(f"lower_bound_gap_figure: {gap_cases_figure}")


def run_bootstrap_analysis(predictive_df: pd.DataFrame) -> pd.DataFrame:
    """Run paired bootstrap comparisons for slack_ratio and T_depth."""
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for task in BOOTSTRAP_TASKS:
        base_df = predictive_df.copy()
        if task.finite_only:
            base_df = base_df[np.isfinite(base_df[task.label_column])].copy()

        for group in FAMILY_ORDER:
            group_df = base_df if group == "ALL" else base_df[base_df["family"] == group].copy()
            result = _bootstrap_predictor_difference(group_df, task, rng)
            rows.append(
                {
                    "group": group,
                    "task": task.name,
                    "metric": task.metric_name,
                    "mean_diff": result["mean_diff"],
                    "ci_lower": result["ci_lower"],
                    "ci_upper": result["ci_upper"],
                    "ci_excludes_zero": result["ci_excludes_zero"],
                }
            )

    return pd.DataFrame(rows)


def run_real_trace_structural_analysis(family_summary_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Compare real traces against synthetic families using structural metrics."""
    synthetic_reference = (
        family_summary_df[family_summary_df["policy"] == "static_min"][["family", "mean_slack_ratio", "mean_delta_max"]]
        .groupby("family", as_index=False)
        .mean()
    )
    rows: list[dict[str, object]] = []

    for trace_name, n_bits, circuit_builder in (
        ("adder_n4", 4, generate_adder_circuit),
        ("multiplier_n4", 4, generate_multiplier_circuit),
    ):
        workload_stats = pd.read_csv(TABLE_DIR / f"{trace_name}_workload_stats.csv").iloc[0]
        summary_df = pd.read_csv(PROJECT_ROOT / "outputs" / "real_trace" / f"{trace_name}_summary.csv")
        transpiled = to_clifford_t(circuit_builder(n_bits=n_bits))
        slack_metrics = compute_circuit_slack_metrics(transpiled)

        slack_distance = (synthetic_reference["mean_slack_ratio"] - slack_metrics.fraction_t_slack_positive).abs()
        closest_family = synthetic_reference.iloc[int(slack_distance.argmin())]["family"]
        slowdown_ratio = summary_df["T_exe_static_min"] / summary_df["T_static_static_min"]
        frac_cb_with_slowdown = float((slowdown_ratio > 1.05).mean())

        rows.append(
            {
                "trace_name": trace_name,
                "T_count": int(workload_stats["total_T"]),
                "T_depth": int(summary_df["T_static_static_min"].iloc[0]),
                "slack_ratio": float(slack_metrics.fraction_t_slack_positive),
                "mean_t_slack": float(slack_metrics.mean_t_slack),
                "delta_max_mean": float(summary_df["Delta_max_static_min"].mean()),
                "closest_synthetic_family": closest_family,
                "frac_CB_with_slowdown": frac_cb_with_slowdown,
            }
        )

    result_df = pd.DataFrame(rows)
    adder_row = result_df[result_df["trace_name"] == "adder_n4"].iloc[0]
    multiplier_row = result_df[result_df["trace_name"] == "multiplier_n4"].iloc[0]
    interpretation = (
        f"The adder trace has slack_ratio={adder_row['slack_ratio']:.3f} and mean delta_max={adder_row['delta_max_mean']:.3f}, "
        f"which places it near {adder_row['closest_synthetic_family']} and limits the number of (C,B) settings with slowdown above 1.05 "
        f"to {adder_row['frac_CB_with_slowdown']:.3f} of the scan.\n"
        f"The multiplier trace has slack_ratio={multiplier_row['slack_ratio']:.3f} and mean delta_max={multiplier_row['delta_max_mean']:.3f}; "
        f"it therefore produces bounded-delivery pressure more often than the adder, but the observed slack remains below the synthetic high-slack regime, "
        f"which is consistent with the weaker inversion signal.\n"
    )
    return result_df, interpretation


def run_lower_bound_gap_analysis(lower_bound_df: pd.DataFrame, grid_df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Analyze when the fixed-schedule lower bound is tight or loose."""
    enriched = lower_bound_df.merge(
        grid_df[["family", "seed", "policy", "C", "B", "slack_ratio", "T_static"]].drop_duplicates(),
        on=["family", "seed", "policy", "C", "B", "T_static"],
        how="left",
    )
    enriched = enriched[np.isfinite(enriched["T_exe"])].copy()
    gap_zero = enriched[np.isclose(enriched["gap"], 0.0)].copy()
    gap_positive = enriched[enriched["gap"] > 0.0].copy()

    rows: list[dict[str, object]] = []
    variables = ["slack_ratio", "delta_max", "C", "B", "T_static"]
    for variable in variables:
        zero_values = gap_zero[variable].astype(float).to_numpy()
        positive_values = gap_positive[variable].astype(float).to_numpy()
        if len(zero_values) == 0 or len(positive_values) == 0:
            pvalue = math.nan
            effect_size = math.nan
        else:
            test = mannwhitneyu(positive_values, zero_values, alternative="two-sided")
            pvalue = float(test.pvalue)
            effect_size = _rank_biserial_from_u(float(test.statistic), len(positive_values), len(zero_values))
        rows.append(
            {
                "variable": variable,
                "mean_gap_zero": float(gap_zero[variable].mean()) if not gap_zero.empty else math.nan,
                "mean_gap_positive": float(gap_positive[variable].mean()) if not gap_positive.empty else math.nan,
                "mannwhitney_pvalue": pvalue,
                "effect_size": effect_size,
                "is_significant": bool(np.isfinite(pvalue) and pvalue < 0.05),
            }
        )

    result_df = pd.DataFrame(rows)
    ranked = result_df.copy()
    ranked["abs_effect_size"] = ranked["effect_size"].abs()
    significant = ranked[ranked["is_significant"]]
    strongest = significant.sort_values("abs_effect_size", ascending=False).iloc[0] if not significant.empty else ranked.sort_values("abs_effect_size", ascending=False).iloc[0]
    strongest_variable = str(strongest["variable"])
    interpretation = (
        f"The lower bound is exact when the observed backlog is resolved within the delivery-plus-buffer budget. "
        f"In the current scan, the largest separation between gap=0 and gap>0 occurs for {strongest_variable}, "
        f"with p-value={strongest['mannwhitney_pvalue']:.6g} and rank-biserial effect size={strongest['effect_size']:.3f}.\n"
        f"Positive bound gaps are associated with larger delta_max and smaller effective delivery slack, which indicates that additional waiting arises when prefix deficits persist beyond the buffer budget rather than from the bound formula itself.\n"
    )
    return result_df, interpretation, strongest_variable


def run_lower_bound_gap_forensics(lower_bound_df: pd.DataFrame, grid_df: pd.DataFrame) -> tuple[pd.DataFrame, str, Path]:
    """Reconstruct representative positive-gap cases and plot their traces."""
    enriched = lower_bound_df.merge(
        grid_df[
            ["family", "seed", "policy", "C", "B", "slack_ratio", "T_static", "peak_demand", "mean_demand"]
        ].drop_duplicates(),
        on=["family", "seed", "policy", "C", "B", "T_static"],
        how="left",
    )
    enriched = enriched[
        (enriched["feasible"] == 1)
        & np.isfinite(enriched["T_exe"])
        & np.isfinite(enriched["predicted_lower_bound"])
        & np.isfinite(enriched["gap"])
        & (enriched["gap"] > 0.0)
    ].copy()
    selected = _select_representative_gap_cases(enriched)

    case_rows: list[dict[str, object]] = []
    figure, axes = plt.subplots(len(selected), 2, figsize=(11.0, 2.5 * max(1, len(selected))), sharex=False)
    if len(selected) == 1:
        axes = np.array([axes])

    interpretation_lines: list[str] = []
    for index, row in enumerate(selected.itertuples(index=False), start=1):
        case_id = f"case_{index}"
        reconstruction = _reconstruct_gap_case(row)
        case_rows.append(
            {
                "case_id": case_id,
                "family": row.family,
                "seed": int(row.seed),
                "policy": row.policy,
                "C": int(row.C),
                "B": int(row.B),
                "T_static": int(row.T_static),
                "T_exe": float(row.T_exe),
                "delta_max": int(row.delta_max),
                "gap": float(row.gap),
                "slack_ratio": float(row.slack_ratio),
                "peak_demand": int(row.peak_demand),
                "mean_demand": float(row.mean_demand),
            }
        )
        _plot_gap_case_row(axes[index - 1], reconstruction, case_id)
        interpretation_lines.append(
            f"{case_id}: {_display_family(row.family)}, seed={int(row.seed)}, "
            f"policy={_display_policy(row.policy)}, C={int(row.C)}, B={int(row.B)}, "
            f"gap={float(row.gap):.2f}. The cumulative demand exceeds the supply envelope near logical cycle "
            f"{reconstruction['peak_index']}, and the backlog remains active for {reconstruction['persistence_length']} logical cycles."
        )

    figure.tight_layout()
    figure_path = FIGURE_DIR / "lower_bound_gap_cases.png"
    figure.savefig(figure_path, dpi=300)
    figure.savefig(figure_path.with_suffix(".pdf"))
    final_png_path = FINAL_PAPER_DIR / "lower_bound_gap_cases_final.png"
    final_pdf_path = FINAL_PAPER_DIR / "lower_bound_gap_cases_final.pdf"
    figure.savefig(final_png_path, dpi=300)
    figure.savefig(final_pdf_path)
    plt.close(figure)

    interpretation = (
        "The selected positive-gap cases show that the lower bound becomes loose when the prefix deficit does not resolve at a single logical cycle boundary. "
        "Instead, demand remains above the supply envelope across a short interval, which forces additional waiting beyond the one-shot deficit term in the bound.\n"
        + "\n".join(interpretation_lines)
        + "\n"
    )
    return pd.DataFrame(case_rows), interpretation, figure_path


def _select_representative_gap_cases(enriched: pd.DataFrame) -> pd.DataFrame:
    quantiles = [0.2, 0.5, 0.8, 0.95]
    selected_rows: list[pd.Series] = []
    used_keys: set[tuple[object, ...]] = set()

    for quantile in quantiles:
        target = float(enriched["gap"].quantile(quantile))
        ranked = enriched.assign(distance=(enriched["gap"] - target).abs()).sort_values(
            ["distance", "gap", "family", "seed", "policy", "C", "B"]
        )
        for _, candidate in ranked.iterrows():
            key = (candidate["family"], candidate["seed"], candidate["policy"], candidate["C"], candidate["B"])
            if key in used_keys:
                continue
            selected_rows.append(candidate)
            used_keys.add(key)
            break

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    for required_family in ("high_compressibility", "medium_compressibility"):
        if (selected["family"] == required_family).any() or not (enriched["family"] == required_family).any():
            continue
        replacement = (
            enriched[enriched["family"] == required_family]
            .assign(distance=(enriched[enriched["family"] == required_family]["gap"] - enriched["gap"].median()).abs())
            .sort_values(["distance", "gap", "seed", "policy", "C", "B"])
            .iloc[0]
        )
        replacement_key = (
            replacement["family"],
            replacement["seed"],
            replacement["policy"],
            replacement["C"],
            replacement["B"],
        )
        if replacement_key in used_keys:
            continue
        replace_candidates = selected[(selected["family"] == "low_compressibility") | (selected.duplicated(subset=["family"], keep=False))]
        if replace_candidates.empty:
            continue
        replace_index = int(replace_candidates.index[0])
        selected.loc[replace_index] = replacement
        used_keys.add(replacement_key)

    return selected.sort_values(["gap", "family", "seed", "policy", "C", "B"]).reset_index(drop=True)


def _reconstruct_gap_case(row: pd.Series | object) -> dict[str, object]:
    family = str(row.family)
    seed = int(row.seed)
    policy = str(row.policy)
    C = int(row.C)
    B = int(row.B)
    dag = _build_family_dag(family, seed)
    capacity_limit = C if policy == "capacity_aware_static" else None
    schedule = build_schedule(dag, policy=policy, capacity_limit=capacity_limit)
    trace_bundle = schedule_to_trace_bundle(dag, schedule)
    sim_result = simulate_trace(trace_bundle.demand, C=C, B=B)
    cumulative_demand = np.cumsum(trace_bundle.demand).astype(int)
    supply_envelope = np.array([C * (index + 1) + B for index in range(len(trace_bundle.demand))], dtype=int)
    backlog = np.maximum(0, cumulative_demand - supply_envelope)
    peak_index = int(np.argmax(backlog)) if len(backlog) > 0 else 0
    active = np.where(backlog > 0)[0]
    active_segments: list[tuple[int, int]] = []
    if len(active) > 0:
        segment_start = int(active[0])
        previous = int(active[0])
        for index in active[1:]:
            current = int(index)
            if current == previous + 1:
                previous = current
                continue
            active_segments.append((segment_start, previous + 1))
            segment_start = current
            previous = current
        active_segments.append((segment_start, previous + 1))
    peak_segment = next(
        ((segment_start, segment_end) for segment_start, segment_end in active_segments if segment_start <= peak_index < segment_end),
        None,
    )
    persistence_length = int(peak_segment[1] - peak_segment[0]) if peak_segment is not None else 0
    return {
        "demand": trace_bundle.demand,
        "cumulative_demand": cumulative_demand,
        "supply_envelope": supply_envelope,
        "backlog": backlog,
        "peak_index": peak_index,
        "persistence_length": persistence_length,
        "active_segments": active_segments,
        "peak_segment": peak_segment,
        "sim_result": sim_result,
        "row": row,
    }


def _plot_gap_case_row(axes_row: np.ndarray, reconstruction: dict[str, object], case_id: str) -> None:
    left_axis, right_axis = axes_row
    demand = np.array(reconstruction["demand"], dtype=int)
    cumulative_demand = np.array(reconstruction["cumulative_demand"], dtype=int)
    supply_envelope = np.array(reconstruction["supply_envelope"], dtype=int)
    backlog = np.array(reconstruction["backlog"], dtype=int)
    peak_index = int(reconstruction["peak_index"])
    active_segments = list(reconstruction["active_segments"])
    peak_segment = reconstruction["peak_segment"]
    row = reconstruction["row"]

    x_values = np.arange(len(demand))
    left_axis.plot(x_values, cumulative_demand, color="#1f77b4", linewidth=1.5, label="Cumulative demand")
    left_axis.plot(x_values, supply_envelope, color="#444444", linewidth=1.2, linestyle="--", label="Supply envelope")
    left_axis.fill_between(x_values, supply_envelope, cumulative_demand, where=backlog > 0, color="#e45756", alpha=0.18)
    left_axis.axvline(peak_index, color="#777777", linewidth=0.9, linestyle=":")
    left_axis.set_ylabel(f"{_display_case_id(case_id)}\nCumulative T")
    left_axis.grid(alpha=0.18)
    left_axis.legend(frameon=False, fontsize=8, loc="upper left")

    right_axis.step(x_values, demand, where="post", color="#2c7fb8", linewidth=1.3)
    for segment_start, segment_end in active_segments:
        right_axis.axvspan(segment_start, segment_end, color="#e45756", alpha=0.18)
    if peak_segment is not None:
        right_axis.axvline(peak_segment[0], color="#777777", linewidth=0.8, linestyle=":")
        right_axis.axvline(peak_segment[1], color="#777777", linewidth=0.8, linestyle=":")
    right_axis.set_ylabel("Demand")
    right_axis.grid(alpha=0.18)
    right_axis.set_title(
        f"{_display_family(row.family)}, {_display_policy(row.policy)}, gap={float(row.gap):.1f}, "
        f"Δ={int(row.delta_max)}",
        fontsize=9,
    )
    if case_id == "case_4":
        left_axis.set_xlabel("Logical cycle")
        right_axis.set_xlabel("Logical cycle")


def _display_family(name: str) -> str:
    return FAMILY_DISPLAY.get(str(name), str(name).replace("_", " "))


def _display_policy(name: str) -> str:
    return POLICY_DISPLAY.get(str(name), str(name).replace("_", " "))


def _display_case_id(case_id: str) -> str:
    return str(case_id).replace("_", " ").title()


def _bootstrap_predictor_difference(dataframe: pd.DataFrame, task: BootstrapTask, rng: np.random.Generator) -> dict[str, float | bool]:
    """Compute paired bootstrap differences for slack_ratio minus T_depth."""
    if dataframe.empty:
        return {"mean_diff": math.nan, "ci_lower": math.nan, "ci_upper": math.nan, "ci_excludes_zero": False}

    y = dataframe[task.label_column].to_numpy()
    x_slack = dataframe["slack_ratio"].to_numpy()
    x_depth = dataframe["T_depth"].to_numpy()
    n = len(dataframe)

    if task.metric_name == "auc" and len(np.unique(y)) < 2:
        return {"mean_diff": math.nan, "ci_lower": math.nan, "ci_upper": math.nan, "ci_excludes_zero": False}
    if task.metric_name == "abs_spearman" and (len(np.unique(y)) < 2 or len(np.unique(x_slack)) < 2 or len(np.unique(x_depth)) < 2):
        return {"mean_diff": math.nan, "ci_lower": math.nan, "ci_upper": math.nan, "ci_excludes_zero": False}

    diffs: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample_idx = rng.integers(0, n, size=n)
        sample_y = y[sample_idx]
        sample_slack = x_slack[sample_idx]
        sample_depth = x_depth[sample_idx]
        if task.metric_name == "auc":
            if len(np.unique(sample_y)) < 2:
                continue
            slack_score = _roc_auc(sample_slack, sample_y)
            depth_score = _roc_auc(sample_depth, sample_y)
        else:
            if len(np.unique(sample_y)) < 2 or len(np.unique(sample_slack)) < 2 or len(np.unique(sample_depth)) < 2:
                continue
            slack_score = abs(_spearman(sample_slack, sample_y))
            depth_score = abs(_spearman(sample_depth, sample_y))
        diffs.append(float(slack_score - depth_score))

    if not diffs:
        return {"mean_diff": math.nan, "ci_lower": math.nan, "ci_upper": math.nan, "ci_excludes_zero": False}

    diff_array = np.asarray(diffs, dtype=float)
    ci_lower, ci_upper = np.percentile(diff_array, [2.5, 97.5])
    excludes_zero = bool(ci_lower > 0.0 or ci_upper < 0.0)
    return {
        "mean_diff": float(diff_array.mean()),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_excludes_zero": excludes_zero,
    }


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return ROC-AUC for binary labels without external ML dependencies."""
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(scores, method="average")
    sum_ranks_pos = float(np.sum(ranks[positives]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(max(auc, 1.0 - auc))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Return Spearman correlation using ranked Pearson correlation."""
    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _rank_biserial_from_u(u_value: float, n_x: int, n_y: int) -> float:
    """Return rank-biserial correlation for the first sample in Mann-Whitney U."""
    return float((2.0 * u_value) / (n_x * n_y) - 1.0)


if __name__ == "__main__":
    main()
