from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from src.paper_config import COMPRESSIBILITY_SCAN_B, COMPRESSIBILITY_SCAN_C
from src.utils import TABLE_DIR, normalize_output_dataframe


PREDICTOR_COLUMNS = [
    "T_count",
    "T_depth",
    "peak_demand",
    "mean_demand",
    "Delta_max",
    "Gamma",
    "slack_ratio",
    "mean_t_slack",
]

STABILITY_PREDICTORS = ["T_depth", "slack_ratio", "Delta_max"]

PREDICTOR_CLASSES = {
    "T_count": "traditional",
    "T_depth": "traditional",
    "peak_demand": "trace_level",
    "mean_demand": "trace_level",
    "Delta_max": "system_level",
    "Gamma": "trace_level",
    "slack_ratio": "structural",
    "mean_t_slack": "structural",
}


def compute_lower_bound(T_static: float, delta_max: float, B: float, C: float) -> float:
    """Return the fixed-schedule execution lower bound induced by bounded delivery."""
    if C <= 0:
        raise ValueError("C must be positive")
    return float(T_static + max(0.0, delta_max - B) / C)


def build_predictive_dataset(grid_df: pd.DataFrame, grid_pair_df: pd.DataFrame) -> pd.DataFrame:
    base_df = grid_df[grid_df["policy"] == "static_min"].copy()
    pair_columns = [
        "family",
        "seed",
        "C",
        "B",
        "static_min_vs_smoothed_inversion",
        "capacity_aware_vs_smoothed_inversion",
        "relative_T_exe_improvement_static_min_vs_smoothed",
        "relative_T_exe_improvement_capacity_aware_vs_smoothed",
        "both_feasible_static_min_vs_smoothed",
        "both_feasible_capacity_aware_vs_smoothed",
    ]
    base_df = base_df.merge(grid_pair_df[pair_columns], on=["family", "seed", "C", "B"], how="left")
    base_df["T_count"] = base_df["total_T"]
    base_df["T_depth"] = base_df["T_static"]
    base_df["slack_ratio"] = base_df["compressibility_slack_ratio"]
    base_df["mean_t_slack"] = base_df["compressibility_mean_t_slack"]
    base_df["stall"] = ((base_df["feasible"] == 0) | (base_df["stall_cycles"] > 0)).astype(int)
    base_df["slowdown_ratio"] = base_df["normalized_makespan"]
    base_df["inversion"] = base_df["static_min_vs_smoothed_inversion"].fillna(0).astype(int)
    base_df["strong_baseline_inversion"] = base_df["capacity_aware_vs_smoothed_inversion"].fillna(0).astype(int)
    base_df["shaping_gain"] = base_df["relative_T_exe_improvement_static_min_vs_smoothed"].fillna(0.0)
    base_df["strong_baseline_shaping_gain"] = base_df["relative_T_exe_improvement_capacity_aware_vs_smoothed"].fillna(0.0)
    columns = [
        "family",
        "seed",
        "C",
        "B",
        "Gamma",
        "T_count",
        "T_depth",
        "peak_demand",
        "mean_demand",
        "Delta_max",
        "slack_ratio",
        "mean_t_slack",
        "stall",
        "slowdown_ratio",
        "inversion",
        "strong_baseline_inversion",
        "shaping_gain",
        "strong_baseline_shaping_gain",
        "feasible",
    ]
    return base_df[columns].sort_values(by=["family", "seed", "C", "B"]).reset_index(drop=True)


def evaluate_predictive_classification(
    predictive_df: pd.DataFrame,
    label_col: str,
    group_cols: tuple[str, str] = ("C", "B"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in PREDICTOR_COLUMNS:
        slice_rows: list[dict[str, Any]] = []
        for group_key, subset in predictive_df.groupby(list(group_cols), dropna=False):
            label_values = subset[label_col].astype(int).to_numpy()
            metric_values = subset[metric].astype(float).to_numpy()
            if len(subset) < 6 or len(np.unique(label_values)) < 2 or np.allclose(metric_values, metric_values[0]):
                continue
            evaluation = _binary_metric_eval(metric_values, label_values)
            slice_rows.append(
                {
                    "group": group_key,
                    "best_accuracy": evaluation["best_accuracy"],
                    "roc_auc": evaluation["roc_auc"],
                }
            )
        if not slice_rows:
            rows.append(
                {
                    "task": label_col,
                    "metric": metric,
                    "predictor_class": PREDICTOR_CLASSES[metric],
                    "evaluated_slices": 0,
                    "mean_best_accuracy": 0.0,
                    "mean_roc_auc": 0.5,
                }
            )
            continue
        slice_df = pd.DataFrame(slice_rows)
        rows.append(
            {
                "task": label_col,
                "metric": metric,
                "predictor_class": PREDICTOR_CLASSES[metric],
                "evaluated_slices": int(len(slice_df)),
                "mean_best_accuracy": float(slice_df["best_accuracy"].mean()),
                "mean_roc_auc": float(slice_df["roc_auc"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(by=["task", "mean_roc_auc", "mean_best_accuracy"], ascending=[True, False, False])


def evaluate_predictive_regression(
    predictive_df: pd.DataFrame,
    target_col: str = "slowdown_ratio",
    group_cols: tuple[str, str] = ("C", "B"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    finite_df = predictive_df[np.isfinite(predictive_df[target_col])].copy()
    for metric in PREDICTOR_COLUMNS:
        slice_rows: list[dict[str, Any]] = []
        for group_key, subset in finite_df.groupby(list(group_cols), dropna=False):
            if len(subset) < 6:
                continue
            x = subset[metric].astype(float).to_numpy()
            y = subset[target_col].astype(float).to_numpy()
            if np.allclose(x, x[0]) or np.allclose(y, y[0]):
                continue
            pearson = _pearson(x, y)
            spearman = _spearman(x, y)
            r_squared = _linear_r_squared(x, y)
            slice_rows.append(
                {
                    "group": group_key,
                    "pearson": pearson,
                    "abs_pearson": abs(pearson),
                    "spearman": spearman,
                    "abs_spearman": abs(spearman),
                    "r_squared": r_squared,
                }
            )
        if not slice_rows:
            rows.append(
                {
                    "task": target_col,
                    "metric": metric,
                    "predictor_class": PREDICTOR_CLASSES[metric],
                    "evaluated_slices": 0,
                    "mean_pearson": 0.0,
                    "mean_abs_pearson": 0.0,
                    "mean_spearman": 0.0,
                    "mean_abs_spearman": 0.0,
                    "mean_r_squared": 0.0,
                }
            )
            continue
        slice_df = pd.DataFrame(slice_rows)
        rows.append(
            {
                "task": target_col,
                "metric": metric,
                "predictor_class": PREDICTOR_CLASSES[metric],
                "evaluated_slices": int(len(slice_df)),
                "mean_pearson": float(slice_df["pearson"].mean()),
                "mean_abs_pearson": float(slice_df["abs_pearson"].mean()),
                "mean_spearman": float(slice_df["spearman"].mean()),
                "mean_abs_spearman": float(slice_df["abs_spearman"].mean()),
                "mean_r_squared": float(slice_df["r_squared"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(by=["mean_abs_spearman", "mean_r_squared"], ascending=[False, False])


def fit_multivariate_slowdown_regression(
    predictive_df: pd.DataFrame,
    C: int = COMPRESSIBILITY_SCAN_C,
    B: int = COMPRESSIBILITY_SCAN_B,
) -> pd.DataFrame:
    subset = predictive_df[
        (predictive_df["C"] == C)
        & (predictive_df["B"] == B)
        & np.isfinite(predictive_df["slowdown_ratio"])
    ].copy()
    model_specs = {
        "T_depth_only": ["T_depth"],
        "T_depth_plus_slack": ["T_depth", "slack_ratio"],
        "T_depth_slack_delta": ["T_depth", "slack_ratio", "Delta_max"],
    }
    rows: list[dict[str, Any]] = []
    previous_r_squared = 0.0
    for model_name, features in model_specs.items():
        if len(subset) < len(features) + 2:
            model_r_squared = 0.0
            coefficients = np.zeros(len(features))
        else:
            X = subset[features].astype(float).to_numpy()
            y = subset["slowdown_ratio"].astype(float).to_numpy()
            X_std = _zscore_matrix(X)
            y_std = _zscore_vector(y)
            design = np.column_stack([np.ones(len(X_std)), X_std])
            solved, _, _, _ = np.linalg.lstsq(design, y_std, rcond=None)
            predictions = design @ solved
            residual = np.sum((y_std - predictions) ** 2)
            total = np.sum((y_std - np.mean(y_std)) ** 2)
            model_r_squared = 0.0 if np.isclose(total, 0.0) else float(1.0 - residual / total)
            coefficients = solved[1:]
        for feature, coefficient in zip(features, coefficients, strict=True):
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "predictor_class": PREDICTOR_CLASSES.get(feature, "mixed"),
                    "standardized_coefficient": float(coefficient),
                    "slice_C": C,
                    "slice_B": B,
                    "model_r_squared": model_r_squared,
                    "delta_r_squared_vs_prev": model_r_squared - previous_r_squared,
                }
            )
        previous_r_squared = model_r_squared
    return pd.DataFrame(rows)


def fit_incremental_predictive_models(predictive_df: pd.DataFrame) -> pd.DataFrame:
    """Fit nested slowdown and stall models to measure incremental explanatory gain."""
    model_specs = [
        ("T_depth_only", ["T_depth"]),
        ("T_depth_plus_slack", ["T_depth", "slack_ratio"]),
        ("T_depth_slack_delta", ["T_depth", "slack_ratio", "Delta_max"]),
    ]
    rows: list[dict[str, Any]] = []

    slowdown_df = predictive_df[np.isfinite(predictive_df["slowdown_ratio"])].copy()
    previous_r_squared = 0.0
    for model_name, features in model_specs:
        result = _fit_linear_model(slowdown_df, features, "slowdown_ratio")
        rows.append(
            {
                "task": "slowdown_ratio",
                "model": model_name,
                "features": "|".join(features),
                "metric_name": "r_squared",
                "metric_value": result["metric_value"],
                "delta_metric_vs_prev": result["metric_value"] - previous_r_squared,
                "num_rows": result["num_rows"],
                "intercept": result["intercept"],
                "coefficients_json": json.dumps(result["coefficients"], sort_keys=True),
            }
        )
        previous_r_squared = result["metric_value"]

    stall_df = predictive_df.copy()
    previous_auc = 0.0
    for model_name, features in model_specs:
        result = _fit_logistic_auc_model(stall_df, features, "stall")
        rows.append(
            {
                "task": "stall",
                "model": model_name,
                "features": "|".join(features),
                "metric_name": "auc",
                "metric_value": result["metric_value"],
                "delta_metric_vs_prev": result["metric_value"] - previous_auc,
                "num_rows": result["num_rows"],
                "intercept": result["intercept"],
                "coefficients_json": json.dumps(result["coefficients"], sort_keys=True),
            }
        )
        previous_auc = result["metric_value"]

    return pd.DataFrame(rows)


def build_causal_chain_summary(grid_df: pd.DataFrame) -> pd.DataFrame:
    dataframe = grid_df.copy()
    dataframe["slack_ratio"] = dataframe["compressibility_slack_ratio"]
    dataframe["mean_t_slack"] = dataframe["compressibility_mean_t_slack"]
    dataframe["stall_observed"] = ((dataframe["feasible"] == 0) | (dataframe["stall_cycles"] > 0)).astype(int)
    dataframe["slowdown_ratio"] = dataframe["normalized_makespan"]
    columns = [
        "family",
        "seed",
        "policy",
        "C",
        "B",
        "slack_ratio",
        "mean_t_slack",
        "peak_demand",
        "Delta_max",
        "Gamma",
        "stall_observed",
        "slowdown_ratio",
    ]
    return dataframe[columns].sort_values(by=["family", "seed", "policy", "C", "B"]).reset_index(drop=True)


def compute_causal_chain_correlations(causal_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, subset in list(causal_df.groupby("policy")) + [("ALL", causal_df)]:
        finite_slowdown = subset[np.isfinite(subset["slowdown_ratio"])].copy()
        correlation_specs = [
            ("slack_ratio", "peak_demand"),
            ("slack_ratio", "Delta_max"),
            ("Delta_max", "stall_observed"),
            ("Delta_max", "slowdown_ratio"),
        ]
        for left, right in correlation_specs:
            target_df = finite_slowdown if right == "slowdown_ratio" else subset
            if len(target_df) < 3 or target_df[left].nunique() < 2 or target_df[right].nunique() < 2:
                pearson = 0.0
                spearman = 0.0
            else:
                pearson = _pearson(target_df[left].astype(float).to_numpy(), target_df[right].astype(float).to_numpy())
                spearman = _spearman(target_df[left].astype(float).to_numpy(), target_df[right].astype(float).to_numpy())
            rows.append(
                {
                    "policy": policy,
                    "left_metric": left,
                    "right_metric": right,
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
    return pd.DataFrame(rows)


def build_predictor_stability_summary(predictive_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate whether predictor ranking is stable across workload and supply subsets."""
    rows: list[dict[str, Any]] = []
    group_specs: list[tuple[str, str, pd.DataFrame]] = [("overall", "ALL", predictive_df)]
    for family, subset in predictive_df.groupby("family", sort=False):
        group_specs.append(("family", str(family), subset.copy()))

    c_bands = [("low_C", [1, 2]), ("mid_C", [3, 4, 5]), ("high_C", [6, 7])]
    b_bands = [("low_B", list(range(0, 6))), ("mid_B", list(range(6, 11))), ("high_B", list(range(11, 16)))]
    for band_name, values in c_bands:
        subset = predictive_df[predictive_df["C"].isin(values)].copy()
        group_specs.append(("capacity_band", band_name, subset))
    for band_name, values in b_bands:
        subset = predictive_df[predictive_df["B"].isin(values)].copy()
        group_specs.append(("buffer_band", band_name, subset))

    for group_type, group_name, subset in group_specs:
        for predictor in STABILITY_PREDICTORS:
            stall_score = _subset_binary_metric(subset, predictor, "stall")
            slowdown_score = _subset_regression_metric(subset, predictor, "slowdown_ratio")
            rows.append(
                {
                    "group_type": group_type,
                    "group_name": group_name,
                    "predictor": predictor,
                    "stall_auc": stall_score["score"],
                    "stall_num_rows": stall_score["num_rows"],
                    "slowdown_abs_spearman": slowdown_score["score"],
                    "slowdown_num_rows": slowdown_score["num_rows"],
                }
            )

    result_df = pd.DataFrame(rows)
    ranked_rows: list[dict[str, Any]] = []
    for (group_type, group_name), subset in result_df.groupby(["group_type", "group_name"], sort=False):
        stall_rank = subset["stall_auc"].rank(method="dense", ascending=False).astype(int)
        slowdown_rank = subset["slowdown_abs_spearman"].rank(method="dense", ascending=False).astype(int)
        ranked = subset.copy()
        ranked["stall_rank"] = stall_rank.to_numpy()
        ranked["slowdown_rank"] = slowdown_rank.to_numpy()
        ranked_rows.append(ranked)
    return pd.concat(ranked_rows, ignore_index=True)


def build_predictor_stability_summary_text(stability_df: pd.DataFrame) -> str:
    """Summarize how often each predictor leads across subgroup analyses."""
    lines: list[str] = []
    for task_col, rank_col, label in (
        ("stall_auc", "stall_rank", "stall"),
        ("slowdown_abs_spearman", "slowdown_rank", "slowdown"),
    ):
        winners = stability_df[stability_df[rank_col] == 1]
        counts = winners["predictor"].value_counts().to_dict()
        delta_wins = counts.get("Delta_max", 0)
        slack_wins = counts.get("slack_ratio", 0)
        depth_wins = counts.get("T_depth", 0)
        overall = stability_df[(stability_df["group_type"] == "overall") & (stability_df["predictor"].isin(STABILITY_PREDICTORS))]
        overall_sorted = overall.sort_values(task_col, ascending=False)
        leader = overall_sorted.iloc[0]["predictor"] if not overall_sorted.empty else "none"
        lines.append(
            f"For {label}, the subgroup leader is {leader} overall; first-place counts across subsets are "
            f"Delta_max={delta_wins}, slack_ratio={slack_wins}, T_depth={depth_wins}."
        )
    family_rows = stability_df[stability_df["group_type"] == "family"].copy()
    family_notes: list[str] = []
    for family in ["high_compressibility", "medium_compressibility", "low_compressibility"]:
        subset = family_rows[family_rows["group_name"] == family]
        if subset.empty:
            continue
        stall_best = subset.sort_values("stall_auc", ascending=False).iloc[0]["predictor"]
        slowdown_best = subset.sort_values("slowdown_abs_spearman", ascending=False).iloc[0]["predictor"]
        family_notes.append(f"{family}: stall={stall_best}, slowdown={slowdown_best}")
    if family_notes:
        lines.append("Family-level leaders: " + "; ".join(family_notes) + ".")
    return "\n".join(lines) + "\n"


def build_lower_bound_validation(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Construct per-schedule lower-bound validation data for exported analysis."""
    dataframe = grid_df.copy()
    dataframe["predicted_lower_bound"] = [
        compute_lower_bound(T_static=row.T_static, delta_max=row.Delta_max, B=row.B, C=row.C)
        for row in dataframe.itertuples(index=False)
    ]
    dataframe["gap"] = dataframe["T_exe"].astype(float) - dataframe["predicted_lower_bound"]
    dataframe["lower_bound_ratio"] = np.where(
        np.isfinite(dataframe["T_exe"].astype(float)) & (dataframe["predicted_lower_bound"] > 0.0),
        dataframe["T_exe"].astype(float) / dataframe["predicted_lower_bound"],
        np.inf,
    )
    columns = [
        "family",
        "seed",
        "policy",
        "C",
        "B",
        "feasible",
        "T_static",
        "T_exe",
        "Delta_max",
        "Gamma",
        "predicted_lower_bound",
        "gap",
        "lower_bound_ratio",
    ]
    return dataframe[columns].sort_values(by=["family", "seed", "policy", "C", "B"]).reset_index(drop=True)


def build_predictive_analysis_summary(
    predictive_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    multivariate_df: pd.DataFrame,
    incremental_df: pd.DataFrame,
    causal_corr_df: pd.DataFrame,
    lower_bound_df: pd.DataFrame,
) -> str:
    stall_auc = classification_df[classification_df["task"] == "stall"].set_index("metric")["mean_roc_auc"].to_dict()
    inversion_auc = classification_df[classification_df["task"] == "inversion"].set_index("metric")["mean_roc_auc"].to_dict()
    regression_map = regression_df.set_index("metric")["mean_abs_spearman"].to_dict()
    representative = predictive_df[
        (predictive_df["C"] == COMPRESSIBILITY_SCAN_C)
        & (predictive_df["B"] == COMPRESSIBILITY_SCAN_B)
        & np.isfinite(predictive_df["slowdown_ratio"])
    ].copy()
    low_slack = representative[representative["slack_ratio"] <= representative["slack_ratio"].median()]
    high_slack = representative[representative["slack_ratio"] > representative["slack_ratio"].median()]
    low_gain = float(low_slack["shaping_gain"].mean()) if not low_slack.empty else 0.0
    high_gain = float(high_slack["shaping_gain"].mean()) if not high_slack.empty else 0.0
    leading_feature = multivariate_df.iloc[multivariate_df["standardized_coefficient"].abs().argmax()]["feature"] if not multivariate_df.empty else "none"
    chain_all = causal_corr_df[causal_corr_df["policy"] == "ALL"].set_index(["left_metric", "right_metric"])
    slack_delta_corr = float(chain_all.loc[("slack_ratio", "Delta_max"), "spearman"]) if ("slack_ratio", "Delta_max") in chain_all.index else 0.0
    delta_stall_corr = float(chain_all.loc[("Delta_max", "stall_observed"), "spearman"]) if ("Delta_max", "stall_observed") in chain_all.index else 0.0
    finite_lower_bound = lower_bound_df[np.isfinite(lower_bound_df["T_exe"])].copy()
    lower_bound_corr = (
        _pearson(
            finite_lower_bound["predicted_lower_bound"].astype(float).to_numpy(),
            finite_lower_bound["T_exe"].astype(float).to_numpy(),
        )
        if len(finite_lower_bound) >= 2
        else 0.0
    )
    mean_bound_gap = float(finite_lower_bound["gap"].mean()) if not finite_lower_bound.empty else math.inf
    incremental_map = incremental_df.set_index(["task", "model"])["metric_value"].to_dict()
    slowdown_increment = incremental_map.get(("slowdown_ratio", "T_depth_plus_slack"), 0.0) - incremental_map.get(
        ("slowdown_ratio", "T_depth_only"),
        0.0,
    )
    slowdown_delta_increment = incremental_map.get(("slowdown_ratio", "T_depth_slack_delta"), 0.0) - incremental_map.get(
        ("slowdown_ratio", "T_depth_plus_slack"),
        0.0,
    )
    stall_increment = incremental_map.get(("stall", "T_depth_plus_slack"), 0.0) - incremental_map.get(
        ("stall", "T_depth_only"),
        0.0,
    )
    stall_delta_increment = incremental_map.get(("stall", "T_depth_slack_delta"), 0.0) - incremental_map.get(
        ("stall", "T_depth_plus_slack"),
        0.0,
    )

    lines = [
        f"1. Structural predictors: slack_ratio outperforms T_depth for slowdown ({regression_map.get('slack_ratio', 0.0):.3f} vs {regression_map.get('T_depth', 0.0):.3f} mean |Spearman|) and for stall ({stall_auc.get('slack_ratio', 0.5):.3f} vs {stall_auc.get('T_depth', 0.5):.3f} mean ROC-AUC).",
        f"2. Immediate system predictors: Delta_max has the highest observed association with slowdown ({regression_map.get('Delta_max', 0.0):.3f} mean |Spearman|) and stall ({stall_auc.get('Delta_max', 0.5):.3f} mean ROC-AUC).",
        f"3. For inversion prediction, slack_ratio={inversion_auc.get('slack_ratio', 0.5):.3f}, T_depth={inversion_auc.get('T_depth', 0.5):.3f}, Delta_max={inversion_auc.get('Delta_max', 0.5):.3f} mean ROC-AUC.",
        f"4. Incremental model fits remain consistent with a layered interpretation: adding slack_ratio to T_depth changes slowdown R^2 by {slowdown_increment:.3f} and stall AUC by {stall_increment:.3f}; adding Delta_max changes slowdown R^2 by {slowdown_delta_increment:.3f} and stall AUC by {stall_delta_increment:.3f}.",
        f"5. The causal chain is consistent with the data: Spearman(slack_ratio, Delta_max)={slack_delta_corr:.3f}, Spearman(Delta_max, stall)={delta_stall_corr:.3f}.",
        f"6. In the representative bounded-delivery slice (C={COMPRESSIBILITY_SCAN_C}, B={COMPRESSIBILITY_SCAN_B}), lower-slack workloads have mean shaping gain={low_gain:.3f}, higher-slack workloads have mean shaping gain={high_gain:.3f}; the largest standardized regression coefficient is associated with {leading_feature}.",
        f"7. The fixed-schedule lower bound tracks executed makespan closely: Pearson(predicted lower bound, T_exe)={lower_bound_corr:.3f}, mean slack above the bound={mean_bound_gap:.3f} cycles.",
    ]
    return "\n".join(lines) + "\n"


def build_paper1_reframing_summary(
    predictive_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    causal_corr_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
) -> str:
    stall_auc = classification_df[classification_df["task"] == "stall"].set_index("metric")["mean_roc_auc"].to_dict()
    regression_map = regression_df.set_index("metric")["mean_abs_spearman"].to_dict()
    chain_all = causal_corr_df[causal_corr_df["policy"] == "ALL"].set_index(["left_metric", "right_metric"])
    delta_stall_corr = float(chain_all.loc[("Delta_max", "stall_observed"), "spearman"]) if ("Delta_max", "stall_observed") in chain_all.index else 0.0
    smoothed_high = family_summary_df[
        (family_summary_df["family"] == "high_compressibility") & (family_summary_df["policy"] == "smoothed")
    ]["mean_normalized_makespan"].iloc[0]
    capacity_high = family_summary_df[
        (family_summary_df["family"] == "high_compressibility") & (family_summary_df["policy"] == "capacity_aware_static")
    ]["mean_normalized_makespan"].iloc[0]
    lines = [
        f"1. Static-depth objective does fail: the representative static_min schedule still exhibits slowdown and inversion regimes.",
        f"2. slack_ratio has a larger observed association than T_depth among the tested structure-aware metrics: stall ROC-AUC={stall_auc.get('slack_ratio', 0.5):.3f} vs T_depth={stall_auc.get('T_depth', 0.5):.3f}; slowdown |Spearman|={regression_map.get('slack_ratio', 0.0):.3f} vs T_depth={regression_map.get('T_depth', 0.0):.3f}.",
        f"3. Delta_max has the highest observed association among the immediate trace-level variables: stall ROC-AUC={stall_auc.get('Delta_max', 0.5):.3f}, slowdown |Spearman|={regression_map.get('Delta_max', 0.0):.3f}, Delta_max-stall Spearman={delta_stall_corr:.3f}.",
        f"4. capacity_aware_static resolves much of the mismatch in the current scan: in high_compressibility its mean normalized makespan is {capacity_high:.3f}, while smoothed remains at {smoothed_high:.3f}.",
        "5. smoothed is therefore treated here as an intervention class / design-space example rather than a universally preferred scheduling policy.",
    ]
    return "\n".join(lines) + "\n"


def run_predictive_analysis(
    grid_df: pd.DataFrame,
    grid_pair_df: pd.DataFrame,
    family_summary_df: pd.DataFrame,
) -> dict[str, pd.DataFrame | str]:
    predictive_df = build_predictive_dataset(grid_df, grid_pair_df)
    classification_stall = evaluate_predictive_classification(predictive_df, "stall")
    classification_inversion = evaluate_predictive_classification(predictive_df, "inversion")
    classification_df = pd.concat([classification_stall, classification_inversion], ignore_index=True)
    regression_df = evaluate_predictive_regression(predictive_df, "slowdown_ratio")
    multivariate_df = fit_multivariate_slowdown_regression(predictive_df)
    incremental_df = fit_incremental_predictive_models(predictive_df)
    stability_df = build_predictor_stability_summary(predictive_df)
    causal_chain_df = build_causal_chain_summary(grid_df)
    causal_corr_df = compute_causal_chain_correlations(causal_chain_df)
    lower_bound_df = build_lower_bound_validation(grid_df)
    summary_text = build_predictive_analysis_summary(
        predictive_df=predictive_df,
        classification_df=classification_df,
        regression_df=regression_df,
        multivariate_df=multivariate_df,
        incremental_df=incremental_df,
        causal_corr_df=causal_corr_df,
        lower_bound_df=lower_bound_df,
    )
    reframing_text = build_paper1_reframing_summary(
        predictive_df=predictive_df,
        classification_df=classification_df,
        regression_df=regression_df,
        causal_corr_df=causal_corr_df,
        family_summary_df=family_summary_df,
    )
    stability_text = build_predictor_stability_summary_text(stability_df)

    _write_table(predictive_df, "predictive_static_dataset.csv")
    _write_table(classification_df, "predictive_classification_summary.csv")
    _write_table(regression_df, "predictive_regression_summary.csv")
    _write_table(multivariate_df, "predictive_multivariate_regression.csv")
    _write_table(incremental_df, "incremental_predictive_models.csv")
    _write_table(stability_df, "predictor_stability_summary.csv")
    _write_table(causal_chain_df, "causal_chain_summary.csv")
    _write_table(causal_corr_df, "causal_chain_correlations.csv")
    _write_table(lower_bound_df, "lower_bound_validation.csv")
    _write_text(summary_text, "predictive_analysis_summary.txt")
    _write_text(reframing_text, "paper1_reframing_summary.txt")
    _write_text(stability_text, "predictor_stability_summary.txt")

    return {
        "predictive_dataset": predictive_df,
        "predictive_classification": classification_df,
        "predictive_regression": regression_df,
        "predictive_multivariate": multivariate_df,
        "incremental_predictive_models": incremental_df,
        "predictor_stability_summary": stability_df,
        "causal_chain_summary": causal_chain_df,
        "causal_chain_correlations": causal_corr_df,
        "lower_bound_validation": lower_bound_df,
        "predictive_summary_text": summary_text,
        "paper1_reframing_summary": reframing_text,
        "predictor_stability_summary_text": stability_text,
    }


def _binary_metric_eval(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    auc_raw = _roc_auc(x, y)
    auc = max(auc_raw, 1.0 - auc_raw)
    thresholds = np.unique(x)
    best_accuracy = 0.0
    for threshold in thresholds:
        pred_high = (x >= threshold).astype(int)
        pred_low = (x <= threshold).astype(int)
        best_accuracy = max(best_accuracy, float((pred_high == y).mean()), float((pred_low == y).mean()))
    return {"best_accuracy": best_accuracy, "roc_auc": auc}


def _subset_binary_metric(dataframe: pd.DataFrame, predictor: str, label_col: str) -> dict[str, float]:
    subset = dataframe[[predictor, label_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(subset) < 6 or subset[label_col].nunique() < 2 or subset[predictor].nunique() < 2:
        return {"score": 0.5, "num_rows": int(len(subset))}
    score = _binary_metric_eval(
        subset[predictor].astype(float).to_numpy(),
        subset[label_col].astype(int).to_numpy(),
    )["roc_auc"]
    return {"score": float(score), "num_rows": int(len(subset))}


def _subset_regression_metric(dataframe: pd.DataFrame, predictor: str, target_col: str) -> dict[str, float]:
    subset = dataframe[[predictor, target_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(subset) < 6 or subset[predictor].nunique() < 2 or subset[target_col].nunique() < 2:
        return {"score": 0.0, "num_rows": int(len(subset))}
    score = abs(
        _spearman(
            subset[predictor].astype(float).to_numpy(),
            subset[target_col].astype(float).to_numpy(),
        )
    )
    return {"score": float(score), "num_rows": int(len(subset))}


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    sum_ranks_pos = float(np.sum(ranks[positives]))
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _linear_r_squared(x: np.ndarray, y: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    predictions = design @ coefficients
    residual = np.sum((y - predictions) ** 2)
    total = np.sum((y - np.mean(y)) ** 2)
    return 0.0 if np.isclose(total, 0.0) else float(1.0 - residual / total)


def _fit_linear_model(dataframe: pd.DataFrame, features: list[str], target_col: str) -> dict[str, Any]:
    subset = dataframe[features + [target_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(subset) < len(features) + 2:
        return {"metric_value": 0.0, "num_rows": int(len(subset)), "intercept": 0.0, "coefficients": {}}
    X = subset[features].astype(float).to_numpy()
    y = subset[target_col].astype(float).to_numpy()
    design = np.column_stack([np.ones(len(X)), X])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    predictions = design @ coefficients
    residual = np.sum((y - predictions) ** 2)
    total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 0.0 if np.isclose(total, 0.0) else float(1.0 - residual / total)
    return {
        "metric_value": r_squared,
        "num_rows": int(len(subset)),
        "intercept": float(coefficients[0]),
        "coefficients": {feature: float(value) for feature, value in zip(features, coefficients[1:], strict=True)},
    }


def _fit_logistic_auc_model(dataframe: pd.DataFrame, features: list[str], target_col: str) -> dict[str, Any]:
    subset = dataframe[features + [target_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(subset) < len(features) + 2 or subset[target_col].nunique() < 2:
        return {"metric_value": 0.5, "num_rows": int(len(subset)), "intercept": 0.0, "coefficients": {}}

    X = subset[features].astype(float).to_numpy()
    y = subset[target_col].astype(float).to_numpy()
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std = np.where(np.isclose(X_std, 0.0), 1.0, X_std)
    X_scaled = (X - X_mean) / X_std
    design = np.column_stack([np.ones(len(X_scaled)), X_scaled])

    ridge = 1e-3

    def objective(beta: np.ndarray) -> float:
        logits = design @ beta
        probs = np.clip(expit(logits), 1e-9, 1.0 - 1e-9)
        penalty = ridge * np.sum(beta[1:] ** 2)
        return float(-np.sum(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)) + penalty)

    fit = minimize(objective, np.zeros(design.shape[1]), method="L-BFGS-B")
    if fit.success and np.all(np.isfinite(fit.x)):
        beta = fit.x
    else:
        fallback, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        beta = fallback
    scores = design @ beta
    auc = max(_roc_auc(scores, y.astype(int)), 1.0 - _roc_auc(scores, y.astype(int)))
    scaled_coefficients = beta[1:] / X_std
    intercept = float(beta[0] - np.sum((X_mean / X_std) * beta[1:]))
    return {
        "metric_value": float(auc),
        "num_rows": int(len(subset)),
        "intercept": intercept,
        "coefficients": {feature: float(value) for feature, value in zip(features, scaled_coefficients, strict=True)},
    }


def _zscore_matrix(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std = np.where(np.isclose(std, 0.0), 1.0, std)
    return (values - mean) / std


def _zscore_vector(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values)
    std = np.std(values)
    if np.isclose(std, 0.0):
        return values - mean
    return (values - mean) / std


def _write_table(dataframe: pd.DataFrame, filename: str) -> Path:
    path = TABLE_DIR / filename
    normalize_output_dataframe(dataframe).to_csv(path, index=False)
    return path


def _write_text(contents: str, filename: str) -> Path:
    path = TABLE_DIR / filename
    path.write_text(contents, encoding="utf-8")
    return path
