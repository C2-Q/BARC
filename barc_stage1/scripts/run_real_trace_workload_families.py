"""Driver for the workload-family extension: CLA adder, modular-arithmetic block, QAOA MaxCut.

This script extends the real-trace evaluation pipeline to three additional
first-class workload families. It uses the same generator -> Clifford+T ->
T-demand -> bounded-delivery scan pipeline that drives the ripple adder,
multiplier and QFT evaluations, but applies a reduced (C, B) grid for the
QAOA family because its Clifford+T synthesis produces long traces. The grid
choice is recorded per row in a `grid_type` column so the two regimes can
never be silently mixed.

Outputs (under `barc_stage1/outputs/`):
  tables/real_trace_workload_families_raw.csv
  tables/real_trace_workload_families_representative_summary.csv
  tables/real_trace_workload_families_scaling_summary.csv
  tables/qaoa_summary.csv
  real_trace/{cla_adder,modular_multiplier,qaoa_maxcut}_*.csv  (per-instance result rows)
  data/real_traces/{family}_n{n}.csv  (raw T-demand traces)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paper_config import B_VALUES, C_VALUES
from src.real_trace.adder_trace import (
    extract_t_demand_trace,
    save_trace,
    to_clifford_t,
)
from src.real_trace.analysis import evaluate_real_trace_policies
from src.real_trace.circuit_slack import compute_circuit_slack_metrics
from src.real_trace.cla_adder_trace import (
    CLA_VARIANT,
    generate_cla_adder_circuit,
)
from src.real_trace.modular_multiplier_trace import (
    MODULAR_MULTIPLIER_VARIANT,
    _select_constants,
    generate_modular_multiplier_circuit,
)
from src.real_trace.qaoa_maxcut_trace import (
    build_qaoa_instance,
    generate_qaoa_maxcut_circuit,
)
from src.utils import OUTPUT_ROOT, TABLE_DIR, ensure_output_dirs


CLA_SIZES = [4, 8, 12, 16]
MODULAR_MULTIPLIER_SIZES = [4, 6, 8]
QAOA_SIZES = [6, 8, 10]
QAOA_DEPTHS = [1, 2]
QAOA_GRAPH_SEEDS = {
    "ring": [0],
    "random_3_regular": [0, 1, 2],
    "erdos_renyi_dense": [0, 1, 2],
}

REDUCED_C_VALUES = [1, 2, 3, 5, 7]
REDUCED_B_VALUES = [0, 4, 8, 12, 15]

TRACE_LENGTH_LIMIT = 600_000
T_COUNT_LIMIT = 1_000_000

SYNTHESIS_PRECISION_TAG = (
    "qiskit_transpile_optimization_level_1_seed_42_basis_h_s_sdg_cx_t_tdg"
)


def _logical_qubit_budget(family: str, n_bits: int) -> int:
    if family == "cla_adder":
        return 2 * n_bits + 1
    if family == "modular_multiplier":
        return 3 * n_bits + 1
    if family == "qaoa_maxcut":
        return n_bits
    raise ValueError(f"unsupported family: {family}")


def _summarize_static_rows(results_df: pd.DataFrame) -> dict[str, float | int]:
    static_rows = results_df[results_df["policy"] == "static_min"].copy()
    static_unique = static_rows.drop_duplicates(subset=["C"])
    slowdown_fraction = float(
        ((static_rows["feasible"] == 1) & (static_rows["normalized_makespan"] > 1.05)).mean()
    )
    stall_fraction = float(
        ((static_rows["feasible"] == 0) | (static_rows["stall_cycles"] > 0)).mean()
    )
    return {
        "mean_delta_max": float(static_unique["Delta_max"].mean()),
        "max_delta_max": float(static_unique["Delta_max"].max()),
        "mean_peak_demand": float(static_rows["peak_demand"].mean()),
        "frac_CB_with_slowdown": slowdown_fraction,
        "frac_CB_with_stall": stall_fraction,
    }


def _build_row(
    workload_family: str,
    workload_name: str,
    instance_id: str,
    n_bits: int,
    trace: list[int],
    slack_metrics,
    results_df: pd.DataFrame | None,
    grid_type: str,
    extra: dict,
) -> dict:
    row = {
        "workload_family": workload_family,
        "workload_name": workload_name,
        "instance_id": instance_id,
        "size_n": n_bits,
        "T_count": int(sum(trace)),
        "T_depth": int(len(trace)),
        "trace_length": int(len(trace)),
        "peak_T_demand": int(max(trace, default=0)),
        "slack_ratio": float(slack_metrics.fraction_t_slack_positive),
        "mean_t_slack": float(slack_metrics.mean_t_slack),
        "grid_type": grid_type,
        "synthesis_precision": SYNTHESIS_PRECISION_TAG,
    }
    row.update(extra)
    if results_df is None:
        row.update(
            {
                "evaluated": 0,
                "skip_reason": "trace_too_large_for_grid",
                "mean_delta_max": math.nan,
                "max_delta_max": math.nan,
                "mean_peak_demand": math.nan,
                "frac_CB_with_slowdown": math.nan,
                "frac_CB_with_stall": math.nan,
            }
        )
    else:
        row["evaluated"] = 1
        row["skip_reason"] = ""
        row.update(_summarize_static_rows(results_df))
    return row


def _evaluate_with_grid(
    trace: list[int],
    trace_name: str,
    logical_qubit_budget: int,
    grid_type: str,
) -> pd.DataFrame | None:
    if len(trace) > TRACE_LENGTH_LIMIT or sum(trace) > T_COUNT_LIMIT:
        return None
    if grid_type == "full":
        c_values, b_values = list(C_VALUES), list(B_VALUES)
    elif grid_type == "reduced":
        c_values, b_values = list(REDUCED_C_VALUES), list(REDUCED_B_VALUES)
    else:
        raise ValueError(f"unsupported grid_type: {grid_type}")
    return evaluate_real_trace_policies(
        trace=trace,
        trace_name=trace_name,
        logical_qubit_budget=logical_qubit_budget,
        C_values=c_values,
        B_values=b_values,
    )


def run_cla_adder() -> tuple[list[dict], list[pd.DataFrame]]:
    rows: list[dict] = []
    raw_dfs: list[pd.DataFrame] = []
    for n_bits in CLA_SIZES:
        instance_id = f"cla_adder_n{n_bits}"
        print(f"  cla_adder n={n_bits}")
        circuit = generate_cla_adder_circuit(n_bits)
        clifford_t = to_clifford_t(circuit)
        trace = extract_t_demand_trace(clifford_t)
        save_trace(trace, PROJECT_ROOT / "data" / "real_traces" / f"{instance_id}.csv")
        slack_metrics = compute_circuit_slack_metrics(clifford_t)
        results_df = _evaluate_with_grid(
            trace=trace,
            trace_name=instance_id,
            logical_qubit_budget=_logical_qubit_budget("cla_adder", n_bits),
            grid_type="full",
        )
        if results_df is not None:
            results_df = results_df.copy()
            results_df["workload_family"] = "cla_adder"
            results_df["workload_name"] = "CLA adder"
            results_df["instance_id"] = instance_id
            results_df["size_n"] = n_bits
            results_df["grid_type"] = "full"
            results_df["synthesis_precision"] = SYNTHESIS_PRECISION_TAG
            results_df["variant"] = CLA_VARIANT
            results_df.to_csv(
                OUTPUT_ROOT / "real_trace" / f"{instance_id}_results.csv",
                index=False,
            )
            raw_dfs.append(results_df)
        rows.append(
            _build_row(
                workload_family="cla_adder",
                workload_name="CLA adder",
                instance_id=instance_id,
                n_bits=n_bits,
                trace=trace,
                slack_metrics=slack_metrics,
                results_df=results_df,
                grid_type="full",
                extra={"variant": CLA_VARIANT},
            )
        )
    return rows, raw_dfs


def run_modular_multiplier() -> tuple[list[dict], list[pd.DataFrame]]:
    rows: list[dict] = []
    raw_dfs: list[pd.DataFrame] = []
    for n_bits in MODULAR_MULTIPLIER_SIZES:
        modulus, multiplier_constant = _select_constants(n_bits)
        instance_id = f"modular_multiplier_n{n_bits}"
        print(f"  modular_multiplier n={n_bits} mod={modulus} c={multiplier_constant}")
        circuit = generate_modular_multiplier_circuit(
            n_bits=n_bits,
            modulus=modulus,
            multiplier_constant=multiplier_constant,
        )
        clifford_t = to_clifford_t(circuit)
        trace = extract_t_demand_trace(clifford_t)
        save_trace(trace, PROJECT_ROOT / "data" / "real_traces" / f"{instance_id}.csv")
        slack_metrics = compute_circuit_slack_metrics(clifford_t)
        results_df = _evaluate_with_grid(
            trace=trace,
            trace_name=instance_id,
            logical_qubit_budget=_logical_qubit_budget("modular_multiplier", n_bits),
            grid_type="full",
        )
        if results_df is not None:
            results_df = results_df.copy()
            results_df["workload_family"] = "modular_multiplier"
            results_df["workload_name"] = "Mod. arith. block"
            results_df["instance_id"] = instance_id
            results_df["size_n"] = n_bits
            results_df["grid_type"] = "full"
            results_df["synthesis_precision"] = SYNTHESIS_PRECISION_TAG
            results_df["variant"] = MODULAR_MULTIPLIER_VARIANT
            results_df["modulus"] = modulus
            results_df["multiplier_constant"] = multiplier_constant
            results_df.to_csv(
                OUTPUT_ROOT / "real_trace" / f"{instance_id}_results.csv",
                index=False,
            )
            raw_dfs.append(results_df)
        rows.append(
            _build_row(
                workload_family="modular_multiplier",
                workload_name="Mod. arith. block",
                instance_id=instance_id,
                n_bits=n_bits,
                trace=trace,
                slack_metrics=slack_metrics,
                results_df=results_df,
                grid_type="full",
                extra={
                    "variant": MODULAR_MULTIPLIER_VARIANT,
                    "modulus": modulus,
                    "multiplier_constant": multiplier_constant,
                },
            )
        )
    return rows, raw_dfs


def run_qaoa_maxcut() -> tuple[list[dict], list[pd.DataFrame]]:
    rows: list[dict] = []
    raw_dfs: list[pd.DataFrame] = []
    for n in QAOA_SIZES:
        for p in QAOA_DEPTHS:
            for graph_type, seeds in QAOA_GRAPH_SEEDS.items():
                for seed in seeds:
                    instance_id = (
                        f"qaoa_maxcut_n{n}_p{p}_{graph_type}_s{0 if seed is None else seed}"
                    )
                    print(f"  qaoa_maxcut n={n} p={p} graph={graph_type} seed={seed}")
                    instance = build_qaoa_instance(
                        n=n, p=p, graph_type=graph_type, seed=seed
                    )
                    circuit = generate_qaoa_maxcut_circuit(
                        n=n, p=p, graph_type=graph_type, seed=seed
                    )
                    clifford_t = to_clifford_t(circuit)
                    trace = extract_t_demand_trace(clifford_t)
                    save_trace(
                        trace,
                        PROJECT_ROOT / "data" / "real_traces" / f"{instance_id}.csv",
                    )
                    slack_metrics = compute_circuit_slack_metrics(clifford_t)
                    results_df = _evaluate_with_grid(
                        trace=trace,
                        trace_name=instance_id,
                        logical_qubit_budget=_logical_qubit_budget("qaoa_maxcut", n),
                        grid_type="reduced",
                    )
                    if results_df is not None:
                        results_df = results_df.copy()
                        results_df["workload_family"] = "qaoa_maxcut"
                        results_df["workload_name"] = "QAOA"
                        results_df["instance_id"] = instance_id
                        results_df["size_n"] = n
                        results_df["qaoa_p"] = p
                        results_df["graph_type"] = graph_type
                        results_df["graph_seed"] = -1 if seed is None else seed
                        results_df["graph_edges"] = instance.graph.number_of_edges()
                        results_df["grid_type"] = "reduced"
                        results_df["synthesis_precision"] = SYNTHESIS_PRECISION_TAG
                        results_df.to_csv(
                            OUTPUT_ROOT / "real_trace" / f"{instance_id}_results.csv",
                            index=False,
                        )
                        raw_dfs.append(results_df)
                    rows.append(
                        _build_row(
                            workload_family="qaoa_maxcut",
                            workload_name="QAOA",
                            instance_id=instance_id,
                            n_bits=n,
                            trace=trace,
                            slack_metrics=slack_metrics,
                            results_df=results_df,
                            grid_type="reduced",
                            extra={
                                "qaoa_p": p,
                                "graph_type": graph_type,
                                "graph_seed": -1 if seed is None else seed,
                                "graph_edges": instance.graph.number_of_edges(),
                            },
                        )
                    )
    return rows, raw_dfs


def build_representative_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Pick one representative instance per family for the headline table."""
    representative_rules = [
        ("ripple_adder", lambda df: df[(df["workload_family"] == "ripple_adder") & (df["size_n"] == 8)]),
        ("multiplier", lambda df: df[(df["workload_family"] == "multiplier") & (df["size_n"] == 8)]),
        ("cla_adder", lambda df: df[(df["workload_family"] == "cla_adder") & (df["size_n"] == 8)]),
        (
            "modular_multiplier",
            lambda df: df[(df["workload_family"] == "modular_multiplier") & (df["size_n"] == 8)],
        ),
        (
            "qaoa_maxcut",
            lambda df: df[
                (df["workload_family"] == "qaoa_maxcut")
                & (df["size_n"] == 8)
                & (df["qaoa_p"] == 2)
                & (df["graph_type"] == "erdos_renyi_dense")
                & (df["graph_seed"] == 0)
            ],
        ),
        ("exact_qft", lambda df: df[(df["workload_family"] == "exact_qft") & (df["size_n"] == 8)]),
    ]
    rows: list[dict] = []
    for family, rule in representative_rules:
        match = rule(summary_df)
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "workload_family": family,
                "workload_name": row.get("workload_name", family),
                "representative_instance": row["instance_id"],
                "size_n": int(row["size_n"]),
                "slack_ratio": float(row["slack_ratio"]),
                "mean_delta_max": float(row.get("mean_delta_max", math.nan)),
                "mean_peak_T_demand": float(row.get("mean_peak_demand", math.nan)),
                "stall_rate": float(row.get("frac_CB_with_stall", math.nan)),
                "slowdown_rate": float(row.get("frac_CB_with_slowdown", math.nan)),
                "grid_type": row.get("grid_type", ""),
                "synthesis_precision": row.get("synthesis_precision", ""),
            }
        )
    return pd.DataFrame(rows)


def build_scaling_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Trim and project the per-instance summary into a scaling view."""
    scaling_rows: list[dict] = []
    for _, row in summary_df.iterrows():
        family = row["workload_family"]
        if family == "qaoa_maxcut":
            if row.get("qaoa_p") != 2 or row.get("graph_type") != "erdos_renyi_dense":
                continue
        scaling_rows.append(
            {
                "workload_family": family,
                "workload_name": row.get("workload_name", family),
                "size_n": int(row["size_n"]),
                "qaoa_p": int(row["qaoa_p"]) if not pd.isna(row.get("qaoa_p")) else None,
                "graph_type": row.get("graph_type"),
                "slack_ratio": float(row["slack_ratio"]),
                "mean_delta_max": float(row.get("mean_delta_max", math.nan)),
                "stall_rate": float(row.get("frac_CB_with_stall", math.nan)),
                "slowdown_rate": float(row.get("frac_CB_with_slowdown", math.nan)),
                "grid_type": row.get("grid_type", ""),
            }
        )
    if not scaling_rows:
        return pd.DataFrame()
    scaling_df = pd.DataFrame(scaling_rows)
    if "qaoa_maxcut" in scaling_df["workload_family"].values:
        seed_avg = (
            scaling_df[scaling_df["workload_family"] == "qaoa_maxcut"]
            .groupby(["workload_family", "workload_name", "size_n", "qaoa_p", "graph_type", "grid_type"], dropna=False)
            .agg(
                slack_ratio=("slack_ratio", "mean"),
                mean_delta_max=("mean_delta_max", "mean"),
                stall_rate=("stall_rate", "mean"),
                slowdown_rate=("slowdown_rate", "mean"),
            )
            .reset_index()
        )
        non_qaoa = scaling_df[scaling_df["workload_family"] != "qaoa_maxcut"]
        scaling_df = pd.concat([non_qaoa, seed_avg], ignore_index=True)
    return scaling_df.sort_values(["workload_family", "size_n"]).reset_index(drop=True)


def build_qaoa_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    qaoa_rows = summary_df[summary_df["workload_family"] == "qaoa_maxcut"].copy()
    if qaoa_rows.empty:
        return pd.DataFrame()
    grouped = (
        qaoa_rows.groupby(["size_n", "qaoa_p", "graph_type"], dropna=False)
        .agg(
            num_instances=("instance_id", "count"),
            mean_edges=("graph_edges", "mean"),
            mean_slack_ratio=("slack_ratio", "mean"),
            mean_delta_max=("mean_delta_max", "mean"),
            mean_peak_T_demand=("mean_peak_demand", "mean"),
            stall_rate=("frac_CB_with_stall", "mean"),
            slowdown_rate=("frac_CB_with_slowdown", "mean"),
        )
        .reset_index()
    )
    return grouped


def _augment_with_existing_real_traces(rows: list[dict]) -> list[dict]:
    """Pull the existing ripple-adder, multiplier, and QFT instance summaries
    into the same schema so the representative table can include all six
    families without re-running the original pipelines.
    """
    legacy_rows: list[dict] = []
    real_scaling_path = TABLE_DIR / "real_trace_scaling_summary.csv"
    qft_summary_path = TABLE_DIR / "qft_real_trace_summary.csv"

    if real_scaling_path.exists():
        legacy_df = pd.read_csv(real_scaling_path)
        for _, lrow in legacy_df.iterrows():
            family_label = (
                "ripple_adder" if lrow["trace_family"] == "adder" else "multiplier"
            )
            display = "Ripple adder" if family_label == "ripple_adder" else "Multiplier"
            legacy_rows.append(
                {
                    "workload_family": family_label,
                    "workload_name": display,
                    "instance_id": lrow["trace_name"],
                    "size_n": int(lrow["n_bits"]),
                    "T_count": int(lrow["T_count"]),
                    "T_depth": int(lrow["T_depth"]),
                    "trace_length": int(lrow["T_depth"]),
                    "peak_T_demand": int(lrow["peak_demand"]),
                    "slack_ratio": float(lrow["slack_ratio"]),
                    "mean_t_slack": float(lrow["mean_t_slack"]),
                    "mean_delta_max": float(lrow["delta_max_mean"]),
                    "max_delta_max": float(lrow["delta_max_max"]),
                    "mean_peak_demand": float(lrow["peak_demand"]),
                    "frac_CB_with_slowdown": float(lrow["frac_CB_with_slowdown"]),
                    "frac_CB_with_stall": float(lrow["frac_CB_with_stall"]),
                    "grid_type": "full",
                    "synthesis_precision": SYNTHESIS_PRECISION_TAG,
                    "variant": "ripple_carry"
                    if family_label == "ripple_adder"
                    else "qiskit_multiplier_gate",
                    "evaluated": 1,
                    "skip_reason": "",
                }
            )

    if qft_summary_path.exists():
        qft_df = pd.read_csv(qft_summary_path)
        for _, qrow in qft_df.iterrows():
            if qrow.get("status") != "full_eval":
                continue
            legacy_rows.append(
                {
                    "workload_family": "exact_qft",
                    "workload_name": "Exact QFT",
                    "instance_id": qrow["trace_name"],
                    "size_n": int(qrow["n_bits"]),
                    "T_count": int(qrow.get("T_count", 0)) if not pd.isna(qrow.get("T_count")) else 0,
                    "T_depth": int(qrow.get("T_depth", 0)) if not pd.isna(qrow.get("T_depth")) else 0,
                    "trace_length": int(qrow.get("trace_length", 0))
                    if not pd.isna(qrow.get("trace_length"))
                    else 0,
                    "peak_T_demand": int(qrow.get("peak_demand", 0))
                    if not pd.isna(qrow.get("peak_demand"))
                    else 0,
                    "slack_ratio": float(qrow["slack_ratio"]) if not pd.isna(qrow["slack_ratio"]) else math.nan,
                    "mean_t_slack": float(qrow.get("mean_t_slack", math.nan)),
                    "mean_delta_max": float(qrow.get("delta_max_mean", math.nan)),
                    "max_delta_max": float(qrow.get("delta_max_max", math.nan)),
                    "mean_peak_demand": float(qrow.get("peak_demand", math.nan)),
                    "frac_CB_with_slowdown": float(qrow.get("frac_CB_with_slowdown", math.nan)),
                    "frac_CB_with_stall": float(qrow.get("frac_CB_with_stall", math.nan)),
                    "grid_type": "full",
                    "synthesis_precision": SYNTHESIS_PRECISION_TAG,
                    "variant": "synth_qft_full",
                    "evaluated": 1,
                    "skip_reason": "",
                }
            )

    return legacy_rows + rows


def main() -> None:
    ensure_output_dirs()
    (PROJECT_ROOT / "data" / "real_traces").mkdir(parents=True, exist_ok=True)

    print("running cla adder family")
    cla_rows, cla_raw = run_cla_adder()
    print("running modular-arithmetic block family")
    mod_rows, mod_raw = run_modular_multiplier()
    print("running qaoa maxcut family")
    qaoa_rows, qaoa_raw = run_qaoa_maxcut()

    summary_rows = cla_rows + mod_rows + qaoa_rows
    summary_df = pd.DataFrame(_augment_with_existing_real_traces(summary_rows))

    raw_path = TABLE_DIR / "real_trace_workload_families_raw.csv"
    summary_df.to_csv(raw_path, index=False)
    print(f"saved: {raw_path}")

    representative_df = build_representative_summary(summary_df)
    rep_path = TABLE_DIR / "real_trace_workload_families_representative_summary.csv"
    representative_df.to_csv(rep_path, index=False)
    print(f"saved: {rep_path}")

    scaling_df = build_scaling_summary(summary_df)
    scaling_path = TABLE_DIR / "real_trace_workload_families_scaling_summary.csv"
    scaling_df.to_csv(scaling_path, index=False)
    print(f"saved: {scaling_path}")

    qaoa_summary = build_qaoa_summary(summary_df)
    if not qaoa_summary.empty:
        qaoa_path = TABLE_DIR / "qaoa_summary.csv"
        qaoa_summary.to_csv(qaoa_path, index=False)
        print(f"saved: {qaoa_path}")


if __name__ == "__main__":
    main()
