from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.dag import DAG, compute_slack_metrics
from src.simulator import SimResult
from src.utils import mean_int


@dataclass(frozen=True)
class MetricBundle:
    cumulative_demand: list[int]
    Delta_max: int
    Gamma: int
    peak_demand: int
    mean_demand: float
    total_T: int
    QTV_proxy: int


@dataclass(frozen=True)
class DAGMetricBundle:
    compressibility_legacy: float
    compressibility_slack_ratio: float
    compressibility_mean_t_slack: float


def compute_trace_statistics(demand: list[int], C: int) -> tuple[list[int], int, int, int, float, int]:
    cumulative_demand = np.cumsum(demand).astype(int).tolist()
    raw_peak_deficit = max(
        (cumulative_demand[idx] - C * (idx + 1) for idx in range(len(cumulative_demand))),
        default=0,
    )
    Delta_max = max(0, int(raw_peak_deficit))
    Gamma = int(sum(max(0, value - C) for value in demand))
    peak_demand = max(demand, default=0)
    mean_demand = mean_int(demand)
    total_T = int(sum(demand))
    return cumulative_demand, Delta_max, Gamma, peak_demand, mean_demand, total_T


def compute_metrics(
    demand: list[int],
    C: int,
    logical_qubits_in_use: list[int],
    sim_result: SimResult,
) -> MetricBundle:
    if len(demand) != len(logical_qubits_in_use):
        raise ValueError("logical_qubits_in_use must align with logical demand cycles")

    cumulative_demand, Delta_max, Gamma, peak_demand, mean_demand, total_T = compute_trace_statistics(demand, C)
    QTV_proxy = int(sum(logical_qubits_in_use[cycle_idx] for cycle_idx in sim_result.logical_cycle_history))

    return MetricBundle(
        cumulative_demand=cumulative_demand,
        Delta_max=Delta_max,
        Gamma=Gamma,
        peak_demand=peak_demand,
        mean_demand=mean_demand,
        total_T=total_T,
        QTV_proxy=QTV_proxy,
    )


def compute_dag_metrics(dag: DAG) -> DAGMetricBundle:
    slack_metrics = compute_slack_metrics(dag)
    return DAGMetricBundle(
        compressibility_legacy=slack_metrics.compressibility_legacy,
        compressibility_slack_ratio=slack_metrics.compressibility_slack_ratio,
        compressibility_mean_t_slack=slack_metrics.compressibility_mean_t_slack,
    )
