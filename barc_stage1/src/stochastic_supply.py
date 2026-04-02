from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.simulator import check_trace_feasibility


@dataclass(frozen=True)
class StochasticSimResult:
    T_static: int
    T_exe: int
    stall_cycles: int
    stall_ratio: float
    buffer_history: list[int]
    served_history: list[int]
    logical_cycle_history: list[int]
    service_history: list[int]
    max_cycle_hit: int


def simulate_trace_stochastic_binomial(
    demand: list[int],
    C: int,
    B: int,
    p_acc: float,
    rng: np.random.Generator,
    max_cycles: int | None = None,
) -> StochasticSimResult:
    if not 0.0 <= p_acc <= 1.0:
        raise ValueError("p_acc must lie in [0, 1]")
    feasible, reason = check_trace_feasibility(demand, C, B)
    if not feasible:
        raise ValueError(f"infeasible trace: {reason}")

    logical_idx = 0
    real_cycles = 0
    stall_cycles = 0
    buffer_stock = 0
    buffer_history: list[int] = []
    served_history: list[int] = []
    logical_cycle_history: list[int] = []
    service_history: list[int] = []
    cycle_limit = max_cycles if max_cycles is not None else max(10 * max(1, len(demand)), 100)

    while logical_idx < len(demand) and real_cycles < cycle_limit:
        current_demand = demand[logical_idx]
        service = int(rng.binomial(C, p_acc))
        available = buffer_stock + service
        real_cycles += 1
        logical_cycle_history.append(logical_idx)
        service_history.append(service)
        if available >= current_demand:
            served_history.append(current_demand)
            buffer_stock = min(B, available - current_demand)
            logical_idx += 1
        else:
            served_history.append(0)
            buffer_stock = min(B, available)
            stall_cycles += 1
        buffer_history.append(buffer_stock)

    max_cycle_hit = int(logical_idx < len(demand))
    stall_ratio = 0.0 if real_cycles == 0 else stall_cycles / real_cycles
    return StochasticSimResult(
        T_static=len(demand),
        T_exe=real_cycles,
        stall_cycles=stall_cycles,
        stall_ratio=stall_ratio,
        buffer_history=buffer_history,
        served_history=served_history,
        logical_cycle_history=logical_cycle_history,
        service_history=service_history,
        max_cycle_hit=max_cycle_hit,
    )


def run_stochastic_supply_trials(
    demand: list[int],
    C: int,
    B: int,
    p_acc: float,
    trials: int,
    seed: int,
    max_cycles: int | None = None,
) -> pd.DataFrame:
    if trials <= 0:
        raise ValueError("trials must be positive")
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(trials)
    rows: list[dict[str, float | int]] = []
    for trial_index, child in enumerate(child_sequences):
        rng = np.random.default_rng(child)
        result = simulate_trace_stochastic_binomial(
            demand=demand,
            C=C,
            B=B,
            p_acc=p_acc,
            rng=rng,
            max_cycles=max_cycles,
        )
        rows.append(
            {
                "trial_index": trial_index,
                "T_static": result.T_static,
                "T_exe": result.T_exe,
                "stall_cycles": result.stall_cycles,
                "stall_ratio": result.stall_ratio,
                "max_cycle_hit": result.max_cycle_hit,
                "mean_service": float(np.mean(result.service_history)) if result.service_history else 0.0,
            }
        )
    return pd.DataFrame(rows)
