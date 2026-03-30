from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimResult:
    T_static: int
    T_exe: int
    stall_cycles: int
    stall_ratio: float
    buffer_history: list[int]
    served_history: list[int]
    logical_cycle_history: list[int]


def check_trace_feasibility(demand: list[int], C: int, B: int) -> tuple[bool, str]:
    if C < 0 or B < 0:
        raise ValueError("C and B must be non-negative")
    if any(value < 0 for value in demand):
        raise ValueError("demand values must be non-negative")
    if not demand:
        return True, "feasible"
    if C == 0:
        if any(value > 0 for value in demand):
            return False, "zero_delivery_positive_demand"
        return True, "feasible"
    if max(demand) > B + C:
        return False, "peak_demand_exceeds_reachable_stock"
    return True, "feasible"


def simulate_trace(demand: list[int], C: int, B: int) -> SimResult:
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

    while logical_idx < len(demand):
        current_demand = demand[logical_idx]
        available = buffer_stock + C
        real_cycles += 1
        logical_cycle_history.append(logical_idx)
        if available >= current_demand:
            served_history.append(current_demand)
            buffer_stock = min(B, available - current_demand)
            logical_idx += 1
        else:
            served_history.append(0)
            buffer_stock = min(B, available)
            stall_cycles += 1
        buffer_history.append(buffer_stock)

    stall_ratio = 0.0 if real_cycles == 0 else stall_cycles / real_cycles
    return SimResult(
        T_static=len(demand),
        T_exe=real_cycles,
        stall_cycles=stall_cycles,
        stall_ratio=stall_ratio,
        buffer_history=buffer_history,
        served_history=served_history,
        logical_cycle_history=logical_cycle_history,
    )
