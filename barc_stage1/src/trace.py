from __future__ import annotations

from dataclasses import dataclass

from src.dag import DAG
from src.schedule import Schedule


@dataclass(frozen=True)
class TraceBundle:
    demand: list[int]
    logical_qubits_in_use: list[int]


def schedule_to_trace(schedule: Schedule) -> list[int]:
    return list(schedule.t_demand_per_cycle)


def schedule_to_trace_bundle(dag: DAG, schedule: Schedule) -> TraceBundle:
    demand = schedule_to_trace(schedule)
    logical_qubits_in_use: list[int] = []
    completed: set[int] = set()

    for cycle_nodes in schedule.cycles:
        live_completed = 0
        for node_id in completed:
            if any(successor not in completed for successor in dag.nodes[node_id].successors):
                live_completed += 1
        logical_qubits_in_use.append(max(1, live_completed + len(cycle_nodes)))
        for node_id in cycle_nodes:
            completed.add(node_id)

    return TraceBundle(demand=demand, logical_qubits_in_use=logical_qubits_in_use)
