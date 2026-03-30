from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag


@dataclass(frozen=True)
class CircuitSlackMetrics:
    mean_t_slack: float
    fraction_t_slack_positive: float
    num_t_nodes: int


def compute_circuit_slack_metrics(circuit: QuantumCircuit) -> CircuitSlackMetrics:
    dag = circuit_to_dag(circuit)
    topo = list(dag.topological_op_nodes())
    if not topo:
        return CircuitSlackMetrics(mean_t_slack=0.0, fraction_t_slack_positive=0.0, num_t_nodes=0)

    es: dict[Any, int] = {}
    for node in topo:
        predecessors = list(dag.op_predecessors(node))
        if not predecessors:
            es[node] = 0
        else:
            es[node] = max(es[pred] + 1 for pred in predecessors)

    critical_last_start = max(es.values())
    ls: dict[Any, int] = {}
    for node in reversed(topo):
        successors = list(dag.op_successors(node))
        if not successors:
            ls[node] = critical_last_start
        else:
            ls[node] = min(ls[succ] - 1 for succ in successors)

    t_nodes = [node for node in topo if node.name in {"t", "tdg"}]
    if not t_nodes:
        return CircuitSlackMetrics(mean_t_slack=0.0, fraction_t_slack_positive=0.0, num_t_nodes=0)

    slacks = [ls[node] - es[node] for node in t_nodes]
    positive = sum(1 for slack in slacks if slack > 0)
    return CircuitSlackMetrics(
        mean_t_slack=sum(slacks) / len(slacks),
        fraction_t_slack_positive=positive / len(slacks),
        num_t_nodes=len(t_nodes),
    )
