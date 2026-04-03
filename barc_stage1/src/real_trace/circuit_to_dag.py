from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from src.dag import DAG, Node


T_GATE_NAMES = {"t", "tdg"}


def quantum_circuit_to_internal_dag(
    circuit: QuantumCircuit,
    family: str,
    seed: int = 0,
) -> DAG:
    dag = circuit_to_dag(circuit)
    topo = list(dag.topological_op_nodes())
    if not topo:
        return DAG(nodes={}, family=family, num_layers=0, width=0, t_ratio=0.0, seed=seed)

    es: dict[Any, int] = {}
    for node in topo:
        predecessors = list(dag.op_predecessors(node))
        if not predecessors:
            es[node] = 0
        else:
            es[node] = max(es[pred] + 1 for pred in predecessors)

    layer_counts = Counter(es.values())
    layer_positions = defaultdict(int)
    node_ids = {node: idx for idx, node in enumerate(topo)}
    nodes: dict[int, Node] = {}

    for node in topo:
        node_id = node_ids[node]
        layer = es[node]
        position = layer_positions[layer]
        layer_positions[layer] += 1
        predecessors = tuple(sorted(node_ids[pred] for pred in dag.op_predecessors(node)))
        successors = tuple(sorted(node_ids[succ] for succ in dag.op_successors(node)))
        nodes[node_id] = Node(
            id=node_id,
            op_type="T" if node.name in T_GATE_NAMES else "C",
            duration=1,
            predecessors=predecessors,
            successors=successors,
            layer=layer,
            position=position,
        )

    return DAG(
        nodes=nodes,
        family=family,
        num_layers=max(es.values()) + 1,
        width=max(layer_counts.values()),
        t_ratio=sum(1 for node in topo if node.name in T_GATE_NAMES) / max(1, len(topo)),
        seed=seed,
    )
