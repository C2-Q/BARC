from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Literal

import numpy as np


OpType = Literal["T", "C"]


@dataclass(frozen=True)
class Node:
    id: int
    op_type: OpType
    duration: int
    predecessors: tuple[int, ...]
    successors: tuple[int, ...]
    layer: int
    position: int


@dataclass(frozen=True)
class DAG:
    nodes: dict[int, Node]
    family: str
    num_layers: int
    width: int
    t_ratio: float
    seed: int

    @property
    def node_ids(self) -> list[int]:
        return sorted(self.nodes)

    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    @property
    def total_t(self) -> int:
        return sum(1 for node in self.nodes.values() if node.op_type == "T")

    def topological_order(self) -> list[int]:
        indegree = {node_id: len(node.predecessors) for node_id, node in self.nodes.items()}
        ready: list[int] = []
        for node_id, degree in indegree.items():
            if degree == 0:
                heappush(ready, node_id)

        ordered: list[int] = []
        while ready:
            node_id = heappop(ready)
            ordered.append(node_id)
            for successor in self.nodes[node_id].successors:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    heappush(ready, successor)

        if len(ordered) != len(self.nodes):
            raise ValueError("DAG contains a cycle or disconnected dependency bookkeeping is invalid")
        return ordered


@dataclass(frozen=True)
class SlackMetrics:
    compressibility_legacy: float
    compressibility_slack_ratio: float
    compressibility_mean_t_slack: float
    t_node_count: int


def generate_dag(
    family: str,
    num_layers: int,
    width: int,
    t_ratio: float,
    seed: int,
) -> DAG:
    """Generate a dependency-aware synthetic DAG with family-specific structure."""
    _validate_generation_inputs(family, num_layers, width, t_ratio)

    rng = np.random.default_rng(seed)
    node_ids_by_layer = [[layer * width + offset for offset in range(width)] for layer in range(num_layers)]
    predecessors: dict[int, set[int]] = {node_id: set() for layer in node_ids_by_layer for node_id in layer}

    if family == "high_compressibility":
        _build_high_compressibility_edges(node_ids_by_layer, predecessors, rng)
    elif family == "medium_compressibility":
        _build_medium_compressibility_edges(node_ids_by_layer, predecessors, rng)
    elif family == "low_compressibility":
        _build_low_compressibility_edges(node_ids_by_layer, predecessors, rng)
    else:
        raise ValueError(f"Unsupported DAG family: {family}")

    total_nodes = num_layers * width
    total_t = int(round(total_nodes * t_ratio))
    successors: dict[int, set[int]] = {node_id: set() for node_id in predecessors}
    for node_id, preds in predecessors.items():
        for predecessor in preds:
            successors[predecessor].add(node_id)

    op_types = np.array(["C"] * total_nodes, dtype=object)
    if total_t > 0:
        t_indices = _select_t_nodes(
            family=family,
            predecessors=predecessors,
            successors=successors,
            num_layers=num_layers,
            width=width,
            total_t=total_t,
            rng=rng,
        )
        op_types[list(t_indices)] = "T"

    nodes: dict[int, Node] = {}
    for layer_idx, layer_nodes in enumerate(node_ids_by_layer):
        for position, node_id in enumerate(layer_nodes):
            nodes[node_id] = Node(
                id=node_id,
                op_type=op_types[node_id],
                duration=1,
                predecessors=tuple(sorted(predecessors[node_id])),
                successors=tuple(sorted(successors[node_id])),
                layer=layer_idx,
                position=position,
            )
    return DAG(
        nodes=nodes,
        family=family,
        num_layers=num_layers,
        width=width,
        t_ratio=t_ratio,
        seed=seed,
    )


def estimate_compressibility(dag: DAG) -> float:
    """Legacy heuristic proxy kept for backward compatibility."""
    if dag.total_t == 0:
        return 0.0

    asap = _compute_asap_times(dag)
    makespan = max(asap.values()) + 1
    alap = _compute_alap_times(dag, makespan)
    t_slacks = [alap[node_id] - asap[node_id] for node_id, node in dag.nodes.items() if node.op_type == "T"]
    normalization = max(1, makespan - 1)
    return float(np.mean(t_slacks) / normalization)


def compute_slack_metrics(dag: DAG) -> SlackMetrics:
    if dag.total_t == 0:
        return SlackMetrics(
            compressibility_legacy=0.0,
            compressibility_slack_ratio=0.0,
            compressibility_mean_t_slack=0.0,
            t_node_count=0,
        )

    asap, alap, makespan = compute_node_slack(dag)
    del makespan
    t_slacks = [alap[node_id] - asap[node_id] for node_id, node in dag.nodes.items() if node.op_type == "T"]
    positive_slack_count = sum(1 for slack in t_slacks if slack > 0)
    return SlackMetrics(
        compressibility_legacy=estimate_compressibility(dag),
        compressibility_slack_ratio=float(positive_slack_count / max(1, len(t_slacks))),
        compressibility_mean_t_slack=float(np.mean(t_slacks)) if t_slacks else 0.0,
        t_node_count=len(t_slacks),
    )


def _validate_generation_inputs(family: str, num_layers: int, width: int, t_ratio: float) -> None:
    valid_families = {"high_compressibility", "medium_compressibility", "low_compressibility"}
    if family not in valid_families:
        raise ValueError(f"family must be one of {sorted(valid_families)}")
    if num_layers < 2:
        raise ValueError("num_layers must be at least 2")
    if width < 1:
        raise ValueError("width must be at least 1")
    if not 0.0 <= t_ratio <= 1.0:
        raise ValueError("t_ratio must be in [0, 1]")


def _build_high_compressibility_edges(
    node_ids_by_layer: list[list[int]],
    predecessors: dict[int, set[int]],
    rng: np.random.Generator,
) -> None:
    for layer_idx in range(1, len(node_ids_by_layer)):
        prev_layer = node_ids_by_layer[layer_idx - 1]
        prev_prev_layer = node_ids_by_layer[layer_idx - 2] if layer_idx >= 2 else []
        for node_id in node_ids_by_layer[layer_idx]:
            primary = int(rng.choice(prev_layer))
            predecessors[node_id].add(primary)
            if prev_prev_layer and rng.random() < 0.25:
                predecessors[node_id].add(int(rng.choice(prev_prev_layer)))


def _build_medium_compressibility_edges(
    node_ids_by_layer: list[list[int]],
    predecessors: dict[int, set[int]],
    rng: np.random.Generator,
) -> None:
    for layer_idx in range(1, len(node_ids_by_layer)):
        prev_layer = node_ids_by_layer[layer_idx - 1]
        for position, node_id in enumerate(node_ids_by_layer[layer_idx]):
            pred_count = min(len(prev_layer), int(rng.integers(1, 4)))
            for predecessor in rng.choice(prev_layer, size=pred_count, replace=False):
                predecessors[node_id].add(int(predecessor))
            if position > 0 and rng.random() < 0.35:
                predecessors[node_id].add(node_ids_by_layer[layer_idx][position - 1])


def _build_low_compressibility_edges(
    node_ids_by_layer: list[list[int]],
    predecessors: dict[int, set[int]],
    rng: np.random.Generator,
) -> None:
    del rng
    previous_serial_tail: int | None = None
    for layer_idx, layer_nodes in enumerate(node_ids_by_layer):
        for position, node_id in enumerate(layer_nodes):
            if previous_serial_tail is not None:
                predecessors[node_id].add(previous_serial_tail)
            if layer_idx > 0:
                predecessors[node_id].add(node_ids_by_layer[layer_idx - 1][position % len(node_ids_by_layer[layer_idx - 1])])
            if position > 0:
                predecessors[node_id].add(layer_nodes[position - 1])
            previous_serial_tail = node_id


def _compute_asap_times(dag: DAG) -> dict[int, int]:
    asap: dict[int, int] = {}
    for node_id in dag.topological_order():
        preds = dag.nodes[node_id].predecessors
        asap[node_id] = 0 if not preds else max(asap[pred] + dag.nodes[pred].duration for pred in preds)
    return asap


def _compute_alap_times(dag: DAG, makespan: int) -> dict[int, int]:
    alap: dict[int, int] = {}
    for node_id in reversed(dag.topological_order()):
        successors = dag.nodes[node_id].successors
        if not successors:
            alap[node_id] = makespan - dag.nodes[node_id].duration
        else:
            latest_successor_start = min(alap[succ] for succ in successors)
            alap[node_id] = latest_successor_start - dag.nodes[node_id].duration
    return alap


def compute_node_slack(dag: DAG) -> tuple[dict[int, int], dict[int, int], int]:
    """Expose ASAP/ALAP-based slack for schedulers."""
    asap = _compute_asap_times(dag)
    makespan = max(asap.values()) + 1
    alap = _compute_alap_times(dag, makespan)
    return asap, alap, makespan


def _select_t_nodes(
    family: str,
    predecessors: dict[int, set[int]],
    successors: dict[int, set[int]],
    num_layers: int,
    width: int,
    total_t: int,
    rng: np.random.Generator,
) -> np.ndarray:
    temp_nodes: dict[int, Node] = {}
    for layer_idx in range(num_layers):
        for position in range(width):
            node_id = layer_idx * width + position
            temp_nodes[node_id] = Node(
                id=node_id,
                op_type="C",
                duration=1,
                predecessors=tuple(sorted(predecessors[node_id])),
                successors=tuple(sorted(successors[node_id])),
                layer=layer_idx,
                position=position,
            )
    temp_dag = DAG(
        nodes=temp_nodes,
        family=family,
        num_layers=num_layers,
        width=width,
        t_ratio=0.0,
        seed=0,
    )
    asap, alap, _ = compute_node_slack(temp_dag)
    candidates = list(temp_nodes)
    slacks = {node_id: alap[node_id] - asap[node_id] for node_id in candidates}

    if family == "high_compressibility":
        ordered = sorted(candidates, key=lambda node_id: (-slacks[node_id], temp_nodes[node_id].layer, node_id))
        return np.array(ordered[:total_t], dtype=int)

    if family == "medium_compressibility":
        ordered = sorted(candidates, key=lambda node_id: (-slacks[node_id], temp_nodes[node_id].layer, node_id))
        top_count = min(total_t, max(1, int(round(total_t * 0.65))))
        selected = ordered[:top_count]
        remaining = [node_id for node_id in candidates if node_id not in selected]
        if total_t > top_count:
            selected.extend(rng.choice(remaining, size=total_t - top_count, replace=False).astype(int).tolist())
        return np.array(selected, dtype=int)

    ordered = sorted(candidates, key=lambda node_id: (slacks[node_id], temp_nodes[node_id].layer, node_id))
    return np.array(ordered[:total_t], dtype=int)
