from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush

from src.dag import DAG, compute_node_slack


@dataclass(frozen=True)
class Schedule:
    policy: str
    cycles: list[list[int]]
    t_demand_per_cycle: list[int]
    node_to_cycle: dict[int, int]
    static_depth: int


def build_schedule(dag: DAG, policy: str, capacity_limit: int | None = None) -> Schedule:
    if policy not in {"static_min", "capacity_aware_static", "smoothed", "delivery_aware_slack"}:
        raise ValueError(
            "policy must be 'static_min', 'capacity_aware_static', 'smoothed', or 'delivery_aware_slack'"
        )

    if policy == "static_min":
        cycles = _build_static_min_cycles(dag)
    elif policy == "capacity_aware_static":
        effective_limit = 1 if capacity_limit is None else capacity_limit
        if effective_limit < 1:
            raise ValueError("capacity_limit must be at least 1 for capacity_aware_static")
        cycles = _build_capacity_aware_static_cycles(dag, capacity_limit=effective_limit)
    elif policy == "smoothed":
        cycles = _build_smoothed_cycles(dag)
    else:
        effective_limit = 1 if capacity_limit is None else capacity_limit
        if effective_limit < 1:
            raise ValueError("capacity_limit must be at least 1 for delivery_aware_slack")
        cycles = _build_delivery_aware_slack_cycles(dag, capacity_limit=effective_limit)

    node_to_cycle: dict[int, int] = {}
    t_demand_per_cycle: list[int] = []
    for cycle_idx, cycle_nodes in enumerate(cycles):
        for node_id in cycle_nodes:
            if node_id in node_to_cycle:
                raise ValueError(f"node {node_id} scheduled more than once")
            node_to_cycle[node_id] = cycle_idx
        t_demand_per_cycle.append(sum(1 for node_id in cycle_nodes if dag.nodes[node_id].op_type == "T"))

    if len(node_to_cycle) != dag.total_nodes:
        missing = sorted(set(dag.node_ids) - set(node_to_cycle))
        raise ValueError(f"schedule is incomplete, missing nodes: {missing[:10]}")

    _validate_schedule(dag, node_to_cycle)
    return Schedule(
        policy=policy,
        cycles=cycles,
        t_demand_per_cycle=t_demand_per_cycle,
        node_to_cycle=node_to_cycle,
        static_depth=len(cycles),
    )


def _build_static_min_cycles(dag: DAG) -> list[list[int]]:
    unscheduled_preds = {node_id: len(node.predecessors) for node_id, node in dag.nodes.items()}
    ready = [node_id for node_id, degree in unscheduled_preds.items() if degree == 0]
    cycles: list[list[int]] = []

    while ready:
        cycle_nodes = sorted(ready, key=lambda node_id: (dag.nodes[node_id].op_type != "T", dag.nodes[node_id].layer, node_id))
        cycles.append(cycle_nodes)
        ready = []
        for node_id in cycle_nodes:
            for successor in dag.nodes[node_id].successors:
                unscheduled_preds[successor] -= 1
                if unscheduled_preds[successor] == 0:
                    ready.append(successor)
    return cycles


def _build_capacity_aware_static_cycles(dag: DAG, capacity_limit: int) -> list[list[int]]:
    unscheduled_preds = {node_id: len(node.predecessors) for node_id, node in dag.nodes.items()}
    ready = [node_id for node_id, degree in unscheduled_preds.items() if degree == 0]
    cycles: list[list[int]] = []

    while ready:
        ordered_ready = sorted(ready, key=lambda node_id: (dag.nodes[node_id].layer, node_id))
        ready_clifford = [node_id for node_id in ordered_ready if dag.nodes[node_id].op_type == "C"]
        ready_t = [node_id for node_id in ordered_ready if dag.nodes[node_id].op_type == "T"]
        chosen_t = ready_t[:capacity_limit]
        cycle_nodes = sorted(ready_clifford + chosen_t, key=lambda node_id: (dag.nodes[node_id].layer, node_id))
        cycles.append(cycle_nodes)

        selected = set(cycle_nodes)
        ready = [node_id for node_id in ordered_ready if node_id not in selected]
        for node_id in cycle_nodes:
            for successor in dag.nodes[node_id].successors:
                unscheduled_preds[successor] -= 1
                if unscheduled_preds[successor] == 0:
                    ready.append(successor)
    return cycles


def _build_smoothed_cycles(dag: DAG) -> list[list[int]]:
    asap, alap, minimal_makespan = compute_node_slack(dag)
    total_t = dag.total_t
    t_slacks = [alap[node_id] - asap[node_id] for node_id, node in dag.nodes.items() if node.op_type == "T"]
    extra_horizon = min(3, max(0, round(sum(t_slacks) / max(1, 2 * len(t_slacks))))) if t_slacks else 0
    smoothing_horizon = minimal_makespan + extra_horizon
    target_t_per_cycle = max(1, round(total_t / max(1, smoothing_horizon)))
    relaxed_deadlines = {
        node_id: alap[node_id] + extra_horizon if dag.nodes[node_id].op_type == "T" else alap[node_id]
        for node_id in dag.nodes
    }

    unscheduled_preds = {node_id: len(node.predecessors) for node_id, node in dag.nodes.items()}
    ready_t: list[tuple[int, int, int]] = []
    ready_c: deque[int] = deque()
    scheduled: set[int] = set()
    scheduled_t = 0

    for node_id, degree in unscheduled_preds.items():
        if degree == 0:
            _push_ready_node(dag, node_id, relaxed_deadlines, ready_t, ready_c)

    cycles: list[list[int]] = []
    current_time = 0
    while len(scheduled) < dag.total_nodes:
        cycle_nodes: list[int] = []

        while ready_c:
            node_id = ready_c.popleft()
            if node_id in scheduled:
                continue
            cycle_nodes.append(node_id)

        chosen_t: list[int] = []
        deferred_heap: list[tuple[int, int, int]] = []
        while ready_t:
            latest_start, layer, node_id = heappop(ready_t)
            if node_id in scheduled:
                continue
            if latest_start <= current_time:
                chosen_t.append(node_id)
            else:
                heappush(deferred_heap, (latest_start, layer, node_id))
        ready_t = deferred_heap

        remaining_t = total_t - scheduled_t
        remaining_min_cycles = max(1, smoothing_horizon - current_time)
        adaptive_target = max(target_t_per_cycle, round(remaining_t / remaining_min_cycles))
        desired_t_this_cycle = max(len(chosen_t), adaptive_target)

        while ready_t and len(chosen_t) < desired_t_this_cycle:
            _, _, node_id = heappop(ready_t)
            if node_id in scheduled:
                continue
            chosen_t.append(node_id)

        cycle_nodes.extend(sorted(chosen_t, key=lambda node_id: (dag.nodes[node_id].layer, node_id)))

        if not cycle_nodes:
            if ready_t:
                _, _, node_id = heappop(ready_t)
                cycle_nodes.append(node_id)
            else:
                raise RuntimeError("scheduler deadlocked with no ready nodes")

        cycle_nodes = sorted(set(cycle_nodes), key=lambda node_id: (dag.nodes[node_id].layer, node_id))
        cycles.append(cycle_nodes)

        new_ready_nodes: list[int] = []
        for node_id in cycle_nodes:
            scheduled.add(node_id)
            if dag.nodes[node_id].op_type == "T":
                scheduled_t += 1
            for successor in dag.nodes[node_id].successors:
                unscheduled_preds[successor] -= 1
                if unscheduled_preds[successor] == 0:
                    new_ready_nodes.append(successor)

        for node_id in new_ready_nodes:
            if node_id not in scheduled:
                _push_ready_node(dag, node_id, relaxed_deadlines, ready_t, ready_c)
        current_time += 1

    return cycles


def _build_delivery_aware_slack_cycles(dag: DAG, capacity_limit: int) -> list[list[int]]:
    asap, alap, _ = compute_node_slack(dag)
    slack = {node_id: alap[node_id] - asap[node_id] for node_id in dag.nodes}
    downstream_t_counts = _compute_downstream_t_counts(dag)

    unscheduled_preds = {node_id: len(node.predecessors) for node_id, node in dag.nodes.items()}
    ready = [node_id for node_id, degree in unscheduled_preds.items() if degree == 0]
    cycles: list[list[int]] = []

    while ready:
        ordered_ready = sorted(ready, key=lambda node_id: (dag.nodes[node_id].layer, node_id))
        ready_clifford = [node_id for node_id in ordered_ready if dag.nodes[node_id].op_type == "C"]
        ready_t = [node_id for node_id in ordered_ready if dag.nodes[node_id].op_type == "T"]
        chosen_t = sorted(
            ready_t,
            key=lambda node_id: (
                slack[node_id],
                -downstream_t_counts[node_id],
                dag.nodes[node_id].layer,
                node_id,
            ),
        )[:capacity_limit]
        cycle_nodes = sorted(ready_clifford + chosen_t, key=lambda node_id: (dag.nodes[node_id].layer, node_id))
        cycles.append(cycle_nodes)

        selected = set(cycle_nodes)
        ready = [node_id for node_id in ordered_ready if node_id not in selected]
        for node_id in cycle_nodes:
            for successor in dag.nodes[node_id].successors:
                unscheduled_preds[successor] -= 1
                if unscheduled_preds[successor] == 0:
                    ready.append(successor)
    return cycles


def _push_ready_node(
    dag: DAG,
    node_id: int,
    latest_start: dict[int, int],
    ready_t: list[tuple[int, int, int]],
    ready_c: deque[int],
) -> None:
    if dag.nodes[node_id].op_type == "T":
        heappush(ready_t, (latest_start[node_id], dag.nodes[node_id].layer, node_id))
    else:
        ready_c.append(node_id)


def _compute_downstream_t_counts(dag: DAG) -> dict[int, int]:
    downstream_t: dict[int, int] = {}
    for node_id in reversed(dag.topological_order()):
        successor_scores = [downstream_t[succ] for succ in dag.nodes[node_id].successors]
        downstream_t[node_id] = (1 if dag.nodes[node_id].op_type == "T" else 0) + (max(successor_scores) if successor_scores else 0)
    return downstream_t


def _validate_schedule(dag: DAG, node_to_cycle: dict[int, int]) -> None:
    for node_id, node in dag.nodes.items():
        cycle = node_to_cycle[node_id]
        for predecessor in node.predecessors:
            if node_to_cycle[predecessor] >= cycle:
                raise ValueError(f"dependency violation: {predecessor} -> {node_id}")
