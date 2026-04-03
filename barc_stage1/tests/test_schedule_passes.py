from __future__ import annotations

import unittest

from src.dag import DAG, Node
from src.metrics import compute_trace_statistics
from src.schedule import build_schedule
from src.trace import schedule_to_trace_bundle


def _build_manual_dag(nodes: dict[int, Node]) -> DAG:
    return DAG(
        nodes=nodes,
        family="test",
        num_layers=1,
        width=len(nodes),
        t_ratio=0.0,
        seed=0,
    )


class TestSchedulePasses(unittest.TestCase):
    def test_delivery_aware_slack_respects_capacity_limit(self) -> None:
        nodes = {
            0: Node(id=0, op_type="C", duration=1, predecessors=(), successors=(1, 2), layer=0, position=0),
            1: Node(id=1, op_type="T", duration=1, predecessors=(0,), successors=(), layer=1, position=0),
            2: Node(id=2, op_type="T", duration=1, predecessors=(0,), successors=(3,), layer=1, position=1),
            3: Node(id=3, op_type="C", duration=1, predecessors=(2,), successors=(), layer=2, position=0),
        }
        dag = _build_manual_dag(nodes)

        schedule = build_schedule(dag, policy="delivery_aware_slack", capacity_limit=1)

        self.assertTrue(all(demand <= 1 for demand in schedule.t_demand_per_cycle))

    def test_delivery_aware_slack_can_improve_static_depth_over_capacity_aware_static(self) -> None:
        nodes = {
            0: Node(id=0, op_type="C", duration=1, predecessors=(), successors=(1, 2), layer=0, position=0),
            1: Node(id=1, op_type="T", duration=1, predecessors=(0,), successors=(), layer=1, position=0),
            2: Node(id=2, op_type="T", duration=1, predecessors=(0,), successors=(3,), layer=1, position=1),
            3: Node(id=3, op_type="C", duration=1, predecessors=(2,), successors=(), layer=2, position=0),
        }
        dag = _build_manual_dag(nodes)

        capacity_aware = build_schedule(dag, policy="capacity_aware_static", capacity_limit=1)
        delivery_aware = build_schedule(dag, policy="delivery_aware_slack", capacity_limit=1)

        self.assertEqual(capacity_aware.static_depth, 4)
        self.assertEqual(delivery_aware.static_depth, 3)

        capacity_trace = schedule_to_trace_bundle(dag, capacity_aware)
        delivery_trace = schedule_to_trace_bundle(dag, delivery_aware)

        self.assertEqual(compute_trace_statistics(capacity_trace.demand, C=1)[1], 0)
        self.assertEqual(compute_trace_statistics(delivery_trace.demand, C=1)[1], 0)


if __name__ == "__main__":
    unittest.main()
