from __future__ import annotations

import unittest

from src.dag import DAG, Node, compute_node_slack, compute_slack_metrics


def _build_manual_dag(nodes: dict[int, Node]) -> DAG:
    return DAG(
        nodes=nodes,
        family="test",
        num_layers=1,
        width=len(nodes),
        t_ratio=0.0,
        seed=0,
    )


class TestDagSlack(unittest.TestCase):
    def test_linear_chain_has_zero_slack(self) -> None:
        nodes = {
            0: Node(id=0, op_type="C", duration=1, predecessors=(), successors=(1,), layer=0, position=0),
            1: Node(id=1, op_type="T", duration=1, predecessors=(0,), successors=(2,), layer=0, position=1),
            2: Node(id=2, op_type="C", duration=1, predecessors=(1,), successors=(3,), layer=0, position=2),
            3: Node(id=3, op_type="T", duration=1, predecessors=(2,), successors=(), layer=0, position=3),
        }
        dag = _build_manual_dag(nodes)
        asap, alap, makespan = compute_node_slack(dag)
        self.assertEqual(makespan, 4)
        for node_id in nodes:
            self.assertEqual(alap[node_id] - asap[node_id], 0)

        slack_metrics = compute_slack_metrics(dag)
        self.assertEqual(slack_metrics.compressibility_mean_t_slack, 0.0)
        self.assertEqual(slack_metrics.compressibility_slack_ratio, 0.0)

    def test_parallel_branch_exposes_positive_slack(self) -> None:
        nodes = {
            0: Node(id=0, op_type="C", duration=1, predecessors=(), successors=(1, 2), layer=0, position=0),
            1: Node(id=1, op_type="T", duration=1, predecessors=(0,), successors=(3,), layer=0, position=1),
            2: Node(id=2, op_type="T", duration=1, predecessors=(0,), successors=(), layer=0, position=2),
            3: Node(id=3, op_type="C", duration=1, predecessors=(1,), successors=(), layer=0, position=3),
        }
        dag = _build_manual_dag(nodes)
        asap, alap, makespan = compute_node_slack(dag)
        self.assertEqual(makespan, 3)
        self.assertEqual(alap[2] - asap[2], 1)
        self.assertEqual(alap[1] - asap[1], 0)

        slack_metrics = compute_slack_metrics(dag)
        self.assertEqual(slack_metrics.t_node_count, 2)
        self.assertAlmostEqual(slack_metrics.compressibility_mean_t_slack, 0.5)
        self.assertAlmostEqual(slack_metrics.compressibility_slack_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
