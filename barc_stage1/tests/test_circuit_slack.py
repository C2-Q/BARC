from __future__ import annotations

import unittest

from qiskit import QuantumCircuit

from src.real_trace.circuit_slack import compute_circuit_slack_metrics


class TestCircuitSlack(unittest.TestCase):
    def test_linear_chain_has_zero_slack(self) -> None:
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.t(0)
        circuit.tdg(0)
        circuit.s(0)

        metrics = compute_circuit_slack_metrics(circuit)
        self.assertEqual(metrics.num_t_nodes, 2)
        self.assertEqual(metrics.mean_t_slack, 0.0)
        self.assertEqual(metrics.fraction_t_slack_positive, 0.0)

    def test_parallel_branch_exposes_positive_slack(self) -> None:
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.t(0)
        circuit.t(1)
        circuit.h(0)

        metrics = compute_circuit_slack_metrics(circuit)
        self.assertEqual(metrics.num_t_nodes, 2)
        self.assertAlmostEqual(metrics.mean_t_slack, 0.5)
        self.assertAlmostEqual(metrics.fraction_t_slack_positive, 0.5)


if __name__ == "__main__":
    unittest.main()
