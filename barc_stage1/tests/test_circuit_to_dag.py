from __future__ import annotations

import unittest

from src.real_trace import quantum_circuit_to_internal_dag
from src.real_trace.adder_trace import generate_adder_circuit, to_clifford_t


class TestCircuitToDag(unittest.TestCase):
    def test_conversion_preserves_node_count(self) -> None:
        circuit = to_clifford_t(generate_adder_circuit(n_bits=4))
        dag = quantum_circuit_to_internal_dag(circuit, family="adder_n4")
        self.assertGreater(dag.total_nodes, 0)
        self.assertEqual(dag.total_nodes, len(circuit.data))

    def test_conversion_produces_dependency_respecting_layers(self) -> None:
        circuit = to_clifford_t(generate_adder_circuit(n_bits=4))
        dag = quantum_circuit_to_internal_dag(circuit, family="adder_n4")
        for node in dag.nodes.values():
            for predecessor in node.predecessors:
                self.assertLess(dag.nodes[predecessor].layer, node.layer)


if __name__ == "__main__":
    unittest.main()
