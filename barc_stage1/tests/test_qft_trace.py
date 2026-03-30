from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.real_trace.analysis import load_real_trace_from_csv
from src.real_trace.qft_trace import generate_and_save_qft_trace, generate_qft_circuit


class TestQftTrace(unittest.TestCase):
    def test_generate_qft_circuit_sets_qubit_count(self) -> None:
        circuit = generate_qft_circuit(2)
        self.assertEqual(circuit.num_qubits, 2)

    def test_generate_and_save_qft_trace_is_non_empty_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "qft_n2.csv"
            trace_first = generate_and_save_qft_trace(2, path)
            trace_second = generate_and_save_qft_trace(2, path)
            loaded = load_real_trace_from_csv(str(path))

        self.assertTrue(trace_first)
        self.assertEqual(trace_first, trace_second)
        self.assertEqual(trace_first, loaded)


if __name__ == "__main__":
    unittest.main()
