from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit.library import MultiplierGate

from src.real_trace.adder_trace import (
    extract_t_demand_trace,
    save_trace,
    to_clifford_t,
)


def generate_multiplier_circuit(n_bits: int = 4) -> QuantumCircuit:
    """Return a small quantum multiplier circuit based on Qiskit's official MultiplierGate."""
    if not 3 <= n_bits <= 16:
        raise ValueError("n_bits must be in [3, 16]")
    gate = MultiplierGate(n_bits)
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.append(gate, range(gate.num_qubits))
    return circuit


def generate_and_save_multiplier_trace(n_bits: int, path: str | Path) -> list[int]:
    circuit = generate_multiplier_circuit(n_bits=n_bits)
    clifford_t = to_clifford_t(circuit)
    trace = extract_t_demand_trace(clifford_t)
    save_trace(trace, path)
    return trace
