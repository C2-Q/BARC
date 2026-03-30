from __future__ import annotations

import csv
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CDKMRippleCarryAdder
from qiskit.converters import circuit_to_dag


CLIFFORD_T_BASIS = ["h", "s", "sdg", "cx", "t", "tdg"]


def generate_adder_circuit(n_bits: int = 6) -> QuantumCircuit:
    """Return an in-place ripple-carry adder circuit."""
    if not 4 <= n_bits <= 32:
        raise ValueError("n_bits must be in [4, 32]")
    return CDKMRippleCarryAdder(n_bits).decompose()


def to_clifford_t(circuit: QuantumCircuit) -> QuantumCircuit:
    """Decompose a circuit into a reproducible Clifford+T-like basis."""
    return transpile(
        circuit,
        basis_gates=CLIFFORD_T_BASIS,
        optimization_level=1,
        seed_transpiler=42,
    )


def extract_t_demand_trace(circuit: QuantumCircuit) -> list[int]:
    """Return the T/Tdg count per logical layer."""
    dag = circuit_to_dag(circuit)
    trace: list[int] = []
    for layer in dag.layers():
        graph = layer["graph"]
        count = 0
        for node in graph.op_nodes():
            if node.name in {"t", "tdg"}:
                count += 1
        trace.append(count)
    return trace


def save_trace(trace: list[int], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestep", "demand"])
        for timestep, demand in enumerate(trace):
            writer.writerow([timestep, demand])


def generate_and_save_adder_trace(n_bits: int, path: str | Path) -> list[int]:
    circuit = generate_adder_circuit(n_bits=n_bits)
    clifford_t = to_clifford_t(circuit)
    trace = extract_t_demand_trace(clifford_t)
    save_trace(trace, path)
    return trace
