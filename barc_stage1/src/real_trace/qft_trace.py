from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.synthesis.qft import synth_qft_full

from src.real_trace.adder_trace import (
    extract_t_demand_trace,
    save_trace,
    to_clifford_t,
)


def generate_qft_circuit(
    n_bits: int,
    do_swaps: bool = False,
    approximation_degree: int = 0,
) -> QuantumCircuit:
    """Return a QFT circuit built from Qiskit's synthesis entrypoint."""
    if not 2 <= n_bits <= 32:
        raise ValueError("n_bits must be in [2, 32]")
    if approximation_degree < 0:
        raise ValueError("approximation_degree must be non-negative")
    return synth_qft_full(
        num_qubits=n_bits,
        do_swaps=do_swaps,
        approximation_degree=approximation_degree,
    )


def generate_and_save_qft_trace(
    n_bits: int,
    path: str | Path,
    approximation_degree: int = 0,
) -> list[int]:
    """Generate, transpile, extract, and save a QFT T-demand trace."""
    circuit = generate_qft_circuit(
        n_bits=n_bits,
        do_swaps=False,
        approximation_degree=approximation_degree,
    )
    clifford_t = to_clifford_t(circuit)
    trace = extract_t_demand_trace(clifford_t)
    save_trace(trace, path)
    return trace
