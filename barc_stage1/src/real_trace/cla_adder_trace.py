from __future__ import annotations

import math
from pathlib import Path

from qiskit import QuantumCircuit, QuantumRegister

from src.real_trace.adder_trace import (
    extract_t_demand_trace,
    save_trace,
    to_clifford_t,
)


CLA_VARIANT = "kogge_stone_parallel_prefix"


def generate_cla_adder_circuit(n_bits: int = 8) -> QuantumCircuit:
    """Return a parallel-prefix carry-lookahead adder built on a Kogge-Stone tree.

    The construction is structurally a carry-lookahead adder: initial generate
    and propagate bits are computed in parallel, prefix combinations are
    performed in a Kogge-Stone tree of depth O(log n_bits), and sum bits are
    written into a fresh result register from the level-zero propagate bits
    XORed with the final carry chain.

    The output is left out-of-place (registers a and b are preserved, the sum
    s = a + b lives in register s with one extra carry-out qubit). Prefix-tree
    ancillae are not uncomputed; this is acceptable for T-demand trace
    extraction, which only inspects gate scheduling rather than uncomputation.
    """
    if not 2 <= n_bits <= 32:
        raise ValueError("n_bits must be in [2, 32]")

    n = n_bits
    levels = max(1, int(math.ceil(math.log2(n)))) if n > 1 else 1

    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    s = QuantumRegister(n + 1, "s")
    g_levels = [QuantumRegister(n, f"g{level}") for level in range(levels + 1)]
    p_levels = [QuantumRegister(n, f"p{level}") for level in range(levels + 1)]

    qc = QuantumCircuit(a, b, s, *g_levels, *p_levels, name=f"cla_adder_n{n}")

    # Level 0: initial generate g_i = a_i AND b_i and propagate p_i = a_i XOR b_i.
    for i in range(n):
        qc.ccx(a[i], b[i], g_levels[0][i])
        qc.cx(a[i], p_levels[0][i])
        qc.cx(b[i], p_levels[0][i])

    # Kogge-Stone prefix tree.
    for level in range(1, levels + 1):
        offset = 2 ** (level - 1)
        for i in range(n):
            if i >= offset:
                # G^(level)_i = G^(level-1)_i OR (P^(level-1)_i AND G^(level-1)_{i-offset}).
                # Implemented as two CCX/CX-based contributions into g_levels[level][i].
                qc.cx(g_levels[level - 1][i], g_levels[level][i])
                qc.ccx(
                    p_levels[level - 1][i],
                    g_levels[level - 1][i - offset],
                    g_levels[level][i],
                )
                # P^(level)_i = P^(level-1)_i AND P^(level-1)_{i-offset}.
                qc.ccx(
                    p_levels[level - 1][i],
                    p_levels[level - 1][i - offset],
                    p_levels[level][i],
                )
            else:
                # Carry-through: G and P unchanged at this level.
                qc.cx(g_levels[level - 1][i], g_levels[level][i])
                qc.cx(p_levels[level - 1][i], p_levels[level][i])

    # Sum bits: s_0 = p_0; s_i = p_i XOR c_i for i in [1, n); s_n = carry-out.
    qc.cx(p_levels[0][0], s[0])
    for i in range(1, n):
        qc.cx(p_levels[0][i], s[i])
        qc.cx(g_levels[levels][i - 1], s[i])
    qc.cx(g_levels[levels][n - 1], s[n])

    return qc


def generate_and_save_cla_adder_trace(n_bits: int, path: str | Path) -> list[int]:
    circuit = generate_cla_adder_circuit(n_bits=n_bits)
    clifford_t = to_clifford_t(circuit)
    trace = extract_t_demand_trace(clifford_t)
    save_trace(trace, path)
    return trace
