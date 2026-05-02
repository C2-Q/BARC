from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from qiskit import QuantumCircuit

from src.real_trace.adder_trace import (
    extract_t_demand_trace,
    save_trace,
    to_clifford_t,
)


QAOA_GRAPH_TYPES = ("ring", "random_3_regular", "erdos_renyi_dense")
ERDOS_RENYI_EDGE_PROBABILITY = 0.5


@dataclass(frozen=True)
class QaoaInstance:
    n: int
    p: int
    graph_type: str
    graph: nx.Graph
    seed: int | None
    gammas: tuple[float, ...]
    betas: tuple[float, ...]


def default_qaoa_angles(p: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return deterministic, non-optimised QAOA angles.

    The schedule is gamma_k = 0.7 / (k+1), beta_k = 0.3 / (k+1) for k=0..p-1.
    The exact values are not important for this study, only that they are
    fixed across runs so that repeated trace extraction is deterministic.
    """
    if p < 1:
        raise ValueError("QAOA depth p must be >= 1")
    gammas = tuple(0.7 / (k + 1) for k in range(p))
    betas = tuple(0.3 / (k + 1) for k in range(p))
    return gammas, betas


def build_maxcut_graph(
    n: int,
    graph_type: str,
    seed: int | None = None,
) -> nx.Graph:
    """Return a deterministic graph instance for QAOA MaxCut.

    The "ring" graph is a cycle on n nodes. The "random_3_regular" graph is
    a random 3-regular graph (requires n*3 even); for n=6,8,10 this is
    always satisfied. The "erdos_renyi_dense" graph is G(n, p_e=0.5).
    """
    if n < 3:
        raise ValueError("n must be at least 3 for a non-trivial MaxCut graph")
    if graph_type == "ring":
        return nx.cycle_graph(n)
    if graph_type == "random_3_regular":
        if (n * 3) % 2 != 0:
            raise ValueError(
                f"random_3_regular requires n*3 to be even (got n={n})"
            )
        rng_seed = 0 if seed is None else int(seed)
        return nx.random_regular_graph(3, n, seed=rng_seed)
    if graph_type == "erdos_renyi_dense":
        rng_seed = 0 if seed is None else int(seed)
        graph = nx.gnp_random_graph(n, ERDOS_RENYI_EDGE_PROBABILITY, seed=rng_seed)
        # Ensure the graph is non-empty; for very small n with bad seeds this
        # can occasionally be empty. Add a deterministic backbone if so.
        if graph.number_of_edges() == 0:
            for i in range(n - 1):
                graph.add_edge(i, i + 1)
        return graph
    raise ValueError(f"unsupported QAOA graph_type: {graph_type}")


def build_qaoa_instance(
    n: int,
    p: int,
    graph_type: str,
    seed: int | None = None,
) -> QaoaInstance:
    graph = build_maxcut_graph(n=n, graph_type=graph_type, seed=seed)
    gammas, betas = default_qaoa_angles(p)
    return QaoaInstance(
        n=n,
        p=p,
        graph_type=graph_type,
        graph=graph,
        seed=seed,
        gammas=gammas,
        betas=betas,
    )


def generate_qaoa_maxcut_circuit(
    n: int = 6,
    p: int = 1,
    graph_type: str = "erdos_renyi_dense",
    seed: int | None = 0,
) -> QuantumCircuit:
    """Return a QAOA MaxCut circuit on n qubits with depth p.

    The circuit has the standard form: an initial Hadamard layer, then for
    each layer k a cost unitary applying exp(-i gamma_k Z_i Z_j) on every
    edge (decomposed as CNOT-RZ(2 gamma_k)-CNOT), and a mixer unitary
    applying RX(2 beta_k) on every node. Angles come from
    default_qaoa_angles. The circuit is unmeasured; downstream code only
    needs the structure for Clifford+T synthesis and T-demand trace
    extraction.
    """
    if graph_type not in QAOA_GRAPH_TYPES:
        raise ValueError(f"graph_type must be one of {QAOA_GRAPH_TYPES}")

    instance = build_qaoa_instance(n=n, p=p, graph_type=graph_type, seed=seed)
    qc = QuantumCircuit(n, name=f"qaoa_n{n}_p{p}_{graph_type}_s{seed}")

    qc.h(range(n))

    for layer in range(p):
        gamma = instance.gammas[layer]
        beta = instance.betas[layer]

        for u, v in instance.graph.edges():
            qc.cx(u, v)
            qc.rz(2.0 * gamma, v)
            qc.cx(u, v)

        for q in range(n):
            qc.rx(2.0 * beta, q)

    return qc


def generate_and_save_qaoa_maxcut_trace(
    n: int,
    p: int,
    graph_type: str,
    seed: int | None,
    path: str | Path,
) -> list[int]:
    circuit = generate_qaoa_maxcut_circuit(
        n=n, p=p, graph_type=graph_type, seed=seed
    )
    clifford_t = to_clifford_t(circuit)
    trace = extract_t_demand_trace(clifford_t)
    save_trace(trace, path)
    return trace
