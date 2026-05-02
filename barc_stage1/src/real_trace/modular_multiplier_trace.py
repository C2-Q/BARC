from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CDKMRippleCarryAdder

from src.real_trace.adder_trace import (
    extract_t_demand_trace,
    save_trace,
    to_clifford_t,
)


MODULAR_MULTIPLIER_VARIANT = "controlled_add_subtract_modulus_chain"


def _bits_of(value: int, width: int) -> list[int]:
    return [(value >> i) & 1 for i in range(width)]


def _select_constants(n_bits: int) -> tuple[int, int]:
    """Return a deterministic (modulus, multiplier_constant) pair.

    Modulus is taken to be 2**n - 1 so the modular reduction always fires
    on the carry-out region for non-trivial inputs. Multiplier constant is
    fixed to 3 (kept below the modulus and coprime with it for n>=3).
    """
    modulus = (1 << n_bits) - 1
    multiplier_constant = 3
    if n_bits == 1:
        modulus = 1
        multiplier_constant = 1
    return modulus, multiplier_constant


def _append_controlled_add_subtract_block(
    circuit: QuantumCircuit,
    control: int,
    target_register: QuantumRegister,
    constant: int,
    modulus: int,
    n_bits: int,
    workspace_register: QuantumRegister,
    helper_qubit: int,
) -> None:
    """Append a controlled add-by-constant followed by an unconditional
    subtract-by-modulus block.

    The structure mirrors the FTQC arithmetic block used in modular
    multiplication: a controlled load of the constant onto a workspace
    register, an addition into a width-(n_bits+1) target register that
    captures the carry-out, an uncompute of the workspace, and a similar
    add-of-(-modulus) phase. The block is intentionally arithmetic-heavy
    so the Clifford+T trace exposes the FTQC delivery pressure of repeated
    controlled modular-style additions; see MODULAR_MULTIPLIER_VARIANT for
    the exact label.
    """
    if target_register.size != n_bits + 1:
        raise ValueError("target_register width must equal n_bits + 1")
    if workspace_register.size != n_bits:
        raise ValueError("workspace_register width must equal n_bits")

    constant_bits = _bits_of(constant % modulus, n_bits)
    adder = CDKMRippleCarryAdder(n_bits, kind="half").decompose()

    # Controlled load of constant onto the workspace register.
    for i, bit in enumerate(constant_bits):
        if bit:
            circuit.cx(control, workspace_register[i])

    # target += workspace (n-bit add with carry-out into target_register[n_bits]).
    circuit.append(
        adder,
        [
            *workspace_register,
            *target_register[:n_bits],
            target_register[n_bits],
            helper_qubit,
        ],
    )

    # Uncompute the constant load.
    for i, bit in enumerate(constant_bits):
        if bit:
            circuit.cx(control, workspace_register[i])

    # Subtract modulus by adding (-modulus mod 2^n) onto target via the
    # same workspace path. We flip workspace bits to encode -modulus,
    # add, and unflip.
    minus_modulus = (-modulus) & ((1 << n_bits) - 1)
    minus_bits = _bits_of(minus_modulus, n_bits)
    for i, bit in enumerate(minus_bits):
        if bit:
            circuit.x(workspace_register[i])
    circuit.append(
        adder,
        [
            *workspace_register,
            *target_register[:n_bits],
            target_register[n_bits],
            helper_qubit,
        ],
    )
    for i, bit in enumerate(minus_bits):
        if bit:
            circuit.x(workspace_register[i])


def generate_modular_multiplier_circuit(
    n_bits: int = 6,
    modulus: int | None = None,
    multiplier_constant: int | None = None,
) -> QuantumCircuit:
    """Return a modular-multiplication-style controlled add/subtract chain.

    For each bit i of the multiplicand register x, the block performs a
    controlled add of c * 2**i (mod N) followed by an unconditional subtract
    of N on a width-(n_bits+1) target register. The chain mirrors the
    structural pattern that appears inside FTQC modular-multiplication blocks
    used by phase-estimation and order-finding workloads, and is intended to
    expose representative T-demand pressure for those FTQC arithmetic stages.

    Note: this implementation does not include the conditional add-back on
    the sign bit that a fully-verified mod-N multiplier requires; it is a
    structural surrogate, labelled accordingly via MODULAR_MULTIPLIER_VARIANT.
    The metric we extract is the T-demand trace of the synthesised circuit,
    so structural representativeness suffices for the bounded-delivery study.
    """
    if not 3 <= n_bits <= 12:
        raise ValueError("n_bits must be in [3, 12]")

    if modulus is None or multiplier_constant is None:
        default_modulus, default_constant = _select_constants(n_bits)
        modulus = default_modulus if modulus is None else modulus
        multiplier_constant = (
            default_constant if multiplier_constant is None else multiplier_constant
        )
    if modulus < 2:
        raise ValueError("modulus must be at least 2")
    if multiplier_constant <= 0 or multiplier_constant >= modulus:
        raise ValueError("multiplier_constant must lie in (0, modulus)")

    x = QuantumRegister(n_bits, "x")
    y = QuantumRegister(n_bits + 1, "y")
    workspace = QuantumRegister(n_bits, "ws")
    helper = QuantumRegister(1, "help")
    qc = QuantumCircuit(
        x,
        y,
        workspace,
        helper,
        name=f"mod_mul_n{n_bits}_c{multiplier_constant}_N{modulus}",
    )

    step_constant = multiplier_constant % modulus
    for bit_index in range(n_bits):
        _append_controlled_add_subtract_block(
            circuit=qc,
            control=x[bit_index],
            target_register=y,
            constant=step_constant,
            modulus=modulus,
            n_bits=n_bits,
            workspace_register=workspace,
            helper_qubit=helper[0],
        )
        step_constant = (step_constant * 2) % modulus

    return qc


def generate_and_save_modular_multiplier_trace(
    n_bits: int,
    path: str | Path,
    modulus: int | None = None,
    multiplier_constant: int | None = None,
) -> list[int]:
    circuit = generate_modular_multiplier_circuit(
        n_bits=n_bits,
        modulus=modulus,
        multiplier_constant=multiplier_constant,
    )
    clifford_t = to_clifford_t(circuit)
    trace = extract_t_demand_trace(clifford_t)
    save_trace(trace, path)
    return trace
