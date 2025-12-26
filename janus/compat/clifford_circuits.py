"""
Circuit simulation for the Clifford class.
"""

from __future__ import annotations

import numpy as np
import sys
import os

# Add janus to path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from circuit.gate import Gate
from circuit.instruction import Instruction
from .exceptions import JanusError


def _append_circuit(clifford, circuit, qargs=None):
    """Update Clifford inplace by applying a Clifford circuit.

    Args:
        clifford (Clifford): The Clifford to update.
        circuit (QuantumCircuit): The circuit to apply.
        qargs (list or None): The qubits to apply circuit to.

    Returns:
        Clifford: the updated Clifford.

    Raises:
        JanusError: if input circuit cannot be decomposed into Clifford operations.
    """
    if qargs is None:
        qargs = list(range(clifford.num_qubits))

    for instruction in circuit.instructions:
        if hasattr(instruction, 'clbits') and instruction.clbits:
            raise JanusError(
                f"Cannot apply Instruction with classical bits: {instruction.operation.name}"
            )
        # Get the integer position of the qubits
        new_qubits = [qargs[q] for q in instruction.qubits]
        clifford = _append_operation(clifford, instruction.operation, new_qubits)
    return clifford


def _append_operation(clifford, operation, qargs=None):
    """Update Clifford inplace by applying a Clifford operation.

    Args:
        clifford (Clifford): The Clifford to update.
        operation (Instruction or Clifford or str): The operation or composite operation to apply.
        qargs (list or None): The qubits to apply operation to.

    Returns:
        Clifford: the updated Clifford.

    Raises:
        JanusError: if input operation cannot be converted into Clifford operations.
    """
    if qargs is None:
        qargs = list(range(clifford.num_qubits))

    gate = operation

    if isinstance(gate, str):
        # Check if gate is a valid Clifford basis gate string
        if gate not in _BASIS_1Q and gate not in _BASIS_2Q:
            raise JanusError(f"Invalid Clifford gate name string {gate}")
        name = gate
    else:
        name = gate.name

    # Apply gate if it is a Clifford basis gate
    if name in _NON_CLIFFORD:
        raise JanusError(f"Cannot update Clifford with non-Clifford gate {name}")
    if name in _BASIS_1Q:
        if len(qargs) != 1:
            raise JanusError("Invalid qubits for 1-qubit gate.")
        return _BASIS_1Q[name](clifford, qargs[0])
    if name in _BASIS_2Q:
        if len(qargs) != 2:
            raise JanusError("Invalid qubits for 2-qubit gate.")
        return _BASIS_2Q[name](clifford, qargs[0], qargs[1])

    # If gate has a definition, try to decompose it
    if hasattr(gate, 'definition') and gate.definition is not None:
        try:
            return _append_circuit(clifford.copy(), gate.definition, qargs)
        except JanusError:
            pass

    raise JanusError(f"Cannot apply {gate}")


def _n_half_pis(param) -> int:
    try:
        param = float(param)
        epsilon = (abs(param) + 0.5 * 1e-10) % (np.pi / 2)
        if epsilon > 1e-10:
            raise ValueError(f"{param} is not to a multiple of pi/2")
        multiple = int(np.round(param / (np.pi / 2)))
        return multiple % 4
    except TypeError as err:
        raise ValueError(f"{param} is not bounded") from err


def _count_y(x, z, dtype=None):
    """Count the number of Y Paulis"""
    return (x & z).sum(axis=0, dtype=dtype)


def _calculate_composed_phase(x1, z1, x2, z2):
    """Direct calculation of the phase of Pauli((x1, z1)).compose(Pauli(x2, z2))"""
    cnt_phase = 2 * _count_y(x2, z1)
    cnt_y1 = _count_y(x1, z1)
    cnt_y2 = _count_y(x2, z2)
    cnt_y = _count_y(x1 ^ x2, z1 ^ z2)
    phase = (cnt_phase + cnt_y - cnt_y1 - cnt_y2) % 4
    return phase


# ---------------------------------------------------------------------
# Helper functions for applying basis gates
# ---------------------------------------------------------------------

def _append_i(clifford, qubit):
    """Apply an I gate to a Clifford."""
    return clifford


def _append_x(clifford, qubit):
    """Apply an X gate to a Clifford."""
    clifford.phase ^= clifford.z[:, qubit]
    return clifford


def _append_y(clifford, qubit):
    """Apply a Y gate to a Clifford."""
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x ^ z
    return clifford


def _append_z(clifford, qubit):
    """Apply an Z gate to a Clifford."""
    clifford.phase ^= clifford.x[:, qubit]
    return clifford


def _append_h(clifford, qubit):
    """Apply a H gate to a Clifford."""
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x & z
    tmp = x.copy()
    x[:] = z
    z[:] = tmp
    return clifford


def _append_s(clifford, qubit):
    """Apply an S gate to a Clifford."""
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x & z
    z ^= x
    return clifford


def _append_sdg(clifford, qubit):
    """Apply an Sdg gate to a Clifford."""
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x & ~z
    z ^= x
    return clifford


def _append_sx(clifford, qubit):
    """Apply an SX gate to a Clifford."""
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= ~x & z
    x ^= z
    return clifford


def _append_sxdg(clifford, qubit):
    """Apply an SXdg gate to a Clifford."""
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x & z
    x ^= z
    return clifford


def _append_cx(clifford, control, target):
    """Apply a CX gate to a Clifford."""
    x0 = clifford.x[:, control]
    z0 = clifford.z[:, control]
    x1 = clifford.x[:, target]
    z1 = clifford.z[:, target]
    clifford.phase ^= (x1 ^ z0 ^ True) & z1 & x0
    x1 ^= x0
    z0 ^= z1
    return clifford


def _append_cz(clifford, control, target):
    """Apply a CZ gate to a Clifford."""
    x0 = clifford.x[:, control]
    z0 = clifford.z[:, control]
    x1 = clifford.x[:, target]
    z1 = clifford.z[:, target]
    clifford.phase ^= x0 & x1 & (z0 ^ z1)
    z1 ^= x0
    z0 ^= x1
    return clifford


def _append_cy(clifford, control, target):
    """Apply a CY gate to a Clifford."""
    clifford = _append_sdg(clifford, target)
    clifford = _append_cx(clifford, control, target)
    clifford = _append_s(clifford, target)
    return clifford


def _append_swap(clifford, qubit0, qubit1):
    """Apply a Swap gate to a Clifford."""
    clifford.x[:, [qubit0, qubit1]] = clifford.x[:, [qubit1, qubit0]]
    clifford.z[:, [qubit0, qubit1]] = clifford.z[:, [qubit1, qubit0]]
    return clifford


# Basis gate lookup tables
_BASIS_1Q = {
    "i": _append_i,
    "id": _append_i,
    "iden": _append_i,
    "x": _append_x,
    "y": _append_y,
    "z": _append_z,
    "h": _append_h,
    "s": _append_s,
    "sdg": _append_sdg,
    "sx": _append_sx,
    "sxdg": _append_sxdg,
}

_BASIS_2Q = {
    "cx": _append_cx,
    "cnot": _append_cx,
    "cz": _append_cz,
    "cy": _append_cy,
    "swap": _append_swap,
}

_NON_CLIFFORD = {
    "t", "tdg", "ccx", "ccz", "rx", "ry", "rz", "u1", "u2", "u3", "u"
}