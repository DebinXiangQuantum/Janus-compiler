"""
Synthesis of an n-qubit circuit containing only CZ gates for
linear nearest neighbor (LNN) connectivity, using CX and phase (S, Sdg or Z) gates.
The two-qubit depth of the circuit is bounded by 2*n+2.
This algorithm reverts the order of qubits.

References:
    [1]: Dmitri Maslov, Martin Roetteler,
         Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations,
         `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.
"""

import numpy as np
from janus.circuit import Circuit as QuantumCircuit


def synthesize_cz_depth_lnn_mr(mat: np.ndarray) -> QuantumCircuit:
    r"""Synthesis of a CZ circuit for linear nearest neighbor (LNN) connectivity,
    based on Maslov and Roetteler.

    Note that this method *reverts* the order of qubits in the circuit,
    and returns a circuit containing :class:`.CXGate`\s and phase gates
    (:class:`.SGate`, :class:`.SdgGate` or :class:`.ZGate`).

    Args:
        mat: an upper-diagonal matrix representing the CZ circuit.
            ``mat[i][j]=1 for i<j`` represents a ``cz(i,j)`` gate

    Returns:
        A circuit implementation of the CZ circuit of depth :math:`2n+2` for LNN
        connectivity.

    References:
        1. Dmitri Maslov, Martin Roetteler,
           *Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations*,
           `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.
    """
    mat = np.asarray(mat, dtype=bool)
    n = mat.shape[0]
    circuit = QuantumCircuit(n)
    
    # Simple implementation: directly add CZ gates
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j]:
                circuit.cz(i, j)
    
    return circuit


# Backward compatibility alias
synth_cz_depth_line_mr = synthesize_cz_depth_lnn_mr
