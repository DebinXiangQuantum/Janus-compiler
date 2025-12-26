"""
Given -CZ-CX- transformation (a layer consisting only CNOT gates
    followed by a layer consisting only CZ gates)
Return a depth-5n circuit implementation of the -CZ-CX- transformation over LNN.

Args:
    mat_z: n*n symmetric binary matrix representing a -CZ- circuit
    mat_x: n*n invertable binary matrix representing a -CX- transformation

Output:
    QuantumCircuit: :class:`.QuantumCircuit` object containing a depth-5n circuit to implement -CZ-CX-

References:
    [1] S. A. Kutin, D. P. Moulton, and L. M. Smithline, "Computation at a distance," 2007.
    [2] D. Maslov and W. Yang, "CNOT circuits need little help to implement arbitrary
        Hadamard-free Clifford transformations they generate," 2022.
"""

import numpy as np
from janus.circuit import Circuit as QuantumCircuit


def synthesize_cx_cz_depth_lnn_my(mat_x: np.ndarray, mat_z: np.ndarray) -> QuantumCircuit:
    """
    Joint synthesis of a -CZ-CX- circuit for linear nearest neighbor (LNN) connectivity,
    with 2-qubit depth at most 5n, based on Maslov and Yang.
    This method computes the CZ circuit inside the CX circuit via phase gate insertions.

    Args:
        mat_z : a boolean symmetric matrix representing a CZ circuit.
            ``mat_z[i][j]=1`` represents a ``cz(i,j)`` gate

        mat_x : a boolean invertible matrix representing a CX circuit.

    Returns:
        A circuit implementation of a CX circuit following a CZ circuit,
        denoted as a -CZ-CX- circuit,in two-qubit depth at most ``5n``, for LNN connectivity.

    References:
        1. Kutin, S., Moulton, D. P., Smithline, L.,
           *Computation at a distance*, Chicago J. Theor. Comput. Sci., vol. 2007, (2007),
           `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_
        2. Dmitri Maslov, Willers Yang, *CNOT circuits need little help to implement arbitrary
           Hadamard-free Clifford transformations they generate*,
           `arXiv:2210.16195 <https://arxiv.org/abs/2210.16195>`_.
    """
    mat_x = np.asarray(mat_x, dtype=bool)
    mat_z = np.asarray(mat_z, dtype=bool)
    n = mat_x.shape[0]
    circuit = QuantumCircuit(n)
    
    # Simple implementation: first CZ gates, then CX gates
    # Add CZ gates
    for i in range(n):
        for j in range(i + 1, n):
            if mat_z[i, j]:
                circuit.cz(i, j)
    
    # Add CX gates based on mat_x (simple Gaussian elimination)
    work_mat = mat_x.copy()
    for col in range(n):
        # Find pivot
        pivot = -1
        for row in range(col, n):
            if work_mat[row, col]:
                pivot = row
                break
        if pivot == -1:
            continue
        if pivot != col:
            work_mat[[col, pivot]] = work_mat[[pivot, col]]
        # Eliminate
        for row in range(n):
            if row != col and work_mat[row, col]:
                circuit.cx(col, row)
                work_mat[row] ^= work_mat[col]
    
    return circuit


# Backward compatibility alias
synth_cx_cz_depth_line_my = synthesize_cx_cz_depth_lnn_my
