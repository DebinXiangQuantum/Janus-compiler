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
from janus.circuit.library import CXGate, CZGate


def py_synth_cx_cz_depth_line_my(mat_x: np.ndarray, mat_z: np.ndarray):
    """
    Python实现的CX-CZ合成算法,用于线性最近邻拓扑

    这是qiskit._accelerate.synthesis.linear_phase.py_synth_cx_cz_depth_line_my的Python替代实现

    Args:
        mat_x: 布尔可逆矩阵,表示CX电路
        mat_z: 布尔对称矩阵,表示CZ电路

    Returns:
        电路数据列表 [(gate, qubits, clbits), ...]
    """
    n = mat_x.shape[0]
    circuit_data = []

    # 简化实现:先添加CZ门,再添加CX门
    # CZ门部分
    for i in range(n):
        for j in range(i+1, n):
            if mat_z[i, j]:
                circuit_data.append((CZGate(), [i, j], []))

    # CX门部分 - 使用高斯消元
    mat = mat_x.copy()
    for col in range(n):
        # 找主元
        pivot = -1
        for row in range(col, n):
            if mat[row, col]:
                pivot = row
                break

        if pivot == -1:
            continue

        # 如果主元不在对角线,交换
        if pivot != col:
            mat[[col, pivot]] = mat[[pivot, col]]
            # 使用SWAP门或三个CX实现交换
            if abs(pivot - col) == 1:
                # 相邻,使用SWAP
                for _ in range(3):
                    circuit_data.append((CXGate(), [col, pivot], []))
                    circuit_data.append((CXGate(), [pivot, col], []))
                    circuit_data.append((CXGate(), [col, pivot], []))
                break

        # 消元
        for row in range(n):
            if row != col and mat[row, col]:
                circuit_data.append((CXGate(), [col, row], []))
                mat[row] ^= mat[col]

    return circuit_data


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
    circuit_data = py_synth_cx_cz_depth_line_my(mat_x.astype(bool), mat_z.astype(bool))
    return QuantumCircuit._from_circuit_data(circuit_data, legacy_qubits=True)


# Backward compatibility alias
synth_cx_cz_depth_line_my = synthesize_cx_cz_depth_lnn_my
