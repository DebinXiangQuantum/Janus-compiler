"""
Implementation of the GraySynth algorithm for synthesizing CNOT-Phase
circuits with efficient CNOT cost, and the Patel-Hayes-Markov algorithm
for optimal synthesis of linear (CNOT-only) reversible circuits.
"""

from __future__ import annotations

import numpy as np
from janus.circuit import Circuit as QuantumCircuit

# FIXME: from # FIXME: qiskit._accelerate.synthesis.linear import synth_cnot_count_full_pmh as fast_pmh
# Python stub implementation
def fast_pmh(matrix, section_size=2):
    """
    Stub: PMH (Patel-Markov-Hayes) CNOT合成算法

    这是一个占位实现，返回简单的CNOT序列
    实际实现需要Rust加速

    Args:
        matrix: 输入的布尔矩阵
        section_size: 分段大小

    Returns:
        list: CNOT门序列 [[control, target], ...]
    """
    # 简单的高斯消元法生成CNOT
    import numpy as np
    mat = np.array(matrix, dtype=bool)
    n = mat.shape[0]
    gates = []

    # 前向消元
    for col in range(n):
        # 找到主元
        pivot = -1
        for row in range(col, n):
            if mat[row, col]:
                pivot = row
                break

        if pivot == -1:
            continue

        # 交换行（使用SWAP或CX）
        if pivot != col:
            for i in range(n):
                if i != col and i != pivot:
                    if mat[i, col]:
                        gates.append([i, col])
                        mat[i] ^= mat[col]

        # 消元
        for row in range(n):
            if row != col and mat[row, col]:
                gates.append([col, row])
                mat[row] ^= mat[col]

    return gates


def synthesize_cnot_count_pmh(
    state: list[list[bool]] | np.ndarray[bool], section_size: int | None = None
) -> QuantumCircuit:
    r"""
    Synthesize linear reversible circuits for all-to-all architecture
    using Patel, Markov and Hayes method.

    This function is an implementation of the Patel, Markov and Hayes algorithm from [1]
    for optimal synthesis of linear reversible circuits for all-to-all architecture,
    as specified by an :math:`n \times n` matrix.

    Args:
        state: :math:`n \times n` boolean invertible matrix, describing
            the state of the input circuit.
        section_size: The size of each section in the Patel–Markov–Hayes algorithm [1].
            If ``None`` it is chosen to be :math:`\max(2, \alpha\log_2(n))` with
            :math:`\alpha = 0.56`, which approximately minimizes the upper bound on the number
            of row operations given in [1] Eq. (3).

    Returns:
        A CX-only circuit implementing the linear transformation.

    Raises:
        ValueError: When ``section_size`` is larger than the number of columns.

    References:
        1. Patel, Ketan N., Igor L. Markov, and John P. Hayes,
           *Optimal synthesis of linear reversible circuits*,
           Quantum Information & Computation 8.3 (2008): 282-294.
           `arXiv:quant-ph/0302002 [quant-ph] <https://arxiv.org/abs/quant-ph/0302002>`_
    """
    normalized = np.asarray(state).astype(bool)
    if section_size is not None and normalized.shape[1] < section_size:
        raise ValueError(
            f"The section_size ({section_size}) cannot be larger than the number of columns "
            f"({normalized.shape[1]})."
        )

    # call Rust implementation with normalized input
    circuit_data = fast_pmh(normalized, section_size)

    # construct circuit from the data
    return QuantumCircuit._from_circuit_data(circuit_data, legacy_qubits=True)


# Backward compatibility alias
synth_cnot_count_full_pmh = synthesize_cnot_count_pmh
