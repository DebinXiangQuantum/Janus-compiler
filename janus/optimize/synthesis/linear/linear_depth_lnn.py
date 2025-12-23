"""
Optimize the synthesis of an n-qubit circuit contains only CX gates for
linear nearest neighbor (LNN) connectivity.
The depth of the circuit is bounded by 5*n, while the gate count is approximately 2.5*n^2

References:
    [1]: Kutin, S., Moulton, D. P., Smithline, L. (2007).
         Computation at a Distance.
         `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_.
"""

from __future__ import annotations
import numpy as np
from janus.compat.exceptions import QiskitError
from janus.circuit import Circuit as QuantumCircuit
from janus.optimize.synthesis.linear.linear_matrix_utils import check_invertible_binary_matrix
from janus.compat.accelerate.synthesis.linear import py_synth_cnot_depth_line_kms as fast_kms


def synthesize_cnot_depth_lnn_kms(mat: np.ndarray) -> QuantumCircuit:
    """
    Synthesize linear reversible circuit for linear nearest-neighbor architectures using
    Kutin, Moulton, Smithline method.

    Synthesis algorithm for linear reversible circuits from [1], section 7.
    This algorithm synthesizes any linear reversible circuit of :math:`n` qubits over
    a linear nearest-neighbor architecture using CX gates with depth at most :math:`5n`.

    Args:
        mat: A boolean invertible matrix.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``mat`` is not invertible.

    References:
        1. Kutin, S., Moulton, D. P., Smithline, L.,
           *Computation at a distance*, Chicago J. Theor. Comput. Sci., vol. 2007, (2007),
           `arXiv:quant-ph/0701194 <https://arxiv.org/abs/quant-ph/0701194>`_
    """
    if not check_invertible_binary_matrix(mat):
        raise QiskitError("The input matrix is not invertible.")

    circuit_data = fast_kms(mat)

    # Build circuit from gate list
    n = mat.shape[0]
    circuit = QuantumCircuit(n)
    
    for gate_name, qubits in circuit_data:
        if gate_name == 'cx':
            circuit.cx(qubits[0], qubits[1])
    
    return circuit


# Backward compatibility alias
synth_cnot_depth_line_kms = synthesize_cnot_depth_lnn_kms
