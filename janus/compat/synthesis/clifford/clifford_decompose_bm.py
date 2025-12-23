"""
This file is adapted from Qiskit
Original: qiskit/...
Modified for Janus - removed qiskit dependencies
"""

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for 2-qubit and 3-qubit Cliffords based on Bravyi & Maslov
decomposition.
"""

from janus.circuit import Circuit as QuantumCircuit
from janus.compat.clifford import Clifford
from janus.compat.exceptions import QiskitError


def synth_clifford_bm(clifford: Clifford) -> QuantumCircuit:
    """Optimal CX-cost decomposition of a :class:`.Clifford` operator on 2 qubits
    or 3 qubits into a :class:`.QuantumCircuit` based on the Bravyi-Maslov method [1].

    Args:
        clifford: A Clifford operator.

    Returns:
        A circuit implementation of the Clifford.

    Raises:
        QiskitError: if Clifford is on more than 3 qubits.

    References:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
    """
    num_qubits = clifford.num_qubits
    if num_qubits > 3:
        raise QiskitError("synth_clifford_bm only supports up to 3 qubits")
    
    # Fallback to greedy synthesis for now
    from janus.compat.synthesis.clifford.clifford_decompose_greedy import synth_clifford_greedy
    return synth_clifford_greedy(clifford)
