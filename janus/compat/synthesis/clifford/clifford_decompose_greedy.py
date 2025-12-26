"""
Compatibility layer for quantum circuit operations

Independent implementation for Janus
"""

# This code is part of Janus.
#
# Copyright Janus Authors.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for the Clifford class.
"""

# ---------------------------------------------------------------------
# Synthesis based on Bravyi et. al. greedy clifford compiler
# ---------------------------------------------------------------------

from janus.circuit import Circuit as QuantumCircuit
from janus.compat.clifford import Clifford

from janus.compat.accelerate.synthesis.clifford import (
    synth_clifford_greedy as synth_clifford_greedy_inner,
)


def synth_clifford_greedy(clifford: Clifford) -> QuantumCircuit:
    """Decompose a :class:`.Clifford` operator into a :class:`.QuantumCircuit` based
    on the greedy Clifford compiler that is described in Appendix A of
    Bravyi, Hu, Maslov and Shaydulin [1].

    This method typically yields better CX cost compared to the Aaronson-Gottesman method.

    Note that this function only implements the greedy Clifford compiler from Appendix A
    of [1], and not the templates and symbolic Pauli gates optimizations
    that are mentioned in the same paper.

    Args:
        clifford: A Clifford operator.

    Returns:
        A circuit implementation of the Clifford.

    Raises:
        JanusError: if symplectic Gaussian elimination fails.

    References:
        1. Sergey Bravyi, Shaohan Hu, Dmitri Maslov, Ruslan Shaydulin,
           *Clifford Circuit Optimization with Templates and Symbolic Pauli Gates*,
           `arXiv:2105.02291 [quant-ph] <https://arxiv.org/abs/2105.02291>`_
    """
    # Get gate list from accelerate module
    gates = synth_clifford_greedy_inner(clifford.tableau.astype(bool))
    
    # Build circuit from gates
    num_qubits = clifford.num_qubits
    circuit = QuantumCircuit(num_qubits, name=str(clifford))
    
    for gate_name, qubits in gates:
        if gate_name == 'h':
            circuit.h(qubits[0])
        elif gate_name == 's':
            circuit.s(qubits[0])
        elif gate_name == 'sdg':
            circuit.sdg(qubits[0])
        elif gate_name == 'cx':
            circuit.cx(qubits[0], qubits[1])
        elif gate_name == 'x':
            circuit.x(qubits[0])
        elif gate_name == 'y':
            circuit.y(qubits[0])
        elif gate_name == 'z':
            circuit.z(qubits[0])
    
    return circuit
