"""
Circuit synthesis for 2-qubit and 3-qubit Cliffords based on Bravyi & Maslov
decomposition.
"""

from janus.circuit import Circuit as QuantumCircuit
from janus.compat.clifford import Clifford

# Import from qiskit._accelerate.synthesis.clifford
from qiskit._accelerate.synthesis.clifford import (
    synth_clifford_bm_inner,
)


def synthesize_clifford_bravyi_maslov(clifford: Clifford) -> QuantumCircuit:
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
    circuit = QuantumCircuit._from_circuit_data(
        synth_clifford_bm_inner(clifford.tableau.astype(bool)),
        legacy_qubits=True,
        name=str(clifford),
    )
    return circuit


# Backward compatibility alias
synth_clifford_bm = synthesize_clifford_bravyi_maslov
