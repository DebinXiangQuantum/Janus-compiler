"""
Circuit synthesis for the Clifford class.
"""

# ---------------------------------------------------------------------
# Synthesis based on Bravyi et. al. greedy clifford compiler
# ---------------------------------------------------------------------

from janus.circuit import Circuit as QuantumCircuit
from janus.compat.clifford import Clifford

# Import from qiskit._accelerate.synthesis.clifford
from qiskit._accelerate.synthesis.clifford import (
    synth_clifford_greedy_inner,
)


def synthesize_clifford_greedy(clifford: Clifford) -> QuantumCircuit:
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
        QiskitError: if symplectic Gaussian elimination fails.

    References:
        1. Sergey Bravyi, Shaohan Hu, Dmitri Maslov, Ruslan Shaydulin,
           *Clifford Circuit Optimization with Templates and Symbolic Pauli Gates*,
           `arXiv:2105.02291 [quant-ph] <https://arxiv.org/abs/2105.02291>`_
    """
    circuit = QuantumCircuit._from_circuit_data(
        synth_clifford_greedy_inner(clifford.tableau.astype(bool)),
        legacy_qubits=True,
        name=str(clifford),
    )
    return circuit


# Backward compatibility alias
synth_clifford_greedy = synthesize_clifford_greedy
