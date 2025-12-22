"""
This file is adapted from Qiskit
Original: qiskit/...
Modified for Janus - removed qiskit dependencies
"""

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper function for converting a dag to a circuit."""

from circuit import Circuit as QuantumCircuit
# Use Janus's native dag_to_circuit instead of Rust accelerated version
from circuit.dag import dag_to_circuit as _janus_dag_to_circuit


def dag_to_circuit(dag, copy_operations=True):
    """Build a ``QuantumCircuit`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): the input dag.
        copy_operations (bool): Deep copy the operation objects
            in the :class:`~.DAGCircuit` for the output :class:`~.QuantumCircuit`.
            This should only be set to ``False`` if the input :class:`~.DAGCircuit`
            will not be used anymore as the operations in the output
            :class:`~.QuantumCircuit` will be shared instances and
            modifications to operations in the :class:`~.DAGCircuit` will
            be reflected in the :class:`~.QuantumCircuit` (and vice versa).

    Return:
        QuantumCircuit: the circuit representing the input dag.

    Example:
        .. plot::
           :alt: Circuit diagram output by the previous code.
           :include-source:

           from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
           from circuit.dag import DAGCircuit
           from compat.converters import circuit_to_dag
           from qiskit.circuit.library.standard_gates import CHGate, U2Gate, CXGate
           from compat.converters import dag_to_circuit

           q = QuantumRegister(3, 'q')
           c = ClassicalRegister(3, 'c')
           circ = QuantumCircuit(q, c)
           circ.h(q[0])
           circ.cx(q[0], q[1])
           circ.measure(q[0], c[0])
           circ.rz(0.5, q[1])
           dag = circuit_to_dag(circ)
           circuit = dag_to_circuit(dag)
           circuit.draw('mpl')
    """

    # Use Janus's native dag_to_circuit (simplified version)
    # TODO: Handle copy_operations if needed
    circuit = _janus_dag_to_circuit(dag)

    # Set additional metadata if available
    if hasattr(dag, 'metadata') and dag.metadata:
        circuit.metadata = dag.metadata
    if hasattr(dag, '_duration'):
        circuit._duration = dag._duration
    if hasattr(dag, '_unit'):
        circuit._unit = dag._unit

    return circuit
