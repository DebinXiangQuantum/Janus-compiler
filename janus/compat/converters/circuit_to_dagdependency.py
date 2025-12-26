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

"""Helper function for converting a circuit to a dag dependency"""

from janus.compat.dagdependency import DAGDependency


def circuit_to_dagdependency(circuit, create_preds_and_succs=True):
    """Build a ``DAGDependency`` object from a circuit.

    Args:
        circuit: the input circuit (janus Circuit or QuantumCircuit).
        create_preds_and_succs (bool): whether to construct lists of
            predecessors and successors for every node.

    Return:
        DAGDependency: the DAG representing the input circuit as a dag dependency.
    """
    dagdependency = DAGDependency()
    dagdependency.name = getattr(circuit, 'name', None)
    dagdependency.metadata = getattr(circuit, 'metadata', {})

    dagdependency.add_qubits(circuit.qubits)
    dagdependency.add_clbits(getattr(circuit, 'clbits', []))

    for register in getattr(circuit, 'qregs', []):
        dagdependency.add_qreg(register)

    for register in getattr(circuit, 'cregs', []):
        dagdependency.add_creg(register)

    # Handle circuit data formats
    for instruction in circuit.data:
        # Get operation, qubits, clbits from instruction
        if hasattr(instruction, 'operation'):
            # janus format or CircuitInstruction
            op = instruction.operation
            qubits = instruction.qubits
            clbits = getattr(instruction, 'clbits', [])
        else:
            # Tuple format (op, qubits, clbits)
            op, qubits, clbits = instruction
        
        dagdependency.add_op_node(op, qubits, clbits)

    if create_preds_and_succs:
        dagdependency._add_predecessors()
        dagdependency._add_successors()

    return dagdependency
