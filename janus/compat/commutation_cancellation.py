"""
Commutation-based gate cancellation - Basic implementation

This implements basic gate cancellation functionality that can eliminate
self-inverse gate pairs (like H-H, X-X, Y-Y, Z-Z).
"""


def cancel_commutations(dag, commutation_checker, basis_gates):
    """
    Cancel commuting gates in the DAG.

    This implementation performs basic gate cancellation:
    1. Identifies self-inverse gates (H, X, Y, Z, CX, CZ)
    2. Finds adjacent pairs of the same gate on the same qubits
    3. Removes these redundant pairs from the DAG

    Args:
        dag: The DAG circuit to optimize
        commutation_checker: Checker for gate commutation relations
        basis_gates: List of basis gate names

    Returns:
        None (modifies dag in place)
    """
    # Self-inverse gates (gates where G * G = I)
    self_inverse_gates = {'h', 'x', 'y', 'z', 'cx', 'cy', 'cz', 's', 'sdg', 't', 'tdg'}

    # Collect all nodes to check
    nodes_to_check = list(dag.topological_op_nodes())

    # Track nodes to remove
    nodes_to_remove = set()

    # Build a mapping from qubits to their operations in order
    qubit_ops = {}
    for node in nodes_to_check:
        if node in nodes_to_remove:
            continue

        gate_name = node.op.name if node.op else ""

        # Only process self-inverse gates
        if gate_name not in self_inverse_gates:
            continue

        # For single-qubit gates
        if len(node.qubits) == 1:
            qubit = node.qubits[0]
            if qubit not in qubit_ops:
                qubit_ops[qubit] = []
            qubit_ops[qubit].append(node)
        # For two-qubit gates
        elif len(node.qubits) == 2:
            qubit_pair = tuple(sorted(node.qubits))
            if qubit_pair not in qubit_ops:
                qubit_ops[qubit_pair] = []
            qubit_ops[qubit_pair].append(node)

    # Find and mark adjacent pairs for removal
    for qubit_key, ops in qubit_ops.items():
        i = 0
        while i < len(ops) - 1:
            node1 = ops[i]
            node2 = ops[i + 1]

            # Check if both nodes are still valid (not marked for removal)
            if node1 in nodes_to_remove or node2 in nodes_to_remove:
                i += 1
                continue

            # Check if they are the same gate
            gate1_name = node1.op.name if node1.op else ""
            gate2_name = node2.op.name if node2.op else ""

            if gate1_name == gate2_name:
                # Check if they operate on the same qubits in the same order
                if node1.qubits == node2.qubits:
                    # Mark both for removal
                    nodes_to_remove.add(node1)
                    nodes_to_remove.add(node2)
                    i += 2  # Skip both nodes
                else:
                    i += 1
            else:
                i += 1

    # Remove marked nodes from DAG
    for node in nodes_to_remove:
        try:
            dag.remove_op_node(node)
        except Exception:
            # Node might have already been removed or invalid
            pass

