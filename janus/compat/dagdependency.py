"""
Simplified DAGDependency stub for Janus
This is a minimal implementation to avoid complex dependencies
Full implementation backed up in dagdependency_full.py.bak
"""
from __future__ import annotations


class DAGDependency:
    """
    Simplified DAG Dependency stub.

    This is a placeholder implementation that provides the minimum interface
    needed for converters. The full implementation requires many dependencies
    that we'll add incrementally as needed.
    """

    def __init__(self):
        """Initialize an empty DAGDependency."""
        self._nodes = []
        self._edges = []
        self.qubits = []
        self.clbits = []
        self._global_phase = 0
        self.qregs = {}
        self.cregs = {}
        self.name = None
        self.metadata = {}

    @property
    def num_qubits(self):
        """Return the number of qubits."""
        return len(self.qubits)

    @property
    def num_clbits(self):
        """Return the number of classical bits."""
        return len(self.clbits)

    @property
    def global_phase(self):
        """Return the global phase."""
        return self._global_phase

    @global_phase.setter
    def global_phase(self, phase):
        """Set the global phase."""
        self._global_phase = phase

    def add_qubits(self, qubits):
        """Add qubits to the DAGDependency."""
        self.qubits.extend(qubits)

    def add_clbits(self, clbits):
        """Add classical bits to the DAGDependency."""
        self.clbits.extend(clbits)

    def add_qreg(self, qreg):
        """Add a quantum register."""
        if hasattr(qreg, 'name'):
            self.qregs[qreg.name] = qreg

    def add_creg(self, creg):
        """Add a classical register."""
        if hasattr(creg, 'name'):
            self.cregs[creg.name] = creg

    def add_op_node(self, op, qargs, cargs):
        """Add an operation node to the DAGDependency."""
        node = DAGDepNode(op=op, qargs=qargs, cargs=cargs)
        self._nodes.append(node)
        return node

    def _add_predecessors(self):
        """Build predecessor information for all nodes (stub)."""
        # Stub implementation - doesn't build actual dependency graph
        pass

    def _add_successors(self):
        """Build successor information for all nodes (stub)."""
        # Stub implementation - doesn't build actual dependency graph
        pass

    def size(self):
        """Return the number of gates/operations in the DAGDependency."""
        return len(self._nodes)

    def __repr__(self):
        return f"DAGDependency(qubits={self.num_qubits}, clbits={self.num_clbits})"


# For compatibility
class DAGDepNode:
    """Simplified DAG Dependency Node stub."""

    def __init__(self, op=None, qargs=None, cargs=None):
        self.op = op
        self.qargs = qargs or []
        self.cargs = cargs or []

    def __repr__(self):
        return f"DAGDepNode(op={self.op})"
