"""
Simplified AnnotatedOperation stub for Janus
This is a minimal implementation to support circuit converters
"""

from janus.circuit.operation import Operation


class AnnotatedOperation(Operation):
    """
    Annotated operation - a simplified stub implementation.

    This wraps an operation with modifiers.
    For Janus, we provide minimal functionality.
    """

    def __init__(self, base_op, modifiers=None):
        """
        Args:
            base_op: The base operation
            modifiers: List of modifiers (ignored for now)
        """
        self.base_op = base_op
        self.modifiers = modifiers or []
        # Initialize with base_op properties
        if hasattr(base_op, 'name'):
            self._name = base_op.name
        if hasattr(base_op, 'n_qubits'):
            self._n_qubits = base_op.n_qubits
        if hasattr(base_op, 'params'):
            self._params = base_op.params if hasattr(base_op, 'params') else []

    @property
    def name(self):
        return getattr(self, '_name', 'annotated')

    @property
    def n_qubits(self):
        return getattr(self, '_n_qubits', 0)

    @property
    def params(self):
        return getattr(self, '_params', [])

    def __repr__(self):
        return f"AnnotatedOperation({self.base_op}, modifiers={self.modifiers})"
