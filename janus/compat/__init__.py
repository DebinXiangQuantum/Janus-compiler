"""
Janus Compatibility Layer
提供与qiskit兼容的接口,完全独立实现
"""

from .exceptions import QiskitError, CircuitError, TranspilerError, DAGCircuitError
from .passmanager import (
    GenericPass,
    AnalysisPass,
    TransformationPass,
    PassManager,
    PropertySet,
    RunState,
    PassManagerState,
)
from .operator import Operator, matrix_equal, is_unitary_matrix
from .quaternion import Quaternion

__all__ = [
    # Exceptions
    'QiskitError',
    'CircuitError',
    'TranspilerError',
    'DAGCircuitError',

    # PassManager
    'GenericPass',
    'AnalysisPass',
    'TransformationPass',
    'PassManager',
    'PropertySet',
    'RunState',
    'PassManagerState',

    # Quantum Info
    'Operator',
    'Quaternion',
    'matrix_equal',
    'is_unitary_matrix',
]
