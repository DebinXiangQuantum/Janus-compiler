"""
Exceptions for the optimize module
完全独立实现,不依赖qiskit
"""

from janus.compat.exceptions import (
    TranspilerError,
    DAGCircuitError,
    QiskitError,
    CircuitError,
)

# 额外的异常类
class TranspilerAccessError(TranspilerError):
    """Transpiler access error"""
    pass

class CouplingError(TranspilerError):
    """Coupling error"""
    pass

class LayoutError(TranspilerError):
    """Layout error"""
    pass

class CircuitTooWideForTarget(TranspilerError):
    """Circuit too wide for target"""
    pass

class InvalidLayoutError(LayoutError):
    """Invalid layout error"""
    pass

class DAGDependencyError(DAGCircuitError):
    """DAG dependency error"""
    pass

__all__ = [
    "TranspilerError",
    "TranspilerAccessError", 
    "CouplingError",
    "LayoutError",
    "CircuitTooWideForTarget",
    "InvalidLayoutError",
    "DAGCircuitError",
    "DAGDependencyError",
    "QiskitError",
    "CircuitError",
]
