"""
Exceptions for the optimize module
完全独立实现
"""

from janus.compat.exceptions import (
    JanusError,
    TranspilerError,
    DAGCircuitError,
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
    "JanusError",
    "TranspilerError",
    "TranspilerAccessError", 
    "CouplingError",
    "LayoutError",
    "CircuitTooWideForTarget",
    "InvalidLayoutError",
    "DAGCircuitError",
    "DAGDependencyError",
    "CircuitError",
]
