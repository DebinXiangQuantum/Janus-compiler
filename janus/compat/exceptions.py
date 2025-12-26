"""
异常类定义
完全独立实现
"""


class JanusError(Exception):
    """Janus基础异常类"""
    pass


class CircuitError(JanusError):
    """电路相关错误"""
    pass


class TranspilerError(JanusError):
    """转译器相关错误"""
    pass


class DAGCircuitError(JanusError):
    """DAG电路相关错误"""
    pass


class OptimizationError(TranspilerError):
    """优化相关错误"""
    pass


class SynthesisError(JanusError):
    """合成相关错误"""
    pass


class DecompositionError(JanusError):
    """分解相关错误"""
    pass
