"""
异常类定义 - 兼容qiskit接口
完全独立实现,不依赖qiskit
"""


class JanusError(Exception):
    """Janus基础异常类"""
    pass


class QiskitError(JanusError):
    """Qiskit兼容的通用错误"""
    pass


class CircuitError(QiskitError):
    """电路相关错误"""
    pass


class TranspilerError(QiskitError):
    """转译器相关错误"""
    pass


class DAGCircuitError(QiskitError):
    """DAG电路相关错误"""
    pass


class OptimizationError(TranspilerError):
    """优化相关错误"""
    pass
