"""
Janus 模拟器异常定义
"""


class SimulatorError(Exception):
    """模拟器基础异常"""
    pass


class InvalidStateError(SimulatorError):
    """无效量子态异常"""
    pass


class InvalidCircuitError(SimulatorError):
    """无效电路异常"""
    pass


class ParameterBindingError(SimulatorError):
    """参数绑定异常"""
    pass
