"""
Janus 分解模块的统一错误类型

本文件定义了分解模块中使用的所有自定义异常类，确保错误类型的一致性。
"""


class DecomposeError(Exception):
    """分解模块的基础异常类"""
    pass


class UnsupportedMethodError(DecomposeError):
    """当使用不支持的分解方法时抛出"""
    pass


class GateNotSupportedError(DecomposeError):
    """当尝试分解不支持的门类型时抛出"""
    pass


class ParameterError(DecomposeError):
    """当参数无效时抛出"""
    pass


class CircuitError(DecomposeError):
    """当电路操作出错时抛出"""
    pass
