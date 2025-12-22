"""
Control flow utilities stub
用于处理量子电路中的控制流
"""

def trivial_recurse(func):
    """
    装饰器: 简单递归处理控制流
    临时stub实现
    """
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


__all__ = ['trivial_recurse']
