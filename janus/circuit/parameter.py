"""
Janus 参数化支持

提供符号参数用于参数化量子电路
"""
from typing import Dict, Set, Union, Optional, List
import uuid
import math


class Parameter:
    """
    符号参数类
    
    用于创建参数化的量子电路，参数可以在后续被赋值
    
    Example:
        theta = Parameter('θ')
        qc.rx(theta, 0)
        qc.assign_parameters({theta: np.pi/2})
    """
    
    def __init__(self, name: str):
        self._name = name
        self._uuid = uuid.uuid4()
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def uuid(self) -> uuid.UUID:
        return self._uuid
    
    def __repr__(self) -> str:
        return f"Parameter({self._name})"
    
    def __str__(self) -> str:
        return self._name
    
    def __hash__(self) -> int:
        return hash(self._uuid)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Parameter):
            return self._uuid == other._uuid
        return False
    
    # 支持基本运算，返回 ParameterExpression
    def __add__(self, other) -> 'ParameterExpression':
        return ParameterExpression({self: 1.0}, other if isinstance(other, (int, float)) else 0.0) + other
    
    def __radd__(self, other) -> 'ParameterExpression':
        return self.__add__(other)
    
    def __sub__(self, other) -> 'ParameterExpression':
        return ParameterExpression({self: 1.0}, 0.0) - other
    
    def __rsub__(self, other) -> 'ParameterExpression':
        return ParameterExpression({self: -1.0}, other if isinstance(other, (int, float)) else 0.0)
    
    def __mul__(self, other) -> 'ParameterExpression':
        if isinstance(other, (int, float)):
            return ParameterExpression({self: float(other)}, 0.0)
        return ParameterExpression({self: 1.0}, 0.0) * other
    
    def __rmul__(self, other) -> 'ParameterExpression':
        return self.__mul__(other)
    
    def __neg__(self) -> 'ParameterExpression':
        return ParameterExpression({self: -1.0}, 0.0)
    
    def __truediv__(self, other) -> 'ParameterExpression':
        if isinstance(other, (int, float)):
            return ParameterExpression({self: 1.0 / float(other)}, 0.0)
        raise TypeError("Cannot divide Parameter by non-numeric type")


class ParameterExpression:
    """
    参数表达式类
    
    表示参数的线性组合: a*p1 + b*p2 + ... + constant
    
    Attributes:
        _coeffs: 参数到系数的映射
        _constant: 常数项
    """
    
    def __init__(self, coeffs: Dict[Parameter, float] = None, constant: float = 0.0):
        self._coeffs = coeffs if coeffs is not None else {}
        self._constant = constant
    
    @property
    def parameters(self) -> Set[Parameter]:
        """获取表达式中的所有参数"""
        return set(self._coeffs.keys())
    
    def is_real(self) -> bool:
        """检查表达式是否为实数（无未绑定参数）"""
        return len(self._coeffs) == 0
    
    def bind(self, parameter_values: Dict[Parameter, float]) -> Union[float, 'ParameterExpression']:
        """
        绑定参数值
        
        Args:
            parameter_values: 参数到值的映射
        
        Returns:
            如果所有参数都被绑定，返回 float；否则返回新的 ParameterExpression
        """
        new_coeffs = {}
        result = self._constant
        
        for param, coeff in self._coeffs.items():
            if param in parameter_values:
                result += coeff * parameter_values[param]
            else:
                new_coeffs[param] = coeff
        
        if len(new_coeffs) == 0:
            return result
        return ParameterExpression(new_coeffs, result)
    
    def __float__(self) -> float:
        if not self.is_real():
            raise TypeError(f"Cannot convert ParameterExpression with unbound parameters to float")
        return self._constant
    
    def __repr__(self) -> str:
        terms = []
        for param, coeff in self._coeffs.items():
            if coeff == 1.0:
                terms.append(str(param))
            elif coeff == -1.0:
                terms.append(f"-{param}")
            else:
                terms.append(f"{coeff}*{param}")
        
        if self._constant != 0.0 or len(terms) == 0:
            terms.append(str(self._constant))
        
        return " + ".join(terms).replace(" + -", " - ")
    
    def __add__(self, other) -> 'ParameterExpression':
        if isinstance(other, (int, float)):
            return ParameterExpression(self._coeffs.copy(), self._constant + other)
        elif isinstance(other, Parameter):
            new_coeffs = self._coeffs.copy()
            new_coeffs[other] = new_coeffs.get(other, 0.0) + 1.0
            return ParameterExpression(new_coeffs, self._constant)
        elif isinstance(other, ParameterExpression):
            new_coeffs = self._coeffs.copy()
            for param, coeff in other._coeffs.items():
                new_coeffs[param] = new_coeffs.get(param, 0.0) + coeff
            return ParameterExpression(new_coeffs, self._constant + other._constant)
        raise TypeError(f"Cannot add ParameterExpression and {type(other)}")
    
    def __radd__(self, other) -> 'ParameterExpression':
        return self.__add__(other)
    
    def __sub__(self, other) -> 'ParameterExpression':
        if isinstance(other, (int, float)):
            return ParameterExpression(self._coeffs.copy(), self._constant - other)
        elif isinstance(other, Parameter):
            new_coeffs = self._coeffs.copy()
            new_coeffs[other] = new_coeffs.get(other, 0.0) - 1.0
            return ParameterExpression(new_coeffs, self._constant)
        elif isinstance(other, ParameterExpression):
            new_coeffs = self._coeffs.copy()
            for param, coeff in other._coeffs.items():
                new_coeffs[param] = new_coeffs.get(param, 0.0) - coeff
            return ParameterExpression(new_coeffs, self._constant - other._constant)
        raise TypeError(f"Cannot subtract {type(other)} from ParameterExpression")
    
    def __rsub__(self, other) -> 'ParameterExpression':
        return (-self).__add__(other)
    
    def __mul__(self, other) -> 'ParameterExpression':
        if isinstance(other, (int, float)):
            new_coeffs = {p: c * other for p, c in self._coeffs.items()}
            return ParameterExpression(new_coeffs, self._constant * other)
        raise TypeError(f"Cannot multiply ParameterExpression by {type(other)}")
    
    def __rmul__(self, other) -> 'ParameterExpression':
        return self.__mul__(other)
    
    def __neg__(self) -> 'ParameterExpression':
        return self.__mul__(-1)
    
    def __truediv__(self, other) -> 'ParameterExpression':
        if isinstance(other, (int, float)):
            return self.__mul__(1.0 / other)
        raise TypeError(f"Cannot divide ParameterExpression by {type(other)}")
    
    def __eq__(self, other) -> bool:
        if isinstance(other, ParameterExpression):
            return self._coeffs == other._coeffs and self._constant == other._constant
        elif isinstance(other, (int, float)) and self.is_real():
            return self._constant == other
        return False
    
    def __hash__(self) -> int:
        return hash((frozenset(self._coeffs.items()), self._constant))


def is_parameterized(value) -> bool:
    """检查值是否包含未绑定的参数"""
    if isinstance(value, Parameter):
        return True
    elif isinstance(value, ParameterExpression):
        return not value.is_real()
    return False
