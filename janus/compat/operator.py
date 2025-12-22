"""
Operator类 - 幺正算子表示
完全独立实现,不依赖qiskit
"""

import numpy as np
from typing import Union, List
from .exceptions import QiskitError


class Operator:
    """
    幺正算子类
    表示量子门的矩阵形式
    """

    def __init__(self, data: Union[np.ndarray, 'Gate', 'Operator'],
                 input_dims: tuple = None, output_dims: tuple = None):
        """
        初始化Operator

        Args:
            data: 矩阵、Gate对象或另一个Operator
            input_dims: 输入维度
            output_dims: 输出维度
        """
        if isinstance(data, Operator):
            self._data = data._data.copy()
            self._input_dims = data._input_dims
            self._output_dims = data._output_dims
        elif isinstance(data, np.ndarray):
            self._data = np.array(data, dtype=complex)
            if self._data.ndim != 2:
                raise QiskitError("Operator data must be 2D array")
            if self._data.shape[0] != self._data.shape[1]:
                raise QiskitError("Operator must be square matrix")

            # 推断维度
            size = self._data.shape[0]
            num_qubits = int(np.log2(size))
            if 2**num_qubits != size:
                raise QiskitError(f"Operator size {size} is not a power of 2")

            self._input_dims = input_dims or (2,) * num_qubits
            self._output_dims = output_dims or (2,) * num_qubits
        else:
            # 尝试从Gate对象获取矩阵
            if hasattr(data, 'to_matrix'):
                self._data = np.array(data.to_matrix(), dtype=complex)
                num_qubits = int(np.log2(self._data.shape[0]))
                self._input_dims = input_dims or (2,) * num_qubits
                self._output_dims = output_dims or (2,) * num_qubits
            else:
                raise QiskitError(f"Cannot create Operator from {type(data)}")

    @property
    def data(self) -> np.ndarray:
        """获取矩阵数据"""
        return self._data

    @property
    def dim(self) -> tuple:
        """获取维度"""
        return (self._input_dims, self._output_dims)

    def to_matrix(self) -> np.ndarray:
        """转换为numpy矩阵"""
        return self._data.copy()

    def conjugate(self):
        """返回共轭"""
        return Operator(np.conj(self._data), self._input_dims, self._output_dims)

    def transpose(self):
        """返回转置"""
        return Operator(self._data.T, self._output_dims, self._input_dims)

    def adjoint(self):
        """返回伴随(共轭转置)"""
        return Operator(self._data.conj().T, self._output_dims, self._input_dims)

    def compose(self, other: 'Operator', qargs: List[int] = None):
        """
        组合两个算子: self @ other

        Args:
            other: 另一个算子
            qargs: 应用other的量子比特索引

        Returns:
            组合后的算子
        """
        if qargs is None:
            # 简单矩阵乘法
            result_data = self._data @ other._data
            return Operator(result_data, other._input_dims, self._output_dims)
        else:
            # 部分量子比特应用
            raise NotImplementedError("Partial qubit compose not yet implemented")

    def tensor(self, other: 'Operator'):
        """
        张量积: self ⊗ other

        Args:
            other: 另一个算子

        Returns:
            张量积算子
        """
        result_data = np.kron(self._data, other._data)
        result_input = self._input_dims + other._input_dims
        result_output = self._output_dims + other._output_dims
        return Operator(result_data, result_input, result_output)

    def power(self, n: int):
        """
        算子的幂: self^n

        Args:
            n: 幂次

        Returns:
            self^n
        """
        if n == 0:
            return Operator(np.eye(self._data.shape[0], dtype=complex),
                          self._input_dims, self._output_dims)
        elif n < 0:
            # 负幂需要求逆
            inv_data = np.linalg.inv(self._data)
            return Operator(inv_data, self._output_dims, self._input_dims).power(-n)
        else:
            result = self
            for _ in range(n - 1):
                result = result.compose(self)
            return result

    def is_unitary(self, atol: float = 1e-8) -> bool:
        """
        检查是否为幺正算子

        Args:
            atol: 绝对容差

        Returns:
            True if幺正
        """
        n = self._data.shape[0]
        product = self._data @ self._data.conj().T
        identity = np.eye(n)
        return np.allclose(product, identity, atol=atol)

    def __matmul__(self, other):
        """矩阵乘法运算符"""
        return self.compose(other)

    def __repr__(self) -> str:
        return f"Operator(shape={self._data.shape}, dims={self.dim})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Operator):
            return False
        return np.allclose(self._data, other._data)


def matrix_equal(mat1: np.ndarray, mat2: np.ndarray,
                 ignore_phase: bool = False,
                 atol: float = 1e-8) -> bool:
    """
    比较两个矩阵是否相等

    Args:
        mat1, mat2: 待比较的矩阵
        ignore_phase: 是否忽略全局相位
        atol: 绝对容差

    Returns:
        True if相等
    """
    if mat1.shape != mat2.shape:
        return False

    if ignore_phase:
        # 忽略全局相位:检查 mat1 = e^{iφ} * mat2
        # 找到第一个非零元素
        for i in range(mat1.size):
            idx = np.unravel_index(i, mat1.shape)
            if abs(mat1[idx]) > atol and abs(mat2[idx]) > atol:
                phase = mat1[idx] / mat2[idx]
                mat2_phased = phase * mat2
                return np.allclose(mat1, mat2_phased, atol=atol)
        # 所有元素都接近零
        return np.allclose(mat1, mat2, atol=atol)
    else:
        return np.allclose(mat1, mat2, atol=atol)


def is_unitary_matrix(mat: np.ndarray, atol: float = 1e-8) -> bool:
    """
    检查矩阵是否为幺正矩阵

    Args:
        mat: 待检查的矩阵
        atol: 绝对容差

    Returns:
        True if幺正
    """
    if mat.shape[0] != mat.shape[1]:
        return False

    n = mat.shape[0]
    product = mat @ mat.conj().T
    identity = np.eye(n)
    return np.allclose(product, identity, atol=atol)
