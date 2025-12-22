"""
Quaternion类 - 四元数表示
用于单量子比特门的优化
完全独立实现,不依赖qiskit
"""

import numpy as np
from typing import Union


class Quaternion:
    """
    四元数类
    表示为 q = a + bi + cj + dk
    用于表示3D旋转,在量子计算中用于单量子比特门
    """

    def __init__(self, data: Union[np.ndarray, list, tuple]):
        """
        初始化四元数

        Args:
            data: [a, b, c, d] 形式的数据
        """
        if isinstance(data, (list, tuple)):
            data = np.array(data, dtype=float)
        elif isinstance(data, np.ndarray):
            data = data.astype(float)
        else:
            raise TypeError("Data must be array-like")

        if data.shape != (4,):
            raise ValueError("Quaternion must have 4 components")

        self._data = data

    @property
    def data(self) -> np.ndarray:
        """获取数据"""
        return self._data

    @classmethod
    def from_axis_rotation(cls, angle: float, axis: np.ndarray):
        """
        从旋转轴和角度创建四元数

        Args:
            angle: 旋转角度(弧度)
            axis: 旋转轴(3D单位向量)

        Returns:
            Quaternion对象
        """
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)  # 归一化

        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)

        return cls([cos_half, axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half])

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
        """
        从2x2幺正矩阵创建四元数
        U = a*I + i*(b*X + c*Y + d*Z)

        Args:
            matrix: 2x2幺正矩阵

        Returns:
            Quaternion对象
        """
        if matrix.shape != (2, 2):
            raise ValueError("Matrix must be 2x2")

        # 提取Pauli分解系数
        # U = [[a+id, -c+ib], [c+ib, a-id]]
        a = (matrix[0, 0].real + matrix[1, 1].real) / 2.0
        b = (matrix[0, 1].imag + matrix[1, 0].imag) / 2.0
        c = (matrix[0, 1].real - matrix[1, 0].real) / 2.0
        d = (matrix[0, 0].imag - matrix[1, 1].imag) / 2.0

        return cls([a, b, c, d])

    def to_matrix(self) -> np.ndarray:
        """
        转换为2x2幺正矩阵

        Returns:
            2x2 numpy数组
        """
        a, b, c, d = self._data
        return np.array([
            [a + 1j*d, -c + 1j*b],
            [c + 1j*b, a - 1j*d]
        ], dtype=complex)

    def norm(self) -> float:
        """计算四元数的范数"""
        return np.linalg.norm(self._data)

    def normalize(self):
        """归一化四元数"""
        norm = self.norm()
        if norm < 1e-10:
            raise ValueError("Cannot normalize zero quaternion")
        return Quaternion(self._data / norm)

    def conjugate(self):
        """
        四元数共轭
        q* = a - bi - cj - dk
        """
        data = self._data.copy()
        data[1:] *= -1
        return Quaternion(data)

    def multiply(self, other: 'Quaternion'):
        """
        四元数乘法

        Args:
            other: 另一个四元数

        Returns:
            乘积四元数
        """
        a1, b1, c1, d1 = self._data
        a2, b2, c2, d2 = other._data

        a = a1*a2 - b1*b2 - c1*c2 - d1*d2
        b = a1*b2 + b1*a2 + c1*d2 - d1*c2
        c = a1*c2 - b1*d2 + c1*a2 + d1*b2
        d = a1*d2 + b1*c2 - c1*b2 + d1*a2

        return Quaternion([a, b, c, d])

    def __mul__(self, other):
        """乘法运算符"""
        if isinstance(other, Quaternion):
            return self.multiply(other)
        elif isinstance(other, (int, float)):
            return Quaternion(self._data * other)
        else:
            raise TypeError(f"Cannot multiply Quaternion with {type(other)}")

    def __add__(self, other):
        """加法运算符"""
        if not isinstance(other, Quaternion):
            raise TypeError(f"Cannot add Quaternion with {type(other)}")
        return Quaternion(self._data + other._data)

    def __repr__(self) -> str:
        a, b, c, d = self._data
        return f"Quaternion({a:.4f} + {b:.4f}i + {c:.4f}j + {d:.4f}k)"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Quaternion):
            return False
        return np.allclose(self._data, other._data)
