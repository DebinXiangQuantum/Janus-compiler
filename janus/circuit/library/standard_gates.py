"""
Janus 标准量子门

包含常用的单比特门和两比特门
"""
import numpy as np
from typing import List, Optional
from ..gate import Gate


# ==================== 单比特门 ====================

class HGate(Gate):
    """Hadamard 门"""
    
    def __init__(self, params: Optional[List] = None, label: Optional[str] = None):
        super().__init__('h', 1, params or [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 1],
            [1, -1]
        ], dtype=complex) / np.sqrt(2)
    
    def inverse(self) -> 'HGate':
        return HGate(self._params.copy(), self._label)
    
    def copy(self) -> 'HGate':
        gate = HGate(self._params.copy(), self._label)
        gate._qubits = self._qubits.copy()
        return gate


class XGate(Gate):
    """Pauli-X 门 (NOT 门)"""
    
    def __init__(self, params: Optional[List] = None, label: Optional[str] = None):
        super().__init__('x', 1, params or [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [0, 1],
            [1, 0]
        ], dtype=complex)
    
    def inverse(self) -> 'XGate':
        return XGate(self._params.copy(), self._label)
    
    def copy(self) -> 'XGate':
        gate = XGate(self._params.copy(), self._label)
        gate._qubits = self._qubits.copy()
        return gate


class YGate(Gate):
    """Pauli-Y 门"""
    
    def __init__(self, params: Optional[List] = None, label: Optional[str] = None):
        super().__init__('y', 1, params or [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [0, -1j],
            [1j, 0]
        ], dtype=complex)
    
    def inverse(self) -> 'YGate':
        return YGate(self._params.copy(), self._label)
    
    def copy(self) -> 'YGate':
        gate = YGate(self._params.copy(), self._label)
        gate._qubits = self._qubits.copy()
        return gate


class ZGate(Gate):
    """Pauli-Z 门"""
    
    def __init__(self, params: Optional[List] = None, label: Optional[str] = None):
        super().__init__('z', 1, params or [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0],
            [0, -1]
        ], dtype=complex)
    
    def inverse(self) -> 'ZGate':
        return ZGate(self._params.copy(), self._label)
    
    def copy(self) -> 'ZGate':
        gate = ZGate(self._params.copy(), self._label)
        gate._qubits = self._qubits.copy()
        return gate


class SGate(Gate):
    """S 门 (sqrt(Z))"""
    
    def __init__(self, params: Optional[List] = None, label: Optional[str] = None):
        super().__init__('s', 1, params or [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0],
            [0, 1j]
        ], dtype=complex)
    
    def inverse(self) -> 'SdgGate':
        return SdgGate(self._params.copy(), self._label)
    
    def copy(self) -> 'SGate':
        gate = SGate(self._params.copy(), self._label)
        gate._qubits = self._qubits.copy()
        return gate


class SdgGate(Gate):
    """S† 门 (S 的共轭转置)"""
    
    def __init__(self, params: Optional[List] = None, label: Optional[str] = None):
        super().__init__('sdg', 1, params or [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0],
            [0, -1j]
        ], dtype=complex)
    
    def inverse(self) -> 'SGate':
        return SGate(self._params.copy(), self._label)
    
    def copy(self) -> 'SdgGate':
        gate = SdgGate(self._params.copy(), self._label)
        gate._qubits = self._qubits.copy()
        return gate


class TGate(Gate):
    """T 门 (sqrt(S))"""
    
    def __init__(self, params: Optional[List] = None, label: Optional[str] = None):
        super().__init__('t', 1, params or [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0],
            [0, np.exp(1j * np.pi / 4)]
        ], dtype=complex)
    
    def inverse(self) -> 'TdgGate':
        return TdgGate(self._params.copy(), self._label)
    
    def copy(self) -> 'TGate':
        gate = TGate(self._params.copy(), self._label)
        gate._qubits = self._qubits.copy()
        return gate


class TdgGate(Gate):
    """T† 门 (T 的共轭转置)"""
    
    def __init__(self, params: Optional[List] = None, label: Optional[str] = None):
        super().__init__('tdg', 1, params or [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0],
            [0, np.exp(-1j * np.pi / 4)]
        ], dtype=complex)
    
    def inverse(self) -> 'TGate':
        return TGate(self._params.copy(), self._label)
    
    def copy(self) -> 'TdgGate':
        gate = TdgGate(self._params.copy(), self._label)
        gate._qubits = self._qubits.copy()
        return gate


# ==================== 参数化单比特门 ====================

class RXGate(Gate):
    """RX 旋转门 - 绕 X 轴旋转"""
    
    def __init__(self, theta: float, label: Optional[str] = None):
        super().__init__('rx', 1, [theta], label)
    
    @property
    def theta(self) -> float:
        return self._params[0]
    
    def to_matrix(self) -> np.ndarray:
        theta = self.theta
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        return np.array([
            [cos, -1j * sin],
            [-1j * sin, cos]
        ], dtype=complex)
    
    def inverse(self) -> 'RXGate':
        return RXGate(-self.theta, self._label)
    
    def copy(self) -> 'RXGate':
        gate = RXGate(self.theta, self._label)
        gate._qubits = self._qubits.copy()
        return gate


class RYGate(Gate):
    """RY 旋转门 - 绕 Y 轴旋转"""
    
    def __init__(self, theta: float, label: Optional[str] = None):
        super().__init__('ry', 1, [theta], label)
    
    @property
    def theta(self) -> float:
        return self._params[0]
    
    def to_matrix(self) -> np.ndarray:
        theta = self.theta
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        return np.array([
            [cos, -sin],
            [sin, cos]
        ], dtype=complex)
    
    def inverse(self) -> 'RYGate':
        return RYGate(-self.theta, self._label)
    
    def copy(self) -> 'RYGate':
        gate = RYGate(self.theta, self._label)
        gate._qubits = self._qubits.copy()
        return gate


class RZGate(Gate):
    """RZ 旋转门 - 绕 Z 轴旋转"""
    
    def __init__(self, theta: float, label: Optional[str] = None):
        super().__init__('rz', 1, [theta], label)
    
    @property
    def theta(self) -> float:
        return self._params[0]
    
    def to_matrix(self) -> np.ndarray:
        theta = self.theta
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)
    
    def inverse(self) -> 'RZGate':
        return RZGate(-self.theta, self._label)
    
    def copy(self) -> 'RZGate':
        gate = RZGate(self.theta, self._label)
        gate._qubits = self._qubits.copy()
        return gate


class UGate(Gate):
    """U 门 - 通用单比特门 U(θ, φ, λ)"""
    
    def __init__(self, theta: float, phi: float, lam: float, label: Optional[str] = None):
        super().__init__('u', 1, [theta, phi, lam], label)
    
    @property
    def theta(self) -> float:
        return self._params[0]
    
    @property
    def phi(self) -> float:
        return self._params[1]
    
    @property
    def lam(self) -> float:
        return self._params[2]
    
    def to_matrix(self) -> np.ndarray:
        theta, phi, lam = self.theta, self.phi, self.lam
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        return np.array([
            [cos, -np.exp(1j * lam) * sin],
            [np.exp(1j * phi) * sin, np.exp(1j * (phi + lam)) * cos]
        ], dtype=complex)
    
    def inverse(self) -> 'UGate':
        return UGate(-self.theta, -self.lam, -self.phi, self._label)
    
    def copy(self) -> 'UGate':
        gate = UGate(self.theta, self.phi, self.lam, self._label)
        gate._qubits = self._qubits.copy()
        return gate


# ==================== 两比特门 ====================

class CXGate(Gate):
    """CNOT (CX) 门 - 受控 X 门"""
    
    def __init__(self, params: Optional[List] = None, label: Optional[str] = None):
        super().__init__('cx', 2, params or [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
    
    def inverse(self) -> 'CXGate':
        return CXGate(self._params.copy(), self._label)
    
    def copy(self) -> 'CXGate':
        gate = CXGate(self._params.copy(), self._label)
        gate._qubits = self._qubits.copy()
        return gate


class CZGate(Gate):
    """CZ 门 - 受控 Z 门"""
    
    def __init__(self, params: Optional[List] = None, label: Optional[str] = None):
        super().__init__('cz', 2, params or [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
    
    def inverse(self) -> 'CZGate':
        return CZGate(self._params.copy(), self._label)
    
    def copy(self) -> 'CZGate':
        gate = CZGate(self._params.copy(), self._label)
        gate._qubits = self._qubits.copy()
        return gate


class CRZGate(Gate):
    """CRZ 门 - 受控 RZ 门"""
    
    def __init__(self, theta: float, label: Optional[str] = None):
        super().__init__('crz', 2, [theta], label)
    
    @property
    def theta(self) -> float:
        return self._params[0]
    
    def to_matrix(self) -> np.ndarray:
        theta = self.theta
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.exp(-1j * theta / 2), 0],
            [0, 0, 0, np.exp(1j * theta / 2)]
        ], dtype=complex)
    
    def inverse(self) -> 'CRZGate':
        return CRZGate(-self.theta, self._label)
    
    def copy(self) -> 'CRZGate':
        gate = CRZGate(self.theta, self._label)
        gate._qubits = self._qubits.copy()
        return gate


class SwapGate(Gate):
    """SWAP 门 - 交换两个量子比特"""
    
    def __init__(self, params: Optional[List] = None, label: Optional[str] = None):
        super().__init__('swap', 2, params or [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    def inverse(self) -> 'SwapGate':
        return SwapGate(self._params.copy(), self._label)
    
    def copy(self) -> 'SwapGate':
        gate = SwapGate(self._params.copy(), self._label)
        gate._qubits = self._qubits.copy()
        return gate


# ==================== 特殊操作 ====================

class Barrier(Gate):
    """Barrier - 用于分隔电路层，不执行任何操作"""
    
    def __init__(self, num_qubits: int, label: Optional[str] = None):
        super().__init__('barrier', num_qubits, [], label)
    
    def to_matrix(self) -> np.ndarray:
        # Barrier 没有矩阵表示
        size = 2 ** self._num_qubits
        return np.eye(size, dtype=complex)
    
    def inverse(self) -> 'Barrier':
        return Barrier(self._num_qubits, self._label)
    
    def copy(self) -> 'Barrier':
        gate = Barrier(self._num_qubits, self._label)
        gate._qubits = self._qubits.copy()
        return gate


class Measure(Gate):
    """测量操作 - 将量子比特测量到经典比特"""
    
    def __init__(self, label: Optional[str] = None):
        super().__init__('measure', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        # 测量不是酉操作，返回投影算符
        raise NotImplementedError("Measure is not a unitary operation")
    
    def inverse(self) -> 'Measure':
        raise NotImplementedError("Measure cannot be inverted")
    
    def copy(self) -> 'Measure':
        gate = Measure(self._label)
        gate._qubits = self._qubits.copy()
        return gate


class Reset(Gate):
    """重置操作 - 将量子比特重置为 |0⟩"""
    
    def __init__(self, label: Optional[str] = None):
        super().__init__('reset', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        raise NotImplementedError("Reset is not a unitary operation")
    
    def inverse(self) -> 'Reset':
        raise NotImplementedError("Reset cannot be inverted")
    
    def copy(self) -> 'Reset':
        gate = Reset(self._label)
        gate._qubits = self._qubits.copy()
        return gate
