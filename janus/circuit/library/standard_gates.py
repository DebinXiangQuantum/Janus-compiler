"""
Janus 标准量子门库

"""
import numpy as np
from typing import List, Optional
from ..gate import Gate


# ==================== 单比特 Pauli 门 ====================

class IGate(Gate):
    """Identity 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('id', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.eye(2, dtype=complex)
    
    def inverse(self) -> 'IGate':
        return IGate(self._label)
    
    def copy(self) -> 'IGate':
        g = IGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class XGate(Gate):
    """Pauli-X 门 (NOT 门)"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('x', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    def inverse(self) -> 'XGate':
        return XGate(self._label)
    
    def copy(self) -> 'XGate':
        g = XGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class YGate(Gate):
    """Pauli-Y 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('y', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    def inverse(self) -> 'YGate':
        return YGate(self._label)
    
    def copy(self) -> 'YGate':
        g = YGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class ZGate(Gate):
    """Pauli-Z 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('z', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    def inverse(self) -> 'ZGate':
        return ZGate(self._label)
    
    def copy(self) -> 'ZGate':
        g = ZGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


# ==================== Hadamard 和 Clifford 门 ====================

class HGate(Gate):
    """Hadamard 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('h', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    def inverse(self) -> 'HGate':
        return HGate(self._label)
    
    def copy(self) -> 'HGate':
        g = HGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class SGate(Gate):
    """S 门 (sqrt(Z)), 也叫 P 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('s', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    
    def inverse(self) -> 'SdgGate':
        return SdgGate(self._label)
    
    def copy(self) -> 'SGate':
        g = SGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class SdgGate(Gate):
    """S† 门 (S 的共轭转置)"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('sdg', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1j]], dtype=complex)
    
    def inverse(self) -> 'SGate':
        return SGate(self._label)
    
    def copy(self) -> 'SdgGate':
        g = SdgGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class TGate(Gate):
    """T 门 (sqrt(S))"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('t', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    
    def inverse(self) -> 'TdgGate':
        return TdgGate(self._label)
    
    def copy(self) -> 'TGate':
        g = TGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class TdgGate(Gate):
    """T† 门 (T 的共轭转置)"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('tdg', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
    
    def inverse(self) -> 'TGate':
        return TGate(self._label)
    
    def copy(self) -> 'TdgGate':
        g = TdgGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class SXGate(Gate):
    """sqrt(X) 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('sx', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=complex) / 2
    
    def inverse(self) -> 'SXdgGate':
        return SXdgGate(self._label)
    
    def copy(self) -> 'SXGate':
        g = SXGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class SXdgGate(Gate):
    """sqrt(X)† 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('sxdg', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([[1-1j, 1+1j], [1+1j, 1-1j]], dtype=complex) / 2
    
    def inverse(self) -> 'SXGate':
        return SXGate(self._label)
    
    def copy(self) -> 'SXdgGate':
        g = SXdgGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


# ==================== 单比特旋转门 ====================

class RXGate(Gate):
    """RX 旋转门 - 绕 X 轴旋转"""
    def __init__(self, theta: float, label: Optional[str] = None):
        super().__init__('rx', 1, [theta], label)
    
    @property
    def theta(self) -> float:
        return self._params[0]
    
    def to_matrix(self) -> np.ndarray:
        c, s = np.cos(self.theta / 2), np.sin(self.theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    
    def inverse(self) -> 'RXGate':
        return RXGate(-self.theta, self._label)
    
    def copy(self) -> 'RXGate':
        g = RXGate(self.theta, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class RYGate(Gate):
    """RY 旋转门 - 绕 Y 轴旋转"""
    def __init__(self, theta: float, label: Optional[str] = None):
        super().__init__('ry', 1, [theta], label)
    
    @property
    def theta(self) -> float:
        return self._params[0]
    
    def to_matrix(self) -> np.ndarray:
        c, s = np.cos(self.theta / 2), np.sin(self.theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    
    def inverse(self) -> 'RYGate':
        return RYGate(-self.theta, self._label)
    
    def copy(self) -> 'RYGate':
        g = RYGate(self.theta, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class RZGate(Gate):
    """RZ 旋转门 - 绕 Z 轴旋转"""
    def __init__(self, theta: float, label: Optional[str] = None):
        super().__init__('rz', 1, [theta], label)
    
    @property
    def theta(self) -> float:
        return self._params[0]
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [np.exp(-1j * self.theta / 2), 0],
            [0, np.exp(1j * self.theta / 2)]
        ], dtype=complex)
    
    def inverse(self) -> 'RZGate':
        return RZGate(-self.theta, self._label)
    
    def copy(self) -> 'RZGate':
        g = RZGate(self.theta, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class PhaseGate(Gate):
    """Phase 门 P(λ) = diag(1, e^{iλ})"""
    def __init__(self, lam: float, label: Optional[str] = None):
        super().__init__('p', 1, [lam], label)
    
    @property
    def lam(self) -> float:
        return self._params[0]
    
    def to_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * self.lam)]], dtype=complex)
    
    def inverse(self) -> 'PhaseGate':
        return PhaseGate(-self.lam, self._label)
    
    def copy(self) -> 'PhaseGate':
        g = PhaseGate(self.lam, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class U1Gate(Gate):
    """U1 门 (等价于 Phase 门)"""
    def __init__(self, lam: float, label: Optional[str] = None):
        super().__init__('u1', 1, [lam], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, np.exp(1j * self._params[0])]], dtype=complex)
    
    def inverse(self) -> 'U1Gate':
        return U1Gate(-self._params[0], self._label)
    
    def copy(self) -> 'U1Gate':
        g = U1Gate(self._params[0], self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class U2Gate(Gate):
    """U2 门 U2(φ, λ) = U(π/2, φ, λ)"""
    def __init__(self, phi: float, lam: float, label: Optional[str] = None):
        super().__init__('u2', 1, [phi, lam], label)
    
    def to_matrix(self) -> np.ndarray:
        phi, lam = self._params[0], self._params[1]
        return np.array([
            [1, -np.exp(1j * lam)],
            [np.exp(1j * phi), np.exp(1j * (phi + lam))]
        ], dtype=complex) / np.sqrt(2)
    
    def inverse(self) -> 'U2Gate':
        return U2Gate(-self._params[1] - np.pi, -self._params[0] + np.pi, self._label)
    
    def copy(self) -> 'U2Gate':
        g = U2Gate(self._params[0], self._params[1], self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class U3Gate(Gate):
    """U3 门 (通用单比特门) U3(θ, φ, λ)"""
    def __init__(self, theta: float, phi: float, lam: float, label: Optional[str] = None):
        super().__init__('u3', 1, [theta, phi, lam], label)
    
    def to_matrix(self) -> np.ndarray:
        theta, phi, lam = self._params
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
        ], dtype=complex)
    
    def inverse(self) -> 'U3Gate':
        return U3Gate(-self._params[0], -self._params[2], -self._params[1], self._label)
    
    def copy(self) -> 'U3Gate':
        g = U3Gate(*self._params, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class UGate(Gate):
    """U 门 (通用单比特门，等价于 U3)"""
    def __init__(self, theta: float, phi: float, lam: float, label: Optional[str] = None):
        super().__init__('u', 1, [theta, phi, lam], label)
    
    def to_matrix(self) -> np.ndarray:
        theta, phi, lam = self._params
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
        ], dtype=complex)
    
    def inverse(self) -> 'UGate':
        return UGate(-self._params[0], -self._params[2], -self._params[1], self._label)
    
    def copy(self) -> 'UGate':
        g = UGate(*self._params, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class RGate(Gate):
    """R 门 R(θ, φ) - 绕任意轴旋转"""
    def __init__(self, theta: float, phi: float, label: Optional[str] = None):
        super().__init__('r', 1, [theta, phi], label)
    
    def to_matrix(self) -> np.ndarray:
        theta, phi = self._params
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, -1j * np.exp(-1j * phi) * s],
            [-1j * np.exp(1j * phi) * s, c]
        ], dtype=complex)
    
    def inverse(self) -> 'RGate':
        return RGate(-self._params[0], self._params[1], self._label)
    
    def copy(self) -> 'RGate':
        g = RGate(*self._params, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class RXXGate(Gate):
    """RXX 门 - 两比特 XX 旋转"""
    def __init__(self, theta: float, label: Optional[str] = None):
        super().__init__('rxx', 2, [theta], label)
    
    def to_matrix(self) -> np.ndarray:
        theta = self._params[0]
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, 0, 0, -1j*s],
            [0, c, -1j*s, 0],
            [0, -1j*s, c, 0],
            [-1j*s, 0, 0, c]
        ], dtype=complex)
    
    def inverse(self) -> 'RXXGate':
        return RXXGate(-self._params[0], self._label)
    
    def copy(self) -> 'RXXGate':
        g = RXXGate(self._params[0], self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class RYYGate(Gate):
    """RYY 门 - 两比特 YY 旋转"""
    def __init__(self, theta: float, label: Optional[str] = None):
        super().__init__('ryy', 2, [theta], label)
    
    def to_matrix(self) -> np.ndarray:
        theta = self._params[0]
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, 0, 0, 1j*s],
            [0, c, -1j*s, 0],
            [0, -1j*s, c, 0],
            [1j*s, 0, 0, c]
        ], dtype=complex)
    
    def inverse(self) -> 'RYYGate':
        return RYYGate(-self._params[0], self._label)
    
    def copy(self) -> 'RYYGate':
        g = RYYGate(self._params[0], self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class RZZGate(Gate):
    """RZZ 门 - 两比特 ZZ 旋转"""
    def __init__(self, theta: float, label: Optional[str] = None):
        super().__init__('rzz', 2, [theta], label)
    
    def to_matrix(self) -> np.ndarray:
        theta = self._params[0]
        return np.diag([
            np.exp(-1j * theta / 2),
            np.exp(1j * theta / 2),
            np.exp(1j * theta / 2),
            np.exp(-1j * theta / 2)
        ])
    
    def inverse(self) -> 'RZZGate':
        return RZZGate(-self._params[0], self._label)
    
    def copy(self) -> 'RZZGate':
        g = RZZGate(self._params[0], self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class RZXGate(Gate):
    """RZX 门 - 两比特 ZX 旋转"""
    def __init__(self, theta: float, label: Optional[str] = None):
        super().__init__('rzx', 2, [theta], label)
    
    def to_matrix(self) -> np.ndarray:
        theta = self._params[0]
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, -1j*s, 0, 0],
            [-1j*s, c, 0, 0],
            [0, 0, c, 1j*s],
            [0, 0, 1j*s, c]
        ], dtype=complex)
    
    def inverse(self) -> 'RZXGate':
        return RZXGate(-self._params[0], self._label)
    
    def copy(self) -> 'RZXGate':
        g = RZXGate(self._params[0], self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


# ==================== 两比特门 ====================

class CXGate(Gate):
    """CNOT (CX) 门 - 受控 X 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('cx', 2, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
    
    def inverse(self) -> 'CXGate':
        return CXGate(self._label)
    
    def copy(self) -> 'CXGate':
        g = CXGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class CYGate(Gate):
    """CY 门 - 受控 Y 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('cy', 2, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0]
        ], dtype=complex)
    
    def inverse(self) -> 'CYGate':
        return CYGate(self._label)
    
    def copy(self) -> 'CYGate':
        g = CYGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class CZGate(Gate):
    """CZ 门 - 受控 Z 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('cz', 2, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.diag([1, 1, 1, -1]).astype(complex)
    
    def inverse(self) -> 'CZGate':
        return CZGate(self._label)
    
    def copy(self) -> 'CZGate':
        g = CZGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class CHGate(Gate):
    """CH 门 - 受控 Hadamard 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('ch', 2, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1/np.sqrt(2), 1/np.sqrt(2)],
            [0, 0, 1/np.sqrt(2), -1/np.sqrt(2)]
        ], dtype=complex)
    
    def inverse(self) -> 'CHGate':
        return CHGate(self._label)
    
    def copy(self) -> 'CHGate':
        g = CHGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class CSGate(Gate):
    """CS 门 - 受控 S 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('cs', 2, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.diag([1, 1, 1, 1j]).astype(complex)
    
    def inverse(self) -> 'CSdgGate':
        return CSdgGate(self._label)
    
    def copy(self) -> 'CSGate':
        g = CSGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class CSdgGate(Gate):
    """CS† 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('csdg', 2, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.diag([1, 1, 1, -1j]).astype(complex)
    
    def inverse(self) -> 'CSGate':
        return CSGate(self._label)
    
    def copy(self) -> 'CSdgGate':
        g = CSdgGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class CSXGate(Gate):
    """CSX 门 - 受控 sqrt(X) 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('csx', 2, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, (1+1j)/2, (1-1j)/2],
            [0, 0, (1-1j)/2, (1+1j)/2]
        ], dtype=complex)
    
    def inverse(self) -> 'CSXGate':
        # CSX^3 = CSX^{-1}
        raise NotImplementedError("CSX inverse requires CSXdg")
    
    def copy(self) -> 'CSXGate':
        g = CSXGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class DCXGate(Gate):
    """DCX 门 - Double CNOT 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('dcx', 2, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=complex)
    
    def inverse(self) -> 'DCXGate':
        return DCXGate(self._label)
    
    def copy(self) -> 'DCXGate':
        g = DCXGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class ECRGate(Gate):
    """ECR 门 - Echoed Cross-Resonance 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('ecr', 2, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [0, 0, 1, 1j],
            [0, 0, 1j, 1],
            [1, -1j, 0, 0],
            [-1j, 1, 0, 0]
        ], dtype=complex) / np.sqrt(2)
    
    def inverse(self) -> 'ECRGate':
        return ECRGate(self._label)
    
    def copy(self) -> 'ECRGate':
        g = ECRGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class SwapGate(Gate):
    """SWAP 门 - 交换两个量子比特"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('swap', 2, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    def inverse(self) -> 'SwapGate':
        return SwapGate(self._label)
    
    def copy(self) -> 'SwapGate':
        g = SwapGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class iSwapGate(Gate):
    """iSWAP 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('iswap', 2, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    def inverse(self) -> 'iSwapGate':
        # iSWAP^{-1} 需要 iSWAP^3
        raise NotImplementedError("iSWAP inverse not implemented")
    
    def copy(self) -> 'iSwapGate':
        g = iSwapGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


# ==================== 受控旋转门 ====================

class CRXGate(Gate):
    """CRX 门 - 受控 RX 门"""
    def __init__(self, theta: float, label: Optional[str] = None):
        super().__init__('crx', 2, [theta], label)
    
    def to_matrix(self) -> np.ndarray:
        theta = self._params[0]
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -1j*s],
            [0, 0, -1j*s, c]
        ], dtype=complex)
    
    def inverse(self) -> 'CRXGate':
        return CRXGate(-self._params[0], self._label)
    
    def copy(self) -> 'CRXGate':
        g = CRXGate(self._params[0], self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class CRYGate(Gate):
    """CRY 门 - 受控 RY 门"""
    def __init__(self, theta: float, label: Optional[str] = None):
        super().__init__('cry', 2, [theta], label)
    
    def to_matrix(self) -> np.ndarray:
        theta = self._params[0]
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -s],
            [0, 0, s, c]
        ], dtype=complex)
    
    def inverse(self) -> 'CRYGate':
        return CRYGate(-self._params[0], self._label)
    
    def copy(self) -> 'CRYGate':
        g = CRYGate(self._params[0], self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class CRZGate(Gate):
    """CRZ 门 - 受控 RZ 门"""
    def __init__(self, theta: float, label: Optional[str] = None):
        super().__init__('crz', 2, [theta], label)
    
    def to_matrix(self) -> np.ndarray:
        theta = self._params[0]
        return np.diag([1, 1, np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])
    
    def inverse(self) -> 'CRZGate':
        return CRZGate(-self._params[0], self._label)
    
    def copy(self) -> 'CRZGate':
        g = CRZGate(self._params[0], self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class CPhaseGate(Gate):
    """CPhase 门 (CP) - 受控 Phase 门"""
    def __init__(self, theta: float, label: Optional[str] = None):
        super().__init__('cp', 2, [theta], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.diag([1, 1, 1, np.exp(1j * self._params[0])])
    
    def inverse(self) -> 'CPhaseGate':
        return CPhaseGate(-self._params[0], self._label)
    
    def copy(self) -> 'CPhaseGate':
        g = CPhaseGate(self._params[0], self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class CU1Gate(Gate):
    """CU1 门 - 受控 U1 门"""
    def __init__(self, lam: float, label: Optional[str] = None):
        super().__init__('cu1', 2, [lam], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.diag([1, 1, 1, np.exp(1j * self._params[0])])
    
    def inverse(self) -> 'CU1Gate':
        return CU1Gate(-self._params[0], self._label)
    
    def copy(self) -> 'CU1Gate':
        g = CU1Gate(self._params[0], self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class CU3Gate(Gate):
    """CU3 门 - 受控 U3 门"""
    def __init__(self, theta: float, phi: float, lam: float, label: Optional[str] = None):
        super().__init__('cu3', 2, [theta, phi, lam], label)
    
    def to_matrix(self) -> np.ndarray:
        theta, phi, lam = self._params
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -np.exp(1j * lam) * s],
            [0, 0, np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
        ], dtype=complex)
    
    def inverse(self) -> 'CU3Gate':
        return CU3Gate(-self._params[0], -self._params[2], -self._params[1], self._label)
    
    def copy(self) -> 'CU3Gate':
        g = CU3Gate(*self._params, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class CUGate(Gate):
    """CU 门 - 受控 U 门 (带全局相位)"""
    def __init__(self, theta: float, phi: float, lam: float, gamma: float, label: Optional[str] = None):
        super().__init__('cu', 2, [theta, phi, lam, gamma], label)
    
    def to_matrix(self) -> np.ndarray:
        theta, phi, lam, gamma = self._params
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.exp(1j * gamma) * c, -np.exp(1j * (gamma + lam)) * s],
            [0, 0, np.exp(1j * (gamma + phi)) * s, np.exp(1j * (gamma + phi + lam)) * c]
        ], dtype=complex)
    
    def inverse(self) -> 'CUGate':
        return CUGate(-self._params[0], -self._params[2], -self._params[1], -self._params[3], self._label)
    
    def copy(self) -> 'CUGate':
        g = CUGate(*self._params, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


# ==================== 三比特门 ====================

class CCXGate(Gate):
    """CCX 门 (Toffoli 门) - 双控制 X 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('ccx', 3, [], label)
    
    def to_matrix(self) -> np.ndarray:
        mat = np.eye(8, dtype=complex)
        mat[6, 6], mat[6, 7] = 0, 1
        mat[7, 6], mat[7, 7] = 1, 0
        return mat
    
    def inverse(self) -> 'CCXGate':
        return CCXGate(self._label)
    
    def copy(self) -> 'CCXGate':
        g = CCXGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class CCZGate(Gate):
    """CCZ 门 - 双控制 Z 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('ccz', 3, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.diag([1, 1, 1, 1, 1, 1, 1, -1]).astype(complex)
    
    def inverse(self) -> 'CCZGate':
        return CCZGate(self._label)
    
    def copy(self) -> 'CCZGate':
        g = CCZGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class CSwapGate(Gate):
    """CSwap 门 (Fredkin 门) - 受控 SWAP 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('cswap', 3, [], label)
    
    def to_matrix(self) -> np.ndarray:
        mat = np.eye(8, dtype=complex)
        mat[5, 5], mat[5, 6] = 0, 1
        mat[6, 5], mat[6, 6] = 1, 0
        return mat
    
    def inverse(self) -> 'CSwapGate':
        return CSwapGate(self._label)
    
    def copy(self) -> 'CSwapGate':
        g = CSwapGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class RCCXGate(Gate):
    """RCCX 门 - 简化 Toffoli 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('rccx', 3, [], label)
    
    def to_matrix(self) -> np.ndarray:
        mat = np.eye(8, dtype=complex)
        mat[6, 6], mat[6, 7] = 0, -1j
        mat[7, 6], mat[7, 7] = -1j, 0
        return mat
    
    def inverse(self) -> 'RCCXGate':
        return RCCXGate(self._label)
    
    def copy(self) -> 'RCCXGate':
        g = RCCXGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class RC3XGate(Gate):
    """RC3X 门 - 简化三控制 X 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('rc3x', 4, [], label)
    
    def to_matrix(self) -> np.ndarray:
        mat = np.eye(16, dtype=complex)
        mat[14, 14], mat[14, 15] = 0, -1j
        mat[15, 14], mat[15, 15] = -1j, 0
        return mat
    
    def inverse(self) -> 'RC3XGate':
        return RC3XGate(self._label)
    
    def copy(self) -> 'RC3XGate':
        g = RC3XGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class C3XGate(Gate):
    """C3X 门 - 三控制 X 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('c3x', 4, [], label)
    
    def to_matrix(self) -> np.ndarray:
        mat = np.eye(16, dtype=complex)
        mat[14, 14], mat[14, 15] = 0, 1
        mat[15, 14], mat[15, 15] = 1, 0
        return mat
    
    def inverse(self) -> 'C3XGate':
        return C3XGate(self._label)
    
    def copy(self) -> 'C3XGate':
        g = C3XGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class C4XGate(Gate):
    """C4X 门 - 四控制 X 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('c4x', 5, [], label)
    
    def to_matrix(self) -> np.ndarray:
        mat = np.eye(32, dtype=complex)
        mat[30, 30], mat[30, 31] = 0, 1
        mat[31, 30], mat[31, 31] = 1, 0
        return mat
    
    def inverse(self) -> 'C4XGate':
        return C4XGate(self._label)
    
    def copy(self) -> 'C4XGate':
        g = C4XGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


# ==================== 特殊门 ====================

class XXMinusYYGate(Gate):
    """XX-YY 门"""
    def __init__(self, theta: float, beta: float = 0, label: Optional[str] = None):
        super().__init__('xx_minus_yy', 2, [theta, beta], label)
    
    def to_matrix(self) -> np.ndarray:
        theta, beta = self._params
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, 0, 0, -1j * np.exp(-1j * beta) * s],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-1j * np.exp(1j * beta) * s, 0, 0, c]
        ], dtype=complex)
    
    def inverse(self) -> 'XXMinusYYGate':
        return XXMinusYYGate(-self._params[0], self._params[1], self._label)
    
    def copy(self) -> 'XXMinusYYGate':
        g = XXMinusYYGate(*self._params, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class XXPlusYYGate(Gate):
    """XX+YY 门"""
    def __init__(self, theta: float, beta: float = 0, label: Optional[str] = None):
        super().__init__('xx_plus_yy', 2, [theta, beta], label)
    
    def to_matrix(self) -> np.ndarray:
        theta, beta = self._params
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, c, -1j * np.exp(-1j * beta) * s, 0],
            [0, -1j * np.exp(1j * beta) * s, c, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    def inverse(self) -> 'XXPlusYYGate':
        return XXPlusYYGate(-self._params[0], self._params[1], self._label)
    
    def copy(self) -> 'XXPlusYYGate':
        g = XXPlusYYGate(*self._params, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class GlobalPhaseGate(Gate):
    """全局相位门"""
    def __init__(self, phase: float, label: Optional[str] = None):
        super().__init__('global_phase', 0, [phase], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([[np.exp(1j * self._params[0])]], dtype=complex)
    
    def inverse(self) -> 'GlobalPhaseGate':
        return GlobalPhaseGate(-self._params[0], self._label)
    
    def copy(self) -> 'GlobalPhaseGate':
        g = GlobalPhaseGate(self._params[0], self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


# ==================== 特殊操作 ====================

class Barrier(Gate):
    """Barrier - 用于分隔电路层"""
    def __init__(self, num_qubits: int = 1, label: Optional[str] = None):
        super().__init__('barrier', num_qubits, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.eye(2 ** self._num_qubits, dtype=complex)
    
    def inverse(self) -> 'Barrier':
        return Barrier(self._num_qubits, self._label)
    
    def copy(self) -> 'Barrier':
        g = Barrier(self._num_qubits, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class Measure(Gate):
    """测量操作"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('measure', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        raise NotImplementedError("Measure is not a unitary operation")
    
    def inverse(self) -> 'Measure':
        raise NotImplementedError("Measure cannot be inverted")
    
    def copy(self) -> 'Measure':
        g = Measure(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class Reset(Gate):
    """重置操作 - 将量子比特重置为 |0⟩"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('reset', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        raise NotImplementedError("Reset is not a unitary operation")
    
    def inverse(self) -> 'Reset':
        raise NotImplementedError("Reset cannot be inverted")
    
    def copy(self) -> 'Reset':
        g = Reset(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class Delay(Gate):
    """延迟操作"""
    def __init__(self, duration: float, unit: str = 'dt', label: Optional[str] = None):
        super().__init__('delay', 1, [duration], label)
        self._unit = unit
    
    @property
    def duration(self) -> float:
        return self._params[0]
    
    @property
    def unit(self) -> str:
        return self._unit
    
    def to_matrix(self) -> np.ndarray:
        return np.eye(2, dtype=complex)
    
    def inverse(self) -> 'Delay':
        return Delay(self._params[0], self._unit, self._label)
    
    def copy(self) -> 'Delay':
        g = Delay(self._params[0], self._unit, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


# ==================== 特殊两比特门 ====================

class XXMinusYYGate(Gate):
    """XX-YY 门"""
    def __init__(self, theta: float, beta: float = 0, label: Optional[str] = None):
        super().__init__('xx_minus_yy', 2, [theta, beta], label)
    
    def to_matrix(self) -> np.ndarray:
        theta, beta = self._params
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [c, 0, 0, -1j * np.exp(-1j * beta) * s],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-1j * np.exp(1j * beta) * s, 0, 0, c]
        ], dtype=complex)
    
    def inverse(self) -> 'XXMinusYYGate':
        return XXMinusYYGate(-self._params[0], self._params[1], self._label)
    
    def copy(self) -> 'XXMinusYYGate':
        g = XXMinusYYGate(*self._params, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class XXPlusYYGate(Gate):
    """XX+YY 门"""
    def __init__(self, theta: float, beta: float = 0, label: Optional[str] = None):
        super().__init__('xx_plus_yy', 2, [theta, beta], label)
    
    def to_matrix(self) -> np.ndarray:
        theta, beta = self._params
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, c, -1j * np.exp(-1j * beta) * s, 0],
            [0, -1j * np.exp(1j * beta) * s, c, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    def inverse(self) -> 'XXPlusYYGate':
        return XXPlusYYGate(-self._params[0], self._params[1], self._label)
    
    def copy(self) -> 'XXPlusYYGate':
        g = XXPlusYYGate(*self._params, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class CSwapGate(Gate):
    """Controlled-SWAP (Fredkin) 门"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('cswap', 3, [], label)
    
    def to_matrix(self) -> np.ndarray:
        mat = np.eye(8, dtype=complex)
        mat[5, 5], mat[5, 6] = 0, 1
        mat[6, 5], mat[6, 6] = 1, 0
        return mat
    
    def inverse(self) -> 'CSwapGate':
        return CSwapGate(self._label)
    
    def copy(self) -> 'CSwapGate':
        g = CSwapGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


# ==================== 特殊操作 ====================

class Barrier(Gate):
    """Barrier - 用于分隔电路层"""
    def __init__(self, num_qubits: int = 1, label: Optional[str] = None):
        super().__init__('barrier', num_qubits, [], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.eye(2 ** self._num_qubits, dtype=complex)
    
    def inverse(self) -> 'Barrier':
        return Barrier(self._num_qubits, self._label)
    
    def copy(self) -> 'Barrier':
        g = Barrier(self._num_qubits, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class Measure(Gate):
    """测量操作"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('measure', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        raise NotImplementedError("Measure is not a unitary operation")
    
    def inverse(self) -> 'Measure':
        raise NotImplementedError("Measure cannot be inverted")
    
    def copy(self) -> 'Measure':
        g = Measure(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class Reset(Gate):
    """重置操作 - 将量子比特重置为 |0⟩"""
    def __init__(self, label: Optional[str] = None):
        super().__init__('reset', 1, [], label)
    
    def to_matrix(self) -> np.ndarray:
        raise NotImplementedError("Reset is not a unitary operation")
    
    def inverse(self) -> 'Reset':
        raise NotImplementedError("Reset cannot be inverted")
    
    def copy(self) -> 'Reset':
        g = Reset(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class Delay(Gate):
    """延迟操作"""
    def __init__(self, duration: float, unit: str = 'dt', label: Optional[str] = None):
        super().__init__('delay', 1, [duration], label)
        self._unit = unit
    
    @property
    def duration(self) -> float:
        return self._params[0]
    
    @property
    def unit(self) -> str:
        return self._unit
    
    def to_matrix(self) -> np.ndarray:
        return np.eye(2, dtype=complex)
    
    def inverse(self) -> 'Delay':
        return Delay(self.duration, self._unit, self._label)
    
    def copy(self) -> 'Delay':
        g = Delay(self.duration, self._unit, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


# ==================== 全局相位门 ====================

class GlobalPhaseGate(Gate):
    """全局相位门"""
    def __init__(self, phase: float, label: Optional[str] = None):
        super().__init__('global_phase', 0, [phase], label)
    
    def to_matrix(self) -> np.ndarray:
        return np.array([[np.exp(1j * self._params[0])]], dtype=complex)
    
    def inverse(self) -> 'GlobalPhaseGate':
        return GlobalPhaseGate(-self._params[0], self._label)
    
    def copy(self) -> 'GlobalPhaseGate':
        g = GlobalPhaseGate(self._params[0], self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


# ==================== 多控制门 ====================

class C3SXGate(Gate):
    """C3SX 门 - 三控制 sqrt(X) 门
    
    基于 Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
    """
    def __init__(self, label: Optional[str] = None):
        super().__init__('c3sx', 4, [], label)
    
    def to_matrix(self) -> np.ndarray:
        """返回 C3SX 门的矩阵表示"""
        mat = np.eye(16, dtype=complex)
        # sqrt(X) 矩阵
        sx = np.array([[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]], dtype=complex)
        # 当所有控制比特为 |1⟩ 时，对目标比特应用 sqrt(X)
        mat[14:16, 14:16] = sx
        return mat
    
    def inverse(self) -> 'C3SXGate':
        # C3SX 的逆需要 C3SXdg，这里简化处理
        raise NotImplementedError("C3SX inverse requires C3SXdg")
    
    def copy(self) -> 'C3SXGate':
        g = C3SXGate(self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class MCXGate(Gate):
    """MCX 门 - 多控制 X 门 (通用)
    
    可以处理任意数量的控制比特
    """
    def __init__(self, num_ctrl_qubits: int, label: Optional[str] = None):
        self._num_ctrl_qubits = num_ctrl_qubits
        super().__init__('mcx', num_ctrl_qubits + 1, [], label)
    
    @property
    def num_ctrl_qubits(self) -> int:
        return self._num_ctrl_qubits
    
    def to_matrix(self) -> np.ndarray:
        """返回 MCX 门的矩阵表示"""
        dim = 2 ** self._num_qubits
        mat = np.eye(dim, dtype=complex)
        # 当所有控制比特为 |1⟩ 时，翻转目标比特
        mat[dim-2, dim-2], mat[dim-2, dim-1] = 0, 1
        mat[dim-1, dim-2], mat[dim-1, dim-1] = 1, 0
        return mat
    
    def inverse(self) -> 'MCXGate':
        return MCXGate(self._num_ctrl_qubits, self._label)
    
    def copy(self) -> 'MCXGate':
        g = MCXGate(self._num_ctrl_qubits, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class MCXGrayCode(MCXGate):
    """MCXGrayCode - 使用 Gray code 实现的多控制 X 门
    
    这种实现方式在某些情况下可以减少门的数量
    """
    def __init__(self, num_ctrl_qubits: int, label: Optional[str] = None):
        super().__init__(num_ctrl_qubits, label)
        self._name = 'mcx_gray'
    
    def copy(self) -> 'MCXGrayCode':
        g = MCXGrayCode(self._num_ctrl_qubits, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class MCXRecursive(MCXGate):
    """MCXRecursive - 递归实现的多控制 X 门
    
    使用递归分解来实现多控制 X 门
    """
    def __init__(self, num_ctrl_qubits: int, label: Optional[str] = None):
        super().__init__(num_ctrl_qubits, label)
        self._name = 'mcx_recursive'
    
    def copy(self) -> 'MCXRecursive':
        g = MCXRecursive(self._num_ctrl_qubits, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class MCXVChain(MCXGate):
    """MCXVChain - 使用 V-chain 实现的多控制 X 门
    
    使用辅助比特的 V-chain 分解
    """
    def __init__(self, num_ctrl_qubits: int, dirty_ancillas: bool = False, label: Optional[str] = None):
        super().__init__(num_ctrl_qubits, label)
        self._name = 'mcx_vchain'
        self._dirty_ancillas = dirty_ancillas
    
    @property
    def dirty_ancillas(self) -> bool:
        return self._dirty_ancillas
    
    def copy(self) -> 'MCXVChain':
        g = MCXVChain(self._num_ctrl_qubits, self._dirty_ancillas, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class MCPhaseGate(Gate):
    """MCPhase 门 - 多控制 Phase 门
    
    对目标比特应用相位旋转，取决于所有控制比特的状态
    """
    def __init__(self, lam: float, num_ctrl_qubits: int, label: Optional[str] = None):
        self._num_ctrl_qubits = num_ctrl_qubits
        super().__init__('mcphase', num_ctrl_qubits + 1, [lam], label)
    
    @property
    def num_ctrl_qubits(self) -> int:
        return self._num_ctrl_qubits
    
    @property
    def lam(self) -> float:
        return self._params[0]
    
    def to_matrix(self) -> np.ndarray:
        """返回 MCPhase 门的矩阵表示"""
        dim = 2 ** self._num_qubits
        mat = np.eye(dim, dtype=complex)
        # 当所有控制比特为 |1⟩ 时，对目标比特应用相位
        mat[dim-1, dim-1] = np.exp(1j * self._params[0])
        return mat
    
    def inverse(self) -> 'MCPhaseGate':
        return MCPhaseGate(-self._params[0], self._num_ctrl_qubits, self._label)
    
    def copy(self) -> 'MCPhaseGate':
        g = MCPhaseGate(self._params[0], self._num_ctrl_qubits, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class MCU1Gate(Gate):
    """MCU1 门 - 多控制 U1 门
    
    等价于 MCPhaseGate，保留用于兼容性
    """
    def __init__(self, lam: float, num_ctrl_qubits: int, label: Optional[str] = None):
        self._num_ctrl_qubits = num_ctrl_qubits
        super().__init__('mcu1', num_ctrl_qubits + 1, [lam], label)
    
    @property
    def num_ctrl_qubits(self) -> int:
        return self._num_ctrl_qubits
    
    @property
    def lam(self) -> float:
        return self._params[0]
    
    def to_matrix(self) -> np.ndarray:
        """返回 MCU1 门的矩阵表示"""
        dim = 2 ** self._num_qubits
        mat = np.eye(dim, dtype=complex)
        # 当所有控制比特为 |1⟩ 时，对目标比特应用相位
        mat[dim-1, dim-1] = np.exp(1j * self._params[0])
        return mat
    
    def inverse(self) -> 'MCU1Gate':
        return MCU1Gate(-self._params[0], self._num_ctrl_qubits, self._label)
    
    def copy(self) -> 'MCU1Gate':
        g = MCU1Gate(self._params[0], self._num_ctrl_qubits, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


# ==================== 多控制旋转门 ====================

class MCRXGate(Gate):
    """MCRX 门 - 多控制 RX 门
    
    当所有控制比特为 |1⟩ 时，对目标比特应用 RX(θ) 旋转
    """
    def __init__(self, theta: float, num_ctrl_qubits: int, label: Optional[str] = None):
        self._num_ctrl_qubits = num_ctrl_qubits
        super().__init__('mcrx', num_ctrl_qubits + 1, [theta], label)
    
    @property
    def num_ctrl_qubits(self) -> int:
        return self._num_ctrl_qubits
    
    @property
    def theta(self) -> float:
        return self._params[0]
    
    def to_matrix(self) -> np.ndarray:
        """返回 MCRX 门的矩阵表示"""
        dim = 2 ** self._num_qubits
        mat = np.eye(dim, dtype=complex)
        theta = self._params[0]
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        # RX 矩阵
        rx = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        # 当所有控制比特为 |1⟩ 时应用 RX
        mat[dim-2:dim, dim-2:dim] = rx
        return mat
    
    def inverse(self) -> 'MCRXGate':
        return MCRXGate(-self._params[0], self._num_ctrl_qubits, self._label)
    
    def copy(self) -> 'MCRXGate':
        g = MCRXGate(self._params[0], self._num_ctrl_qubits, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class MCRYGate(Gate):
    """MCRY 门 - 多控制 RY 门
    
    当所有控制比特为 |1⟩ 时，对目标比特应用 RY(θ) 旋转
    """
    def __init__(self, theta: float, num_ctrl_qubits: int, label: Optional[str] = None):
        self._num_ctrl_qubits = num_ctrl_qubits
        super().__init__('mcry', num_ctrl_qubits + 1, [theta], label)
    
    @property
    def num_ctrl_qubits(self) -> int:
        return self._num_ctrl_qubits
    
    @property
    def theta(self) -> float:
        return self._params[0]
    
    def to_matrix(self) -> np.ndarray:
        """返回 MCRY 门的矩阵表示"""
        dim = 2 ** self._num_qubits
        mat = np.eye(dim, dtype=complex)
        theta = self._params[0]
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        # RY 矩阵
        ry = np.array([[c, -s], [s, c]], dtype=complex)
        # 当所有控制比特为 |1⟩ 时应用 RY
        mat[dim-2:dim, dim-2:dim] = ry
        return mat
    
    def inverse(self) -> 'MCRYGate':
        return MCRYGate(-self._params[0], self._num_ctrl_qubits, self._label)
    
    def copy(self) -> 'MCRYGate':
        g = MCRYGate(self._params[0], self._num_ctrl_qubits, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g


class MCRZGate(Gate):
    """MCRZ 门 - 多控制 RZ 门
    
    当所有控制比特为 |1⟩ 时，对目标比特应用 RZ(θ) 旋转
    """
    def __init__(self, theta: float, num_ctrl_qubits: int, label: Optional[str] = None):
        self._num_ctrl_qubits = num_ctrl_qubits
        super().__init__('mcrz', num_ctrl_qubits + 1, [theta], label)
    
    @property
    def num_ctrl_qubits(self) -> int:
        return self._num_ctrl_qubits
    
    @property
    def theta(self) -> float:
        return self._params[0]
    
    def to_matrix(self) -> np.ndarray:
        """返回 MCRZ 门的矩阵表示"""
        dim = 2 ** self._num_qubits
        mat = np.eye(dim, dtype=complex)
        theta = self._params[0]
        # RZ 矩阵
        rz = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)
        # 当所有控制比特为 |1⟩ 时应用 RZ
        mat[dim-2:dim, dim-2:dim] = rz
        return mat
    
    def inverse(self) -> 'MCRZGate':
        return MCRZGate(-self._params[0], self._num_ctrl_qubits, self._label)
    
    def copy(self) -> 'MCRZGate':
        g = MCRZGate(self._params[0], self._num_ctrl_qubits, self._label)
        g._qubits = self._qubits.copy()
        g._params = self._params.copy()
        return g
