"""
Janus 噪声模型

提供常见的量子噪声信道
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


class NoiseChannel:
    """
    噪声信道基类
    
    使用 Kraus 表示：ρ → Σ_k K_k ρ K_k†
    """
    
    def __init__(self, kraus_ops: List[np.ndarray], num_qubits: int = 1):
        """
        初始化噪声信道
        
        Args:
            kraus_ops: Kraus 算符列表
            num_qubits: 作用的量子比特数
        """
        self._kraus_ops = [np.asarray(K, dtype=complex) for K in kraus_ops]
        self._num_qubits = num_qubits
        
        # 验证完备性：Σ_k K_k† K_k = I
        identity = np.eye(2 ** num_qubits, dtype=complex)
        completeness = sum(K.conj().T @ K for K in self._kraus_ops)
        if not np.allclose(completeness, identity, atol=1e-10):
            raise ValueError("Kraus operators do not satisfy completeness relation")
    
    @property
    def kraus_ops(self) -> List[np.ndarray]:
        """获取 Kraus 算符"""
        return self._kraus_ops
    
    @property
    def num_qubits(self) -> int:
        """获取量子比特数"""
        return self._num_qubits
    
    def compose(self, other: 'NoiseChannel') -> 'NoiseChannel':
        """组合两个噪声信道"""
        if self._num_qubits != other._num_qubits:
            raise ValueError("Cannot compose channels with different qubit counts")
        
        new_kraus = []
        for K1 in self._kraus_ops:
            for K2 in other._kraus_ops:
                new_kraus.append(K1 @ K2)
        
        return NoiseChannel(new_kraus, self._num_qubits)
    
    def tensor(self, other: 'NoiseChannel') -> 'NoiseChannel':
        """张量积两个噪声信道"""
        new_kraus = []
        for K1 in self._kraus_ops:
            for K2 in other._kraus_ops:
                new_kraus.append(np.kron(K1, K2))
        
        return NoiseChannel(new_kraus, self._num_qubits + other._num_qubits)


# ==================== 常见噪声信道 ====================

def depolarizing_channel(p: float, num_qubits: int = 1) -> NoiseChannel:
    """
    去极化信道
    
    ρ → (1-p)ρ + p/3 (XρX + YρY + ZρZ)  (单比特)
    ρ → (1-p)ρ + p/(4^n-1) Σ_P PρP      (多比特)
    
    Args:
        p: 去极化概率
        num_qubits: 量子比特数
    
    Returns:
        NoiseChannel: 去极化信道
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Probability p={p} must be in [0, 1]")
    
    dim = 2 ** num_qubits
    
    if num_qubits == 1:
        # 单比特去极化
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        kraus_ops = [
            np.sqrt(1 - p) * I,
            np.sqrt(p / 3) * X,
            np.sqrt(p / 3) * Y,
            np.sqrt(p / 3) * Z
        ]
    else:
        # 多比特去极化
        num_paulis = dim ** 2 - 1
        kraus_ops = [np.sqrt(1 - p) * np.eye(dim, dtype=complex)]
        
        # 生成所有非恒等 Pauli 算符
        paulis = _generate_paulis(num_qubits)
        for P in paulis[1:]:  # 跳过恒等
            kraus_ops.append(np.sqrt(p / num_paulis) * P)
    
    return NoiseChannel(kraus_ops, num_qubits)


def amplitude_damping_channel(gamma: float) -> NoiseChannel:
    """
    振幅阻尼信道（能量弛豫）
    
    模拟 |1⟩ → |0⟩ 的自发衰减
    
    Args:
        gamma: 衰减概率
    
    Returns:
        NoiseChannel: 振幅阻尼信道
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"Gamma={gamma} must be in [0, 1]")
    
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    
    return NoiseChannel([K0, K1], 1)


def phase_damping_channel(gamma: float) -> NoiseChannel:
    """
    相位阻尼信道（纯退相干）
    
    Args:
        gamma: 退相干概率
    
    Returns:
        NoiseChannel: 相位阻尼信道
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"Gamma={gamma} must be in [0, 1]")
    
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=complex)
    
    return NoiseChannel([K0, K1], 1)


def bit_flip_channel(p: float) -> NoiseChannel:
    """
    比特翻转信道
    
    ρ → (1-p)ρ + p XρX
    
    Args:
        p: 翻转概率
    
    Returns:
        NoiseChannel: 比特翻转信道
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Probability p={p} must be in [0, 1]")
    
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    
    kraus_ops = [np.sqrt(1 - p) * I, np.sqrt(p) * X]
    return NoiseChannel(kraus_ops, 1)


def phase_flip_channel(p: float) -> NoiseChannel:
    """
    相位翻转信道
    
    ρ → (1-p)ρ + p ZρZ
    
    Args:
        p: 翻转概率
    
    Returns:
        NoiseChannel: 相位翻转信道
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Probability p={p} must be in [0, 1]")
    
    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    kraus_ops = [np.sqrt(1 - p) * I, np.sqrt(p) * Z]
    return NoiseChannel(kraus_ops, 1)


def bit_phase_flip_channel(p: float) -> NoiseChannel:
    """
    比特-相位翻转信道
    
    ρ → (1-p)ρ + p YρY
    
    Args:
        p: 翻转概率
    
    Returns:
        NoiseChannel: 比特-相位翻转信道
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Probability p={p} must be in [0, 1]")
    
    I = np.eye(2, dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    kraus_ops = [np.sqrt(1 - p) * I, np.sqrt(p) * Y]
    return NoiseChannel(kraus_ops, 1)


def thermal_relaxation_channel(
    t1: float, 
    t2: float, 
    time: float,
    excited_state_population: float = 0.0
) -> NoiseChannel:
    """
    热弛豫信道
    
    结合 T1（能量弛豫）和 T2（退相干）效应
    
    Args:
        t1: T1 时间
        t2: T2 时间
        time: 门操作时间
        excited_state_population: 热平衡时激发态布居
    
    Returns:
        NoiseChannel: 热弛豫信道
    """
    if t2 > 2 * t1:
        raise ValueError(f"T2={t2} cannot be greater than 2*T1={2*t1}")
    
    # 计算概率
    p_reset = 1 - np.exp(-time / t1)
    
    if t1 == t2:
        # 纯 T1 弛豫
        gamma = p_reset
        return amplitude_damping_channel(gamma)
    
    # T2 退相干
    exp_t2 = np.exp(-time / t2)
    exp_t1 = np.exp(-time / t1)
    
    # 组合 T1 和 T2 效应
    p0 = 1 - excited_state_population
    p1 = excited_state_population
    
    # Kraus 算符
    K0 = np.array([
        [np.sqrt(p0) * np.sqrt(1 - p_reset), 0],
        [0, np.sqrt(p1) * exp_t2 / np.sqrt(exp_t1)]
    ], dtype=complex)
    
    K1 = np.array([
        [np.sqrt(p0) * np.sqrt(p_reset), 0],
        [0, 0]
    ], dtype=complex)
    
    K2 = np.array([
        [0, np.sqrt(p1) * np.sqrt(p_reset)],
        [0, 0]
    ], dtype=complex)
    
    K3 = np.array([
        [0, 0],
        [np.sqrt(p0) * np.sqrt(1 - exp_t2**2 / exp_t1), 0]
    ], dtype=complex)
    
    # 过滤零算符
    kraus_ops = [K for K in [K0, K1, K2, K3] if np.linalg.norm(K) > 1e-10]
    
    return NoiseChannel(kraus_ops, 1)


def readout_error_channel(p0_given_1: float, p1_given_0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    读出错误（经典噪声）
    
    返回混淆矩阵，用于后处理测量结果
    
    Args:
        p0_given_1: 测量 |1⟩ 得到 0 的概率
        p1_given_0: 测量 |0⟩ 得到 1 的概率
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (混淆矩阵, 逆混淆矩阵)
    """
    # 混淆矩阵 M[i,j] = P(测量结果=i | 真实状态=j)
    confusion_matrix = np.array([
        [1 - p1_given_0, p0_given_1],
        [p1_given_0, 1 - p0_given_1]
    ])
    
    # 逆矩阵用于误差缓解
    try:
        inverse_matrix = np.linalg.inv(confusion_matrix)
    except np.linalg.LinAlgError:
        inverse_matrix = None
    
    return confusion_matrix, inverse_matrix


# ==================== 辅助函数 ====================

def _generate_paulis(num_qubits: int) -> List[np.ndarray]:
    """生成所有 n-qubit Pauli 算符"""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    single_paulis = [I, X, Y, Z]
    
    if num_qubits == 1:
        return single_paulis
    
    paulis = [np.array([[1]], dtype=complex)]
    for _ in range(num_qubits):
        new_paulis = []
        for P in paulis:
            for sigma in single_paulis:
                new_paulis.append(np.kron(P, sigma))
        paulis = new_paulis
    
    return paulis


class NoiseModel:
    """
    噪声模型
    
    定义电路中各种门的噪声
    
    Example:
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            depolarizing_channel(0.01), ['cx']
        )
        noise_model.add_readout_error(0.02, 0.03)
    """
    
    def __init__(self):
        self._gate_errors = {}  # gate_name -> NoiseChannel
        self._qubit_gate_errors = {}  # (gate_name, qubit) -> NoiseChannel
        self._readout_error = None
        self._readout_error_qubits = {}  # qubit -> (p0_given_1, p1_given_0)
    
    def add_quantum_error(
        self,
        error: NoiseChannel,
        gate_names: List[str],
        qubits: Optional[List[int]] = None
    ):
        """
        添加量子错误
        
        Args:
            error: 噪声信道
            gate_names: 门名称列表
            qubits: 特定量子比特（None 表示所有）
        """
        for gate in gate_names:
            if qubits is None:
                self._gate_errors[gate] = error
            else:
                for q in qubits:
                    self._qubit_gate_errors[(gate, q)] = error
    
    def add_all_qubit_quantum_error(
        self,
        error: NoiseChannel,
        gate_names: List[str]
    ):
        """为所有量子比特添加量子错误"""
        self.add_quantum_error(error, gate_names, None)
    
    def add_readout_error(
        self,
        p0_given_1: float,
        p1_given_0: float,
        qubits: Optional[List[int]] = None
    ):
        """
        添加读出错误
        
        Args:
            p0_given_1: 测量 |1⟩ 得到 0 的概率
            p1_given_0: 测量 |0⟩ 得到 1 的概率
            qubits: 特定量子比特（None 表示所有）
        """
        if qubits is None:
            self._readout_error = (p0_given_1, p1_given_0)
        else:
            for q in qubits:
                self._readout_error_qubits[q] = (p0_given_1, p1_given_0)
    
    def get_gate_error(
        self, 
        gate_name: str, 
        qubits: List[int]
    ) -> Optional[NoiseChannel]:
        """获取门的噪声信道"""
        # 先检查特定量子比特的错误
        for q in qubits:
            if (gate_name, q) in self._qubit_gate_errors:
                return self._qubit_gate_errors[(gate_name, q)]
        
        # 再检查全局门错误
        return self._gate_errors.get(gate_name)
    
    def get_readout_error(self, qubit: int) -> Optional[Tuple[float, float]]:
        """获取读出错误"""
        if qubit in self._readout_error_qubits:
            return self._readout_error_qubits[qubit]
        return self._readout_error
