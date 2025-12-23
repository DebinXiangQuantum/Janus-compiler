"""
Janus 密度矩阵类

表示混合量子态，支持噪声模拟
"""
from __future__ import annotations
from typing import List, Optional, Union, Dict, Tuple
import numpy as np

from .statevector import Statevector
from .result import Counts
from .exceptions import SimulatorError, InvalidStateError


class DensityMatrix:
    """
    密度矩阵类
    
    表示混合量子态 ρ，支持：
    - 噪声信道演化
    - 部分迹
    - 保真度计算
    
    Example:
        # 从纯态创建
        dm = DensityMatrix.from_statevector(sv)
        
        # 应用去极化噪声
        dm = dm.apply_channel(depolarizing_channel(0.01), [0])
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, 'DensityMatrix'],
        num_qubits: Optional[int] = None
    ):
        """
        初始化密度矩阵
        
        Args:
            data: 密度矩阵数据
            num_qubits: 量子比特数
        """
        if isinstance(data, DensityMatrix):
            self._data = data._data.copy()
            self._num_qubits = data._num_qubits
        else:
            self._data = np.asarray(data, dtype=complex)
            
            if self._data.ndim != 2:
                raise InvalidStateError("Density matrix must be 2-dimensional")
            if self._data.shape[0] != self._data.shape[1]:
                raise InvalidStateError("Density matrix must be square")
            
            dim = self._data.shape[0]
            if dim == 0 or (dim & (dim - 1)) != 0:
                raise InvalidStateError(f"Dimension {dim} is not a power of 2")
            
            self._num_qubits = int(np.log2(dim))
        
        self._rng = np.random.default_rng()
    
    @classmethod
    def from_statevector(cls, sv: Union[Statevector, np.ndarray]) -> 'DensityMatrix':
        """从纯态创建密度矩阵 ρ = |ψ⟩⟨ψ|"""
        if isinstance(sv, Statevector):
            data = np.outer(sv.data, np.conj(sv.data))
            return cls(data, sv.num_qubits)
        else:
            sv_arr = np.asarray(sv, dtype=complex).flatten()
            data = np.outer(sv_arr, np.conj(sv_arr))
            return cls(data)
    
    @classmethod
    def from_label(cls, label: str) -> 'DensityMatrix':
        """从标签创建密度矩阵"""
        sv = Statevector.from_label(label)
        return cls.from_statevector(sv)
    
    @classmethod
    def maximally_mixed(cls, num_qubits: int) -> 'DensityMatrix':
        """创建最大混合态 ρ = I/d"""
        dim = 2 ** num_qubits
        data = np.eye(dim, dtype=complex) / dim
        return cls(data, num_qubits)
    
    @property
    def data(self) -> np.ndarray:
        """获取密度矩阵数据"""
        return self._data
    
    @property
    def num_qubits(self) -> int:
        """获取量子比特数"""
        return self._num_qubits
    
    @property
    def dim(self) -> int:
        """获取维度"""
        return self._data.shape[0]
    
    def seed(self, value: Optional[Union[int, np.random.Generator]] = None):
        """设置随机数种子"""
        if value is None:
            self._rng = np.random.default_rng()
        elif isinstance(value, np.random.Generator):
            self._rng = value
        else:
            self._rng = np.random.default_rng(value)
    
    def copy(self) -> 'DensityMatrix':
        """创建副本"""
        dm = DensityMatrix(self._data.copy(), self._num_qubits)
        dm._rng = self._rng
        return dm
    
    def is_valid(self, atol: float = 1e-10) -> bool:
        """检查是否为有效密度矩阵"""
        # 检查厄米性
        if not np.allclose(self._data, self._data.conj().T, atol=atol):
            return False
        
        # 检查迹为 1
        if not np.isclose(np.trace(self._data), 1.0, atol=atol):
            return False
        
        # 检查半正定性
        eigenvalues = np.linalg.eigvalsh(self._data)
        if np.any(eigenvalues < -atol):
            return False
        
        return True
    
    def trace(self) -> complex:
        """计算迹"""
        return np.trace(self._data)
    
    def purity(self) -> float:
        """计算纯度 Tr(ρ²)"""
        return np.real(np.trace(self._data @ self._data))
    
    def is_pure(self, atol: float = 1e-10) -> bool:
        """检查是否为纯态"""
        return np.isclose(self.purity(), 1.0, atol=atol)
    
    # ==================== 演化方法 ====================
    
    def evolve(
        self,
        operator: np.ndarray,
        qargs: Optional[List[int]] = None
    ) -> 'DensityMatrix':
        """
        酉演化 ρ → U ρ U†
        
        Args:
            operator: 酉矩阵
            qargs: 作用的量子比特
        
        Returns:
            DensityMatrix: 演化后的密度矩阵
        """
        if qargs is None:
            self._data = operator @ self._data @ operator.conj().T
        else:
            self._data = self._apply_unitary(operator, qargs)
        return self
    
    def _apply_unitary(self, U: np.ndarray, qargs: List[int]) -> np.ndarray:
        """将酉算符应用到指定量子比特"""
        n = self._num_qubits
        num_qargs = len(qargs)
        
        # 将密度矩阵重塑为张量
        tensor_shape = [2] * (2 * n)
        rho_tensor = self._data.reshape(tensor_shape)
        
        # 计算轴
        row_axes = [n - 1 - q for q in qargs]
        col_axes = [n + n - 1 - q for q in qargs]
        
        # 应用 U 到行索引
        rho_tensor = self._contract_tensor(rho_tensor, U, row_axes, n)
        
        # 应用 U† 到列索引
        rho_tensor = self._contract_tensor(rho_tensor, U.conj().T, col_axes, n)
        
        return rho_tensor.reshape(self.dim, self.dim)
    
    def _contract_tensor(
        self, 
        tensor: np.ndarray, 
        matrix: np.ndarray, 
        axes: List[int],
        n: int
    ) -> np.ndarray:
        """张量收缩辅助函数"""
        num_axes = len(axes)
        total_axes = 2 * n
        
        # 将目标轴移到前面
        other_axes = [i for i in range(total_axes) if i not in axes]
        perm = axes + other_axes
        tensor = np.transpose(tensor, perm)
        
        # 重塑并乘法
        op_dim = 2 ** num_axes
        other_dim = tensor.size // op_dim
        tensor = tensor.reshape(op_dim, other_dim)
        tensor = matrix @ tensor
        
        # 恢复形状
        tensor = tensor.reshape([2] * num_axes + [2] * (total_axes - num_axes))
        
        # 逆转置
        inv_perm = np.argsort(perm)
        tensor = np.transpose(tensor, inv_perm)
        
        return tensor
    
    def apply_channel(
        self,
        kraus_ops: List[np.ndarray],
        qargs: Optional[List[int]] = None
    ) -> 'DensityMatrix':
        """
        应用量子信道（Kraus 表示）
        
        ρ → Σ_k K_k ρ K_k†
        
        Args:
            kraus_ops: Kraus 算符列表
            qargs: 作用的量子比特
        
        Returns:
            DensityMatrix: 演化后的密度矩阵
        """
        if qargs is None:
            new_data = sum(K @ self._data @ K.conj().T for K in kraus_ops)
        else:
            new_data = np.zeros_like(self._data)
            for K in kraus_ops:
                # 对每个 Kraus 算符，应用 K ρ K†
                temp_data = self._apply_kraus_single(K, qargs)
                new_data += temp_data
        
        self._data = new_data
        return self
    
    def _apply_kraus_single(self, K: np.ndarray, qargs: List[int]) -> np.ndarray:
        """应用单个 Kraus 算符 K ρ K†"""
        n = self._num_qubits
        num_qargs = len(qargs)
        
        # 将密度矩阵重塑为张量 [2]*n (行) + [2]*n (列)
        tensor_shape = [2] * (2 * n)
        rho_tensor = self._data.reshape(tensor_shape)
        
        # 计算行和列的轴索引
        # 对于 n 量子比特，行轴是 0 到 n-1，列轴是 n 到 2n-1
        # qargs 中的量子比特 q 对应行轴 n-1-q 和列轴 2n-1-q
        row_axes = [n - 1 - q for q in qargs]
        col_axes = [2*n - 1 - q for q in qargs]
        
        # 应用 K 到行索引（左乘）
        # 将目标行轴移到前面
        other_row_axes = [i for i in range(n) if i not in row_axes]
        other_col_axes = [i for i in range(n, 2*n) if i not in col_axes]
        
        perm = row_axes + other_row_axes + col_axes + other_col_axes
        rho_tensor = np.transpose(rho_tensor, perm)
        
        # 重塑为矩阵形式
        k_dim = 2 ** num_qargs
        other_dim = 2 ** (n - num_qargs)
        
        # 形状: [k_dim, other_dim, k_dim, other_dim]
        rho_tensor = rho_tensor.reshape(k_dim, other_dim, k_dim, other_dim)
        
        # 应用 K ρ K†
        # K @ rho @ K† 在目标子空间
        result = np.zeros_like(rho_tensor)
        for i in range(other_dim):
            for j in range(other_dim):
                sub_rho = rho_tensor[:, i, :, j]  # k_dim x k_dim
                result[:, i, :, j] = K @ sub_rho @ K.conj().T
        
        # 恢复形状
        result = result.reshape([2] * num_qargs + [2] * (n - num_qargs) + 
                                [2] * num_qargs + [2] * (n - num_qargs))
        
        # 逆转置
        inv_perm = np.argsort(perm)
        result = np.transpose(result, inv_perm)
        
        return result.reshape(self.dim, self.dim)
    
    def evolve_circuit(self, circuit) -> 'DensityMatrix':
        """通过电路演化密度矩阵"""
        from ..circuit.parameter import is_parameterized
        
        for inst in circuit.instructions:
            if is_parameterized(inst.operation):
                raise SimulatorError(
                    f"Circuit contains unbound parameter in gate {inst.name}"
                )
            
            matrix = inst.operation.to_matrix()
            self.evolve(matrix, inst.qubits)
        
        return self
    
    # ==================== 测量方法 ====================
    
    def probabilities(self, qargs: Optional[List[int]] = None) -> np.ndarray:
        """计算测量概率分布"""
        probs = np.real(np.diag(self._data))
        
        if qargs is None:
            return probs
        
        return self._marginalize_probabilities(probs, qargs)
    
    def _marginalize_probabilities(
        self, 
        probs: np.ndarray, 
        qargs: List[int]
    ) -> np.ndarray:
        """边缘化概率分布"""
        n = self._num_qubits
        probs_tensor = probs.reshape([2] * n)
        
        keep_axes = [n - 1 - q for q in qargs]
        sum_axes = tuple(i for i in range(n) if i not in keep_axes)
        
        if sum_axes:
            probs_tensor = np.sum(probs_tensor, axis=sum_axes)
        
        return probs_tensor.flatten()
    
    def sample_counts(
        self, 
        shots: int, 
        qargs: Optional[List[int]] = None
    ) -> Counts:
        """采样测量结果"""
        probs = self.probabilities(qargs)
        num_bits = len(qargs) if qargs else self._num_qubits
        
        indices = self._rng.choice(len(probs), size=shots, p=probs)
        samples = [format(i, f'0{num_bits}b') for i in indices]
        
        unique, counts = np.unique(samples, return_counts=True)
        return Counts(dict(zip(unique, counts.astype(int))))
    
    # ==================== 信息度量 ====================
    
    def expectation_value(
        self, 
        operator: np.ndarray,
        qargs: Optional[List[int]] = None
    ) -> complex:
        """计算期望值 Tr(ρO)"""
        if qargs is None:
            return np.trace(self._data @ operator)
        else:
            # 扩展算符到全系统
            full_op = self._expand_operator(operator, qargs)
            return np.trace(self._data @ full_op)
    
    def _expand_operator(self, op: np.ndarray, qargs: List[int]) -> np.ndarray:
        """将子系统算符扩展到全系统"""
        n = self._num_qubits
        num_qargs = len(qargs)
        
        # 构建全系统算符
        full_dim = 2 ** n
        full_op = np.zeros((full_dim, full_dim), dtype=complex)
        
        op_dim = 2 ** num_qargs
        other_dim = full_dim // op_dim
        
        for i in range(other_dim):
            for j in range(other_dim):
                for a in range(op_dim):
                    for b in range(op_dim):
                        # 计算全系统索引
                        row = self._compute_index(a, i, qargs, n)
                        col = self._compute_index(b, j, qargs, n)
                        if i == j:
                            full_op[row, col] = op[a, b]
        
        return full_op
    
    def _compute_index(
        self, 
        qarg_bits: int, 
        other_bits: int, 
        qargs: List[int], 
        n: int
    ) -> int:
        """计算全系统索引"""
        result = 0
        other_idx = 0
        qarg_idx = 0
        
        for q in range(n):
            if q in qargs:
                bit = (qarg_bits >> qargs.index(q)) & 1
            else:
                bit = (other_bits >> other_idx) & 1
                other_idx += 1
            result |= (bit << q)
        
        return result
    
    def partial_trace(self, qargs: List[int]) -> 'DensityMatrix':
        """
        计算部分迹，保留指定量子比特
        
        Args:
            qargs: 要保留的量子比特索引
        
        Returns:
            DensityMatrix: 约化密度矩阵
        """
        n = self._num_qubits
        keep_qubits = sorted(qargs)
        trace_qubits = sorted([q for q in range(n) if q not in keep_qubits])
        
        if not trace_qubits:
            return self.copy()
        
        if not keep_qubits:
            # 迹掉所有量子比特，返回标量
            return DensityMatrix(np.array([[self.trace()]]), 0)
        
        # 使用更直接的方法计算部分迹
        # 将密度矩阵重塑为张量，然后重新排列轴
        
        # 重塑为张量: [2]*n (行索引) + [2]*n (列索引)
        # 轴顺序: 行轴 0..n-1 对应量子比特 n-1..0
        #         列轴 n..2n-1 对应量子比特 n-1..0
        tensor_shape = [2] * (2 * n)
        rho_tensor = self._data.reshape(tensor_shape)
        
        # 重新排列轴，使得要迹掉的量子比特的行列轴相邻
        # 新顺序: [keep_row_axes] + [trace_row_axes] + [keep_col_axes] + [trace_col_axes]
        keep_row_axes = [n - 1 - q for q in keep_qubits]
        trace_row_axes = [n - 1 - q for q in trace_qubits]
        keep_col_axes = [2*n - 1 - q for q in keep_qubits]
        trace_col_axes = [2*n - 1 - q for q in trace_qubits]
        
        # 重新排列，使得可以方便地求迹
        # 目标: [keep_row] + [keep_col] + [trace_row] + [trace_col]
        # 然后对 trace_row 和 trace_col 求迹
        
        n_keep = len(keep_qubits)
        n_trace = len(trace_qubits)
        
        # 转置到新顺序
        new_order = keep_row_axes + keep_col_axes + trace_row_axes + trace_col_axes
        rho_tensor = np.transpose(rho_tensor, new_order)
        
        # 重塑为 [keep_dim, keep_dim, trace_dim, trace_dim]
        keep_dim = 2 ** n_keep
        trace_dim = 2 ** n_trace
        rho_tensor = rho_tensor.reshape(keep_dim, keep_dim, trace_dim, trace_dim)
        
        # 对 trace 维度求迹: sum over diagonal elements
        # 即 result[i,j] = sum_k rho_tensor[i,j,k,k]
        result = np.trace(rho_tensor, axis1=2, axis2=3)
        
        return DensityMatrix(result, n_keep)
    
    def fidelity(self, other: Union['DensityMatrix', Statevector]) -> float:
        """
        计算保真度
        
        F(ρ, σ) = (Tr√(√ρ σ √ρ))²
        F(ρ, |ψ⟩) = ⟨ψ|ρ|ψ⟩
        """
        if isinstance(other, Statevector):
            return np.real(np.vdot(other.data, self._data @ other.data))
        else:
            # 一般情况
            sqrt_rho = self._matrix_sqrt(self._data)
            product = sqrt_rho @ other._data @ sqrt_rho
            sqrt_product = self._matrix_sqrt(product)
            return np.real(np.trace(sqrt_product)) ** 2
    
    def _matrix_sqrt(self, A: np.ndarray) -> np.ndarray:
        """计算矩阵平方根"""
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        eigenvalues = np.maximum(eigenvalues, 0)  # 数值稳定性
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T
    
    def von_neumann_entropy(self) -> float:
        """计算冯诺依曼熵 S(ρ) = -Tr(ρ log ρ)"""
        eigenvalues = np.linalg.eigvalsh(self._data)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    # ==================== 特殊方法 ====================
    
    def __repr__(self) -> str:
        return f"DensityMatrix(shape={self._data.shape}, num_qubits={self._num_qubits})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, DensityMatrix):
            return False
        return np.allclose(self._data, other._data)
