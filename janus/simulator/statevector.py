"""
Janus 状态向量类

表示量子态的状态向量，支持演化、测量、期望值计算等操作
"""
from __future__ import annotations
import copy
import math
from typing import List, Optional, Union, Dict, Tuple
import numpy as np

from .exceptions import SimulatorError, InvalidStateError
from .result import Counts


class Statevector:
    """
    状态向量类
    
    表示纯量子态 |ψ⟩，支持：
    - 从电路演化
    - 测量采样
    - 期望值计算
    - 子系统操作
    
    Example:
        # 创建 |00⟩ 态
        sv = Statevector.from_label('00')
        
        # 从电路创建
        sv = Statevector.from_circuit(circuit)
        
        # 测量
        counts = sv.sample_counts(shots=1000)
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, List, 'Statevector'],
        num_qubits: Optional[int] = None
    ):
        """
        初始化状态向量
        
        Args:
            data: 状态向量数据，可以是数组、列表或另一个 Statevector
            num_qubits: 量子比特数（可选，自动推断）
        """
        if isinstance(data, Statevector):
            self._data = data._data.copy()
            self._num_qubits = data._num_qubits
        elif isinstance(data, (list, np.ndarray)):
            self._data = np.asarray(data, dtype=complex).flatten()
        else:
            raise InvalidStateError(f"Invalid data type: {type(data)}")
        
        # 推断量子比特数
        dim = len(self._data)
        if dim == 0 or (dim & (dim - 1)) != 0:
            raise InvalidStateError(f"Statevector dimension {dim} is not a power of 2")
        
        inferred_qubits = int(np.log2(dim))
        if num_qubits is not None and num_qubits != inferred_qubits:
            raise InvalidStateError(
                f"num_qubits={num_qubits} does not match data dimension {dim}"
            )
        self._num_qubits = inferred_qubits
        
        # 随机数生成器
        self._rng = np.random.default_rng()
    
    @classmethod
    def from_int(cls, i: int, num_qubits: int) -> 'Statevector':
        """
        从计算基态创建状态向量
        
        Args:
            i: 基态索引
            num_qubits: 量子比特数
        
        Returns:
            Statevector: |i⟩ 态
        
        Example:
            sv = Statevector.from_int(0, 2)  # |00⟩
            sv = Statevector.from_int(3, 2)  # |11⟩
        """
        dim = 2 ** num_qubits
        if i < 0 or i >= dim:
            raise InvalidStateError(f"Index {i} out of range for {num_qubits} qubits")
        
        data = np.zeros(dim, dtype=complex)
        data[i] = 1.0
        return cls(data, num_qubits)
    
    @classmethod
    def from_label(cls, label: str) -> 'Statevector':
        """
        从标签创建状态向量
        
        支持的标签：
        - '0', '1': 计算基态
        - '+', '-': X 基态
        - 'r', 'l': Y 基态
        
        Args:
            label: 状态标签字符串
        
        Returns:
            Statevector: 对应的状态向量
        
        Example:
            sv = Statevector.from_label('00')   # |00⟩
            sv = Statevector.from_label('+0')   # |+0⟩ = (|00⟩ + |10⟩)/√2
        """
        # 验证标签
        valid_chars = set('01+-rl')
        if not all(c in valid_chars for c in label):
            raise InvalidStateError(f"Invalid label characters in '{label}'")
        
        num_qubits = len(label)
        
        # 先创建 Z 基态
        z_label = label.replace('+', '0').replace('-', '1').replace('r', '0').replace('l', '1')
        idx = int(z_label, 2)
        sv = cls.from_int(idx, num_qubits)
        
        # 应用 Hadamard 和 S 门转换到正确的基态
        h_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        sh_mat = np.dot(np.diag([1, 1j]), h_mat)  # S·H
        
        for qubit, char in enumerate(reversed(label)):
            if char in ['+', '-']:
                sv = sv.evolve(h_mat, [qubit])
            elif char in ['r', 'l']:
                sv = sv.evolve(sh_mat, [qubit])
        
        return sv
    
    @classmethod
    def from_circuit(cls, circuit) -> 'Statevector':
        """
        从电路创建状态向量
        
        Args:
            circuit: Janus Circuit 对象
        
        Returns:
            Statevector: 电路作用后的状态向量
        """
        sv = cls.from_int(0, circuit.n_qubits)
        return sv.evolve_circuit(circuit)
    
    @property
    def data(self) -> np.ndarray:
        """获取状态向量数据"""
        return self._data
    
    @property
    def num_qubits(self) -> int:
        """获取量子比特数"""
        return self._num_qubits
    
    @property
    def dim(self) -> int:
        """获取状态向量维度"""
        return len(self._data)
    
    def seed(self, value: Optional[Union[int, np.random.Generator]] = None):
        """设置随机数种子"""
        if value is None:
            self._rng = np.random.default_rng()
        elif isinstance(value, np.random.Generator):
            self._rng = value
        else:
            self._rng = np.random.default_rng(value)
    
    def copy(self) -> 'Statevector':
        """创建副本"""
        new_sv = Statevector(self._data.copy(), self._num_qubits)
        new_sv._rng = self._rng
        return new_sv
    
    def is_valid(self, atol: float = 1e-10) -> bool:
        """检查是否为有效的归一化状态向量"""
        norm = np.linalg.norm(self._data)
        return np.isclose(norm, 1.0, atol=atol)
    
    def normalize(self) -> 'Statevector':
        """归一化状态向量"""
        norm = np.linalg.norm(self._data)
        if norm < 1e-15:
            raise InvalidStateError("Cannot normalize zero vector")
        self._data = self._data / norm
        return self
    
    # ==================== 演化方法 ====================
    
    def evolve(
        self,
        operator: np.ndarray,
        qargs: Optional[List[int]] = None
    ) -> 'Statevector':
        """
        通过算符演化状态向量
        
        Args:
            operator: 酉矩阵
            qargs: 作用的量子比特索引列表
        
        Returns:
            Statevector: 演化后的状态向量（原地修改）
        """
        if qargs is None:
            # 全系统演化
            if operator.shape[0] != self.dim:
                raise SimulatorError(
                    f"Operator dimension {operator.shape[0]} does not match "
                    f"statevector dimension {self.dim}"
                )
            self._data = operator @ self._data
        else:
            # 子系统演化
            self._data = self._apply_operator(operator, qargs)
        
        return self
    
    def _apply_operator(self, operator: np.ndarray, qargs: List[int]) -> np.ndarray:
        """
        将算符应用到指定量子比特
        
        使用张量收缩实现高效的子系统操作
        """
        n = self._num_qubits
        num_qargs = len(qargs)
        
        # 验证量子比特索引
        for q in qargs:
            if q < 0 or q >= n:
                raise SimulatorError(f"Qubit index {q} out of range [0, {n})")
        
        # 将状态向量重塑为张量
        tensor_shape = [2] * n
        state_tensor = self._data.reshape(tensor_shape)
        
        # 计算转置轴：将目标量子比特移到前面
        # 注意：Janus 使用小端序（qubit 0 是最低位）
        axes = list(range(n))
        target_axes = [n - 1 - q for q in qargs]  # 转换为张量索引
        other_axes = [i for i in range(n) if i not in target_axes]
        
        # 重排轴
        perm = target_axes + other_axes
        state_tensor = np.transpose(state_tensor, perm)
        
        # 重塑为矩阵形式进行乘法
        op_dim = 2 ** num_qargs
        other_dim = 2 ** (n - num_qargs)
        state_matrix = state_tensor.reshape(op_dim, other_dim)
        
        # 应用算符
        result_matrix = operator @ state_matrix
        
        # 恢复张量形状
        result_tensor = result_matrix.reshape([2] * num_qargs + [2] * (n - num_qargs))
        
        # 逆转置恢复原始顺序
        inv_perm = np.argsort(perm)
        result_tensor = np.transpose(result_tensor, inv_perm)
        
        return result_tensor.flatten()
    
    def evolve_circuit(self, circuit) -> 'Statevector':
        """
        通过电路演化状态向量
        
        Args:
            circuit: Janus Circuit 对象
        
        Returns:
            Statevector: 演化后的状态向量
        """
        from ..circuit.parameter import is_parameterized
        
        for inst in circuit.instructions:
            # 检查参数化
            if is_parameterized(inst.operation):
                raise SimulatorError(
                    f"Circuit contains unbound parameter in gate {inst.name}. "
                    "Please bind parameters before simulation."
                )
            
            # 获取门矩阵
            try:
                matrix = inst.operation.to_matrix()
            except NotImplementedError:
                raise SimulatorError(
                    f"Gate {inst.name} does not have a matrix representation"
                )
            
            # 应用门
            self.evolve(matrix, inst.qubits)
        
        return self

    # ==================== 测量方法 ====================
    
    def probabilities(self, qargs: Optional[List[int]] = None) -> np.ndarray:
        """
        计算测量概率分布
        
        Args:
            qargs: 要测量的量子比特索引，None 表示全部
        
        Returns:
            np.ndarray: 概率分布数组
        """
        probs = np.abs(self._data) ** 2
        
        if qargs is None:
            return probs
        
        # 边缘化到指定量子比特
        return self._marginalize_probabilities(probs, qargs)
    
    def _marginalize_probabilities(
        self, 
        probs: np.ndarray, 
        qargs: List[int]
    ) -> np.ndarray:
        """边缘化概率分布到指定量子比特"""
        n = self._num_qubits
        
        # 重塑为张量
        probs_tensor = probs.reshape([2] * n)
        
        # 计算要求和的轴（不在 qargs 中的量子比特）
        # 注意张量索引和量子比特索引的对应关系
        keep_axes = [n - 1 - q for q in qargs]
        sum_axes = tuple(i for i in range(n) if i not in keep_axes)
        
        # 求和边缘化
        if sum_axes:
            probs_tensor = np.sum(probs_tensor, axis=sum_axes)
        
        # 重排轴以匹配 qargs 顺序
        current_order = sorted(range(len(qargs)), key=lambda i: keep_axes[i])
        target_order = list(range(len(qargs)))
        
        if current_order != target_order:
            # 需要转置
            inv_order = [current_order.index(i) for i in target_order]
            probs_tensor = np.transpose(probs_tensor, inv_order)
        
        return probs_tensor.flatten()
    
    def probabilities_dict(
        self, 
        qargs: Optional[List[int]] = None,
        decimals: Optional[int] = None
    ) -> Dict[str, float]:
        """
        获取概率分布字典
        
        Args:
            qargs: 要测量的量子比特索引
            decimals: 小数位数
        
        Returns:
            Dict[str, float]: 比特串到概率的映射
        """
        probs = self.probabilities(qargs)
        num_bits = len(qargs) if qargs else self._num_qubits
        
        result = {}
        for i, p in enumerate(probs):
            if p > 1e-15:
                bitstring = format(i, f'0{num_bits}b')
                if decimals is not None:
                    p = round(p, decimals)
                result[bitstring] = p
        
        return result
    
    def sample_memory(
        self, 
        shots: int, 
        qargs: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        采样测量结果（保留顺序）
        
        Args:
            shots: 采样次数
            qargs: 要测量的量子比特索引
        
        Returns:
            np.ndarray: 测量结果数组，每个元素是比特串
        """
        probs = self.probabilities(qargs)
        num_bits = len(qargs) if qargs else self._num_qubits
        
        # 采样索引
        indices = self._rng.choice(len(probs), size=shots, p=probs)
        
        # 转换为比特串
        return np.array([format(i, f'0{num_bits}b') for i in indices])
    
    def sample_counts(
        self, 
        shots: int, 
        qargs: Optional[List[int]] = None
    ) -> Counts:
        """
        采样测量结果（返回计数）
        
        Args:
            shots: 采样次数
            qargs: 要测量的量子比特索引
        
        Returns:
            Counts: 测量计数
        """
        samples = self.sample_memory(shots, qargs)
        unique, counts = np.unique(samples, return_counts=True)
        return Counts(dict(zip(unique, counts.astype(int))))
    
    def measure(
        self, 
        qargs: Optional[List[int]] = None
    ) -> Tuple[str, 'Statevector']:
        """
        执行测量并返回结果和坍缩后的状态
        
        Args:
            qargs: 要测量的量子比特索引
        
        Returns:
            Tuple[str, Statevector]: (测量结果, 坍缩后的状态)
        """
        probs = self.probabilities(qargs)
        num_bits = len(qargs) if qargs else self._num_qubits
        
        # 采样一个结果
        outcome_idx = self._rng.choice(len(probs), p=probs)
        outcome = format(outcome_idx, f'0{num_bits}b')
        
        # 计算坍缩后的状态
        if qargs is None:
            # 全系统测量，坍缩到计算基态
            new_data = np.zeros_like(self._data)
            new_data[outcome_idx] = 1.0
        else:
            # 部分测量，需要投影
            new_data = self._project_measurement(qargs, outcome_idx)
        
        new_sv = Statevector(new_data, self._num_qubits)
        new_sv._rng = self._rng
        
        return outcome, new_sv
    
    def _project_measurement(self, qargs: List[int], outcome: int) -> np.ndarray:
        """投影测量后的状态"""
        n = self._num_qubits
        num_measured = len(qargs)
        
        # 创建投影后的状态
        new_data = np.zeros_like(self._data)
        
        # 遍历所有基态，保留与测量结果一致的分量
        for i in range(self.dim):
            # 提取测量比特的值
            measured_bits = 0
            for j, q in enumerate(qargs):
                bit = (i >> q) & 1
                measured_bits |= (bit << j)
            
            if measured_bits == outcome:
                new_data[i] = self._data[i]
        
        # 归一化
        norm = np.linalg.norm(new_data)
        if norm > 1e-15:
            new_data = new_data / norm
        
        return new_data
    
    # ==================== 期望值计算 ====================
    
    def expectation_value(
        self, 
        operator: np.ndarray,
        qargs: Optional[List[int]] = None
    ) -> complex:
        """
        计算可观测量的期望值 ⟨ψ|O|ψ⟩
        
        Args:
            operator: 可观测量矩阵
            qargs: 作用的量子比特索引
        
        Returns:
            complex: 期望值
        """
        if qargs is None:
            # 全系统算符
            return np.vdot(self._data, operator @ self._data)
        else:
            # 子系统算符
            evolved = self.copy()
            evolved._data = evolved._apply_operator(operator, qargs)
            return np.vdot(self._data, evolved._data)
    
    # ==================== 状态操作 ====================
    
    def tensor(self, other: 'Statevector') -> 'Statevector':
        """
        张量积 self ⊗ other
        
        Args:
            other: 另一个状态向量
        
        Returns:
            Statevector: 张量积状态
        """
        new_data = np.kron(self._data, other._data)
        return Statevector(new_data)
    
    def inner(self, other: 'Statevector') -> complex:
        """
        内积 ⟨self|other⟩
        
        Args:
            other: 另一个状态向量
        
        Returns:
            complex: 内积值
        """
        if self.dim != other.dim:
            raise SimulatorError("Statevector dimensions do not match")
        return np.vdot(self._data, other._data)
    
    def conjugate(self) -> 'Statevector':
        """返回共轭状态向量"""
        return Statevector(np.conj(self._data), self._num_qubits)
    
    def equiv(
        self, 
        other: 'Statevector', 
        atol: float = 1e-10
    ) -> bool:
        """
        检查两个状态向量是否等价（忽略全局相位）
        
        Args:
            other: 另一个状态向量
            atol: 绝对容差
        
        Returns:
            bool: 是否等价
        """
        if self.dim != other.dim:
            return False
        
        # 找到第一个非零元素
        for i in range(self.dim):
            if abs(self._data[i]) > atol:
                if abs(other._data[i]) < atol:
                    return False
                # 计算相位差
                phase = other._data[i] / self._data[i]
                # 检查所有元素是否相差相同的相位
                return np.allclose(self._data * phase, other._data, atol=atol)
        
        # self 是零向量
        return np.allclose(other._data, 0, atol=atol)
    
    # ==================== 特殊方法 ====================
    
    def __repr__(self) -> str:
        return f"Statevector({self._data}, num_qubits={self._num_qubits})"
    
    def __len__(self) -> int:
        return self.dim
    
    def __getitem__(self, key: Union[int, str]) -> complex:
        """
        获取状态向量元素
        
        Args:
            key: 索引（整数）或比特串
        """
        if isinstance(key, str):
            key = int(key, 2)
        return self._data[key]
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Statevector):
            return False
        return np.allclose(self._data, other._data)
    
    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)
    
    def to_dict(self, decimals: Optional[int] = None) -> Dict[str, complex]:
        """
        转换为字典表示
        
        Args:
            decimals: 小数位数
        
        Returns:
            Dict[str, complex]: 比特串到振幅的映射
        """
        result = {}
        for i, amp in enumerate(self._data):
            if abs(amp) > 1e-15:
                bitstring = format(i, f'0{self._num_qubits}b')
                if decimals is not None:
                    amp = complex(round(amp.real, decimals), round(amp.imag, decimals))
                result[bitstring] = amp
        return result
    
    def draw(self, output: str = 'text', **kwargs) -> str:
        """
        绘制状态向量
        
        Args:
            output: 输出格式 ('text', 'latex')
        
        Returns:
            str: 状态向量的字符串表示
        """
        if output == 'text':
            return self._draw_text(**kwargs)
        elif output == 'latex':
            return self._draw_latex(**kwargs)
        else:
            raise ValueError(f"Unknown output format: {output}")
    
    def _draw_text(self, decimals: int = 4) -> str:
        """文本格式绘制"""
        lines = []
        for i, amp in enumerate(self._data):
            if abs(amp) > 1e-10:
                bitstring = format(i, f'0{self._num_qubits}b')
                if abs(amp.imag) < 1e-10:
                    amp_str = f"{amp.real:.{decimals}f}"
                elif abs(amp.real) < 1e-10:
                    amp_str = f"{amp.imag:.{decimals}f}j"
                else:
                    amp_str = f"({amp.real:.{decimals}f}{amp.imag:+.{decimals}f}j)"
                lines.append(f"{amp_str}|{bitstring}⟩")
        
        return " + ".join(lines) if lines else "0"
    
    def _draw_latex(self, decimals: int = 4) -> str:
        """LaTeX 格式绘制"""
        terms = []
        for i, amp in enumerate(self._data):
            if abs(amp) > 1e-10:
                bitstring = format(i, f'0{self._num_qubits}b')
                if abs(amp.imag) < 1e-10:
                    amp_str = f"{amp.real:.{decimals}f}"
                elif abs(amp.real) < 1e-10:
                    amp_str = f"{amp.imag:.{decimals}f}i"
                else:
                    amp_str = f"({amp.real:.{decimals}f}{amp.imag:+.{decimals}f}i)"
                terms.append(f"{amp_str}|{bitstring}\\rangle")
        
        return " + ".join(terms) if terms else "0"
