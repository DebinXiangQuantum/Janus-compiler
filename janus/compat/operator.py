"""
Janus Operator 类实现

用于表示量子操作的矩阵形式
"""
import numpy as np
from typing import Union, List, Optional
import sys
import os

# Add janus to path for absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from circuit.gate import Gate


class Operator:
    """
    量子操作符类
    
    表示量子操作的酉矩阵形式
    """
    
    def __init__(self, data: Union[Gate, np.ndarray, List]):
        """
        初始化操作符
        
        Args:
            data: 门对象、矩阵或数据
        """
        if isinstance(data, Gate):
            self._data = data.to_matrix()
        elif hasattr(data, 'to_matrix'):
            # 处理具有 to_matrix 方法的对象
            self._data = data.to_matrix()
        elif isinstance(data, np.ndarray):
            self._data = data.astype(complex)
        elif isinstance(data, (list, tuple)):
            self._data = np.array(data, dtype=complex)
        else:
            # 尝试处理 Circuit 对象
            try:
                # 检查是否是 Circuit 类型（通过类名检查，避免循环导入）
                type_name = type(data).__name__
                if type_name == 'Circuit':
                    # 将电路转换为酉矩阵
                    self._data = self._circuit_to_unitary(data)
                elif type_name == 'QuantumCircuit':
                    # 处理外部 QuantumCircuit
                    self._data = self._external_circuit_to_unitary(data)
                else:
                    raise ValueError(f"Unsupported data type: {type(data)}")
            except Exception as e:
                raise ValueError(f"Unsupported data type: {type(data)}, error: {e}")
        
        # 确保是二维矩阵
        if self._data.ndim != 2:
            raise ValueError("Operator data must be a 2D matrix")
        
        # 确保是方阵
        if self._data.shape[0] != self._data.shape[1]:
            raise ValueError("Operator matrix must be square")
    
    def _circuit_to_unitary(self, circuit) -> np.ndarray:
        """
        将电路转换为酉矩阵
        
        Args:
            circuit: Circuit 对象
            
        Returns:
            电路对应的酉矩阵
        """
        n_qubits = circuit.n_qubits
        dim = 2 ** n_qubits
        
        # 初始化为单位矩阵
        unitary = np.eye(dim, dtype=complex)
        
        # 逐个应用门操作
        for instruction in circuit.instructions:
            gate = instruction.operation
            qubits = instruction.qubits
            
            # 获取门的矩阵
            gate_matrix = gate.to_matrix()
            
            # 构造作用在整个系统上的矩阵
            full_matrix = self._expand_gate_to_full_system(gate_matrix, qubits, n_qubits)
            
            # 应用门操作
            unitary = full_matrix @ unitary
        
        return unitary
    
    def _external_circuit_to_unitary(self, circuit) -> np.ndarray:
        """
        将外部 QuantumCircuit 转换为酉矩阵
        
        Args:
            circuit: 外部 QuantumCircuit 对象
            
        Returns:
            电路对应的酉矩阵
        """
        n_qubits = circuit.num_qubits
        dim = 2 ** n_qubits
        
        # 初始化为单位矩阵
        unitary = np.eye(dim, dtype=complex)
        
        # 创建 qubit 到索引的映射
        qubit_indices = {qubit: i for i, qubit in enumerate(circuit.qubits)}
        
        # 逐个应用门操作
        for instruction in circuit.data:
            op = instruction.operation
            qubits = [qubit_indices[q] for q in instruction.qubits]
            
            # 获取门的矩阵
            gate_matrix = self._get_external_gate_matrix(op)
            
            if gate_matrix is not None:
                # 构造作用在整个系统上的矩阵
                full_matrix = self._expand_gate_to_full_system(gate_matrix, qubits, n_qubits)
                
                # 应用门操作
                unitary = full_matrix @ unitary
        
        return unitary
    
    def _get_external_gate_matrix(self, gate) -> np.ndarray:
        """获取外部门的矩阵表示"""
        gate_name = gate.name.lower()
        
        # 标准门矩阵
        gate_matrices = {
            'id': np.eye(2, dtype=complex),
            'i': np.eye(2, dtype=complex),
            'x': np.array([[0, 1], [1, 0]], dtype=complex),
            'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'z': np.array([[1, 0], [0, -1]], dtype=complex),
            'h': np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
            's': np.array([[1, 0], [0, 1j]], dtype=complex),
            'sdg': np.array([[1, 0], [0, -1j]], dtype=complex),
            't': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
            'tdg': np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex),
            'cx': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
            'cnot': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
            'cz': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex),
            'swap': np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex),
        }
        
        if gate_name in gate_matrices:
            return gate_matrices[gate_name]
        
        # 尝试使用 to_matrix 方法
        if hasattr(gate, 'to_matrix'):
            try:
                return np.array(gate.to_matrix(), dtype=complex)
            except:
                pass
        
        # 返回单位矩阵作为默认值
        return None
    
    def _expand_gate_to_full_system(self, gate_matrix: np.ndarray, qubits: List[int], n_qubits: int) -> np.ndarray:
        """
        将门矩阵扩展到整个量子系统
        
        Args:
            gate_matrix: 门的矩阵
            qubits: 门作用的量子比特
            n_qubits: 总量子比特数
            
        Returns:
            扩展后的矩阵
        """
        dim = 2 ** n_qubits
        full_matrix = np.eye(dim, dtype=complex)
        
        # 对于单比特门
        if len(qubits) == 1:
            qubit = qubits[0]
            # 使用 Kronecker 积构造完整矩阵
            matrices = []
            for i in range(n_qubits):
                if i == qubit:
                    matrices.append(gate_matrix)
                else:
                    matrices.append(np.eye(2, dtype=complex))
            
            # 计算 Kronecker 积
            result = matrices[0]
            for i in range(1, len(matrices)):
                result = np.kron(result, matrices[i])
            return result
        
        # 对于多比特门，使用更通用的方法
        elif len(qubits) == 2:
            # 两比特门的实现
            q0, q1 = sorted(qubits)
            
            # 构造基态映射
            for i in range(dim):
                for j in range(dim):
                    # 提取相关比特的状态
                    state_i = [(i >> k) & 1 for k in range(n_qubits)]
                    state_j = [(j >> k) & 1 for k in range(n_qubits)]
                    
                    # 检查非相关比特是否相同
                    non_gate_qubits_same = True
                    for k in range(n_qubits):
                        if k not in qubits and state_i[k] != state_j[k]:
                            non_gate_qubits_same = False
                            break
                    
                    if non_gate_qubits_same:
                        # 计算门矩阵的索引
                        gate_i = state_i[q1] * 2 + state_i[q0]  # 注意比特顺序
                        gate_j = state_j[q1] * 2 + state_j[q0]
                        
                        full_matrix[i, j] = gate_matrix[gate_i, gate_j]
                    else:
                        full_matrix[i, j] = 0.0
            
            return full_matrix
        
        else:
            # 多比特门的通用实现（简化版）
            # 这里可以根据需要实现更复杂的多比特门
            return full_matrix
    
    @property
    def data(self) -> np.ndarray:
        """获取矩阵数据"""
        return self._data
    
    @property
    def dim(self) -> int:
        """获取矩阵维度"""
        return self._data.shape[0]
    
    @property
    def num_qubits(self) -> int:
        """获取量子比特数"""
        return int(np.log2(self.dim))
    
    def __matmul__(self, other: 'Operator') -> 'Operator':
        """矩阵乘法"""
        if isinstance(other, Operator):
            return Operator(self._data @ other._data)
        return NotImplemented
    
    def __mul__(self, scalar: Union[int, float, complex]) -> 'Operator':
        """标量乘法"""
        return Operator(self._data * scalar)
    
    def __rmul__(self, scalar: Union[int, float, complex]) -> 'Operator':
        """右标量乘法"""
        return self.__mul__(scalar)
    
    def conjugate(self) -> 'Operator':
        """共轭"""
        return Operator(np.conjugate(self._data))
    
    def transpose(self) -> 'Operator':
        """转置"""
        return Operator(self._data.T)
    
    def adjoint(self) -> 'Operator':
        """厄米共轭（转置共轭）"""
        return Operator(self._data.T.conj())
    
    def dagger(self) -> 'Operator':
        """厄米共轭的别名"""
        return self.adjoint()
    
    def dot(self, other: 'Operator') -> 'Operator':
        """矩阵乘法（dot product）"""
        if isinstance(other, Operator):
            return Operator(self._data @ other._data)
        return NotImplemented
    
    def trace(self) -> complex:
        """矩阵的迹"""
        return np.trace(self._data)
    
    def is_unitary(self, atol: float = 1e-10) -> bool:
        """检查是否为酉矩阵"""
        product = self._data @ self._data.T.conj()
        identity = np.eye(self.dim, dtype=complex)
        return np.allclose(product, identity, atol=atol)
    
    def __repr__(self) -> str:
        return f"Operator({self._data.shape[0]}x{self._data.shape[1]} matrix)"
    
    def __str__(self) -> str:
        return f"Operator:\n{self._data}"


def random_unitary(dim: int, seed: Optional[int] = None) -> Operator:
    """
    生成随机酉矩阵
    
    Args:
        dim: 矩阵维度
        seed: 随机种子
    
    Returns:
        随机酉操作符
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 使用 QR 分解生成随机酉矩阵
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    Q, R = np.linalg.qr(A)
    
    # 确保对角元素为正
    d = np.diagonal(R)
    ph = d / np.abs(d)
    Q = Q @ np.diag(ph)
    
    return Operator(Q)


def matrix_equal(matrix1: np.ndarray, matrix2: np.ndarray, 
                 ignore_phase: bool = False, atol: float = 1e-10) -> bool:
    """
    比较两个矩阵是否相等
    
    Args:
        matrix1: 第一个矩阵
        matrix2: 第二个矩阵
        ignore_phase: 是否忽略全局相位
        atol: 绝对容差
    
    Returns:
        是否相等
    """
    if matrix1.shape != matrix2.shape:
        return False
    
    if ignore_phase:
        # 忽略全局相位：找到第一个非零元素并归一化
        for i in range(matrix1.size):
            idx = np.unravel_index(i, matrix1.shape)
            if abs(matrix1[idx]) > atol:
                phase1 = matrix1[idx] / abs(matrix1[idx])
                break
        else:
            phase1 = 1.0
            
        for i in range(matrix2.size):
            idx = np.unravel_index(i, matrix2.shape)
            if abs(matrix2[idx]) > atol:
                phase2 = matrix2[idx] / abs(matrix2[idx])
                break
        else:
            phase2 = 1.0
        
        # 归一化后比较
        normalized1 = matrix1 / phase1
        normalized2 = matrix2 / phase2
        return np.allclose(normalized1, normalized2, atol=atol)
    else:
        return np.allclose(matrix1, matrix2, atol=atol)


def is_unitary_matrix(matrix: np.ndarray, atol: float = 1e-10) -> bool:
    """
    检查矩阵是否为酉矩阵
    
    Args:
        matrix: 要检查的矩阵
        atol: 绝对容差
    
    Returns:
        是否为酉矩阵
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    product = matrix @ matrix.T.conj()
    identity = np.eye(matrix.shape[0], dtype=complex)
    return np.allclose(product, identity, atol=atol)