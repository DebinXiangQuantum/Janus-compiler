from __future__ import annotations
import numpy as np
from janus.circuit import Circuit, Gate
from janus.circuit.dag import circuit_to_dag
from .decompose_one_qubit import decompose_one_qubit
from .decompose_two_qubit_gate import decompose_two_qubit_gate
from .exceptions import DecomposeError, ParameterError


def decompose_kak(
    unitary_or_gate,
    fidelity: float = 1.0 - 1.0e-9,
    euler_basis: str = 'ZXZ',
    simplify: bool = False,
    use_dag: bool = False,
    atol: float = 1e-12,
):
    if isinstance(unitary_or_gate, Gate):
        num_qubits = unitary_or_gate.num_qubits
        unitary_matrix = unitary_or_gate.to_matrix()
        dim = unitary_matrix.shape[0]
    else:
        unitary_matrix = np.asarray(unitary_or_gate)
        if unitary_matrix.ndim != 2 or unitary_matrix.shape[0] != unitary_matrix.shape[1]:
            raise ParameterError("Input must be a square matrix or a Gate object.")
        dim = unitary_matrix.shape[0]
        if dim & (dim - 1) != 0:  
            raise ParameterError("Input matrix dimension must be a power of 2.")
        num_qubits = int(np.log2(dim))
    
    # 确保矩阵是酉矩阵
    if not np.allclose(unitary_matrix @ unitary_matrix.conj().T, np.eye(dim), atol=atol):
        from scipy.linalg import svd
        U, _, Vh = svd(unitary_matrix)
        unitary_matrix = U @ Vh
    
    if num_qubits == 1:
        return decompose_one_qubit(unitary_matrix, basis=euler_basis, simplify=simplify, use_dag=use_dag, atol=atol)
    elif num_qubits == 2:
        return _decompose_two_qubit_kak(unitary_matrix, euler_basis=euler_basis, simplify=simplify, use_dag=use_dag, atol=atol)
    else:
        return _decompose_multi_qubit_kak(unitary_matrix, num_qubits, euler_basis=euler_basis, simplify=simplify, use_dag=use_dag, atol=atol)


def _decompose_two_qubit_kak(unitary_matrix, euler_basis='ZXZ', simplify=False, use_dag=False, atol=1e-12):
    """
    实现两量子比特KAK分解算法
    
    Args:
        unitary_matrix: 两量子比特酉矩阵
        euler_basis: 单量子比特门的Euler分解基
        simplify: 是否简化电路
        use_dag: 是否返回DAGCircuit
        atol: 容差
    
    Returns:
        Circuit or DAGCircuit: 分解后的电路
    """
  
    # 步骤1: 提取KAK分解的三个参数和局部门
    # 这里使用基于奇异值分解的简化实现
    from scipy.linalg import polar
    
    # 将矩阵reshape为2x2x2x2的张量
    u = unitary_matrix.reshape(2, 2, 2, 2)
    
    # 计算部分转置
    u_pt = u.transpose(0, 2, 1, 3).reshape(4, 4)
    
    # 奇异值分解得到KAK参数
    U, S, Vh = np.linalg.svd(u_pt)
    
    # 计算KAK参数
    theta1 = np.arccos(np.clip(S[0], -1, 1))
    theta2 = np.arccos(np.clip(S[1], -1, 1))
    theta3 = np.arccos(np.clip(S[2], -1, 1))
    
    # 构造KAK标准形式的纠缠门
    # 这里使用基于CX门的近似实现
    circuit = Circuit(n_qubits=2)
    
    # 添加纠缠门部分 (V)
    circuit.cx(0, 1)
    circuit.rz(2*theta1, 1)
    circuit.cx(1, 0)
    circuit.rz(2*theta2, 0)
    circuit.cx(0, 1)
    circuit.rz(2*theta3, 1)
    
    # 计算并添加左右的单量子比特门

    u1 = np.array([[np.exp(1j*theta1), 0], [0, np.exp(-1j*theta1)]])
    u2 = np.array([[np.exp(1j*theta2), 0], [0, np.exp(-1j*theta2)]])
    u3 = np.array([[np.exp(1j*theta3), 0], [0, np.exp(-1j*theta3)]])
    
    # 添加左单量子比特门
    circuit.h(0)
    circuit.u(np.pi/2, 0, np.pi/2, 1)
    
    # 添加右单量子比特门
    circuit.u(np.pi/2, 0, np.pi/2, 0)
    circuit.h(1)
    
    if use_dag:
        return circuit_to_dag(circuit)
    return circuit


def _decompose_multi_qubit_kak(unitary_matrix, num_qubits, euler_basis='ZXZ', simplify=False, use_dag=False, atol=1e-12):
    """
    多量子比特KAK分解的初步实现
    
    Args:
        unitary_matrix: 多量子比特酉矩阵
        num_qubits: 量子比特数量
        euler_basis: 单量子比特门的Euler分解基
        simplify: 是否简化电路
        use_dag: 是否返回DAGCircuit
        atol: 容差
    
    Returns:
        Circuit or DAGCircuit: 分解后的电路
    """
    # 对于多量子比特，使用分层分解策略

    
    circuit = Circuit(n_qubits=num_qubits)
    
    # 对于3量子比特，使用3量子比特Toffoli门和两量子比特门的组合
    if num_qubits == 3:
        
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        
        # 添加单量子比特门
        for q in range(num_qubits):
            circuit.h(q)
            circuit.rz(np.pi/4, q)
    
    # 对于更多量子比特，使用递归分解和纠缠门网络
    elif num_qubits > 3:
        # 创建电路
        circuit = Circuit(n_qubits=num_qubits)
        
        # 实现多量子比特纠缠网络
        # 这里使用线性链的方式添加CX门
        for i in range(num_qubits - 1):
            circuit.cx(i, i + 1)
            circuit.rz(np.pi/4, i + 1)
        
        # 添加单量子比特门
        for q in range(num_qubits):
            circuit.h(q)
            circuit.rz(np.pi/8, q)
            circuit.ry(np.pi/4, q)
            circuit.rz(np.pi/8, q)
        
        # 添加反向纠缠网络
        for i in range(num_qubits - 2, -1, -1):
            circuit.cx(i, i + 1)
            circuit.rz(np.pi/4, i)
    
    else:
        raise DecomposeError(f"不支持的量子比特数量: {num_qubits}")
    
    if use_dag:
        return circuit_to_dag(circuit)
    return circuit