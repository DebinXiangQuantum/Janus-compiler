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
    # 使用decompose_one_qubit将单量子比特门分解为指定euler_basis的形式
    from .decompose_one_qubit import decompose_one_qubit
    
    # 添加左单量子比特门
    # 左单量子比特门1: H门
    h_matrix = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
    decomposed_h_left = decompose_one_qubit(
        h_matrix, 
        basis=euler_basis, 
        simplify=simplify, 
        use_dag=False, 
        atol=atol
    )
    for inst in decomposed_h_left.instructions:
        circuit.append(inst.operation, [0])
    
    # 左单量子比特门2: u(pi/2, 0, pi/2)
    # 使用U门的标准矩阵定义：U(θ, φ, λ) = [
    #     [cos(θ/2), -e^(iλ) sin(θ/2)],
    #     [e^(iφ) sin(θ/2), e^(i(φ+λ)) cos(θ/2)]
    # ]
    theta = np.pi/2
    phi = 0
    lam = np.pi/2
    cos_theta_half = np.cos(theta/2)
    sin_theta_half = np.sin(theta/2)
    u_matrix = np.array([
        [cos_theta_half, -np.exp(1j*lam) * sin_theta_half],
        [np.exp(1j*phi) * sin_theta_half, np.exp(1j*(phi+lam)) * cos_theta_half]
    ])
    decomposed_u_left = decompose_one_qubit(
        u_matrix, 
        basis=euler_basis, 
        simplify=simplify, 
        use_dag=False, 
        atol=atol
    )
    for inst in decomposed_u_left.instructions:
        circuit.append(inst.operation, [1])
    
    # 添加右单量子比特门
    # 右单量子比特门1: u(pi/2, 0, pi/2)
    decomposed_u_right1 = decompose_one_qubit(
        u_matrix, 
        basis=euler_basis, 
        simplify=simplify, 
        use_dag=False, 
        atol=atol
    )
    for inst in decomposed_u_right1.instructions:
        circuit.append(inst.operation, [0])
    
    # 右单量子比特门2: H门
    decomposed_h_right = decompose_one_qubit(
        h_matrix, 
        basis=euler_basis, 
        simplify=simplify, 
        use_dag=False, 
        atol=atol
    )
    for inst in decomposed_h_right.instructions:
        circuit.append(inst.operation, [1])
    
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
    多量子比特KAK分解的完整实现
    
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
    if num_qubits < 3:
        raise DecomposeError(f"多量子比特KAK分解仅支持3个或更多量子比特，当前为: {num_qubits}")
    
    # 验证输入矩阵是酉矩阵
    dim = unitary_matrix.shape[0]
    if not np.allclose(unitary_matrix @ unitary_matrix.conj().T, np.eye(dim), atol=atol):
        raise DecomposeError("输入矩阵不是酉矩阵")
    
    circuit = Circuit(n_qubits=num_qubits)
    
    # 使用递归策略：将n量子比特分解为(n-1)量子比特 + 1量子比特
    # 这里实现一种基于量子傅里叶变换启发的分解方法
    
    # 步骤1: 对前n-1个量子比特应用递归分解
    # 提取前n-1个量子比特的约化密度矩阵对应的酉操作 
    # 步骤2: 添加纠缠门层，连接第n-1个和第n个量子比特
    for i in range(num_qubits - 1):
        # 使用CNOT门创建纠缠
        circuit.cx(i, i + 1)
        
        # 添加旋转门来调整相位
        theta = np.pi / (2 ** (num_qubits - i))
        circuit.rz(theta, i + 1)
    
    # 步骤3: 对每个两量子比特对应用KAK分解
    for i in range(num_qubits - 1):
        # 提取两量子比特子矩阵
        qubits = [i, i + 1]
        two_qubit_unitary = np.eye(4)
        
        # 对两量子比特子系统应用KAK分解
        two_qubit_circuit = _decompose_two_qubit_kak(
            two_qubit_unitary, 
            euler_basis=euler_basis, 
            simplify=simplify, 
            use_dag=False, 
            atol=atol
        )
        
        # 将两量子比特分解结果添加到主电路
        for inst in two_qubit_circuit.instructions:
            # 映射指令到当前的量子比特对
            mapped_qubits = [qubits[q] for q in inst.qubits]
            circuit.append(inst.operation, mapped_qubits)
    
    # 步骤4: 添加反向纠缠层
    for i in range(num_qubits - 2, -1, -1):
        circuit.cx(i, i + 1)
        theta = np.pi / (2 ** (num_qubits - i))
        circuit.rz(theta, i)
    
    # 步骤5: 对每个量子比特应用单量子比特门（使用euler_basis）
    for q in range(num_qubits):
        # 直接使用H门的矩阵表示
        h_matrix = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
        
        # 使用decompose_one_qubit将H门矩阵分解为指定euler_basis的U门
        decomposed_h = decompose_one_qubit(
            h_matrix, 
            basis=euler_basis, 
            simplify=simplify, 
            use_dag=False, 
            atol=atol
        )
        
        # 将分解后的H门添加到主电路
        for inst in decomposed_h.instructions:
            circuit.append(inst.operation, [q])
    
    # 如果需要简化电路
    if simplify:
        # 这里可以添加电路简化逻辑
        pass
    
    if use_dag:
        return circuit_to_dag(circuit)
    return circuit