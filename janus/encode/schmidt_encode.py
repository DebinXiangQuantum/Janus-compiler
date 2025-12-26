from typing import List, Dict, Tuple
import numpy as np
from math import log2, ceil, floor, pi, acos, asin
from janus.circuit import Circuit
from janus.circuit.library import U3Gate


class SchmidtEncodeResult:
    """Schmidt编码结果封装类"""
    def __init__(self, circuit, out_qubits):
        self._circuit = circuit
        self._out_qubits = out_qubits
    
    @property
    def circuit(self):
        return self._circuit
    
    @property
    def out_qubits(self):
        return self._out_qubits
    
    def measure(self):
        n_clbits = len(self._out_qubits)
        measured_circuit = Circuit(self._circuit.num_qubits, n_clbits)
        for inst in self._circuit.instructions:
            measured_circuit.append(inst.operation, inst.qubits, inst.clbits)
        for i, qubit in enumerate(self._out_qubits):
            measured_circuit.measure(qubit, i)
        return measured_circuit


# ========== Schmidt分解的量子态制备辅助类 ==========

class QuantumStateManager:
    """量子态管理器，用于跟踪和操作量子态的比特串表示"""
    
    def __init__(self, amplitudes: List[float]):
        self.n_qubits = int(np.ceil(np.log2(len(amplitudes))))
        self.basis_states = {}
        
        # 初始化基态映射
        for idx, amp in enumerate(amplitudes):
            if abs(amp) > 1e-14:
                basis = format(idx, f'0{self.n_qubits}b')
                self.basis_states[basis] = float(amp)
        
        # 归一化
        self._normalize()
    
    def _normalize(self):
        """归一化量子态"""
        total_prob = sum(amp**2 for amp in self.basis_states.values())
        if abs(1.0 - total_prob) > 1e-13 and total_prob > 1e-13:
            norm_factor = np.sqrt(total_prob)
            self.basis_states = {k: v / norm_factor for k, v in self.basis_states.items()}
    
    def get_active_bases(self):
        """获取所有活跃的基态"""
        return list(self.basis_states.keys())
    
    def get_amplitude(self, basis: str):
        """获取指定基态的振幅"""
        return self.basis_states.get(basis, 0.0)
    
    def flip_bit(self, basis: str, pos: int) -> str:
        """翻转指定位置的比特"""
        bits = list(basis)
        bits[pos] = '0' if bits[pos] == '1' else '1'
        return ''.join(bits)
    
    def apply_pauli_x(self, qubit_pos: int):
        """应用Pauli-X门到所有基态"""
        new_states = {}
        for basis, amp in self.basis_states.items():
            new_basis = self.flip_bit(basis, qubit_pos)
            new_states[new_basis] = amp
        self.basis_states = new_states
    
    def apply_cnot(self, ctrl_pos: int, tgt_pos: int):
        """应用CNOT门到所有基态"""
        new_states = {}
        for basis, amp in self.basis_states.items():
            if basis[ctrl_pos] == '1':
                new_basis = self.flip_bit(basis, tgt_pos)
                new_states[new_basis] = amp
            else:
                new_states[basis] = amp
        self.basis_states = new_states
    
    def collapse_bases(self, target_basis: str, partner_basis: str):
        """合并两个基态"""
        amp1 = self.basis_states[target_basis]
        amp2 = self.basis_states[partner_basis]
        combined_amp = np.sqrt(amp1**2 + amp2**2)
        
        del self.basis_states[partner_basis]
        self.basis_states[target_basis] = combined_amp


class BasisPairSelector:
    """基态对选择器，用于选择待合并的基态对"""
    
    @staticmethod
    def partition_by_bit(bases: List[str], bit_pos: int, excluded: List[int]):
        """按指定比特位分割基态集合"""
        available_bits = [i for i in range(len(bases[0])) if i not in excluded]
        
        best_pos = bit_pos
        best_split = ([], [])
        max_imbalance = -1
        
        for pos in available_bits:
            zeros = [b for b in bases if b[pos] == '0']
            ones = [b for b in bases if b[pos] == '1']
            
            if zeros and ones:
                imbalance = abs(len(zeros) - len(ones))
                if imbalance > max_imbalance:
                    max_imbalance = imbalance
                    best_pos = pos
                    best_split = (zeros, ones)
        
        return best_pos, best_split[0], best_split[1]
    
    @staticmethod
    def isolate_single_basis(candidates: List[str], constraint_bits: List[int], constraint_vals: List[int]):
        """从候选集中隔离出单个基态"""
        remaining = candidates[:]
        
        while len(remaining) > 1:
            pos, group0, group1 = BasisPairSelector.partition_by_bit(remaining, 0, constraint_bits)
            
            if pos not in constraint_bits:
                constraint_bits.append(pos)
            
            if len(group0) < len(group1):
                constraint_vals.append(0)
                remaining = group0
            else:
                constraint_vals.append(1)
                remaining = group1
        
        return remaining
    
    @staticmethod
    def filter_compatible_bases(all_bases: List[str], exclude: str, 
                                constraint_bits: List[int], constraint_vals: List[int]):
        """筛选满足约束条件的基态"""
        compatible = []
        
        for basis in all_bases:
            if basis == exclude:
                continue
            
            is_compatible = True
            for bit_idx, val in zip(constraint_bits, constraint_vals):
                if int(basis[bit_idx]) != val:
                    is_compatible = False
                    break
            
            if is_compatible:
                compatible.append(basis)
        
        return compatible
    
    @staticmethod
    def select_pair(state_mgr: QuantumStateManager):
        """选择一对待合并的基态"""
        active_bases = state_mgr.get_active_bases()
        
        if len(active_bases) == 2:
            pivot_bit, g0, g1 = BasisPairSelector.partition_by_bit(active_bases, 0, [])
            return g1[0], g0[0], pivot_bit, []
        
        # 多基态情况
        constraints_bits = []
        constraints_vals = []
        
        # 选择第一个基态
        first_group = BasisPairSelector.isolate_single_basis(
            active_bases, constraints_bits, constraints_vals
        )
        first_basis = first_group[0]
        
        # 移除最后一个约束
        pivot_bit = constraints_bits.pop()
        constraints_vals.pop()
        
        # 选择第二个基态
        compatible = BasisPairSelector.filter_compatible_bases(
            active_bases, first_basis, constraints_bits, constraints_vals
        )
        second_group = BasisPairSelector.isolate_single_basis(
            compatible, constraints_bits, constraints_vals
        )
        second_basis = second_group[0]
        
        return first_basis, second_basis, pivot_bit, constraints_bits


class RotationCalculator:
    """旋转角度计算器"""
    
    @staticmethod
    def compute_u3_params(amplitude_a: float, amplitude_b: float):
        """计算U3门的参数"""
        magnitude = np.sqrt(amplitude_a**2 + amplitude_b**2)
        
        if magnitude < 1e-14:
            return 0.0, 0.0, 0.0
        
        ratio = abs(amplitude_b / magnitude)
        ratio = max(0.0, min(1.0, ratio))
        theta = 2 * asin(ratio)
        
        return float(theta), 0.0, 0.0


class CircuitBuilder:
    """电路构建器，负责生成量子电路"""
    
    def __init__(self, num_qubits: int):
        self.working_circuit = Circuit(num_qubits)
        self.qubit_indices = list(range(num_qubits))
    
    def add_x_gate(self, qubit: int):
        """添加X门"""
        self.working_circuit.x(qubit)
    
    def add_cx_gate(self, control: int, target: int):
        """添加CNOT门"""
        self.working_circuit.cx(control, target)
    
    def add_controlled_rotation(self, target: int, controls: List[int], theta: float):
        """添加受控旋转门"""
        if not controls:
            self.working_circuit.u3(theta, 0.0, 0.0, target)
        else:
            u3_gate = U3Gate(theta, 0.0, 0.0)
            ctrl_qubits = [controls[i] for i in range(len(controls))]
            self.working_circuit.gate(u3_gate, target).control(ctrl_qubits)
    
    def get_inverted_circuit(self):
        """获取反转后的电路"""
        return self.working_circuit.inverse()


def _execute_basis_reduction(state_mgr: QuantumStateManager, builder: CircuitBuilder):
    """执行基态约简过程"""
    # 选择待处理的基态对
    basis_a, basis_b, pivot, ctrl_bits = BasisPairSelector.select_pair(state_mgr)
    
    # 步骤1: 确保basis_a在pivot位为'1'
    if basis_a[pivot] != '1':
        builder.add_x_gate(pivot)
        state_mgr.apply_pauli_x(pivot)
        basis_a = state_mgr.flip_bit(basis_a, pivot)
        basis_b = state_mgr.flip_bit(basis_b, pivot)
    
    # 步骤2: 对齐两个基态（除pivot位外）
    for bit_idx in range(len(basis_a)):
        if bit_idx != pivot and basis_a[bit_idx] != basis_b[bit_idx]:
            builder.add_cx_gate(pivot, bit_idx)
            state_mgr.apply_cnot(pivot, bit_idx)
            basis_a = state_mgr.flip_bit(basis_a, bit_idx) if basis_a[pivot] == '1' else basis_a
            basis_b = state_mgr.flip_bit(basis_b, bit_idx) if basis_b[pivot] == '1' else basis_b
    
    # 步骤3: 设置控制位
    for ctrl_bit in ctrl_bits:
        if basis_b[ctrl_bit] != '1':
            builder.add_x_gate(ctrl_bit)
            state_mgr.apply_pauli_x(ctrl_bit)
            basis_a = state_mgr.flip_bit(basis_a, ctrl_bit)
            basis_b = state_mgr.flip_bit(basis_b, ctrl_bit)
    
    # 步骤4: 应用旋转门
    amp_a = state_mgr.get_amplitude(basis_a)
    amp_b = state_mgr.get_amplitude(basis_b)
    theta, _, _ = RotationCalculator.compute_u3_params(amp_a, amp_b)
    builder.add_controlled_rotation(pivot, ctrl_bits, theta)
    
    # 步骤5: 合并基态
    state_mgr.collapse_bases(basis_a, basis_b)


def _prepare_quantum_state(circuit: Circuit, qubits: List[int], amplitudes: List[float]):
    """
    量子态制备的主函数
    使用迭代约简方法将多个基态逐步合并
    """
    # 初始化状态管理器
    state_mgr = QuantumStateManager(amplitudes)
    
    if not state_mgr.basis_states:
        return
    
    # 创建电路构建器
    builder = CircuitBuilder(len(qubits))
    
    # 迭代约简基态
    while len(state_mgr.basis_states) > 1:
        _execute_basis_reduction(state_mgr, builder)
    
    # 处理最终基态
    final_basis = next(iter(state_mgr.basis_states.keys()))
    for idx, bit_val in enumerate(final_basis):
        if bit_val == '1':
            builder.add_x_gate(idx)
    
    # 获取反转电路并添加到目标电路
    inverted = builder.get_inverted_circuit()
    for inst in inverted.instructions:
        mapped_qubits = [qubits[q] for q in inst.qubits]
        circuit.append(inst.operation, mapped_qubits, inst.clbits)


# ========== Schmidt分解的递归实现 ==========

def _handle_vector_decomposition(circuit: Circuit, qubits: List[int], 
                                 matrix: np.ndarray, threshold: float):
    """处理向量形式的矩阵分解"""
    matrix = np.asarray(matrix, dtype=complex)
    
    if len(matrix.shape) == 1:
        matrix = matrix.reshape(-1, 1)
    
    rows, cols = matrix.shape
    qubit_count = len(qubits)
    
    # 跳过单位矩阵
    if np.allclose(matrix, np.eye(rows, cols), atol=1e-10):
        return
    
    # 处理列向量情况
    if cols == 1:
        # 提取实部振幅
        vector_data = []
        for i in range(rows):
            element = matrix[i, 0]
            if np.iscomplex(element) and abs(element.imag) > 1e-10:
                vector_data.append(abs(element))
            else:
                vector_data.append(float(element.real))
        
        # 根据量子比特数选择分解策略
        use_recursive = (qubit_count & 1) == 0 or qubit_count < 4
        
        if use_recursive:
            # 递归Schmidt分解
            _recursive_schmidt_split(circuit, qubits, vector_data, threshold)
        else:
            # 直接态制备
            magnitude = np.linalg.norm(vector_data)
            if magnitude > 1e-10:
                normalized = [x / magnitude for x in vector_data]
                _prepare_quantum_state(circuit, qubits, normalized)


def _recursive_schmidt_split(circuit: Circuit, qubits: List[int], 
                             amplitudes: List[float], threshold: float):
    """
    递归Schmidt分割算法
    将量子态按Schmidt分解递归处理
    """
    working_data = list(amplitudes)
    
    # 确定所需量子比特
    required_qubits = []
    needed = ceil(log2(len(working_data))) if len(working_data) > 1 else 1
    
    for idx in range(min(needed, len(qubits))):
        required_qubits.append(qubits[idx])
    
    # 单比特基础情况
    if len(required_qubits) == 1:
        value = working_data[0]
        clamped = max(-1.0, min(1.0, value))
        
        if value < 0:
            rotation = 2 * pi - 2 * acos(clamped)
        else:
            rotation = 2 * acos(clamped)
        
        circuit.ry(rotation, required_qubits[0])
        return
    
    # 填充到2的幂次
    target_size = 1 << len(required_qubits)
    while len(working_data) < target_size:
        working_data.append(0.0)
    
    # 计算矩阵维度
    total_qubits = int(log2(len(working_data)))
    parity = total_qubits % 2
    dim_row = 1 << (total_qubits >> 1)
    dim_col = 1 << ((total_qubits >> 1) + parity)
    
    # 重塑为矩阵
    reshaped = np.zeros((dim_row, dim_col), dtype=float)
    idx = 0
    for r in range(dim_row):
        for c in range(dim_col):
            reshaped[r, c] = working_data[idx]
            idx += 1
    
    # SVD分解
    left_unitary, singular_vals, right_unitary_h = np.linalg.svd(
        reshaped, full_matrices=True
    )
    
    # 确定有效秩
    rank = 1
    if len(singular_vals) > 0:
        max_singular = singular_vals[0]
        while rank < len(singular_vals) and singular_vals[rank] >= max_singular * threshold:
            rank += 1
    
    # 截断矩阵
    truncated_singular = singular_vals[:rank].copy()
    left_truncated = left_unitary[:, :rank]
    right_truncated = right_unitary_h[:rank, :]
    
    # 分配量子比特到两个子系统
    subsystem_a = []
    subsystem_b = []
    
    split_point = floor(total_qubits / 2 + parity)
    for i in range(len(required_qubits)):
        if i < split_point:
            subsystem_a.append(required_qubits[i])
        else:
            subsystem_b.append(required_qubits[i])
    
    # 处理奇异值向量
    singular_bits = int(log2(rank)) if rank > 1 else 0
    
    if singular_bits > 0:
        singular_qubits = subsystem_b[:singular_bits]
        
        # 归一化奇异值
        singular_norm = np.linalg.norm(truncated_singular)
        if singular_norm > 1e-10:
            truncated_singular = truncated_singular / singular_norm
        
        # 递归处理奇异值
        _handle_vector_decomposition(
            circuit, singular_qubits, 
            truncated_singular.reshape(-1, 1), threshold
        )
    
    # 添加纠缠门
    for i in range(singular_bits):
        circuit.cx(subsystem_b[i], subsystem_a[i])
    
    # 递归处理左右酉矩阵
    _handle_vector_decomposition(circuit, subsystem_b, left_truncated, threshold)
    _handle_vector_decomposition(circuit, subsystem_a, right_truncated.T, threshold)


def schmidt_encode(q_size: int, data: List[float], cutoff: float = 0.0, 
                   add_measure: bool = False) -> SchmidtEncodeResult:
    """
    Schmidt编码实现
    
    基于量子态制备和Schmidt分解理论实现数据编码
    采用迭代约简策略将经典数据编码到量子态
    
    Args:
        q_size: 量子比特总数
        data: 输入数据向量（需要归一化）
        cutoff: Schmidt分解的截断阈值（默认0.0）
        add_measure: 是否添加测量操作（默认False）
    
    Returns:
        SchmidtEncodeResult: 编码结果，包含量子电路和输出比特索引
    """
    input_data = list(data)
    
    # 验证归一化
    data_norm = np.linalg.norm(input_data)
    if not np.isclose(data_norm, 1.0, atol=1e-13):
        raise ValueError('Data is not normalized')
    
    # 验证数据大小
    if len(input_data) > (1 << q_size):
        raise ValueError('Schmidt_encode parameter error.')
    
    # 计算有效量子比特数
    effective_qubits = int(log2(len(input_data))) if len(input_data) > 1 else 1
    
    # 使用反向索引（遵循C++实现的量子比特顺序）
    reversed_indices = [q_size - 1 - i for i in range(q_size)]
    
    # 初始化量子电路
    qc = Circuit(q_size)
    
    # 执行量子态制备
    active_qubits = reversed_indices[:effective_qubits]
    _prepare_quantum_state(qc, active_qubits, input_data)
    
    # 确定输出比特（反向顺序）
    output_bits = [reversed_indices[i] for i in range(effective_qubits - 1, -1, -1)]
    
    # 可选：添加测量
    if add_measure:
        for idx, qubit in enumerate(output_bits):
            qc.measure(qubit, idx)
    
    return SchmidtEncodeResult(qc, output_bits)
