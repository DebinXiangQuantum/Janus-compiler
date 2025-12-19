"""
高效稀疏编码模块

使用 Janus 电路库实现高效稀疏状态编码，逻辑同步自 Encode.cpp。
"""
from typing import Union, List, Dict, Tuple, Optional
import numpy as np
from math import log2, ceil, asin, pi

from janus.circuit import Circuit
from .utils import _build_state_dict
from janus.circuit.library import U3Gate,RXGate, HGate

def _maximizing_difference_bit_search(b_strings: List[str], dif_qubits: List[int]) -> Tuple[int, List[str], List[str]]:
    """寻找最大化集合差异的比特位"""
    bit_index = 0
    set_difference = -1
    t0_res, t1_res = [], []
    
    n_bits = len(b_strings[0])
    bit_search_space = [i for i in range(n_bits) if i not in dif_qubits]
            
    for bit in bit_search_space:
        temp_t0 = [s for s in b_strings if s[bit] == '0']
        temp_t1 = [s for s in b_strings if s[bit] == '1']
        
        if temp_t0 and temp_t1:
            temp_difference = abs(len(temp_t0) - len(temp_t1))
            if set_difference == -1 or temp_difference > set_difference:
                t0_res, t1_res = temp_t0, temp_t1
                bit_index = bit
                set_difference = temp_difference
                
    return bit_index, t0_res, t1_res


def _build_bit_string_set(b_strings: List[str], bitstr1: str, dif_qubits: List[int], dif_values: List[int]) -> List[str]:
    """构建满足特定比特值条件的字符串集合"""
    bit_string_set = []
    for b_string in b_strings:
        if b_string == bitstr1:
            continue
        
        match = True
        for i, qubit_idx in enumerate(dif_qubits):
            if int(b_string[qubit_idx]) != dif_values[i]:
                match = False
                break
        
        if match:
            bit_string_set.append(b_string)
            
    return bit_string_set


def _bit_string_search(b_strings: List[str], dif_qubits: List[int], dif_values: List[int]) -> List[str]:
    """搜索满足条件的单一比特字符串"""
    temp_strings = b_strings[:]
    while len(temp_strings) > 1:
        bit, t0, t1 = _maximizing_difference_bit_search(temp_strings, dif_qubits)
        if bit not in dif_qubits:
            dif_qubits.append(bit)
        
        if len(t0) < len(t1):
            dif_values.append(0)
            temp_strings = t0
        else:
            dif_values.append(1)
            temp_strings = t1
    return temp_strings


def _search_bit_strings_for_merging(state: Dict[str, complex]) -> Tuple[str, str, int, List[int]]:
    """寻找待合并的两个比特字符串及差异位"""
    b_strings = list(state.keys())
    dif_qubits = []
    dif_values = []
    
    if len(b_strings) == 2:
        bit, t0, t1 = _maximizing_difference_bit_search(b_strings, [])
        return t1[0], t0[0], bit, []
    else:
        # 寻找第一个字符串
        res1 = _bit_string_search(b_strings, dif_qubits, dif_values)
        bitstr1 = res1[0]
        
        # 弹出最后一个差异位，用于后续寻找第二个字符串
        dif_qubit = dif_qubits.pop()
        dif_values.pop()
        
        # 寻找第二个字符串
        b_strings2 = _build_bit_string_set(b_strings, bitstr1, dif_qubits, dif_values)
        res2 = _bit_string_search(b_strings2, dif_qubits, dif_values)
        bitstr2 = res2[0]
        
        return bitstr1, bitstr2, dif_qubit, dif_qubits


def _apply_x_to_bit_string(b_string: str, qubit_idx: int) -> str:
    """对比特字符串应用 X 门效果"""
    s_list = list(b_string)
    s_list[qubit_idx] = '1' if s_list[qubit_idx] == '0' else '0'
    return "".join(s_list)


def _apply_cx_to_bit_string(b_string: str, control: int, target: int) -> str:
    """对比特字符串应用 CX 门效果"""
    if b_string[control] == '1':
        return _apply_x_to_bit_string(b_string, target)
    return b_string


def _update_state_dict(state: Dict[str, complex], operation: str, 
                       qubit_idx: Optional[int] = None, 
                       control: Optional[int] = None, 
                       target: Optional[int] = None,
                       merge_strings: Optional[Tuple[str, str]] = None) -> Dict[str, complex]:
    """更新状态字典以反映门操作或合并"""
    new_state = {}
    if operation == "merge":
        if merge_strings:
            s1, s2 = merge_strings
            amp1, amp2 = state[s1], state[s2]
            norm = (abs(amp1)**2 + abs(amp2)**2)**0.5
            new_state = state.copy()
            del new_state[s2]
            new_state[s1] = complex(norm)
    elif operation == "x":
        for k, v in state.items():
            new_state[_apply_x_to_bit_string(k, qubit_idx)] = v
    elif operation == "cx":
        for k, v in state.items():
            new_state[_apply_cx_to_bit_string(k, control, target)] = v
    return new_state


def _compute_angles(amp1: complex, amp2: complex) -> Tuple[float, float, float]:
    """计算 U3 旋转角度"""
    norm = (abs(amp1)**2 + abs(amp2)**2)**0.5
    if norm < 1e-14:
        return (0.0, 0.0, 0.0)
    
    # 逻辑同步自 Encode.cpp:2266
    theta = 2 * asin(max(0.0, min(1.0, abs(amp2 / norm))))
    
    # 使用 np.angle 获取相位
    phi = -np.angle(amp2 / norm)
    lam = -np.angle(amp1 / norm) - phi
    
    return (float(theta), float(phi), float(lam))


def _merging_procedure(state: Dict[str, complex], circuit: Circuit, q_indices: List[int]) -> Dict[str, complex]:
    """核心合并程序，逻辑同步自 Encode.cpp:2286"""
    # 1. 搜索待合并的比特串
    bitstr1, bitstr2, dif, dif_qubits = _search_bit_strings_for_merging(state)
    
    # 2. 预处理 (同步自 _preprocess_states_for_merging)
    # 确保 bitstr1 在 dif 位为 '1'
    if bitstr1[dif] != '1':
        circuit.x(q_indices[dif])
        bitstr1 = _apply_x_to_bit_string(bitstr1, dif)
        bitstr2 = _apply_x_to_bit_string(bitstr2, dif)
        state = _update_state_dict(state, "x", qubit_idx=dif)
        
    # 使两个比特串在除 dif 以外的位上相同 (同步自 _equalize_bit_string_states)
    for i in range(len(bitstr1)):
        if i != dif and bitstr1[i] != bitstr2[i]:
            circuit.cx(q_indices[dif], q_indices[i])
            bitstr1 = _apply_cx_to_bit_string(bitstr1, dif, i)
            bitstr2 = _apply_cx_to_bit_string(bitstr2, dif, i)
            state = _update_state_dict(state, "cx", control=dif, target=i)
            
    # 将 bitstr2 在 dif_qubits 位上设为 '1' 以满足控制条件 (同步自 _apply_not_gates_to_qubit_index_list)
    for b_idx in dif_qubits:
        if bitstr2[b_idx] != '1':
            circuit.x(q_indices[b_idx])
            bitstr1 = _apply_x_to_bit_string(bitstr1, b_idx)
            bitstr2 = _apply_x_to_bit_string(bitstr2, b_idx)
            state = _update_state_dict(state, "x", qubit_idx=b_idx)
            
    # 3. 计算并应用受控旋转
    angles = _compute_angles(state[bitstr1], state[bitstr2])
    control_qubits = [q_indices[i] for i in dif_qubits]
    
    if not control_qubits:
        circuit.u3(*angles, q_indices[dif])
    else:
        # 使用 Janus 的受控门
        circuit.gate(U3Gate(*angles), q_indices[dif]).control(control_qubits)
        
    # 4. 更新状态字典
    state = _update_state_dict(state, "merge", merge_strings=(bitstr1, bitstr2))
    return state


def efficient_sparse(q_size: int, data: Union[List[float], List[complex], Dict[str, Union[float, complex]]]) -> Circuit:
    """
    高效稀疏编码 (Efficient Sparse Encoding)
    
    通过一系列合并操作将状态简化为基态，然后对整个电路求逆。
    """
    # 输入转换
    if isinstance(data, (list, np.ndarray)):
        state = _build_state_dict(data)
    elif isinstance(data, dict):
        state = {k: complex(v) for k, v in data.items()}
    else:
        raise TypeError("输入数据必须是列表、数组或字典")
        
    if not state:
        raise ValueError("输入数据不能为空")
        
    # 验证归一化
    tmp_sum = sum(abs(amp)**2 for amp in state.values())
    if abs(1.0 - tmp_sum) > 1e-13:
        if tmp_sum < 1e-13:
            raise ValueError("输入向量为零")
        # 自动归一化
        factor = np.sqrt(tmp_sum)
        state = {k: v / factor for k, v in state.items()}
        
    first_key = next(iter(state.keys()))
    n_qubits = len(first_key)
    if n_qubits > q_size:
        raise ValueError(f"需要 {n_qubits} 个量子比特，但只有 {q_size} 个可用")
        
    # 这里的 q_indices 对应 C++ 中的 reverse_q
    # Encode.cpp:1836 reverse_q[i] = q[q.size()-1-i]
    q_indices = [q_size - 1 - i for i in range(q_size)]
    
    circuit = Circuit(q_size)
    
    # 循环合并直到只剩一个状态
    current_state = state.copy()
    while len(current_state) > 1:
        current_state = _merging_procedure(current_state, circuit, q_indices)
        
    # 处理最后一个基态对应的 X 门
    final_bitstr = next(iter(current_state.keys()))
    for i, bit in enumerate(final_bitstr):
        if bit == '1':
            circuit.x(q_indices[i])
            
    # 全局求逆得到制备电路
    return circuit.inverse()
