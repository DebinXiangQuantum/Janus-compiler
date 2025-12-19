from __future__ import annotations
from typing import Optional, List, Union
import numpy as np
from janus.circuit import Circuit, Gate
from janus.circuit.converters import GATE_MAP
from janus.circuit.dag import circuit_to_dag
from janus.circuit.library.standard_gates import (
    HGate, XGate, YGate, ZGate, SGate, TGate,
    RXGate, RYGate, RZGate, UGate,
    CXGate, CZGate, CRZGate, SwapGate
)
from .exceptions import ParameterError, GateNotSupportedError

# 导入其他分解函数
from .decompose_one_qubit import OneQubitEulerDecomposer, decompose_one_qubit
from .decompose_two_qubit_gate import decompose_two_qubit_gate
from .decompose_controlled_gate import decompose_controlled_gate


def convert_circuit_to_instruction_set(
    circuit: Circuit,
    instruction_set: List[str],
    use_dag: bool = False,
) -> Circuit:
    """
    将电路转换为指定的指令集
    
    Args:
        circuit: Janus电路对象
        instruction_set: 目标指令集（门名称列表）
        use_dag: 是否返回DAGCircuit
        
    Returns:
        转换后的电路
    """
    if not isinstance(circuit, Circuit):
        raise ParameterError("Input must be a valid Janus Circuit object.")
    
    if not isinstance(instruction_set, list) or not instruction_set:
        raise ParameterError("instruction_set must be a non-empty list of strings.")
    
    if not all(isinstance(gate, str) for gate in instruction_set):
        raise ParameterError("All elements in instruction_set must be strings.")
    
    # 检查指令集中的门是否都支持
    for gate_name in instruction_set:
        if gate_name not in GATE_MAP:
            raise GateNotSupportedError(f"Unsupported gate type: {gate_name}")
    
    # 创建新电路
    converted_circuit = Circuit(n_qubits=circuit.n_qubits)
    
    # 转换每个门
    for inst in circuit.instructions:
        gate_name = inst.name.lower()
        qubits = inst.qubits
        params = inst.params
        
        # 如果门已经在目标指令集中，直接添加
        if gate_name in instruction_set:
            gate_cls = GATE_MAP[gate_name]
            if gate_cls == UGate:
                # U门需要三个参数：theta, phi, lam
                if len(params) == 0:
                    # 默认参数
                    converted_circuit.append(gate_cls(0, 0, 0), qubits)
                elif len(params) == 1:
                    # 只有theta
                    converted_circuit.append(gate_cls(params[0], 0, 0), qubits)
                elif len(params) == 2:
                    # theta和phi
                    converted_circuit.append(gate_cls(params[0], params[1], 0), qubits)
                else:
                    # 完整参数
                    converted_circuit.append(gate_cls(params[0], params[1], params[2]), qubits)
            elif gate_cls in [RXGate, RYGate, RZGate]:
                # 旋转门需要一个角度参数
                if params:
                    converted_circuit.append(gate_cls(params[0]), qubits)
                else:
                    converted_circuit.append(gate_cls(0), qubits)
            else:
                # 其他门直接添加
                converted_circuit.append(gate_cls(params), qubits)
        else:
            # 分解实现
            if gate_name in ['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz', 'u']:
                # 单比特门，使用分解函数
                if 'u' in instruction_set:
                    # 使用OneQubitEulerDecomposer分解为U门
                    decomposer = OneQubitEulerDecomposer(basis="U", use_dag=False)
                    decomposed_circuit = decomposer(inst.operation)
                    
                    # 将分解后的门添加到目标电路中，注意保持原始比特位置
                    for decomposed_inst in decomposed_circuit.instructions:
                        converted_circuit.append(decomposed_inst.operation, qubits)
                elif all(g in instruction_set for g in ['rx', 'ry', 'rz']):
                    # 使用OneQubitEulerDecomposer分解为RX/RY/RZ组合
                    decomposer = OneQubitEulerDecomposer(basis="ZYZ", use_dag=False)
                    decomposed_circuit = decomposer(inst.operation)
                    
                    # 将分解后的门添加到目标电路中，注意保持原始比特位置
                    for decomposed_inst in decomposed_circuit.instructions:
                        converted_circuit.append(decomposed_inst.operation, qubits)
                else:
                    raise ValueError(f"Cannot decompose {gate_name} into target instruction set")
            elif gate_name == 'cz':
                # 两比特门: CZ分解
                if 'cx' in instruction_set and 'u' in instruction_set:
                    # CZ门分解: CX = H*CZ*H
                    # 所以 CZ = H*CX*H
                    control, target = qubits
                    # 目标比特上应用H门（转换为U门）
                    converted_circuit.append(UGate(np.pi/2, 0, np.pi), [target])
                    # 应用CX门
                    converted_circuit.append(CXGate(), [control, target])
                    # 目标比特上应用H门（转换为U门）
                    converted_circuit.append(UGate(np.pi/2, 0, np.pi), [target])
                else:
                    raise ValueError(f"Cannot decompose cz into target instruction set. Need 'cx' and 'u' gates.")
            elif gate_name == 'swap':
                # 两比特门: SWAP分解为三个CX门
                if 'cx' in instruction_set:
                    # 使用decompose_two_qubit_gate函数分解为CX门
                    decomposed_circuit = decompose_two_qubit_gate(inst.operation, basis_gate='cx', use_dag=False)
                    
                    # 将分解后的门添加到目标电路中，注意保持原始比特位置
                    qubit1, qubit2 = qubits
                    for decomposed_inst in decomposed_circuit.instructions:
                        # 映射分解电路的比特到原始比特
                        decomposed_qubits = decomposed_inst.qubits
                        if len(decomposed_qubits) == 2:
                            # 两比特门
                            mapped_qubits = [qubit1 if q == 0 else qubit2 for q in decomposed_qubits]
                        else:
                            # 单比特门
                            mapped_qubits = [qubit1 if q == 0 else qubit2 for q in decomposed_qubits]
                        converted_circuit.append(decomposed_inst.operation, mapped_qubits)
                else:
                    raise ValueError(f"Cannot decompose swap into target instruction set. Need 'cx' gate.")
            elif gate_name == 'crz':
                # 两比特门: CRZ分解为RZ和CX门
                if 'cx' in instruction_set and 'rz' in instruction_set:
                    control, target = qubits
                    theta = params[0] if params else 0
                    # CRZ门分解: CRZ(θ) = (I⊗H) * CNOT * (I⊗RZ(θ/2)) * CNOT * (I⊗RZ(-θ/2)) * (I⊗H)
                    if 'u' in instruction_set:
                        # H门转换为U门
                        converted_circuit.append(UGate(np.pi/2, 0, np.pi), [target])
                    else:
                        converted_circuit.append(HGate(), [target])
                    converted_circuit.append(CXGate(), [control, target])
                    converted_circuit.append(RZGate(theta/2), [target])
                    converted_circuit.append(CXGate(), [control, target])
                    converted_circuit.append(RZGate(-theta/2), [target])
                    if 'u' in instruction_set:
                        # H门转换为U门
                        converted_circuit.append(UGate(np.pi/2, 0, np.pi), [target])
                    else:
                        converted_circuit.append(HGate(), [target])
                else:
                    raise ValueError(f"Cannot decompose crz into target instruction set. Need 'cx' and 'rz' gates.")
            else:  # gate_name not in ['cz', 'swap', 'crz']
                # 检查是否是多受控门
                # 多受控门的名称通常以多个'c'开头，如'ccx'表示双受控X门
                if gate_name.startswith('c'):
                    # 计算控制比特数量
                    # 例如：'ccx'表示双受控X门，控制比特数为2
                    # 计算前缀'c'的数量
                    num_ctrl_qubits = len(gate_name) - len(gate_name.lstrip('c'))
                    
                    if num_ctrl_qubits > 1:
                        # 多受控门，使用decompose_controlled_gate函数
                        # 获取目标门名称（去掉'c'前缀）
                        target_gate_name = gate_name.lstrip('c')
                        
                        # 根据目标门名称创建目标门对象
                        if target_gate_name == 'x':
                            target_gate = XGate()
                        elif target_gate_name == 'y':
                            target_gate = YGate()
                        elif target_gate_name == 'z':
                            target_gate = ZGate()
                        elif target_gate_name == 'rx':
                            target_gate = RXGate(params[0])
                        elif target_gate_name == 'ry':
                            target_gate = RYGate(params[0])
                        elif target_gate_name == 'rz':
                            target_gate = RZGate(params[0])
                        else:
                            raise ValueError(f"Cannot decompose multi-controlled {target_gate_name} into target instruction set")
                        
                        # 分解多受控门
                        decomposed_circuit = decompose_controlled_gate(target_gate, num_ctrl_qubits=num_ctrl_qubits)
                        
                        # 递归地将分解后的电路转换为目标指令集
                        decomposed_circuit = convert_circuit_to_instruction_set(decomposed_circuit, instruction_set, use_dag=False)
                        
                        # 将分解后的门添加到目标电路中，注意保持原始比特位置
                        ctrl_qubits = qubits[:num_ctrl_qubits]
                        target_qubit = qubits[num_ctrl_qubits]
                        
                        for decomposed_inst in decomposed_circuit.instructions:
                            decomposed_qubits = decomposed_inst.qubits
                            
                            # 映射分解电路的比特到原始比特
                            mapped_qubits = []
                            for q in decomposed_qubits:
                                if q < num_ctrl_qubits:
                                    # 前num_ctrl_qubits个比特是控制比特
                                    mapped_qubits.append(ctrl_qubits[q])
                                elif q == num_ctrl_qubits:
                                    # 第num_ctrl_qubits个比特是目标比特
                                    mapped_qubits.append(target_qubit)
                                else:
                                    # 不应该有其他比特
                                    raise ValueError(f"Unexpected qubit index {q} in decomposed circuit")
                            
                            # 添加分解后的门
                            converted_circuit.append(decomposed_inst.operation, mapped_qubits)
                    else:
                        # 单受控门，已经在前面的逻辑中处理
                        raise ValueError(f"Cannot decompose {gate_name} into target instruction set")
                else:
                    # 尝试使用通用分解函数
                    if len(qubits) == 1:
                        # 单比特门
                        if 'u' in instruction_set:
                            decomposer = OneQubitEulerDecomposer(basis="U", use_dag=False)
                            decomposed_circuit = decomposer(inst.operation)
                            
                            # 将分解后的门添加到目标电路中
                            for decomposed_inst in decomposed_circuit.instructions:
                                converted_circuit.append(decomposed_inst.operation, qubits)
                        else:
                            raise ValueError(f"Cannot decompose {gate_name} into target instruction set")
                    elif len(qubits) == 2:
                        # 两比特门
                        if 'cx' in instruction_set:
                            decomposed_circuit = decompose_two_qubit_gate(inst.operation, basis_gate='cx', use_dag=False)
                            
                            # 将分解后的门添加到目标电路中，注意保持原始比特位置
                            qubit1, qubit2 = qubits
                            for decomposed_inst in decomposed_circuit.instructions:
                                # 映射分解电路的比特到原始比特
                                decomposed_qubits = decomposed_inst.qubits
                                if len(decomposed_qubits) == 2:
                                    # 两比特门
                                    mapped_qubits = [qubit1 if q == 0 else qubit2 for q in decomposed_qubits]
                                else:
                                    # 单比特门
                                    mapped_qubits = [qubit1 if q == 0 else qubit2 for q in decomposed_qubits]
                                converted_circuit.append(decomposed_inst.operation, mapped_qubits)
                        else:
                            raise ValueError(f"Cannot decompose {gate_name} into target instruction set")
                    else:
                        raise ValueError(f"Cannot decompose {gate_name} with {len(qubits)} qubits into target instruction set")
    
    if use_dag:
        return circuit_to_dag(converted_circuit)
    return converted_circuit