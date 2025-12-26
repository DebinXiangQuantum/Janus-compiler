#!/usr/bin/env python3
"""
大规模量子门分解测试脚本
"""

import argparse
import json
import sys
import os
from pathlib import Path
import numpy as np
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from janus.decompose.decompose_kak import decompose_two_qubit_gate
from janus.decompose.decompose_controlled_gate import decompose_controlled_gate
from janus.decompose.convert_circuit_to_instruction_set import convert_circuit_to_instruction_set
from janus.circuit import Circuit
from janus.circuit.library.standard_gates import (
    HGate, XGate, YGate, ZGate, 
    RXGate, RYGate, RZGate, UGate,
    CXGate, CZGate, SwapGate,
    RXXGate, RYYGate, RZZGate,
    SGate, TGate  # 添加缺失的门类
)


def parse_gate_from_json_with_control_info(gate_info):
    """从JSON数据解析单个门，并返回是否是多受控门的信息"""
    if isinstance(gate_info, dict):
        gate_name = gate_info.get('name', '').lower()
        qubits = gate_info.get('qubits', [])
        params = gate_info.get('params', [])
        controls = gate_info.get('controls', [])
    elif isinstance(gate_info, list):
        gate_name = gate_info[0].lower()
        qubits = gate_info[1] if len(gate_info) > 1 else []
        params = gate_info[2] if len(gate_info) > 2 else []
        controls = []  # 列表格式中不包含controls字段
    else:
        raise ValueError(f"Invalid gate format: {gate_info}")
    
    # 如果有controls字段，将控制比特添加到qubits前面
    if controls:
        # 多受控门：将控制比特放在前面，目标比特放在后面
        all_qubits = controls + qubits
        # 根据控制比特数量创建门名称
        num_controls = len(controls)
        if gate_name == 'x':
            gate_name = 'c' * num_controls + 'x'  # 如: ccx, cccx等
        elif gate_name == 'y':
            gate_name = 'c' * num_controls + 'y'
        elif gate_name == 'z':
            gate_name = 'c' * num_controls + 'z'
        elif gate_name == 'rx':
            gate_name = 'c' * num_controls + 'rx'
        elif gate_name == 'ry':
            gate_name = 'c' * num_controls + 'ry'
        elif gate_name == 'rz':
            gate_name = 'c' * num_controls + 'rz'
        else:
            # 对于其他门类型，我们暂时不处理多控制版本
            all_qubits = qubits  # 只使用原始qubits
    else:
        all_qubits = qubits
    
    # 检查是否是多受控门
    is_multi_controlled = (len(controls) > 1) or (gate_name.startswith('c') and len(gate_name) > 2 and gate_name.lstrip('c') in ['x', 'y', 'z', 'rx', 'ry', 'rz'])
    
    # 根据门名称创建对应的门对象
    gate_class_map = {
        'h': HGate,
        'x': XGate,
        'y': YGate,
        'z': ZGate,
        's': SGate,
        't': TGate,  # 添加T门支持
        'rx': RXGate,
        'ry': RYGate,
        'rz': RZGate,
        'u': UGate,
        'cx': CXGate,
        'cz': CZGate,
        'swap': SwapGate,
        'rxx': RXXGate,
        'ryy': RYYGate,
        'rzz': RZZGate
    }
    
    # 如果门名是多受控门（以多个'c'开头），我们需要特殊处理
    if gate_name.startswith('c') and len(gate_name) > 2:
        # 多受控门，我们创建目标门（去掉开头的'c'）
        target_gate_name = gate_name.lstrip('c')
        if target_gate_name in gate_class_map:
            target_gate_class = gate_class_map[target_gate_name]
            if target_gate_name in ['rx', 'ry', 'rz'] and params:
                gate = target_gate_class(params[0])
            else:
                gate = target_gate_class()
        else:
            raise ValueError(f"不支持的多受控门目标类型: {target_gate_name}")
    elif gate_name not in gate_class_map:
        raise ValueError(f"不支持的门类型: {gate_name}")
    else:
        gate_class = gate_class_map[gate_name]
        
        # 创建门实例
        if gate_name in ['h', 'x', 'y', 'z', 's', 't', 'swap']:  # 无参数门
            gate = gate_class()
        elif gate_name in ['rx', 'ry', 'rz']:  # 单参数门
            angle = params[0] if params else 0
            gate = gate_class(angle)
        elif gate_name == 'u':  # 三参数门
            theta = params[0] if len(params) > 0 else 0
            phi = params[1] if len(params) > 1 else 0
            lam = params[2] if len(params) > 2 else 0
            gate = gate_class(theta, phi, lam)
        elif gate_name in ['cx', 'cz']:  # 两比特门
            gate = gate_class()
        elif gate_name in ['rxx', 'ryy', 'rzz']:  # 两比特双参数门
            angle = params[0] if params else 0
            gate = gate_class(angle)
        else:
            raise ValueError(f"不支持的门类型: {gate_name}")
    
    return gate, all_qubits, is_multi_controlled


def parse_gate_from_json(gate_info):
    """解析JSON格式的门信息，返回门对象和量子比特列表"""
    # 定义门类映射
    gate_class_map = {
        'h': HGate,
        'x': XGate,
        'y': YGate,
        'z': ZGate,
        's': SGate,
        't': TGate,  # 添加T门支持
        'rx': RXGate,
        'ry': RYGate,
        'rz': RZGate,
        'u': UGate,
        'cx': CXGate,
        'cz': CZGate,
        'swap': SwapGate,
        'rxx': RXXGate,
        'ryy': RYYGate,
        'rzz': RZZGate
    }
    
    if isinstance(gate_info, dict):
        gate_name = gate_info.get('name', '').lower()
        all_qubits = gate_info.get('qubits', [])
        params = gate_info.get('params', [])
        controls = gate_info.get('controls', [])  # 提取controls字段
    else:
        gate_name = gate_info[0].lower()
        all_qubits = gate_info[1] if len(gate_info) > 1 else []
        params = gate_info[2] if len(gate_info) > 2 else []
        controls = []  # 列表格式中不包含controls字段

    # 检查是否是多受控门（有controls字段且控制比特数大于1）
    if len(controls) > 1:
        # 对于多受控门，我们不在这里处理，而是在build_circuit_from_data中处理
        # 因为多受控门需要分解，这里直接返回信息
        raise ValueError(f"多受控门应由build_circuit_from_data处理: {gate_name} with controls {controls}")
    
    # 检查是否是多受控门（以多个'c'开头）
    if gate_name.startswith('c') and len(gate_name) > 2:
        # 多受控门，我们创建目标门（去掉开头的'c'）
        target_gate_name = gate_name.lstrip('c')
        if target_gate_name in gate_class_map:
            target_gate_class = gate_class_map[target_gate_name]
            if target_gate_name in ['rx', 'ry', 'rz'] and params:
                gate = target_gate_class(params[0])
            else:
                gate = target_gate_class()
        else:
            raise ValueError(f"不支持的多受控门目标类型: {target_gate_name}")
    elif gate_name not in gate_class_map:
        raise ValueError(f"不支持的门类型: {gate_name}")
    else:
        gate_class = gate_class_map[gate_name]
        
        # 创建门实例
        if gate_name in ['h', 'x', 'y', 'z', 's', 't', 'swap']:  # 无参数门
            gate = gate_class()
        elif gate_name in ['rx', 'ry', 'rz']:  # 单参数门
            angle = params[0] if params else 0
            gate = gate_class(angle)
        elif gate_name == 'u':  # 三参数门
            theta = params[0] if len(params) > 0 else 0
            phi = params[1] if len(params) > 1 else 0
            lam = params[2] if len(params) > 2 else 0
            gate = gate_class(theta, phi, lam)
        elif gate_name in ['cx', 'cz']:  # 两比特门
            gate = gate_class()
        elif gate_name in ['rxx', 'ryy', 'rzz']:  # 两比特双参数门
            angle = params[0] if params else 0
            gate = gate_class(angle)
        else:
            raise ValueError(f"不支持的门类型: {gate_name}")
    
    return gate, all_qubits


def parse_circuit_from_json(file_path):
    """从JSON文件解析电路信息"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 根据文件格式解析电路
    circuit_data = None
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        # 格式为 [[gate_info, ...]]
        circuit_data = data[0]
    elif isinstance(data, dict) and 'circuit' in data:
        # 格式为 {'circuit': [gate_info, ...]}
        circuit_data = data['circuit']
    else:
        circuit_data = data  # 直接是门的列表
    
    return circuit_data


def build_circuit_from_data(circuit_data, n_qubits):
    """根据电路数据构建Circuit对象，对多受控门进行预处理"""
    circuit = Circuit(n_qubits=n_qubits)
    
    for gate_info in circuit_data:
        if isinstance(gate_info, dict):
            gate_name = gate_info.get('name', '').lower()
            qubits = gate_info.get('qubits', [])
            params = gate_info.get('params', [])
            controls = gate_info.get('controls', [])
        else:
            gate_name = gate_info[0].lower()
            qubits = gate_info[1] if len(gate_info) > 1 else []
            params = gate_info[2] if len(gate_info) > 2 else []
            controls = []  # 列表格式中不包含controls字段
        
        # 检查是否是多受控门（有controls字段且控制比特数大于1）
        is_multi_controlled = len(controls) > 1
        
        if is_multi_controlled:
            # 多受控门，需要进行分解
            # 创建目标门对象
            target_gate_class_map = {
                'x': XGate,
                'y': YGate,
                'z': ZGate,
                'rx': RXGate,
                'ry': RYGate,
                'rz': RZGate,
                'h': HGate,
                's': SGate,
                't': TGate,
                'u': UGate,
                'cx': CXGate,  # 添加CX门支持（虽然它本身不是多受控门，但以防万一）
                'cz': CZGate
            }
            
            if gate_name in target_gate_class_map:
                target_gate_class = target_gate_class_map[gate_name]
                if gate_name in ['rx', 'ry', 'rz'] and params:
                    target_gate = target_gate_class(params[0])
                elif gate_name == 'u' and params and len(params) >= 3:
                    target_gate = target_gate_class(params[0], params[1], params[2])
                elif gate_name == 'u':
                    target_gate = target_gate_class(0, 0, 0)  # 默认参数
                elif gate_name in ['cx', 'cz']:
                    target_gate = target_gate_class()  # 无参数门
                else:
                    target_gate = target_gate_class()
            else:
                raise ValueError(f"不支持的多受控门类型: {gate_name}")
            
            # 获取控制比特数
            num_controls = len(controls)
            
            # 使用decompose_controlled_gate函数进行分解
            decomposed_subcircuit = decompose_controlled_gate(target_gate, num_controls)
            
            # 需要映射分解后电路的量子比特到原始电路的量子比特
            # 分解后的电路包含num_controls个控制比特和1个目标比特
            qubit_mapping = {}
            for i in range(num_controls):
                qubit_mapping[i] = controls[i]
            # 目标比特是最后一个量子比特
            qubit_mapping[num_controls] = qubits[0]  # 假设只有一个目标比特
            
            # 将分解后的门添加到电路中
            for sub_inst in decomposed_subcircuit.instructions:
                mapped_qubits = [qubit_mapping[qubit_idx] for qubit_idx in sub_inst.qubits]
                circuit.append(sub_inst.operation, mapped_qubits)
        else:
            # 普通门，使用原来的解析方法
            gate, qubits = parse_gate_from_json(gate_info)
            circuit.append(gate, qubits)
    
    return circuit


def detect_multi_controlled_gates(circuit):
    """检测电路中的多受控门"""
    multi_controlled_gates = []
    for inst in circuit.instructions:
        gate_name = inst.name.lower()
        # 检查是否是多受控门（以多个'c'开头）
        if gate_name.startswith('c') and len(gate_name) > 2:  # 至少是ccx这样的双受控门
            # 计算控制比特数：统计开头连续的'c'
            num_ctrl = 0
            for char in gate_name:
                if char == 'c':
                    num_ctrl += 1
                else:
                    break
            if num_ctrl > 1:  # 多于1个控制比特
                multi_controlled_gates.append((inst, num_ctrl))
        # 检查是否有控制比特字段（在某些表示中，控制比特和目标比特可能以不同方式存储）
        elif hasattr(inst, 'controls') and len(inst.controls) > 1:
            # 如果指令对象有controls属性且控制比特数大于1
            num_ctrl = len(inst.controls)
            multi_controlled_gates.append((inst, num_ctrl))
    return multi_controlled_gates


def decompose_multi_controlled_gates(circuit):
    """分解电路中的多受控门"""
    # 检测多受控门
    multi_controlled_gates = detect_multi_controlled_gates(circuit)
    
    if not multi_controlled_gates:
        print("No multi-controlled gates found in the circuit.")
        return circuit
    
    print(f"Found {len(multi_controlled_gates)} multi-controlled gates to decompose.")
    
    # 创建新电路，用于存储分解后的门
    decomposed_circuit = Circuit(n_qubits=circuit.n_qubits)
    
    for inst in circuit.instructions:
        gate_name = inst.name.lower()
        
        # 检查是否是多受控门
        if gate_name.startswith('c') and len(gate_name) > 2:
            num_ctrl = 0
            for char in gate_name:
                if char == 'c':
                    num_ctrl += 1
                else:
                    break
            
            if num_ctrl > 1:  # 多于1个控制比特
                print(f"Decomposing multi-controlled gate: {gate_name} with {num_ctrl} control qubits on qubits {inst.qubits}")
                
                # 获取目标门类型（去掉开头的'c'字符）
                target_gate_name = gate_name[num_ctrl:]  # 例如，'ccx' -> 'x'
                
                # 根据目标门类型创建对应的门
                target_gates_map = {
                    'x': XGate(),
                    'y': YGate(),
                    'z': ZGate(),
                    'rx': RXGate(0),  # 需要传入参数
                    'ry': RYGate(0),
                    'rz': RZGate(0),
                    'h': HGate(),
                    's': SGate(),
                    't': TGate()
                }
                
                # 如果门需要参数，获取参数
                if target_gate_name in ['rx', 'ry', 'rz'] and inst.params:
                    target_gate = target_gates_map[target_gate_name].__class__(inst.params[0])
                elif target_gate_name in target_gates_map:
                    target_gate = target_gates_map[target_gate_name]
                else:
                    # 如果目标门类型不在预定义列表中，尝试使用通用方法
                    # 这里假设我们只处理已知的门类型
                    print(f"Warning: Unknown target gate type {target_gate_name}, skipping decomposition")
                    decomposed_circuit.append(inst.operation, inst.qubits)
                    continue
                
                # 获取控制比特和目标比特
                control_qubits = inst.qubits[:num_ctrl]
                target_qubits = inst.qubits[num_ctrl:]  # 可能有多个目标比特，但通常只有一个
                
                # 调用decompose_controlled_gate进行分解
                decomposed_subcircuit = decompose_controlled_gate(target_gate, num_ctrl)
                
                # 需要映射分解后电路的量子比特到原始电路的量子比特
                # 分解后的电路包含num_ctrl个控制比特和1个目标比特
                # 我们需要将这些映射到原始电路中的对应比特
                qubit_mapping = {}
                for i in range(num_ctrl):
                    qubit_mapping[i] = control_qubits[i]
                for i in range(len(target_qubits)):
                    qubit_mapping[num_ctrl + i] = target_qubits[i]
                
                # 将分解后的门添加到新电路中
                for sub_inst in decomposed_subcircuit.instructions:
                    mapped_qubits = [qubit_mapping[qubit_idx] for qubit_idx in sub_inst.qubits]
                    decomposed_circuit.append(sub_inst.operation, mapped_qubits)
            else:
                # 不是多受控门，直接添加
                decomposed_circuit.append(inst.operation, inst.qubits)
        else:
            # 不是受控门，直接添加
            decomposed_circuit.append(inst.operation, inst.qubits)
    
    return decomposed_circuit


def decompose_circuit_gates(circuit):
    """分解电路中的两比特门"""
    decomposed_circuit = Circuit(n_qubits=circuit.n_qubits)
    
    for inst in circuit.instructions:
        gate = inst.operation
        qubits = inst.qubits
        
        # 检查是否为两比特门
        if len(qubits) == 2:
            print(f"Decomposing two-qubit gate: {gate.name} on qubits {qubits}")
            
            # 使用KAK分解将两比特门分解为CNOT门
            decomposed_subcircuit = decompose_two_qubit_gate(gate, basis_gate='cx')
            
            # 将分解后的子电路中的门添加到新电路中，映射到原始量子比特
            for sub_inst in decomposed_subcircuit.instructions:
                sub_qubits = sub_inst.qubits
                # 映射到原始量子比特
                mapped_qubits = [qubits[i] for i in sub_qubits]
                decomposed_circuit.append(sub_inst.operation, mapped_qubits)
        else:
            # 单比特门直接添加
            decomposed_circuit.append(gate, qubits)
    
    return decomposed_circuit


def save_circuit_to_json(circuit, output_file):
    """将电路的to_dict_list结果保存到JSON文件"""
    circuit_dict_list = circuit.to_dict_list()
    
    # 准备要保存的数据
    data = {
        'circuit': circuit_dict_list,
        'num_gates': len(circuit_dict_list),
        'qubits_used': circuit.n_qubits,
        'qubit_count': circuit.n_qubits  # 明确标示量子比特总数
    }
    
    # 保存到JSON文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Circuit representation saved to {output_file}")
    return circuit_dict_list


def main():
    parser = argparse.ArgumentParser(description='Test large-scale quantum gate decomposition')
    parser.add_argument('--file', type=str, required=True, help='JSON file containing the circuit')
    parser.add_argument('--output', type=str, help='Output JSON file for decomposed circuit representation')
    parser.add_argument('--instruction-set', type=str, help='Target instruction set for circuit conversion (comma-separated gate names)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.file):
        # 尝试在benchmark目录下查找
        benchmark_file = os.path.join(os.path.dirname(__file__), 'benchmark', os.path.basename(args.file))
        if os.path.exists(benchmark_file):
            args.file = benchmark_file
        else:
            raise FileNotFoundError(f"File {args.file} not found")
    
    # 如果没有指定输出文件，则生成默认输出文件名
    if not args.output:
        # 获取原始文件名（不含扩展名）和扩展名
        file_path_obj = Path(args.file)
        file_stem = file_path_obj.stem  # 文件名不含扩展名
        file_suffix = file_path_obj.suffix  # 包括点号的扩展名
        
        # 构造默认输出文件名：原始文件名_decomposed.json
        default_output_filename = f"{file_stem}_decomposed{file_suffix}"
        
        # 输出到test/decompose/result目录
        output_dir = os.path.join(os.path.dirname(__file__), 'result')
        args.output = os.path.join(output_dir, default_output_filename)
    
    # 解析电路信息
    circuit_data = parse_circuit_from_json(args.file)
    
    # 确定电路的量子比特数
    max_qubit = 0
    for gate_info in circuit_data:
        qubits = gate_info['qubits']
        max_qubit = max(max_qubit, max(qubits))
    n_qubits = max_qubit + 1  # 量子比特数是最大索引+1
    
    print(f"Original circuit has {len(circuit_data)} gates and {n_qubits} qubits")
    
    # 构建原始电路
    original_circuit = build_circuit_from_data(circuit_data, n_qubits)
    
    # 首先分解多受控门
    print("Checking for multi-controlled gates...")
    circuit_after_multi_control_decomp = decompose_multi_controlled_gates(original_circuit)
    
    # 如果指定了指令集，则使用convert_circuit_to_instruction_set进行转换
    if args.instruction_set:
        instruction_set = [gate.strip() for gate in args.instruction_set.split(',')]
        print(f"Converting circuit to instruction set: {instruction_set}")
        start_time = time.time()
        converted_circuit = convert_circuit_to_instruction_set(circuit_after_multi_control_decomp, instruction_set)
        end_time = time.time()
        print(f"Circuit conversion completed in {end_time - start_time:.2f} seconds")
        
        decomposed_circuit = converted_circuit
    else:
        # 否则执行原有的两比特门分解
        print("Starting gate decomposition...")
        start_time = time.time()
        
        decomposed_circuit = decompose_circuit_gates(circuit_after_multi_control_decomp)
        
        end_time = time.time()
        print(f"Decomposition completed in {end_time - start_time:.2f} seconds")
    
    print(f"Number of gates in decomposed circuit: {len(decomposed_circuit.instructions)}")
    print(f"Number of qubits in circuit: {decomposed_circuit.n_qubits}")
    
    # 打印部分转换后的量子电路（仅前20个门，避免输出过多）
    print("\nFirst 20 gates of decomposed quantum circuit:")
    for i, inst in enumerate(decomposed_circuit.instructions[:20]):
        if i >= len(decomposed_circuit.instructions[:20]) - 1:
            if len(decomposed_circuit.instructions) > 20:
                print("  ...")
                print(f"  And {len(decomposed_circuit.instructions) - 20} more gates")
            break
        params_str = f" params={inst.params}" if inst.params else ""
        # 获取所有量子比特的索引
        if hasattr(inst.qubits[0], 'index'):
            qubit_indices = [q.index for q in inst.qubits]
        else:
            qubit_indices = inst.qubits
        qubit_info = ', '.join([f"qubit {idx}" for idx in qubit_indices])
        print(f"  {i+1}. {inst.name}{params_str} on {qubit_info}")
    
    # 使用to_dict_list方法获取电路的字典表示
    circuit_dict_list = decomposed_circuit.to_dict_list()
    print(f"\nCircuit representation (first 5 gates):")
    for i, gate_dict in enumerate(circuit_dict_list[:5]):
        print(f"  {i+1}. {gate_dict}")
    if len(circuit_dict_list) > 5:
        print(f"  ... and {len(circuit_dict_list) - 5} more gates")
    
    # 保存电路表示到JSON文件
    save_circuit_to_json(decomposed_circuit, args.output)
    
    print(f"\n✓ Large-scale circuit decomposition completed successfully!")
    print(f"  Original gates: {len(circuit_data)}")
    print(f"  Decomposed gates: {len(decomposed_circuit.instructions)}")
    print(f"  Qubits: {decomposed_circuit.n_qubits}")


if __name__ == "__main__":
    main()