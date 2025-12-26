#!/usr/bin/env python3
"""
电路转换测试脚本
"""

import argparse
import json
import sys
import os
from pathlib import Path
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from janus.decompose.convert_circuit_to_instruction_set import convert_circuit_to_instruction_set
from janus.circuit import Circuit
from janus.circuit.io import load_circuit


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
    """根据电路数据构建Circuit对象"""
    circuit = Circuit(n_qubits=n_qubits)
    
    for gate_info in circuit_data:
        gate_name = gate_info['name']
        qubits = gate_info['qubits']
        params = gate_info.get('params', [])
        
        # 根据门名称和参数构建门并添加到电路
        if gate_name == 'h':
            from janus.circuit.library.standard_gates import HGate
            gate = HGate()
        elif gate_name == 'x':
            from janus.circuit.library.standard_gates import XGate
            gate = XGate()
        elif gate_name == 'y':
            from janus.circuit.library.standard_gates import YGate
            gate = YGate()
        elif gate_name == 'z':
            from janus.circuit.library.standard_gates import ZGate
            gate = ZGate()
        elif gate_name == 'rx':
            from janus.circuit.library.standard_gates import RXGate
            gate = RXGate(params[0] if params else 0)
        elif gate_name == 'ry':
            from janus.circuit.library.standard_gates import RYGate
            gate = RYGate(params[0] if params else 0)
        elif gate_name == 'rz':
            from janus.circuit.library.standard_gates import RZGate
            gate = RZGate(params[0] if params else 0)
        elif gate_name == 'u':
            from janus.circuit.library.standard_gates import UGate
            gate = UGate(params[0] if len(params) > 0 else 0, 
                        params[1] if len(params) > 1 else 0, 
                        params[2] if len(params) > 2 else 0)
        elif gate_name == 'cx':
            from janus.circuit.library.standard_gates import CXGate
            gate = CXGate()
        elif gate_name == 'cz':
            from janus.circuit.library.standard_gates import CZGate
            gate = CZGate()
        elif gate_name == 'swap':
            from janus.circuit.library.standard_gates import SwapGate
            gate = SwapGate()
        elif gate_name == 'rxx':
            from janus.circuit.library.standard_gates import RXXGate
            gate = RXXGate(params[0] if params else 0)
        elif gate_name == 'ryy':
            from janus.circuit.library.standard_gates import RYYGate
            gate = RYYGate(params[0] if params else 0)
        elif gate_name == 'rzz':
            from janus.circuit.library.standard_gates import RZZGate
            gate = RZZGate(params[0] if params else 0)
        elif gate_name == 'ccx':
            from janus.circuit.library.standard_gates import CCXGate
            gate = CCXGate()
        elif gate_name == 'ccz':
            from janus.circuit.library.standard_gates import CCZGate
            gate = CCZGate()
        elif gate_name == 'c3x' or gate_name == 'cccx':
            from janus.circuit.library.standard_gates import C3XGate
            gate = C3XGate()
        elif gate_name == 'c4x' or gate_name == 'ccccx':
            from janus.circuit.library.standard_gates import C4XGate
            gate = C4XGate()
        elif gate_name.startswith('mcx'):
            # 处理多控制X门 (MCXGate)
            from janus.circuit.library.standard_gates import MCXGate
            num_ctrl = len(gate_name) - len(gate_name.lstrip('mcx'))  # 计算控制位数量
            gate = MCXGate(num_ctrl)
        else:
            raise ValueError(f"不支持的门类型: {gate_name}")
        
        circuit.append(gate, qubits)
    
    return circuit


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test circuit conversion to specific instruction sets')
    parser.add_argument('--file', type=str, required=True, help='JSON file containing the circuit')
    parser.add_argument('--output', type=str, help='Output JSON file for converted circuit representation')
    parser.add_argument('--basis', type=str, required=True, help='Basis instruction set (0 for u3+cx, 1 for u+cx+cz)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.file):
        # 尝试在benchmark目录下查找
        benchmark_file = os.path.join(os.path.dirname(__file__), 'benchmark', os.path.basename(args.file))
        if os.path.exists(benchmark_file):
            args.file = benchmark_file
        else:
            # 尝试在banchmark目录下查找（处理拼写错误）
            banchmark_file = os.path.join(os.path.dirname(__file__), 'banchmark', os.path.basename(args.file))
            if os.path.exists(banchmark_file):
                args.file = banchmark_file
            else:
                raise FileNotFoundError(f"File {args.file} not found")
    
    # 如果没有指定输出文件，则生成默认输出文件名
    if not args.output:
        # 获取原始文件名（不含扩展名）和扩展名
        file_path_obj = Path(args.file)
        file_stem = file_path_obj.stem  # 文件名不含扩展名
        file_suffix = file_path_obj.suffix  # 包括点号的扩展名
        
        # 构造默认输出文件名：原始文件名_converted.json
        if args.basis == '0':
            basis_name = 'u3cx'
        elif args.basis == '1':
            basis_name = 'ucxcz'
        else:
            basis_name = f'basis{args.basis}'
            
        default_output_filename = f"{file_stem}_converted_{basis_name}{file_suffix}"
        
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
    
    # 构建原始电路
    original_circuit = build_circuit_from_data(circuit_data, n_qubits)
    print(f"Original circuit has {len(original_circuit.instructions)} gates and {n_qubits} qubits")
    
    # 根据basis参数选择目标指令集
    if args.basis == '0':
        # u3+cx指令集 (u3实际上是U门，对应U(theta, phi, lambda))
        instruction_set = ['u', 'cx']
        print(f"Converting circuit to U+CNOT instruction set (from file: {args.file})")
    elif args.basis == '1':
        # u+cx+cz指令集
        instruction_set = ['u', 'cx', 'cz']
        print(f"Converting circuit to U+CNOT+CZ instruction set (from file: {args.file})")
    else:
        raise ValueError(f"Unsupported basis: {args.basis}. Use '0' for u+cx or '1' for u+cx+cz")
    
    # 执行电路转换
    try:
        converted_circuit = convert_circuit_to_instruction_set(original_circuit, instruction_set)
        
        print(f"Circuit conversion successful!")
        print(f"Number of gates in converted circuit: {len(converted_circuit.instructions)}")
        print(f"Number of qubits in circuit: {converted_circuit.n_qubits}")
        
        # 打印转换后的量子电路
        print("\nConverted quantum circuit:")
        for i, inst in enumerate(converted_circuit.instructions):
            params_str = f" params={inst.params}" if inst.params else ""
            # 获取所有量子比特的索引
            if hasattr(inst.qubits[0], 'index'):
                qubit_indices = [q.index for q in inst.qubits]
            else:
                qubit_indices = inst.qubits
            qubit_info = ', '.join([f"qubit {idx}" for idx in qubit_indices])
            print(f"  {i+1}. {inst.name}{params_str} on {qubit_info}")
        
        # 使用to_dict_list方法获取电路的字典表示
        circuit_dict_list = converted_circuit.to_dict_list()
        print(f"\nCircuit representation (to_dict_list format):")
        for i, gate_dict in enumerate(circuit_dict_list):
            print(f"  {i+1}. {gate_dict}")
        
        # 保存电路表示到JSON文件（无论是否指定输出文件）
        save_circuit_to_json(converted_circuit, args.output)
        
        print(f"\n✓ Circuit conversion to {', '.join(instruction_set)} instruction set completed successfully!")
            
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()