#!/usr/bin/env python3
"""
多控制Toffoli门分解测试脚本
"""

import argparse
import json
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from janus.decompose.decompose_multi_control_toffoli import decompose_multi_control_toffoli
from janus.circuit import Circuit
from janus.circuit.io import load_circuit


def save_circuit_to_json(circuit, output_file):
    """将电路的to_dict_list结果保存到JSON文件"""
    circuit_dict_list = circuit.to_dict_list()
    
    # 准备要保存的数据
    data = {
        'circuit': circuit_dict_list,
        'num_gates': len(circuit_dict_list),
        'qubits_used': circuit.n_qubits
    }
    
    # 保存到JSON文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Circuit representation saved to {output_file}")
    return circuit_dict_list


def parse_toffoli_from_json(file_path):
    """从JSON文件解析Toffoli门信息"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if 'circuit' in data and len(data['circuit']) > 0:
        gate_info = data['circuit'][0]  # 获取第一个门的信息
        if gate_info['name'] == 'x' and 'controls' in gate_info:
            num_ctrl_qubits = len(gate_info['controls'])
            return num_ctrl_qubits
        else:
            raise ValueError("File does not contain a proper multi-controlled X gate")
    else:
        raise ValueError("File does not contain valid circuit data")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test multi-controlled Toffoli gate decomposition')
    parser.add_argument('--file', type=str, required=True, help='JSON file containing the multi-controlled Toffoli gate')
    parser.add_argument('--output', type=str, help='Output JSON file for circuit representation')
    
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
        
        # 构造默认输出文件名：原始文件名_decomposed.json
        default_output_filename = f"{file_stem}_decomposed{file_suffix}"
        
        # 输出到test/decompose/result目录
        output_dir = os.path.join(os.path.dirname(__file__), 'result')
        args.output = os.path.join(output_dir, default_output_filename)
    
    # 解析Toffoli门信息
    num_ctrl_qubits = parse_toffoli_from_json(args.file)
    print(f"Decomposing {num_ctrl_qubits}-controlled Toffoli gate (from file: {args.file})")
    
    # 执行多控制Toffoli门分解
    try:
        decomposed_circuit = decompose_multi_control_toffoli(num_ctrl_qubits)
        
        print(f"Decomposition successful!")
        print(f"Number of gates in decomposed circuit: {len(decomposed_circuit.instructions)}")
        print(f"Number of qubits in circuit: {decomposed_circuit.n_qubits}")
        
        # 打印分解后的量子电路
        print("\nDecomposed quantum circuit:")
        for i, inst in enumerate(decomposed_circuit.instructions):
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
        print(f"\nCircuit representation (to_dict_list format):")
        for i, gate_dict in enumerate(circuit_dict_list):
            print(f"  {i+1}. {gate_dict}")
        
        # 保存电路表示到JSON文件（无论是否指定输出文件）
        save_circuit_to_json(decomposed_circuit, args.output)
        
        print(f"\n✓ {num_ctrl_qubits}-controlled Toffoli gate decomposition completed successfully!")
            
    except Exception as e:
        print(f"Error during decomposition: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()