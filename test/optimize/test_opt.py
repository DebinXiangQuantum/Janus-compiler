"""
电路深度优化测试脚本
Usage: 
  电路深度优化规模: python test/optimize/test_opt.py --file benchmark/200_10000_opt.json
  电路深度优化代价: python test/optimize/test_opt.py --file benchmark/200_10000_opt.json
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from janus.circuit import Circuit as QuantumCircuit
from janus.circuit import circuit_to_dag, dag_to_circuit
from janus.optimize import (
    # 技术1: Clifford+Rz优化
    TChinMerger, CliffordMerger,
    # 技术2: 门融合优化
    SingleQubitGateOptimizer, SingleQubitRunCollector,
    # 技术3: 交换性优化
    CommutativeGateCanceller, InverseGateCanceller, CommutativeInverseGateCanceller,
    # 技术10: 智能优化
    smart_optimize,
)


def load_circuit_from_json(filepath):
    """从JSON文件加载量子电路"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    n_qubits = data.get('n_qubits', 4)
    qc = QuantumCircuit(n_qubits)
    
    for gate_info in data.get('gates', []):
        gate_name = gate_info['gate'].lower()
        qubits = gate_info.get('qubits', [0])
        params = gate_info.get('params', [])
        
        if gate_name == 'h':
            qc.h(qubits[0])
        elif gate_name == 't':
            qc.t(qubits[0])
        elif gate_name == 'tdg':
            qc.tdg(qubits[0])
        elif gate_name == 's':
            qc.s(qubits[0])
        elif gate_name == 'sdg':
            qc.sdg(qubits[0])
        elif gate_name == 'x':
            qc.x(qubits[0])
        elif gate_name == 'y':
            qc.y(qubits[0])
        elif gate_name == 'z':
            qc.z(qubits[0])
        elif gate_name == 'cx':
            qc.cx(qubits[0], qubits[1])
        elif gate_name == 'cz':
            qc.cz(qubits[0], qubits[1])
        elif gate_name == 'rx':
            qc.rx(params[0], qubits[0])
        elif gate_name == 'ry':
            qc.ry(params[0], qubits[0])
        elif gate_name == 'rz':
            qc.rz(params[0], qubits[0])
    
    return qc, data.get('description', '')


def count_ops(circuit):
    """统计电路中各类门的数量"""
    ops_count = {}
    for instruction in circuit.data:
        gate_name = instruction.operation.name.lower()
        ops_count[gate_name] = ops_count.get(gate_name, 0) + 1
    return ops_count


def run_optimization(qc):
    """运行全面优化并返回优化后的电路
    
    使用多种优化技术组合:
    - 技术1: T门合并 (TChinMerger)
    - 技术1: Clifford门合并 (CliffordMerger)
    - 技术2: 单比特门优化 (SingleQubitGateOptimizer)
    - 技术3: 交换性消除 (CommutativeGateCanceller)
    - 技术3: 逆门消除 (InverseGateCanceller)
    - 技术3: 交换性逆门消除 (CommutativeInverseGateCanceller)
    """
    dag = circuit_to_dag(qc)
    
    # 技术1: Clifford+Rz优化
    dag = TChinMerger().run(dag)           # T门合并: T+T→S, T+T+T+T→Z
    dag = CliffordMerger().run(dag)        # Clifford门合并
    
    # 技术2: 门融合优化
    dag = SingleQubitRunCollector().run(dag)   # 收集单比特门序列
    dag = SingleQubitGateOptimizer().run(dag)  # 优化单比特门
    
    # 技术3: 交换性优化
    dag = CommutativeGateCanceller().run(dag)         # 交换性消除
    dag = InverseGateCanceller().run(dag)             # 逆门消除: H·H=I, X·X=I
    dag = CommutativeInverseGateCanceller().run(dag)  # 交换性逆门消除: T·Tdg=I
    
    # 再次执行基础优化（清理残余）
    dag = InverseGateCanceller().run(dag)
    
    return dag_to_circuit(dag)


def print_circuit_text(circuit, title, max_qubits=10):
    """使用文本方式输出电路图"""
    print(f"\n{'='*70}")
    print(f"【{title}】- 文本电路图 (draw output='text')")
    print(f"{'='*70}")
    
    if circuit.n_qubits > max_qubits:
        print(f"(电路有 {circuit.n_qubits} 个量子比特，仅显示前 {max_qubits} 个比特的示意)")
        # 创建一个小规模的示例电路用于展示
        small_qc = QuantumCircuit(min(max_qubits, circuit.n_qubits))
        gate_count = 0
        for inst in circuit.data:
            if gate_count >= 20:  # 只显示前20个门
                break
            qubits = inst.qubits
            if all(q < max_qubits for q in qubits):
                try:
                    small_qc.append(inst.operation, qubits)
                    gate_count += 1
                except:
                    pass
        print(small_qc.draw(output='text'))
        print(f"... (共 {len(circuit.data)} 个门)")
    else:
        print(circuit.draw(output='text'))


def print_circuit_str(circuit, title):
    """使用 str() 方式输出电路"""
    print(f"\n{'='*70}")
    print(f"【{title}】- 字符串表示 (str(circuit))")
    print(f"{'='*70}")
    
    # 限制输出长度
    str_output = str(circuit)
    lines = str_output.split('\n')
    if len(lines) > 25:
        print('\n'.join(lines[:25]))
        print(f"... (共 {len(lines)} 行，已截断)")
    else:
        print(str_output)


def print_circuit_repr(circuit, title):
    """使用 repr() 方式输出电路"""
    print(f"\n{'='*70}")
    print(f"【{title}】- repr表示 (repr(circuit))")
    print(f"{'='*70}")
    print(repr(circuit))


def print_circuit_layers(circuit, title, max_layers=5):
    """使用 to_layers() 方式输出电路"""
    print(f"\n{'='*70}")
    print(f"【{title}】- 分层数据 (to_layers())")
    print(f"{'='*70}")
    
    layers = circuit.to_layers()
    print(f"总层数: {len(layers)}")
    print(f"前 {min(max_layers, len(layers))} 层内容:")
    for i, layer in enumerate(layers[:max_layers]):
        print(f"  Layer {i}: {len(layer)} 个门")
        for gate in layer[:3]:  # 每层最多显示3个门
            print(f"    - {gate}")
        if len(layer) > 3:
            print(f"    ... (共 {len(layer)} 个门)")


def print_circuit_instructions(circuit, title, max_instructions=10):
    """使用 to_instructions() 方式输出电路"""
    print(f"\n{'='*70}")
    print(f"【{title}】- 指令列表 (to_instructions())")
    print(f"{'='*70}")
    
    instructions = circuit.to_instructions()
    print(f"总指令数: {len(instructions)}")
    print(f"前 {min(max_instructions, len(instructions))} 条指令:")
    for i, inst in enumerate(instructions[:max_instructions]):
        print(f"  {i}: {inst}")
    if len(instructions) > max_instructions:
        print(f"  ... (共 {len(instructions)} 条指令)")


def print_circuit_tuple_list(circuit, title, max_tuples=10):
    """使用 to_tuple_list() 方式输出电路"""
    print(f"\n{'='*70}")
    print(f"【{title}】- 元组列表 (to_tuple_list())")
    print(f"{'='*70}")
    
    tuples = circuit.to_tuple_list()
    print(f"总元组数: {len(tuples)}")
    print(f"前 {min(max_tuples, len(tuples))} 个元组:")
    for i, t in enumerate(tuples[:max_tuples]):
        print(f"  {i}: {t}")
    if len(tuples) > max_tuples:
        print(f"  ... (共 {len(tuples)} 个元组)")


def print_circuit_gates(circuit, title, max_gates=10):
    """使用 gates 属性输出电路"""
    print(f"\n{'='*70}")
    print(f"【{title}】- 门列表 (circuit.gates)")
    print(f"{'='*70}")
    
    gates = circuit.gates
    print(f"总门数: {len(gates)}")
    print(f"前 {min(max_gates, len(gates))} 个门:")
    for i, gate in enumerate(gates[:max_gates]):
        print(f"  {i}: {gate}")
    if len(gates) > max_gates:
        print(f"  ... (共 {len(gates)} 个门)")


def save_circuit_png(circuit, filename, title):
    """使用 matplotlib 保存电路为 PNG 图片"""
    print(f"\n{'='*70}")
    print(f"【{title}】- PNG图片输出 (draw output='png')")
    print(f"{'='*70}")
    
    try:
        # 根据电路规模动态计算图片大小
        n_layers = circuit.depth
        n_qubits = circuit.n_qubits
        
        # 计算合适的图片尺寸
        width = max(20, n_layers * 1.8 + 3)
        height = max(10, n_qubits * 0.8 + 2)
        
        print(f"电路规模: {n_qubits} 比特, {len(circuit.data)} 门, 深度 {n_layers}")
        print(f"图片尺寸: {width:.0f} x {height:.0f} 英寸")
        print(f"正在生成完整电路图...")
        
        circuit.draw(output='png', filename=filename, figsize=(width, height), dpi=100)
        print(f"电路图已保存到: {filename}")
    except ImportError:
        print("警告: matplotlib 未安装，无法生成 PNG 图片")
        print("请运行: pip install matplotlib")
    except Exception as e:
        print(f"生成 PNG 时出错: {e}")


def print_optimization_summary(original, optimized, description):
    """输出优化摘要"""
    original_size = len(original.data)
    optimized_size = len(optimized.data)
    original_depth = original.depth
    optimized_depth = optimized.depth
    
    reduction = original_size - optimized_size
    reduction_rate = reduction / original_size * 100 if original_size > 0 else 0
    depth_reduction = original_depth - optimized_depth
    depth_reduction_rate = depth_reduction / original_depth * 100 if original_depth > 0 else 0
    
    print(f"\n{'='*70}")
    print("输出优化后的量子电路，")
    print("和优化前后电路数据。")
    print(f"{'='*70}")
    print(f"电路描述: {description}")
    print(f"\n【优化前电路数据】")
    print(f"  量子比特数: {original.n_qubits}")
    print(f"  门数量: {original_size}")
    print(f"  电路深度: {original_depth}")
    print(f"  门分布: {count_ops(original)}")
    print(f"\n【优化后电路数据】")
    print(f"  量子比特数: {optimized.n_qubits}")
    print(f"  门数量: {optimized_size}")
    print(f"  电路深度: {optimized_depth}")
    print(f"  门分布: {count_ops(optimized)}")
    print(f"\n【优化效果统计】")
    print(f"  门减少数量: {reduction}")
    print(f"  门减少比例: {reduction_rate:.2f}%")
    print(f"  深度减少数量: {depth_reduction}")
    print(f"  深度减少比例: {depth_reduction_rate:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='电路深度优化测试')
    parser.add_argument('--file', type=str, required=True, help='输入电路JSON文件路径')
    parser.add_argument('--no-png', action='store_true', help='跳过PNG输出')
    args = parser.parse_args()

    print("="*70)
    print("电路深度优化测试 - 多种输出方式演示")
    print("="*70)

    # 解析文件路径
    filepath = args.file
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.path.dirname(__file__), '..', '..', filepath)
    
    if not os.path.exists(filepath):
        alt_path = os.path.join(os.path.dirname(__file__), '..', '..', 'benchmark', os.path.basename(args.file))
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            print(f"Error: 文件未找到 - {args.file}")
            sys.exit(1)
    
    # 加载电路
    print(f"\n加载电路: {filepath}")
    original_circuit, description = load_circuit_from_json(filepath)
    print(f"电路描述: {description}")
    
    # 运行优化
    print("\n正在优化电路...")
    optimized_circuit = run_optimization(original_circuit)
    print("优化完成!")

    # ==================== 输出优化前电路 ====================
    print("\n" + "="*70)
    print("                    优化前电路输出")
    print("="*70)
    
    # 1. repr 输出
    print_circuit_repr(original_circuit, "优化前电路")
    
    # 2. str 输出
    print_circuit_str(original_circuit, "优化前电路")
    
    # 3. 文本电路图
    print_circuit_text(original_circuit, "优化前电路")
    
    # 4. 分层数据
    print_circuit_layers(original_circuit, "优化前电路")
    
    # 5. 指令列表
    print_circuit_instructions(original_circuit, "优化前电路")
    
    # 6. 元组列表
    print_circuit_tuple_list(original_circuit, "优化前电路")
    
    # 7. 门列表
    print_circuit_gates(original_circuit, "优化前电路")
    
    # 8. PNG 输出
    if not args.no_png:
        output_dir = os.path.dirname(filepath)
        save_circuit_png(original_circuit, 
                        os.path.join(output_dir, "original_circuit.png"),
                        "优化前电路")

    # ==================== 输出优化后电路 ====================
    print("\n" + "="*70)
    print("                    优化后电路输出")
    print("="*70)
    
    # 1. repr 输出
    print_circuit_repr(optimized_circuit, "优化后电路")
    
    # 2. str 输出
    print_circuit_str(optimized_circuit, "优化后电路")
    
    # 3. 文本电路图
    print_circuit_text(optimized_circuit, "优化后电路")
    
    # 4. 分层数据
    print_circuit_layers(optimized_circuit, "优化后电路")
    
    # 5. 指令列表
    print_circuit_instructions(optimized_circuit, "优化后电路")
    
    # 6. 元组列表
    print_circuit_tuple_list(optimized_circuit, "优化后电路")
    
    # 7. 门列表
    print_circuit_gates(optimized_circuit, "优化后电路")
    
    # 8. PNG 输出
    if not args.no_png:
        save_circuit_png(optimized_circuit,
                        os.path.join(output_dir, "optimized_circuit.png"),
                        "优化后电路")

    # ==================== 优化摘要 ====================
    print_optimization_summary(original_circuit, optimized_circuit, description)


if __name__ == '__main__':
    main()
