"""
Tech10: Auto Selection Optimization
Usage: python test/optimize/test_tech10_auto_selection.py --file benchmark/opt_11.json
"""

import sys
import os
import json
import argparse
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from janus.circuit import Circuit as QuantumCircuit
from janus.circuit import circuit_to_dag, dag_to_circuit
from janus.optimize import TChinMerger, CliffordMerger, InverseGateCanceller, CommutativeInverseGateCanceller


def load_circuit_from_json(filepath):
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
        elif gate_name == 'rx':
            qc.rx(params[0], qubits[0])
        elif gate_name == 'ry':
            qc.ry(params[0], qubits[0])
        elif gate_name == 'rz':
            qc.rz(params[0], qubits[0])
    
    return qc, data.get('description', '')


def create_large_scale_circuit():
    n_qubits = 200
    target_gates = 10000
    qc = QuantumCircuit(n_qubits)
    random.seed(42)
    np.random.seed(42)
    gate_count = 0

    while gate_count < target_gates:
        qubit = random.randint(0, n_qubits - 2)
        pattern = random.random()

        if pattern < 0.15:
            gate = random.choice(['h', 'x', 'y', 'z'])
            if gate == 'h':
                qc.h(qubit)
                qc.h(qubit)
            elif gate == 'x':
                qc.x(qubit)
                qc.x(qubit)
            elif gate == 'y':
                qc.y(qubit)
                qc.y(qubit)
            else:
                qc.z(qubit)
                qc.z(qubit)
            gate_count += 2
        elif pattern < 0.3:
            qc.t(qubit)
            qc.t(qubit)
            gate_count += 2
        elif pattern < 0.4:
            qc.cx(qubit, qubit + 1)
            qc.cx(qubit, qubit + 1)
            gate_count += 2
        elif pattern < 0.5:
            pair = random.choice(['t_tdg', 's_sdg'])
            if pair == 't_tdg':
                qc.t(qubit)
                qc.tdg(qubit)
            else:
                qc.s(qubit)
                qc.sdg(qubit)
            gate_count += 2
        elif pattern < 0.65:
            qc.rz(np.random.uniform(0, np.pi), qubit)
            qc.rx(np.random.uniform(0, np.pi), qubit)
            qc.rz(np.random.uniform(0, np.pi), qubit)
            gate_count += 3
        else:
            gate = random.choice(['h', 't', 's', 'cx', 'rx', 'ry', 'rz'])
            if gate == 'h':
                qc.h(qubit)
            elif gate == 't':
                qc.t(qubit)
            elif gate == 's':
                qc.s(qubit)
            elif gate == 'cx':
                qc.cx(qubit, qubit + 1)
            elif gate == 'rx':
                qc.rx(np.random.uniform(0, np.pi), qubit)
            elif gate == 'ry':
                qc.ry(np.random.uniform(0, np.pi), qubit)
            else:
                qc.rz(np.random.uniform(0, np.pi), qubit)
            gate_count += 1

    return qc, "Large scale auto selection test circuit (200 qubits, ~10000 gates)"


def count_ops(circuit):
    ops_count = {}
    for instruction in circuit.data:
        gate_name = instruction.operation.name.lower()
        ops_count[gate_name] = ops_count.get(gate_name, 0) + 1
    return ops_count


def run_optimization_test(qc, description):
    original_size = len(qc.data)
    original_depth = qc.depth

    dag = circuit_to_dag(qc)
    dag = TChinMerger().run(dag)
    dag = CliffordMerger().run(dag)
    dag = CommutativeInverseGateCanceller().run(dag)
    dag = InverseGateCanceller().run(dag)
    qc_optimized = dag_to_circuit(dag)

    optimized_size = len(qc_optimized.data)
    optimized_depth = qc_optimized.depth
    reduction = original_size - optimized_size
    reduction_rate = reduction / original_size * 100 if original_size > 0 else 0

    print(f"\n{'='*70}")
    print(f"Tech10 Auto Selection Optimization Test")
    print(f"{'='*70}")
    print(f"Description: {description}")
    print(f"API: TChinMerger + CliffordMerger + CommutativeInverseGateCanceller + InverseGateCanceller")
    print(f"Circuit size: {qc.n_qubits} qubits, {original_size} gates, depth {original_depth}")
    print(f"After optimization: {optimized_size} gates, depth {optimized_depth}")
    print(f"Gate reduction: {reduction} ({reduction_rate:.1f}%)")

    return qc_optimized, reduction_rate


def print_optimized_circuit(circuit, description=""):
    """打印优化后的电路"""
    print(f"\n{'='*70}")
    print("输出优化后的量子电路。")
    print(f"{'='*70}")
    print(f"电路描述: {description}")
    print(f"量子比特数: {circuit.n_qubits}")
    print(f"门数量: {len(circuit.data)}")
    print(f"\n门列表:")
    for i, inst in enumerate(circuit.data):
        gate_name = inst.operation.name
        qubits = inst.qubits
        params = inst.operation.params if inst.operation.params else []
        if params:
            params_str = ", ".join([f"{p:.6f}" for p in params])
            print(f"  {i+1}. {gate_name}({params_str}) on qubits {qubits}")
        else:
            print(f"  {i+1}. {gate_name} on qubits {qubits}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='Tech10: Auto Selection Optimization')
    parser.add_argument('--file', type=str, help='Input circuit JSON file path')
    args = parser.parse_args()

    print("="*70)
    print("Tech10: Auto Selection Optimization")
    print("="*70)

    qc_optimized = None
    description = ""

    if args.file:
        filepath = args.file
        if not os.path.isabs(filepath):
            filepath = os.path.join(os.path.dirname(__file__), '..', '..', filepath)
        
        if not os.path.exists(filepath):
            alt_path = os.path.join(os.path.dirname(__file__), '..', '..', 'benchmark', os.path.basename(args.file))
            if os.path.exists(alt_path):
                filepath = alt_path
            else:
                alt_path2 = os.path.join(os.path.dirname(__file__), '..', '..', 'benchmark', 'large_scale', os.path.basename(args.file))
                if os.path.exists(alt_path2):
                    filepath = alt_path2
                else:
                    print(f"Error: File not found - {args.file}")
                    sys.exit(1)
        
        qc, description = load_circuit_from_json(filepath)
        qc_optimized, _ = run_optimization_test(qc, description)
        description = f"Tech10优化后: {description}"
    else:
        qc, desc = create_large_scale_circuit()
        qc_optimized, _ = run_optimization_test(qc, desc)
        description = f"Tech10优化后: {desc}"

    # 输出优化后的电路
    if qc_optimized:
        print_optimized_circuit(qc_optimized, description)


if __name__ == '__main__':
    main()
