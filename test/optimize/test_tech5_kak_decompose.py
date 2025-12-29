"""
Tech5: KAK Decomposition Optimization
Usage: python test/optimize/test_tech5_kak_decompose.py --file benchmark/opt_6.json
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
from janus.optimize import TwoQubitBlockCollector, InverseGateCanceller


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
        elif gate_name == 's':
            qc.s(qubits[0])
        elif gate_name == 'x':
            qc.x(qubits[0])
        elif gate_name == 'cx':
            qc.cx(qubits[0], qubits[1])
        elif gate_name == 'rx':
            qc.rx(params[0], qubits[0])
        elif gate_name == 'ry':
            qc.ry(params[0], qubits[0])
        elif gate_name == 'rz':
            qc.rz(params[0], qubits[0])
        elif gate_name == 'u':
            qc.u(params[0], params[1], params[2], qubits[0])
    
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

        if pattern < 0.4:
            qc.cx(qubit, qubit + 1)
            qc.cx(qubit, qubit + 1)
            gate_count += 2
        elif pattern < 0.6:
            qc.h(qubit)
            qc.h(qubit)
            gate_count += 2
        elif pattern < 0.8:
            qc.t(qubit)
            qc.t(qubit)
            gate_count += 2
        else:
            gate = random.choice(['h', 'rx', 'ry', 'rz'])
            if gate == 'h':
                qc.h(qubit)
            elif gate == 'rx':
                qc.rx(np.random.uniform(0, np.pi), qubit)
            elif gate == 'ry':
                qc.ry(np.random.uniform(0, np.pi), qubit)
            else:
                qc.rz(np.random.uniform(0, np.pi), qubit)
            gate_count += 1

    return qc, "Large scale KAK decomposition test circuit (200 qubits, ~10000 gates)"


def count_ops(circuit):
    ops_count = {}
    for instruction in circuit.data:
        gate_name = instruction.operation.name.lower()
        ops_count[gate_name] = ops_count.get(gate_name, 0) + 1
    return ops_count


def run_optimization_test(qc, description):
    original_size = len(qc.data)
    original_cx = count_ops(qc).get('cx', 0)

    dag = circuit_to_dag(qc)
    dag = TwoQubitBlockCollector().run(dag)
    dag = InverseGateCanceller().run(dag)
    qc_optimized = dag_to_circuit(dag)

    optimized_size = len(qc_optimized.data)
    optimized_cx = count_ops(qc_optimized).get('cx', 0)
    reduction = original_size - optimized_size
    reduction_rate = reduction / original_size * 100 if original_size > 0 else 0

    print(f"\n{'='*70}")
    print(f"Tech5 KAK Decomposition Optimization Test")
    print(f"{'='*70}")
    print(f"Description: {description}")
    print(f"API: TwoQubitBlockCollector + InverseGateCanceller")
    print(f"Circuit size: {qc.n_qubits} qubits, {original_size} gates")
    print(f"Original CX gates: {original_cx}")
    print(f"After optimization: {optimized_size} gates, CX: {optimized_cx}")
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
    parser = argparse.ArgumentParser(description='Tech5: KAK Decomposition Optimization')
    parser.add_argument('--file', type=str, help='Input circuit JSON file path')
    args = parser.parse_args()

    print("="*70)
    print("Tech5: KAK Decomposition Optimization")
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
        description = f"Tech5优化后: {description}"
    else:
        qc, desc = create_large_scale_circuit()
        qc_optimized, _ = run_optimization_test(qc, desc)
        description = f"Tech5优化后: {desc}"

    # 输出优化后的电路
    if qc_optimized:
        print_optimized_circuit(qc_optimized, description)


if __name__ == '__main__':
    main()
