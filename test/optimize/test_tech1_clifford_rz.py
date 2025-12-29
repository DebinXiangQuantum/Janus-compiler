"""
Tech1: Clifford+RZ Decomposition Optimization - Large Scale Test (200 qubits, 10000 gates)
Usage: python test/optimize/test_tech1_clifford_rz.py --file benchmark/opt_1.json
"""

import sys
import os
import json
import argparse
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from janus.circuit import Circuit as QuantumCircuit
from janus.circuit import circuit_to_dag, dag_to_circuit
from janus.optimize import TChinMerger, CliffordMerger, InverseGateCanceller


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
        elif gate_name == 'cz':
            qc.cz(qubits[0], qubits[1])
        elif gate_name == 'rx':
            qc.rx(params[0], qubits[0])
        elif gate_name == 'ry':
            qc.ry(params[0], qubits[0])
        elif gate_name == 'rz':
            qc.rz(params[0], qubits[0])
    
    return qc, data.get('description', '')


def create_large_scale_t_gate_circuit():
    n_qubits = 200
    target_gates = 10000
    qc = QuantumCircuit(n_qubits)
    random.seed(42)
    gate_count = 0

    while gate_count < target_gates:
        qubit = random.randint(0, n_qubits - 1)
        pattern = random.random()

        if pattern < 0.6:
            num_t = random.choice([2, 4, 8])
            for _ in range(num_t):
                qc.t(qubit)
            gate_count += num_t
        elif pattern < 0.8:
            qc.t(qubit)
            qc.tdg(qubit)
            gate_count += 2
        else:
            gate = random.choice(['h', 's', 'x'])
            if gate == 'h':
                qc.h(qubit)
            elif gate == 's':
                qc.s(qubit)
            else:
                qc.x(qubit)
            gate_count += 1

    return qc, "Large scale T gate merge test circuit (200 qubits, ~10000 gates)"


def create_large_scale_clifford_t_circuit():
    n_qubits = 200
    target_gates = 10000
    qc = QuantumCircuit(n_qubits)
    random.seed(123)
    gate_count = 0

    while gate_count < target_gates:
        qubit = random.randint(0, n_qubits - 2)
        pattern = random.random()

        if pattern < 0.3:
            num_t = random.choice([2, 4])
            for _ in range(num_t):
                qc.t(qubit)
            gate_count += num_t
        elif pattern < 0.5:
            qc.h(qubit)
            qc.h(qubit)
            gate_count += 2
        elif pattern < 0.7:
            qc.cx(qubit, qubit + 1)
            qc.cx(qubit, qubit + 1)
            gate_count += 2
        else:
            gate = random.choice(['h', 't', 's', 'cx'])
            if gate == 'h':
                qc.h(qubit)
            elif gate == 't':
                qc.t(qubit)
            elif gate == 's':
                qc.s(qubit)
            else:
                qc.cx(qubit, qubit + 1)
            gate_count += 1

    return qc, "Large scale Clifford+T optimization test circuit (200 qubits, ~10000 gates)"


def count_ops(circuit):
    ops_count = {}
    for instruction in circuit.data:
        gate_name = instruction.operation.name.lower()
        ops_count[gate_name] = ops_count.get(gate_name, 0) + 1
    return ops_count


def run_t_gate_merging_test(qc, description):
    original_size = len(qc.data)
    original_t = count_ops(qc).get('t', 0)

    dag = circuit_to_dag(qc)
    optimizer = TChinMerger()
    dag_opt = optimizer.run(dag)
    qc_optimized = dag_to_circuit(dag_opt)

    optimized_size = len(qc_optimized.data)
    optimized_t = count_ops(qc_optimized).get('t', 0)
    optimized_s = count_ops(qc_optimized).get('s', 0)
    optimized_z = count_ops(qc_optimized).get('z', 0)
    reduction = original_size - optimized_size
    reduction_rate = reduction / original_size * 100 if original_size > 0 else 0

    print(f"\n{'='*70}")
    print(f"Tech1 Large Scale T Gate Merge Test")
    print(f"{'='*70}")
    print(f"Description: {description}")
    print(f"API: TChinMerger().run(dag)")
    print(f"Circuit size: {qc.n_qubits} qubits, {original_size} gates")
    print(f"Original T gates: {original_t}")
    print(f"After optimization: {optimized_size} gates")
    print(f"  - T gates: {optimized_t}")
    print(f"  - S gates: {optimized_s} (T+T merged)")
    print(f"  - Z gates: {optimized_z} (T+T+T+T merged)")
    print(f"Gate reduction: {reduction} ({reduction_rate:.1f}%)")

    return qc_optimized, reduction_rate


def run_clifford_t_optimization_test(qc, description):
    original_size = len(qc.data)
    original_depth = qc.depth

    dag = circuit_to_dag(qc)
    dag = TChinMerger().run(dag)
    dag = CliffordMerger().run(dag)
    dag = InverseGateCanceller().run(dag)
    qc_optimized = dag_to_circuit(dag)

    optimized_size = len(qc_optimized.data)
    optimized_depth = qc_optimized.depth
    reduction = original_size - optimized_size
    reduction_rate = reduction / original_size * 100 if original_size > 0 else 0

    print(f"\n{'='*70}")
    print(f"Tech1 Large Scale Clifford+T Optimization Test")
    print(f"{'='*70}")
    print(f"Description: {description}")
    print(f"API: TChinMerger + CliffordMerger + InverseGateCanceller")
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
    parser = argparse.ArgumentParser(description='Tech1: Clifford+RZ Decomposition Optimization')
    parser.add_argument('--file', type=str, help='Input circuit JSON file path')
    args = parser.parse_args()

    print("="*70)
    print("Tech1 Large Scale Test: Clifford+RZ Decomposition Optimization")
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
        qc_optimized, _ = run_t_gate_merging_test(qc, description)
        description = f"Tech1优化后: {description}"
    else:
        qc1, desc1 = create_large_scale_t_gate_circuit()
        qc_optimized1, _ = run_t_gate_merging_test(qc1, desc1)

        qc2, desc2 = create_large_scale_clifford_t_circuit()
        qc_optimized, _ = run_clifford_t_optimization_test(qc2, desc2)
        description = f"Tech1优化后: {desc2}"

    # 输出优化后的电路
    if qc_optimized:
        print_optimized_circuit(qc_optimized, description)


if __name__ == '__main__':
    main()
