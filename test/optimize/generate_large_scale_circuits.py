"""
Generate large scale test circuit JSON files
Output directory: benchmark/large_scale/
Usage: python generate_large_scale_circuits.py
"""

import sys
import os
import json
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from janus.circuit import Circuit as QuantumCircuit

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'benchmark', 'large_scale')


def circuit_to_json(qc, description):
    gates = []
    for instruction in qc.data:
        gate_name = instruction.operation.name
        qubits = []
        for q in instruction.qubits:
            if hasattr(q, 'index'):
                qubits.append(q.index)
            else:
                qubits.append(int(q))
        
        gate_info = {"gate": gate_name, "qubits": qubits}
        
        if hasattr(instruction.operation, 'params') and instruction.operation.params:
            gate_info["params"] = [float(p) for p in instruction.operation.params]
        
        gates.append(gate_info)
    
    return {
        "description": description,
        "n_qubits": qc.n_qubits,
        "gates": gates
    }


def save_circuit(qc, description, filename):
    data = circuit_to_json(qc, description)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Generated: {filename} ({qc.n_qubits} qubits, {len(qc.data)} gates)")


def create_tech1_circuit():
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

    return qc, "Tech1: Large scale T gate merge test circuit"


def main():
    print("Generating large scale test circuits...")
    
    qc, desc = create_tech1_circuit()
    save_circuit(qc, desc, "tech1_t_gate_circuit.json")
    
    print("Done!")


if __name__ == '__main__':
    main()
