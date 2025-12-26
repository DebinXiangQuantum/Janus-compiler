"""
Pure Python implementation of greedy Clifford synthesis.
Based on the algorithm described in Appendix A of Bravyi & Maslov (2020).
"""

import numpy as np
from typing import List, Tuple


def _calc_cost(tableau: np.ndarray, num_qubits: int) -> int:
    """Calculate the cost function for greedy synthesis."""
    cost = 0
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i != j:
                # Count non-zero entries in destabilizer and stabilizer parts
                cost += int(tableau[i, j]) + int(tableau[i, j + num_qubits])
                cost += int(tableau[i + num_qubits, j]) + int(tableau[i + num_qubits, j + num_qubits])
    return cost


def _apply_h(tableau: np.ndarray, qubit: int, num_qubits: int) -> None:
    """Apply Hadamard gate to tableau."""
    for i in range(2 * num_qubits):
        # Swap X and Z columns
        tableau[i, qubit], tableau[i, qubit + num_qubits] = \
            tableau[i, qubit + num_qubits], tableau[i, qubit]
        # Update phase
        tableau[i, 2 * num_qubits] ^= tableau[i, qubit] & tableau[i, qubit + num_qubits]


def _apply_s(tableau: np.ndarray, qubit: int, num_qubits: int) -> None:
    """Apply S gate to tableau."""
    for i in range(2 * num_qubits):
        # Update phase: r -> r XOR x*z
        tableau[i, 2 * num_qubits] ^= tableau[i, qubit] & tableau[i, qubit + num_qubits]
        # Update Z: z -> z XOR x
        tableau[i, qubit + num_qubits] ^= tableau[i, qubit]


def _apply_cx(tableau: np.ndarray, ctrl: int, tgt: int, num_qubits: int) -> None:
    """Apply CNOT gate to tableau."""
    for i in range(2 * num_qubits):
        # Update phase
        tableau[i, 2 * num_qubits] ^= (
            tableau[i, ctrl] & tableau[i, tgt + num_qubits] &
            (tableau[i, tgt] ^ tableau[i, ctrl + num_qubits] ^ 1)
        )
        # Update X of target
        tableau[i, tgt] ^= tableau[i, ctrl]
        # Update Z of control
        tableau[i, ctrl + num_qubits] ^= tableau[i, tgt + num_qubits]


def synth_clifford_greedy(tableau: np.ndarray) -> List[Tuple[str, List[int]]]:
    """
    Greedy synthesis of a Clifford operator.
    
    Args:
        tableau: Boolean numpy array representing the Clifford tableau.
                 Shape should be (2n, 2n+1) where n is the number of qubits.
    
    Returns:
        List of (gate_name, qubits) tuples representing the circuit.
    """
    tableau = tableau.astype(bool).copy()
    num_qubits = tableau.shape[0] // 2
    gates = []
    
    # Greedy algorithm: repeatedly apply gates that reduce the cost
    max_iterations = 10000
    for _ in range(max_iterations):
        cost = _calc_cost(tableau, num_qubits)
        if cost == 0:
            break
        
        best_gate = None
        best_cost = cost
        
        # Try all single-qubit gates
        for q in range(num_qubits):
            # Try H gate
            test_tableau = tableau.copy()
            _apply_h(test_tableau, q, num_qubits)
            new_cost = _calc_cost(test_tableau, num_qubits)
            if new_cost < best_cost:
                best_cost = new_cost
                best_gate = ('h', [q])
            
            # Try S gate
            test_tableau = tableau.copy()
            _apply_s(test_tableau, q, num_qubits)
            new_cost = _calc_cost(test_tableau, num_qubits)
            if new_cost < best_cost:
                best_cost = new_cost
                best_gate = ('s', [q])
        
        # Try all CNOT gates
        for ctrl in range(num_qubits):
            for tgt in range(num_qubits):
                if ctrl != tgt:
                    test_tableau = tableau.copy()
                    _apply_cx(test_tableau, ctrl, tgt, num_qubits)
                    new_cost = _calc_cost(test_tableau, num_qubits)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_gate = ('cx', [ctrl, tgt])
        
        if best_gate is None:
            # No improvement found, try random gate
            break
        
        # Apply the best gate
        gate_name, qubits = best_gate
        if gate_name == 'h':
            _apply_h(tableau, qubits[0], num_qubits)
        elif gate_name == 's':
            _apply_s(tableau, qubits[0], num_qubits)
        elif gate_name == 'cx':
            _apply_cx(tableau, qubits[0], qubits[1], num_qubits)
        
        gates.append(best_gate)
    
    # Reverse the gates to get the synthesis
    return list(reversed(gates))
