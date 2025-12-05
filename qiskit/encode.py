import numpy as np
from circuit import QuantumCircuit, QuantumRegister
from circuit.exceptions import QiskitError
from circuit.quantum_info import Operator, Statevector
import scipy.linalg
from circuit.library import UGate, Initialize
from math import log2, ceil, floor, pi, acos
from typing import Dict, List, Union
def bid_amplitude_encode_qiskit(data: list[float], split: int = 0) -> QuantumCircuit:
    data_temp = np.array(data, dtype=float)

    if not np.isclose(np.sum(data_temp**2), 1.0):
        raise ValueError("Data is not normalized (sum of squared amplitudes must equal 1).")

    n_qubits = int(np.ceil(np.log2(len(data_temp))))

    split_temp = split
    if split == 0:
        split_temp = (n_qubits + 1) // 2

    if split_temp > n_qubits:
        raise ValueError("Bid_Amplitude_encode parameter error: split is greater than required qubits.")
    
    size_next = 1 << n_qubits
    if len(data_temp) < size_next:
        padding = np.zeros(size_next - len(data_temp))
        data_temp = np.concatenate((data_temp, padding))
    
    if not np.isclose(np.linalg.norm(data_temp), 1.0):
         raise ValueError("Internal error: Padded data is not normalized.")

    qr = QuantumRegister(n_qubits, name='q')
    circuit = QuantumCircuit(qr)
    
    initializer = Initialize(data_temp)
    
    circuit.append(initializer, qr)
    
    return circuit
def _complete_to_unitary(matrix: np.ndarray, target_num_qubits: int) -> np.ndarray:
    target_dim = 2**target_num_qubits
    rows, cols = matrix.shape
   
    if rows == target_dim and cols == target_dim:
          return matrix

    if rows == target_dim and cols < target_dim:
        Q, _ = scipy.linalg.qr(matrix, mode='full')
        if Q.shape != (target_dim, target_dim):
            raise RuntimeError(f"SciPy QR completion produced unexpected shape: {Q.shape}, expected {(target_dim, target_dim)}")
        return Q
    
    raise ValueError(f"Unexpected matrix shape {matrix.shape} for target_num_qubits {target_num_qubits}. Expected rows={target_dim}")

def _apply_unitary_qiskit(circuit: QuantumCircuit, qubits: list, matrix: np.ndarray, cutoff: float):
    if not qubits:
        return

    target_num_qubits = len(qubits)
    completed_matrix = _complete_to_unitary(matrix, target_num_qubits)

    if completed_matrix.shape[0] != completed_matrix.shape[1] or completed_matrix.shape[0] != 2**target_num_qubits:
        raise RuntimeError(f"Completed matrix is not square (2^N x 2^N). Shape: {completed_matrix.shape}, expected ({2**target_num_qubits}, {2**target_num_qubits})")

    input_dims = (2,) * target_num_qubits

    try:
        unitary_op = Operator(completed_matrix, input_dims=input_dims)
        circuit.unitary(unitary_op, qubits, label=f"Unitary_{len(qubits)}")
    except Exception as e:
        raise

def _schmidt_qiskit(circuit: QuantumCircuit, qubits: list, data: List[float], cutoff: float):
    data_temp = np.array(data, dtype=float)
    n_qubits = len(qubits)

    if n_qubits == 1:
        a0 = data_temp[0] 
        clamped_a0 = np.fmin(np.fmax(a0, -1.0), 1.0)

        if clamped_a0 < 0:
            angle = 2 * pi - 2 * acos(clamped_a0)
        else:
            angle = 2 * acos(clamped_a0)

        circuit.ry(angle, qubits[0])
        return

    size = 1 << n_qubits
    if len(data_temp) < size:
        padding = np.zeros(size - len(data_temp))
        data_temp = np.concatenate((data_temp, padding))

    r = n_qubits % 2
    row = 1 << (n_qubits >> 1)
    col = 1 << ((n_qubits >> 1) + r)

    eigen_matrix = data_temp.reshape((row, col), order='C')

    U, S, Vh = np.linalg.svd(eigen_matrix, full_matrices=False)

    V = Vh

    A = S
    length = 0
    while length < len(A) and (A[length] >= A[0] * cutoff or length == 0):
        length += 1

    A_cut = A[:length]
    PartU = U[:, :length]
    PartV = V[:length, :]

    A_qubits_indices = slice(0, floor(n_qubits / 2) + r)
    B_qubits_indices = slice(floor(n_qubits / 2) + r, n_qubits)

    A_qubits = qubits[A_qubits_indices]
    B_qubits = qubits[B_qubits_indices]

    bit = int(log2(length)) if length > 0 else 0

    if bit > 0:
        reg_tmp = B_qubits[:bit]
        A_cut_normalized = A_cut / np.linalg.norm(A_cut)
        _schmidt_qiskit(circuit, reg_tmp, A_cut_normalized.tolist(), cutoff)

    for i in range(bit):
        circuit.cx(B_qubits[i], A_qubits[i])

    _apply_unitary_qiskit(circuit, B_qubits, PartU, cutoff)

    _apply_unitary_qiskit(circuit, A_qubits, PartV.T, cutoff)

def schmidt_encode_qiskit(q_size: int, data: List[float], cutoff: float = 1e-4) -> QuantumCircuit:
    data_temp = np.array(data, dtype=float)

    if not np.isclose(np.linalg.norm(data_temp), 1.0):
        raise ValueError("Data is not normalized (L2 norm must equal 1).")

    n_required_qubits = ceil(log2(len(data_temp)))
    if n_required_qubits > q_size:
        raise ValueError("Schmidt_encode parameter error: Qubits available are less than required.")

    qr = QuantumRegister(q_size, 'q')
    circuit = QuantumCircuit(qr)

    qubits_to_use = qr[:n_required_qubits]
    _schmidt_qiskit(circuit, qubits_to_use, data_temp.tolist(), cutoff)

    print(f"--- 编码所使用的量子比特数量: {n_required_qubits} ---")

    return circuit

QComplex = complex

def _build_state_dict(data: Union[List[float], List[QComplex]]) -> Dict[str, QComplex]:
    if not data:
        return {}

    n_qubits = int(ceil(log2(len(data))))
    state_dict: Dict[str, QComplex] = {}

    for cnt, amp in enumerate(data):
        amp_complex = QComplex(amp)

        if np.isclose(np.abs(amp_complex)**2, 0.0):
            continue

        binary_string = format(cnt, f'0{n_qubits}b')
        state_dict[binary_string] = amp_complex

    return state_dict

def _merging_procedure(state: Dict[str, QComplex], circuit: QuantumCircuit, q_reg: QuantumRegister, reverse_q: List[int]) -> Dict[str, QComplex]:
    new_state: Dict[str, QComplex] = {}

    if not state:
        return new_state

    n_qubits = len(list(state.keys())[0])
    prefix_length = n_qubits - 1

    q_index_to_operate = reverse_q[prefix_length]

    groups: Dict[str, Dict[str, QComplex]] = {}
    for key, amp in state.items():
        prefix = key[:prefix_length]
        suffix = key[prefix_length:]

        if prefix not in groups:
            groups[prefix] = {'0': 0.0 + 0.0j, '1': 0.0 + 0.0j}

        groups[prefix][suffix] = amp

    for prefix, amps in groups.items():
        amp0 = amps['0']
        amp1 = amps['1']

        new_amp = np.sqrt(np.abs(amp0)**2 + np.abs(amp1)**2)

        if np.isclose(new_amp, 0.0):
            continue

        relative_phase = 0.0
        if not np.isclose(amp0, 0.0):
            relative_phase = np.angle(amp1 / amp0)

        ry_angle = 0.0
        if not np.isclose(np.abs(amp0), 0.0):
            ry_angle = 2 * np.arccos(np.abs(amp0) / new_amp)
        elif not np.isclose(np.abs(amp1), 0.0):
             ry_angle = pi

        control_qubits_indices = [reverse_q[i] for i in range(n_qubits) if i != prefix_length]

        if not control_qubits_indices:
            circuit.ry(-ry_angle, q_reg[q_index_to_operate])
            circuit.rz(-relative_phase, q_reg[q_index_to_operate])

        else:
            u_gate = UGate(-ry_angle, -relative_phase, 0.0)
            control_states = [c for c in prefix]

            control_indices = control_qubits_indices
            target_index = q_index_to_operate # Corrected line: removed .index from q_reg[...]

            circuit.append(u_gate.control(len(control_indices),
                                          ctrl_state=''.join(control_states)),
                           control_indices + [target_index])

        new_state[prefix] = new_amp

    return new_state

def efficient_sparse_qiskit(q_size: int, data: Union[List[float], List[QComplex], Dict[str, Union[float, QComplex]]]) -> QuantumCircuit:

    if isinstance(data, (list, np.ndarray)):
        state = _build_state_dict(data)
    elif isinstance(data, dict):
        state = {k: QComplex(v) for k, v in data.items()}
    else:
        raise TypeError("Input data must be a list of amplitudes or a dictionary of states.")

    if not state:
        raise ValueError("Error: The input map data must not be null.")

    first_key = next(iter(state.keys()))
    n_qubits = len(first_key)

    for key in state.keys():
        if len(key) != n_qubits:
            raise ValueError("Error: The input map data.key must have same dimension.")

    tmp_sum = sum(np.abs(amp)**2 for amp in state.values())
    max_precision = 1e-13

    if abs(1.0 - tmp_sum) > max_precision:
        if tmp_sum < max_precision:
             raise ValueError("Error: The input vector b is zero.")
        raise ValueError("Error: The input vector b must satisfy the normalization condition.")

    if 1 << n_qubits > 1 << q_size:
        raise ValueError("Error: The input qubits size error (required > available).")

    qr = QuantumRegister(q_size, 'q')
    circuit = QuantumCircuit(qr)

    reverse_q_indices = list(range(n_qubits - 1, -1, -1))

    temp_circuit = QuantumCircuit(qr)
    current_state = state.copy()

    for i in range(n_qubits, 0, -1):
        qubit_indices_to_merge = reverse_q_indices[:i]
        current_state = _merging_procedure(current_state, temp_circuit, qr, qubit_indices_to_merge)

    # The following lines caused IndexError and are removed as _merging_procedure already reduces to |0...0>.
    # b_string = next(iter(current_state.keys()))
    # for i in range(n_qubits):
    #     if b_string[i] == '1':
    #         temp_circuit.x(qr[i])

    final_circuit = temp_circuit.inverse()

    return final_circuit

