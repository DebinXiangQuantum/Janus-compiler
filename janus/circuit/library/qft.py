"""
Quantum Fourier Transform (QFT) gate
"""
from ..circuit import Circuit
from ..gate import Gate
import numpy as np


class QFT(Gate):
    """Quantum Fourier Transform gate

    The QFT is the quantum analogue of the discrete Fourier transform.
    """

    def __init__(self, num_qubits: int, approximation_degree: int = 0,
                 inverse: bool = False, insert_barriers: bool = False,
                 do_swaps: bool = True, name: str = "qft"):
        """Create new QFT gate.

        Args:
            num_qubits: The number of qubits the QFT acts on
            approximation_degree: Degree of approximation (0 for exact)
            inverse: If True, the inverse Fourier transform is returned
            insert_barriers: If True, barriers are inserted as visualization improvement
            do_swaps: If True, perform the final swaps of the QFT
            name: Name of the gate
        """
        super().__init__(name, num_qubits, [])
        self._qft_num_qubits = num_qubits
        self.approximation_degree = approximation_degree
        self._inverse = inverse
        self.insert_barriers = insert_barriers
        self.do_swaps = do_swaps

    def _define(self):
        """Define the QFT circuit"""
        from .standard_gates import HGate, CPhaseGate, SwapGate

        circuit = Circuit(self._qft_num_qubits)

        # Build QFT circuit
        for j in range(self._qft_num_qubits):
            circuit.append(HGate(), [j])

            # Controlled phase rotations
            for k in range(j + 1, self._qft_num_qubits):
                if self.approximation_degree == 0 or (k - j) <= self.approximation_degree:
                    lam = np.pi / (2 ** (k - j))
                    circuit.append(CPhaseGate(lam), [k, j])

            if self.insert_barriers:
                circuit.barrier()

        # Swap qubits
        if self.do_swaps:
            for i in range(self._qft_num_qubits // 2):
                circuit.append(SwapGate(), [i, self._qft_num_qubits - i - 1])

        # Inverse if requested
        if self._inverse:
            circuit = circuit.inverse()

        return circuit

    def inverse(self):
        """Return inverse QFT gate"""
        return QFT(
            num_qubits=self._qft_num_qubits,
            approximation_degree=self.approximation_degree,
            inverse=not self._inverse,
            insert_barriers=self.insert_barriers,
            do_swaps=self.do_swaps,
            name="qft_dg" if not self._inverse else "qft"
        )

    def decompose(self):
        """Decompose QFT gate into basic gates"""
        return self._define()
