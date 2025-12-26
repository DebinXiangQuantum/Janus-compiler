# This code is part of Janus Quantum Compiler.
"""
Janus Quantum Circuit Synthesis Module

This module provides quantum circuit synthesis algorithms for various gate sets:
- Clifford circuits (Technology 6)
- CNOT circuits (Technology 7)
- Two-qubit decompositions (Technology 5)
"""

# Technology 5: KAK Decomposition
from .two_qubit import (
    TwoQubitWeylDecomposition,
    TwoQubitBasisDecomposer,
    two_qubit_cnot_decompose,
    TwoQubitControlledUDecomposer,
)

# Technology 6: Clifford Synthesis
from .clifford import (
    synthesize_clifford_circuit,
    synthesize_clifford_aaronson_gottesman,
    synthesize_clifford_bravyi_maslov,
    synthesize_clifford_greedy,
    synthesize_clifford_layered,
    synthesize_clifford_depth_lnn,
)

# Technology 7: CNOT Optimization
from .linear import (
    synthesize_cnot_count_pmh,
    synthesize_cnot_depth_lnn_kms,
)

from .linear_phase import (
    synthesize_cnot_phase_aam,
    synthesize_cx_cz_depth_lnn_my,
    synthesize_cz_depth_lnn_mr,
)

# Helper utilities
from .one_qubit import OneQubitEulerDecomposer, DEFAULT_ATOL
