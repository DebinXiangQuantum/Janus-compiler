# This code is part of Janus Quantum Compiler.
"""Module containing linear (CNOT) circuit synthesis."""

# Technology 7: CNOT Optimization (Linear synthesis)
from .cnot_synth import synthesize_cnot_count_pmh
from .linear_depth_lnn import synthesize_cnot_depth_lnn_kms

# Helper utilities
from .linear_matrix_utils import (
    calc_inverse_matrix,
    random_invertible_binary_matrix,
    check_invertible_binary_matrix,
    binary_matmul,
)
from .linear_circuits_utils import transpose_cx_circ
