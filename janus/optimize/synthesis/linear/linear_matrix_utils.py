"""Utility functions for handling binary matrices."""

# pylint: disable=unused-import
# Import from janus.compat.accelerate.synthesis.linear which has pure Python implementations
from janus.compat.accelerate.synthesis.linear import (
    gauss_elimination,
    gauss_elimination_with_perm,
    compute_rank_after_gauss_elim,
    compute_rank,
    calc_inverse_matrix,
    binary_matmul,
    random_invertible_binary_matrix,
    check_invertible_binary_matrix,
    row_op,
    col_op,
)

__all__ = [
    'gauss_elimination',
    'gauss_elimination_with_perm',
    'compute_rank_after_gauss_elim',
    'compute_rank',
    'calc_inverse_matrix',
    'binary_matmul',
    'random_invertible_binary_matrix',
    'check_invertible_binary_matrix',
    'row_op',
    'col_op',
]
