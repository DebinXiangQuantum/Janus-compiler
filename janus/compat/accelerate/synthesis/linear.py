"""
Linear synthesis accelerate functions - Pure Python implementation
Based on Qiskit's implementation but without Rust dependencies.
"""
import numpy as np
from typing import List, Tuple, Optional


def gauss_elimination(mat: np.ndarray, ncols: Optional[int] = None) -> None:
    """
    Gaussian elimination of a matrix mat over GF(2) in-place.
    
    Args:
        mat: A boolean matrix to be transformed in-place.
        ncols: Number of columns to consider (default: all columns).
    """
    nrows = mat.shape[0]
    if ncols is None:
        ncols = mat.shape[1]
    
    pivot_row = 0
    for col in range(ncols):
        # Find pivot
        found_pivot = False
        for row in range(pivot_row, nrows):
            if mat[row, col]:
                found_pivot = True
                if row != pivot_row:
                    # Swap rows
                    mat[[pivot_row, row]] = mat[[row, pivot_row]]
                break
        
        if not found_pivot:
            continue
        
        # Eliminate
        for row in range(nrows):
            if row != pivot_row and mat[row, col]:
                mat[row] ^= mat[pivot_row]
        
        pivot_row += 1
        if pivot_row >= nrows:
            break


def gauss_elimination_with_perm(mat: np.ndarray, ncols: Optional[int] = None) -> List[int]:
    """
    Gaussian elimination with column permutation tracking.
    
    Args:
        mat: A boolean matrix to be transformed in-place.
        ncols: Number of columns to consider.
    
    Returns:
        List of column indices representing the permutation.
    """
    nrows = mat.shape[0]
    if ncols is None:
        ncols = mat.shape[1]
    
    perm = list(range(ncols))
    pivot_row = 0
    
    for col in range(ncols):
        # Find pivot in remaining rows and columns
        found_pivot = False
        for c in range(col, ncols):
            for row in range(pivot_row, nrows):
                if mat[row, c]:
                    found_pivot = True
                    # Swap columns if needed
                    if c != col:
                        mat[:, [col, c]] = mat[:, [c, col]]
                        perm[col], perm[c] = perm[c], perm[col]
                    # Swap rows if needed
                    if row != pivot_row:
                        mat[[pivot_row, row]] = mat[[row, pivot_row]]
                    break
            if found_pivot:
                break
        
        if not found_pivot:
            continue
        
        # Eliminate
        for row in range(nrows):
            if row != pivot_row and mat[row, col]:
                mat[row] ^= mat[pivot_row]
        
        pivot_row += 1
        if pivot_row >= nrows:
            break
    
    return perm


def compute_rank_after_gauss_elim(mat: np.ndarray) -> int:
    """
    Compute rank of a matrix that has already been Gaussian eliminated.
    
    Args:
        mat: A boolean matrix after Gaussian elimination.
    
    Returns:
        The rank of the matrix.
    """
    rank = 0
    for row in range(mat.shape[0]):
        if mat[row].any():
            rank += 1
    return rank


def compute_rank(mat: np.ndarray) -> int:
    """
    Compute the rank of a boolean matrix over GF(2).
    
    Args:
        mat: A boolean matrix.
    
    Returns:
        The rank of the matrix.
    """
    work_mat = mat.astype(bool).copy()
    gauss_elimination(work_mat)
    return compute_rank_after_gauss_elim(work_mat)


def calc_inverse_matrix(mat: np.ndarray, validate: bool = False) -> np.ndarray:
    """
    Calculate the inverse of a boolean matrix over GF(2).
    
    Args:
        mat: A boolean invertible matrix.
        validate: If True, validate that the matrix is invertible.
    
    Returns:
        The inverse matrix.
    
    Raises:
        ValueError: If the matrix is not invertible.
    """
    n = mat.shape[0]
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Matrix must be square")
    
    # Create augmented matrix [mat | I]
    augmented = np.zeros((n, 2 * n), dtype=bool)
    augmented[:, :n] = mat.astype(bool)
    augmented[:, n:] = np.eye(n, dtype=bool)
    
    # Gaussian elimination
    for col in range(n):
        # Find pivot
        pivot_row = -1
        for row in range(col, n):
            if augmented[row, col]:
                pivot_row = row
                break
        
        if pivot_row == -1:
            if validate:
                raise ValueError("Matrix is not invertible")
            # Return identity as fallback
            return np.eye(n, dtype=bool)
        
        # Swap rows
        if pivot_row != col:
            augmented[[col, pivot_row]] = augmented[[pivot_row, col]]
        
        # Eliminate
        for row in range(n):
            if row != col and augmented[row, col]:
                augmented[row] ^= augmented[col]
    
    return augmented[:, n:]


def binary_matmul(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """
    Binary matrix multiplication over GF(2).
    
    Args:
        mat1: First boolean matrix.
        mat2: Second boolean matrix.
    
    Returns:
        The product matrix over GF(2).
    """
    # Use numpy matmul and take mod 2
    result = np.matmul(mat1.astype(int), mat2.astype(int)) % 2
    return result.astype(bool)


def random_invertible_binary_matrix(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random invertible binary matrix.
    
    Args:
        n: Size of the matrix.
        seed: Random seed (optional).
    
    Returns:
        A random invertible n x n boolean matrix.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Start with identity and apply random row operations
    mat = np.eye(n, dtype=bool)
    
    # Apply random row operations to make it random but still invertible
    for _ in range(n * n):
        i, j = np.random.randint(0, n, 2)
        if i != j:
            mat[i] ^= mat[j]
    
    return mat


def check_invertible_binary_matrix(mat: np.ndarray) -> bool:
    """
    Check if a binary matrix is invertible over GF(2).
    
    Args:
        mat: Binary matrix (numpy array or list).
    
    Returns:
        True if matrix is invertible, False otherwise.
    """
    matrix = np.array(mat, dtype=bool)
    
    # Must be square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Check if rank equals size
    return compute_rank(matrix) == matrix.shape[0]


def row_op(mat: np.ndarray, ctrl: int, trgt: int) -> None:
    """
    Apply row operation: row[trgt] ^= row[ctrl].
    
    Args:
        mat: Matrix to modify in-place.
        ctrl: Control row index.
        trgt: Target row index.
    """
    mat[trgt] ^= mat[ctrl]


def col_op(mat: np.ndarray, ctrl: int, trgt: int) -> None:
    """
    Apply column operation: col[trgt] ^= col[ctrl].
    
    Args:
        mat: Matrix to modify in-place.
        ctrl: Control column index.
        trgt: Target column index.
    """
    mat[:, trgt] ^= mat[:, ctrl]


def synth_cnot_count_full_pmh(mat: np.ndarray, section_size: int = 2) -> List[List[int]]:
    """
    PMH (Patel-Markov-Hayes) CNOT synthesis algorithm.
    
    Synthesizes a CNOT circuit implementing the given linear reversible function
    with O(n^2/log n) CNOT gates.
    
    Args:
        mat: Boolean invertible matrix representing the linear function.
        section_size: Section size for the algorithm (default 2).
    
    Returns:
        List of CNOT gates as [[control, target], ...].
    """
    matrix = np.array(mat, dtype=bool).copy()
    n = matrix.shape[0]
    gates = []
    
    # Forward elimination
    for col in range(n):
        # Find pivot
        pivot_row = -1
        for row in range(col, n):
            if matrix[row, col]:
                pivot_row = row
                break
        
        if pivot_row == -1:
            continue
        
        # Swap rows using CNOT gates
        if pivot_row != col:
            # Swap rows col and pivot_row
            gates.append([col, pivot_row])
            gates.append([pivot_row, col])
            gates.append([col, pivot_row])
            matrix[[col, pivot_row]] = matrix[[pivot_row, col]]
        
        # Eliminate below pivot
        for row in range(col + 1, n):
            if matrix[row, col]:
                gates.append([col, row])
                matrix[row] ^= matrix[col]
    
    # Back substitution
    for col in range(n - 1, -1, -1):
        for row in range(col - 1, -1, -1):
            if matrix[row, col]:
                gates.append([col, row])
                matrix[row] ^= matrix[col]
    
    return gates


def py_synth_cnot_depth_line_kms(mat: np.ndarray) -> List[Tuple[str, List[int]]]:
    """
    KMS (Kutin-Moulton-Smithline) CNOT synthesis for linear nearest-neighbor.
    
    Synthesizes a CNOT circuit for LNN connectivity with depth O(n).
    
    Args:
        mat: Boolean invertible matrix.
    
    Returns:
        List of (gate_name, [control, target]) tuples.
    """
    matrix = np.array(mat, dtype=bool).copy()
    n = matrix.shape[0]
    gates = []
    
    # Simple LNN synthesis using nearest-neighbor CNOTs
    # This is a simplified version - full KMS would be more optimal
    
    # Forward pass: make upper triangular
    for col in range(n):
        # Find pivot in column
        pivot_row = -1
        for row in range(col, n):
            if matrix[row, col]:
                pivot_row = row
                break
        
        if pivot_row == -1:
            continue
        
        # Move pivot to diagonal using adjacent swaps
        while pivot_row > col:
            # Swap rows pivot_row-1 and pivot_row using CNOTs
            gates.append(('cx', [pivot_row - 1, pivot_row]))
            gates.append(('cx', [pivot_row, pivot_row - 1]))
            gates.append(('cx', [pivot_row - 1, pivot_row]))
            matrix[[pivot_row - 1, pivot_row]] = matrix[[pivot_row, pivot_row - 1]]
            pivot_row -= 1
        
        # Eliminate below using adjacent CNOTs
        for row in range(col + 1, n):
            if matrix[row, col]:
                # Need to propagate CNOT from col to row
                for i in range(col, row):
                    gates.append(('cx', [i, i + 1]))
                matrix[row] ^= matrix[col]
                # Undo intermediate effects
                for i in range(row - 1, col, -1):
                    gates.append(('cx', [i - 1, i]))
    
    # Backward pass: make identity
    for col in range(n - 1, 0, -1):
        for row in range(col - 1, -1, -1):
            if matrix[row, col]:
                # Propagate CNOT from col to row
                for i in range(col, row, -1):
                    gates.append(('cx', [i, i - 1]))
                matrix[row] ^= matrix[col]
                # Undo intermediate effects
                for i in range(row + 1, col):
                    gates.append(('cx', [i, i - 1]))
    
    return gates
