import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.linalg import inv
import warnings

def smart_inverse(A, sparse_threshold=0.1):
    """
    Compute the inverse of a matrix A.
    If A is sparse and density < sparse_threshold, use spsolve.
    Otherwise, use dense inverse.
    Returns a dense or sparse matrix as appropriate.
    """
    if sp.issparse(A):
        density = A.nnz / (A.shape[0] * A.shape[1])
        if density < sparse_threshold:
            n = A.shape[0]
            I = sp.identity(n, dtype=A.dtype, format='csc')
            try:
                A_inv = spsolve(A, I)
                return sp.csc_matrix(A_inv)
            except Exception:
                warnings.warn("spsolve failed, falling back to dense inverse.")
                A = A.toarray()
        else:
            A = A.toarray()
    # Now A is dense
    try:
        return inv(A)
    except Exception:
        warnings.warn("Dense inverse failed, using pseudo-inverse.")
        return np.linalg.pinv(A)
def FD_half(x):
    '''
    Approximation of the Fermi-Dirac integral of order 1/2.
    Reference: http://dx.doi.org/10.1063/1.4825209
    '''
    v = x**4 + 50 + 33.6 * x * (1 - 0.68 * np.exp(-0.17 * (x + 1)**2))
    return 1 / (np.exp(-x) + 3 * np.pi**0.5 / 4 * v**(-3/8))

# Define F_-1/2(x) as the derivative of F_1/2(x)
def FD_minus_half(x):
    dx = x * 1e-6  
    return (FD_half(x + dx) - FD_half(x - dx)) / (2 * dx)

def sparse_diag_product(A, B):
    """
    Compute diagonal elements of C = A * B efficiently for sparse matrices.

    Parameters:
        A (csr_matrix): sparse matrix in CSR format
        B (csc_matrix): sparse matrix in CSC format

    Returns:
        numpy.ndarray: diagonal elements of A*B
    """
    from scipy.sparse import csr_matrix, csc_matrix
    import numpy as np

    # Ensure A is CSR and B is CSC for efficient indexing
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A)
    if not isinstance(B, csc_matrix):
        B = csc_matrix(B)

    n = A.shape[0]
    diag = np.zeros(n, dtype=complex)

    for i in range(n):
        # Get row i from A (CSR format)
        A_row_start, A_row_end = A.indptr[i], A.indptr[i+1]
        A_cols = A.indices[A_row_start:A_row_end]
        A_vals = A.data[A_row_start:A_row_end]

        # Get column i from B (CSC format)

        B_rows = B.indices[B.indptr[i]:B.indptr[i+1]]
        B_vals = B.data[B.indptr[i]:B.indptr[i+1]]

        # Compute intersection of indices efficiently
        ptr_a, ptr_b = 0, 0
        sum_diagonal = 0.0
        while ptr_a < len(A_cols) and ptr_b < len(B_rows):
            col_a, row_b = A_cols[ptr_a], B_rows[ptr_b]
            if col_a == row_b:
                sum_diagonal += A_vals[ptr_a] * B_vals[ptr_b]
                ptr_a += 1
                ptr_b += 1
            elif col_a < row_b:
                ptr_a += 1
            else:
                ptr_b += 1

        diag[i] = sum_diagonal

    return diag