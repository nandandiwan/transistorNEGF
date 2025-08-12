import numpy as np
from scipy.optimize import brentq
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

def chandrupatla(f, x0, x1, verbose=False,
                 eps_m=None, eps_a=None,
                 rtol=1e-5, atol=0.0,
                 maxiter=50, return_iter=False, args=()):
    """Vectorized Chandrupatla root-finding method.

    Parameters
    ----------
    f : callable
        Function handle. Must accept (x,*args) where x can be a scalar or a
        NumPy array. The function must be vectorized w.r.t. x (element-wise).
    x0, x1 : scalar or array_like
        Bracketing interval bounds for each root. These can be scalars (then
        broadcast to the output shape) or arrays of the same shape as f(x0).
        For array use each pair (x0[i], x1[i]) must bracket a root: f(x0)*f(x1)<=0.
    verbose : bool or file-like, optional
        If True prints iteration diagnostics. If file-like writes there.
    eps_m, eps_a : float or array_like, optional (deprecated in favor of rtol/atol)
        Legacy names kept for backward compatibility. If provided they override
        rtol / atol. Relative (multiplicative) and absolute components of the
        stopping tolerance  tol = 2*eps_m*|x| + eps_a.
    rtol, atol : float or array_like, optional
        Preferred interface for relative/absolute tolerances. Default rtol=1e-5,
        atol=0.0. Can be broadcast to the shape of x. Effective tolerance used:
            tol = 2*rtol*|x| + atol
    maxiter : int, optional
        Maximum number of iterations.
    return_iter : bool, optional
        If True also return per-root iteration counts.
    args : tuple, optional
        Extra arguments passed to f.

    Returns
    -------
    roots : ndarray or scalar
        Approximated roots.
    iterations : ndarray (optional)
        Number of iterations performed for each root (only if return_iter).
    """
    # Evaluate function at initial end points (we copy to avoid mutating inputs)

    a0 = np.asarray(x0, dtype=float)
    b0 = np.asarray(x1, dtype=float)

    # Evaluate function at initial end points
    fa = f(a0, *args)
    fb = f(b0, *args)
    fa = np.asarray(fa)
    fb = np.asarray(fb)

    shape = fa.shape
    scalar_output = (shape == ())

    # Broadcasting diagnostics and actions
    if fa.shape != fb.shape:
        if verbose:
            print(f"Broadcasting fb and b0 from shape {fb.shape} to {fa.shape}")
        try:
            fb = np.broadcast_to(fb, fa.shape)
            b0 = np.broadcast_to(b0, fa.shape)
        except ValueError:
            raise ValueError("f(x0) and f(x1) shapes not broadcastable")

    if np.shape(a0) != shape:
        if verbose:
            print(f"Broadcasting a0 from shape {np.shape(a0)} to {shape}")
        a0 = np.broadcast_to(a0, shape).astype(float)
    if np.shape(b0) != shape:
        if verbose:
            print(f"Broadcasting b0 from shape {np.shape(b0)} to {shape}")
        b0 = np.broadcast_to(b0, shape).astype(float)


    a = a0.copy()
    b = b0.copy()

    # Check for valid brackets: allow equality (root at endpoint)
    if not np.all(np.sign(fa) * np.sign(fb) <= 0):
        bad = np.where(np.sign(fa) * np.sign(fb) > 0)
        if verbose:
            print(f"Bracket check failed at indices {bad}")
        #raise ValueError(f"Chandrupatla: supplied bounds do not bracket a root at indices {bad}")
        print("doubling range")
        chandrupatla(f, 3 * x0, 3*x1)

    c = a.copy()
    fc = fa.copy()

    # Determine effective tolerances (prefer new rtol/atol unless legacy given)
    if eps_m is not None:
        rtol = eps_m
    if eps_a is not None:
        atol = eps_a
    # Ensure arrays
    eps_m = np.asarray(rtol)  # rename for internal use
    eps_a = np.asarray(atol)
    if eps_m.shape not in ((), shape):
        if verbose:
            print(f"Broadcasting eps_m from shape {eps_m.shape} to {shape}")
        eps_m = np.broadcast_to(eps_m, shape)
    if eps_a.shape not in ((), shape):
        if verbose:
            print(f"Broadcasting eps_a from shape {eps_a.shape} to {shape}")
        eps_a = np.broadcast_to(eps_a, shape)

    t = np.full(shape, 0.5) if shape else 0.5
    terminate = np.zeros(shape, dtype=bool) if shape else False
    iterations = np.zeros(shape, dtype=int) if shape else 0

    for _ in range(maxiter):
        # Candidate new point
        xt = a + t * (b - a)
        ft = f(xt, *args)

        if verbose:
            output = f"t={t}\nxt={xt}\nft={ft}\na={a}\nfa={fa}\nb={b}\nfb={fb}\nc={c}\nfc={fc}"
            if verbose is True:
                print(output)
            else:
                try:
                    print(output, file=verbose)
                except Exception:
                    print(output)

        # Determine sign agreement
        samesign = np.sign(ft) == np.sign(fa)

        # Save old values for those not overwritten
        a_old, fa_old = a, fa
        b_old, fb_old = b.copy(), fb.copy()
        c_old, fc_old = c.copy(), fc.copy()

        # Update per scalar formulation using boolean masks
        # where samesign: c <- a_old; fc <- fa_old
        # else:          c <- b_old; fc <- fb_old; b <- a_old; fb <- fa_old
        if shape:
            c = np.where(samesign, a_old, b_old)
            fc = np.where(samesign, fa_old, fb_old)
            b = np.where(samesign, b_old, a_old)
            fb = np.where(samesign, fb_old, fa_old)
        else:
            if samesign:
                c = a_old
                fc = fa_old
            else:
                c = b_old
                fc = fb_old
                b = a_old
                fb = fa_old

        a = xt
        fa = ft

        # xm: point with smaller |f|
        fa_is_smaller = np.abs(fa) < np.abs(fb)
        if shape:
            xm = np.where(fa_is_smaller, a, b)
            fm = np.where(fa_is_smaller, fa, fb)
        else:
            xm = a if fa_is_smaller else b
            fm = fa if fa_is_smaller else fb

        tol = 2 * eps_m * np.abs(xm) + eps_a
        denom = np.abs(b - c)
        denom = np.where(denom == 0, 1.0, denom)  # avoid divide by zero
        tlim = tol / denom
        new_terminate = (fm == 0) | (tlim > 0.5) | terminate
        if shape:
            iterations[~new_terminate] += 1
        else:
            if not new_terminate:
                iterations += 1
        terminate = new_terminate
        if np.all(terminate):
            break

        # Compute xi, phi
        with np.errstate(divide='ignore', invalid='ignore'):
            xi = (a - b) / (c - b)
            phi = (fa - fb) / (fc - fb)

        # Inverse quadratic interpolation condition
        iqi = (phi ** 2 < xi) & ((1 - phi) ** 2 < 1 - xi)

        if shape:
            t = np.full(shape, 0.5)
            if np.any(iqi):
                # Only compute for those indices to avoid invalid divisions
                ai = a[iqi]; bi = b[iqi]; ci = c[iqi]
                fai = fa[iqi]; fbi = fb[iqi]; fci = fc[iqi]
                with np.errstate(divide='ignore', invalid='ignore'):
                    ti = (fai / (fbi - fai)) * (fci / (fbi - fci)) + \
                         ((ci - ai) / (bi - ai)) * (fai / (fci - fai)) * (fbi / (fci - fbi))
                # Fallback to bisection if nan/inf
                bad = ~np.isfinite(ti)
                if np.any(bad):
                    ti[bad] = 0.5
                t[iqi] = ti
        else:
            if iqi and np.isfinite(fa) and np.isfinite(fb) and np.isfinite(fc):
                t = (fa / (fb - fa)) * (fc / (fb - fc)) + ((c - a) / (b - a)) * (fa / (fc - fa)) * (fb / (fc - fb))
            else:
                t = 0.5

        # Clamp t to (tlim, 1 - tlim)
        if shape:
            t = np.minimum(1 - tlim, np.maximum(tlim, t))
        else:
            t = min(1 - tlim, max(tlim, t))

    roots = xm if 'xm' in locals() else a  # fallback
    if scalar_output:
        roots = np.asarray(roots).item()
        iterations = int(iterations)
    if return_iter:
        return roots, iterations
    return roots
    
    