import numpy as np
from scipy.optimize import brentq
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.linalg import inv
import warnings
from __future__ import print_function
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

def chandrupatla(f,x0,x1,verbose=False, 
                 eps_m = None, eps_a = None, 
                 maxiter=50, return_iter=False, args=(),):
    # copied from https://github.com/scipy/scipy/issues/7242#issuecomment-290548427
    
    # Initialization
    b = x0
    a = x1
    fa = f(a, *args)
    fb = f(b, *args)
    
    # Make sure we know the size of the result
    shape = np.shape(fa)
    assert shape == np.shape(fb)
        
    # In case x0, x1 are scalars, make sure we broadcast them to become the size of the result
    b += np.zeros(shape)
    a += np.zeros(shape)

    fc = fa
    c = a
    
    # Make sure we are bracketing a root in each case
    assert (np.sign(fa) * np.sign(fb) <= 0).all()
    t = 0.5
    # Initialize an array of False,
    # determines whether we should do inverse quadratic interpolation
    iqi = np.zeros(shape, dtype=bool)
    
    # jms: some guesses for default values of the eps_m and eps_a settings
    # based on machine precision... not sure exactly what to do here
    eps = np.finfo(float).eps
    if eps_m is None:
        eps_m = eps
    if eps_a is None:
        eps_a = 2*eps
    
    iterations = 0
    terminate = False
    
    while maxiter > 0:
        maxiter -= 1
        # use t to linearly interpolate between a and b,
        # and evaluate this function as our newest estimate xt
        xt = a + t*(b-a)
        ft = f(xt, *args)
        if verbose:
            output = 'IQI? %s\nt=%s\nxt=%s\nft=%s\na=%s\nb=%s\nc=%s' % (iqi,t,xt,ft,a,b,c)
            if verbose == True:
                print(output)
            else:
                print(output,file=verbose)
        # update our history of the last few points so that
        # - a is the newest estimate (we're going to update it from xt)
        # - c and b get the preceding two estimates
        # - a and b maintain opposite signs for f(a) and f(b)
        samesign = np.sign(ft) == np.sign(fa)
        c  = np.choose(samesign, [b,a])
        b  = np.choose(samesign, [a,b])
        fc = np.choose(samesign, [fb,fa])
        fb = np.choose(samesign, [fa,fb])
        a  = xt
        fa = ft
        
        # set xm so that f(xm) is the minimum magnitude of f(a) and f(b)
        fa_is_smaller = np.abs(fa) < np.abs(fb)
        xm = np.choose(fa_is_smaller, [b,a])
        fm = np.choose(fa_is_smaller, [fb,fa])
        
        """
        the preceding lines are a vectorized version of:

        samesign = np.sign(ft) == np.sign(fa)        
        if samesign
            c = a
            fc = fa
        else:
            c = b
            b = a
            fc = fb
            fb = fa

        a = xt
        fa = ft
        # set xm so that f(xm) is the minimum magnitude of f(a) and f(b)
        if np.abs(fa) < np.abs(fb):
            xm = a
            fm = fa
        else:
            xm = b
            fm = fb
        """
        
        tol = 2*eps_m*np.abs(xm) + eps_a
        tlim = tol/np.abs(b-c)
        terminate = np.logical_or(terminate, np.logical_or(fm==0, tlim > 0.5))
        if verbose:            
            output = "fm=%s\ntlim=%s\nterm=%s" % (fm,tlim,terminate)
            if verbose == True:
                print(output)
            else:
                print(output, file=verbose)

        if np.all(terminate):
            break
        iterations += 1-terminate
        
        # Figure out values xi and phi 
        # to determine which method we should use next
        xi  = (a-b)/(c-b)
        phi = (fa-fb)/(fc-fb)
        iqi = np.logical_and(phi**2 < xi, (1-phi)**2 < 1-xi)
            
        if not shape:
            # scalar case
            if iqi:
                # inverse quadratic interpolation
                t = fa / (fb-fa) * fc / (fb-fc) + (c-a)/(b-a)*fa/(fc-fa)*fb/(fc-fb)
            else:
                # bisection
                t = 0.5
        else:
            # array case
            t = np.full(shape, 0.5)
            a2,b2,c2,fa2,fb2,fc2 = a[iqi],b[iqi],c[iqi],fa[iqi],fb[iqi],fc[iqi]
            t[iqi] = fa2 / (fb2-fa2) * fc2 / (fb2-fc2) + (c2-a2)/(b2-a2)*fa2/(fc2-fa2)*fb2/(fc2-fb2)
        
        # limit to the range (tlim, 1-tlim)
        t = np.minimum(1-tlim, np.maximum(tlim, t))
        
    # done!
    if return_iter:
        return xm, iterations
    else:
        return xm
    
    
def f(x,a):
    return a-x*x

k=np.arange(1,8)
y = chandrupatla(f,0,3,args=(k,))
print(y)
print(k-y**2)

y = chandrupatla(f,0,3,args=(7.0,))
print(y)
print(7-y**2)
