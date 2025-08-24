import numpy as np
from functools import lru_cache

@lru_cache(maxsize=None)
def get_ozaki_poles_residues(cutoff:int, kT:float, T_key:float|None=None):
    """Return (poles,residues) for Ozaki continued-fraction Fermi approximation.
    Cached globally by (cutoff, kT, T_key). T_key can be an external temperature marker
    (e.g. physical temperature in K) to force regeneration when temperature changes even
    if kT matches numerically.
    """
    # Build symmetric tridiagonal Jacobi matrix
    j = np.arange(cutoff - 1)
    b = 1.0 / (2.0 * np.sqrt((2 * (j + 1) - 1) * (2 * (j + 1) + 1)))
    J = np.zeros((cutoff, cutoff), dtype=float)
    J[j, j + 1] = b
    J[j + 1, j] = b
    vals, vecs = np.linalg.eigh(J)
    mask = vals > 0
    poles = vals[mask]
    v0 = vecs[0, mask]
    residues = 0.25 * (np.abs(v0) ** 2) / (poles ** 2)
    return poles, residues

def fermi_cfr(E: np.ndarray, mu: float, poles: np.ndarray, residues: np.ndarray, kT: float) -> np.ndarray:
    """Ozaki CFR approximation to Fermi-Dirac distribution.
    E, mu in eV; kT in eV.
    """
    x = (E - mu) / kT
    x_col = x[:, None]
    aj = poles[None, :]
    rj = residues[None, :]
    # (1/(x - i/a) + 1/(x + i/a)) simplifies to 2x/(x^2 + 1/a^2)
    inv_a2 = (1.0 / aj) ** 2
    denom = x_col**2 + inv_a2
    s = 2.0 * x_col * rj / denom
    f = 0.5 - np.sum(s, axis=1)
    return f.real

def fermi_derivative_cfr_abs(E: np.ndarray, V_vec: np.ndarray, Efn_vec: np.ndarray,
                             poles: np.ndarray, residues: np.ndarray, kT: float) -> np.ndarray:
    """Return positive quantity corresponding to -df/dx (the usual |df/dx| of Fermi) for each site.
    We evaluate for each energy E a vector over sites: x_i = (E - V_i - Efn_i)/kT.
    Output shape: (n_sites,) per energy call.
    Formula: f(x) = 1/2 - Σ_j 2 r_j x / (x^2 + β_j), β_j = 1/a_j^2
    df/dx = - Σ_j 2 r_j (β_j - x^2)/(x^2 + β_j)^2, so |df/dx| = Σ_j 2 r_j (β_j - x^2)/(x^2 + β_j)^2
    which is positive near x=0; if numerical negatives arise (far tails) we clamp to >=0.
    """
    x = (E - V_vec - Efn_vec) / kT  # shape (n_sites,)
    x2 = x * x
    beta = (1.0 / poles) ** 2  # shape (m,)
    # Compute contribution sum_j 2 r_j (beta_j - x^2)/(x^2 + beta_j)^2 for each site.
    # Vectorize: expand x2 to (n_sites,1)
    x2_col = x2[:, None]
    beta_row = beta[None, :]
    num = beta_row - x2_col
    denom = (x2_col + beta_row) ** 2
    contrib = 2.0 * residues[None, :] * num / denom
    val = np.sum(contrib, axis=1)
    # Clamp tiny negative due to rounding
    val[val < 0] = 0.0
    # |df/dx| = val; derivative wrt V: df/dV = +|df/dx|/kT as in existing implementation
    return val
