import os
import numpy as np
import pytest

# Limit threading for deterministic, fast tests
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from hamiltonian import Hamiltonian  # noqa: E402
from rgf import GreensFunction       # noqa: E402


def _conduction_edge(eigs, tol=1e-9):
    """Return smallest positive eigenvalue as conduction band edge."""
    pos = eigs[eigs > tol]
    if pos.size == 0:
        raise RuntimeError("No positive eigenvalues found for conduction edge.")
    return float(np.min(pos))


def build_ssh_gf(N=20, t1=1.0, t2=0.5, energy_span=3.0, nE=121):
    ham = Hamiltonian("ssh", periodic=False)
    # The Hamiltonian class appears to use ham.N for total sites; ensure even.
    ham.N = N if N % 2 == 0 else N + 1
    energy = np.linspace(-energy_span, energy_span, nE)
    gf = GreensFunction(ham, energy_grid=energy)
    # Force serial to avoid multiprocessing overhead in tests
    gf.force_serial = True
    return ham, gf


@pytest.mark.fast
def test_fermi_energy_consistent_mode_converges_for_ssh():
    ham, gf = build_ssh_gf()
    # Full matrix for eigenvalues
    H_full = ham.create_hamiltonian(blocks=False).toarray()
    eigs = np.linalg.eigvalsh(H_full)
    Ec = _conduction_edge(eigs) - 1e-6  # a hair below conduction edge

    V = np.zeros(ham.get_num_sites())
    lower = np.full_like(V, Ec - 0.4)
    upper = np.full_like(V, Ec + 0.4)

    # Solve in consistent mode (target density defined via same method)
    efn = gf.fermi_energy(V, lower_bound=lower, upper_bound=upper, Ec=Ec,
                          mode='consistent', f_tol=1e-6,
                          get_n_kwargs={'method': 'gauss_fermi'})

    assert np.all(np.isfinite(efn)), "Efn contains NaN/inf in consistent mode"
    # Reconstruct midpoint used internally
    mid = 0.5 * (lower + upper)
    n_root = gf.get_n(V=V, Efn=efn, Ec=Ec, method='gauss_fermi')
    n_target = gf.get_n(V=V, Efn=mid, Ec=Ec, method='gauss_fermi')
    residual = n_root - n_target
    max_resid = np.max(np.abs(residual))
    assert max_resid < 5e-6, f"Residual too large after convergence: {max_resid}"


@pytest.mark.fast
def test_fermi_energy_inconsistent_mode_flags_bad_Ec():
    ham, gf = build_ssh_gf()
    H_full = ham.create_hamiltonian(blocks=False).toarray()
    eigs = np.linalg.eigvalsh(H_full)
    Ec_true = _conduction_edge(eigs) - 1e-6
    # Pick a bad Ec well inside the gap (shift upward by half the gap so conduction states excluded)
    gap = np.min(eigs[eigs > 0]) - np.max(eigs[eigs < 0])
    Ec_bad = Ec_true + 0.5 * gap

    V = np.zeros(ham.get_num_sites())
    lower = np.full_like(V, Ec_bad - 0.4)
    upper = np.full_like(V, Ec_bad + 0.4)

    efn_bad = gf.fermi_energy(V, lower_bound=lower, upper_bound=upper, Ec=Ec_bad,
                              mode='inconsistent', f_tol=1e-8,
                              get_n_kwargs={'method': 'gauss_fermi'})

    # Evaluate residual against full lesser density target
    n_target_full = gf.compute_charge_density()
    n_est = gf.get_n(V=V, Efn=np.nan_to_num(efn_bad, nan=Ec_bad), Ec=Ec_bad, method='gauss_fermi')
    residual = n_est - n_target_full
    print(residual)
    # Expect either NaNs (marked) or large residual indicating mismatch
    large_resid = np.max(np.abs(residual))
    assert (np.any(~np.isfinite(efn_bad)) or large_resid > 1e-3), (
        "Inconsistent mode with bad Ec neither produced NaNs nor large residual; detection failed"
    )


if __name__ == "__main__":  # Manual quick run
    test_fermi_energy_consistent_mode_converges_for_ssh()
    test_fermi_energy_inconsistent_mode_flags_bad_Ec()
    print("Tests executed manually.")
