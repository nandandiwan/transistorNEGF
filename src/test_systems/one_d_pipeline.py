import os
import time
from typing import Tuple, List

# Limit threading to keep runs reproducible/sane on laptops
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt

from hamiltonian import Hamiltonian
from rgf import GreensFunction
import poisson


def self_consistent_poisson_negf(ham: Hamiltonian,
                                 gf: GreensFunction,
                                 Vs: float,
                                 Vd: float,
                                 Vg: float,
                                 tol: float = 1e-4,
                                 max_iter: int = 50,
                                 verbose: bool = True,
                                 V_init: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a self-consistent Poisson–NEGF loop at a given (Vs, Vd, Vg).

    Returns (V_converged, charge_density_last).
    """
    ham.set_voltage(Vs=Vs, Vd=Vd, Vg=Vg)

    # Initial Poisson solve (as in notebook snippet)

    ham.clear_potential()
    V_old, charge_density = poisson.solve_poisson_nonlinear(ham, gf, np.zeros(ham.N))


    err = np.inf
    it = 0
    t0 = time.time()
    while err > tol and it < max_iter:
        it += 1
        gf.clear_ldos_cache()
        ham.set_potential(V_old)
        # Compute quasi-Fermi level self-consistently for this potential
        Efn = gf.fermi_energy(V_old)
        V_new, charge_density = poisson.solve_poisson_nonlinear(ham, gf, Efn)
        err = float(np.max(np.abs(V_new - V_old)))
        V_old = V_new
        if verbose:
            print(f"SCF iter {it:02d} | err = {err:.3e}")

    if verbose:
        print(f"SCF {'converged' if err <= tol else 'stopped'} in {it} iters, {time.time()-t0:.2f}s")

    return V_old, charge_density


def compute_current(ham: Hamiltonian, gf: GreensFunction) -> float:
    """Compute total current using Landauer integration (uses ham.mu1/mu2)."""
    I = float(gf.compute_total_current())
    return I


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sweep_id_vg(ham: Hamiltonian, gf: GreensFunction,
                Vds: float = 0.5,
                Vg_list: np.ndarray | List[float] = None,
                out_dir: str = "pipeline_plots",
                reuse_prev_potential: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if Vg_list is None:
        Vg_list = np.linspace(-1.0, 1.0, 11)
    Vg_list = np.asarray(Vg_list, dtype=float)

    ensure_dir(out_dir)

    Id = np.zeros_like(Vg_list, dtype=float)
    V_prev = None
    for i, Vg in enumerate(Vg_list):
        # Symmetric biasing around Ef
        Vs, Vd = +0.5 * Vds, -0.5 * Vds
        V_init = V_prev if reuse_prev_potential else None
        print(f"\n=== Id–Vg sweep: Vds={Vds:.3f} V, Vg={Vg:.3f} V ===")
        V_conv, _ = self_consistent_poisson_negf(ham, gf, Vs, Vd, Vg, V_init=V_init, verbose=False)
        ham.set_potential(V_conv)
        Id[i] = compute_current(ham, gf)
        V_prev = V_conv

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(Vg_list, Id, marker='o')
    plt.xlabel('Vg (V)')
    plt.ylabel('Id (A)')
    plt.title(f'Id–Vg at Vds={Vds:.3f} V')
    plt.grid(True)
    fpath = os.path.join(out_dir, f"id_vs_vg_vds_{Vds:.3f}.png")
    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()

    # Save raw data
    np.savez(os.path.join(out_dir, f"id_vs_vg_vds_{Vds:.3f}.npz"), Vg=Vg_list, Id=Id)
    return Vg_list, Id


def sweep_id_vds(ham: Hamiltonian, gf: GreensFunction,
                 Vg: float = 0.0,
                 Vds_list: np.ndarray | List[float] = None,
                 out_dir: str = "pipeline_plots",
                 reuse_prev_potential: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if Vds_list is None:
        Vds_list = np.linspace(0.0, 1.0, 11)
    Vds_list = np.asarray(Vds_list, dtype=float)

    ensure_dir(out_dir)

    Id = np.zeros_like(Vds_list, dtype=float)
    V_prev = None
    for i, Vds in enumerate(Vds_list):
        Vs, Vd = +0.5 * Vds, -0.5 * Vds
        V_init = V_prev if reuse_prev_potential else None
        print(f"\n=== Id–Vds sweep: Vg={Vg:.3f} V, Vds={Vds:.3f} V ===")
        V_conv, _ = self_consistent_poisson_negf(ham, gf, Vs, Vd, Vg, V_init=V_init, verbose=False)
        ham.set_potential(V_conv)
        Id[i] = compute_current(ham, gf)
        V_prev = V_conv

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(Vds_list, Id, marker='o')
    plt.xlabel('Vds (V)')
    plt.ylabel('Id (A)')
    plt.title(f'Id–Vds at Vg={Vg:.3f} V')
    plt.grid(True)
    fpath = os.path.join(out_dir, f"id_vs_vds_vg_{Vg:.3f}.png")
    plt.tight_layout()
    plt.savefig(fpath, dpi=150)
    plt.close()

    # Save raw data
    np.savez(os.path.join(out_dir, f"id_vs_vds_vg_{Vg:.3f}.npz"), Vds=Vds_list, Id=Id)
    return Vds_list, Id


def main():
    # Device setup (mirrors the notebook snippet defaults)
    ham = Hamiltonian("one_d_wire")
    ham.o = 0.0
    ham.t = 1.0

    gf = GreensFunction(ham)

    # Example single SCF run and potential plot
    Vs0, Vd0, Vg0 = 2.0, 0.0, 3.0
    V_conv, _ = self_consistent_poisson_negf(ham, gf, Vs0, Vd0, Vg0, verbose=True)
    ham.set_potential(V_conv)

    out_dir = "pipeline_plots"
    ensure_dir(out_dir)

    # Save potential profile plot
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(V_conv)), V_conv)
    plt.xlabel('Device Index')
    plt.ylabel('Potential V (eV)')
    plt.title('Converged Potential Profile')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "potential_profile.png"), dpi=150)
    plt.close()

    # Compute one current as a sanity check
    I0 = compute_current(ham, gf)
    print(f"Current at (Vs={Vs0}, Vd={Vd0}, Vg={Vg0}): {I0:.6e} A")

    # Id–Vg at fixed Vds
    _ = sweep_id_vg(ham, gf, Vds=0.5, Vg_list=np.linspace(-1, 1, 9))

    # Id–Vds at fixed Vg
    _ = sweep_id_vds(ham, gf, Vg=0.0, Vds_list=np.linspace(0, 1.0, 9))


if __name__ == "__main__":
    main()
