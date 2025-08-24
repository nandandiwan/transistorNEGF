from __future__ import annotations
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import dolfinx.fem as fem
from rgf import GreensFunction
import dolfinx.fem.petsc
import ufl
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple

q = 1.602176634e-19
k_B = 1.380649e-23

@dataclass
class GummelDiagnostics:
    residual_norms: List[float] = field(default_factory=list)
    update_norms: List[float] = field(default_factory=list)
    mixing_factors: List[float] = field(default_factory=list)
    anderson_used: List[bool] = field(default_factory=list)

class CoupledNEGFPoisson:
    def __init__(self,
                 mesh: dolfinx.mesh.Mesh,
                 V_space: fem.FunctionSpace,
                 gf : GreensFunction,
                 eps_rel: float,
                 N_D_profile: float | np.ndarray | Callable[[np.ndarray], np.ndarray],
                 site_volume_m3: float,
                 comm: MPI.Comm | None = None):
        self.mesh = mesh
        self.V_space = V_space
        self.gf = gf
        self.comm = comm or mesh.comm
        self.eps = eps_rel  
        self.site_volume_m3 = site_volume_m3

        # Doping representation
        self.N_D = fem.Function(V_space, name="DonorDoping")
        coords = V_space.tabulate_dof_coordinates()[:, 0]
        if np.isscalar(N_D_profile):
            self.N_D.x.array[:] = float(N_D_profile)
        elif callable(N_D_profile):
            self.N_D.x.array[:] = N_D_profile(coords)
        else:
            arr = np.asarray(N_D_profile)
            if arr.size == self.N_D.x.array.size:
                self.N_D.x.array[:] = arr
            else:
                raise ValueError("N_D_profile size mismatch with DOFs")
        self.N_D.x.scatter_forward()

        # FEM functions
        self.V_func = fem.Function(V_space, name="Potential")
        self.rho_func = fem.Function(V_space, name="ChargeDensity")
        self.drho_dV_func = fem.Function(V_space, name="ChargeDensityDerivative")

        # Test/Trial for dynamic assembly per iteration
        self.v_test = ufl.TestFunction(V_space)
        self.u_trial = ufl.TrialFunction(V_space)

        # Dirichlet BC container
        self.bcs: List[fem.DirichletBC] = []

        # Diagnostics
        self.diag = GummelDiagnostics()

    def set_dirichlet_bcs(self, left_value: float, right_value: float,
                          left_marker: Callable[[np.ndarray], np.ndarray],
                          right_marker: Callable[[np.ndarray], np.ndarray],
                          debug: bool = False):
        """Define Dirichlet boundary conditions for 1D interval mesh.
        Falls back to first/last DOF if geometric location returns empty.
        """
        mesh = self.mesh
        V_space = self.V_space
        fdim = mesh.topology.dim - 1
        # Robust DOF identification: simply pick min/max coordinate DOFs (1D interval)
        coords = V_space.tabulate_dof_coordinates()[:, 0]
        left_dof = int(np.argmin(coords))
        right_dof = int(np.argmax(coords))
        left_dofs = np.array([left_dof], dtype=np.int32)
        right_dofs = np.array([right_dof], dtype=np.int32)
        if left_dof == right_dof:
            raise RuntimeError("Left and right boundary DOFs coincide; mesh seems degenerate.")
        bc_left = fem.dirichletbc(PETSc.ScalarType(left_value), left_dofs, V_space)
        bc_right = fem.dirichletbc(PETSc.ScalarType(right_value), right_dofs, V_space)
        self.bcs = [bc_left, bc_right]
        # Persist values & DOFs for explicit re-application
        self._left_bc_value = float(left_value)
        self._right_bc_value = float(right_value)
        self._left_dofs = np.array(left_dofs, copy=True)
        self._right_dofs = np.array(right_dofs, copy=True)
        if debug and self.comm.rank == 0:
            print(f"[BC Debug] left_dof={left_dof} coord={coords[left_dof]:.3e} value={self._left_bc_value}; right_dof={right_dof} coord={coords[right_dof]:.3e} value={self._right_bc_value}")

    def debug_boundary_values(self):
        """Print current potential at boundary DOFs (for troubleshooting)."""
        coords = self.V_space.tabulate_dof_coordinates()[:,0]
        if hasattr(self, '_left_dofs'):
            print("[BC Debug] Current left DOF potentials:", list(zip(self._left_dofs.tolist(), coords[self._left_dofs], self.V_func.x.array[self._left_dofs])))
        if hasattr(self, '_right_dofs'):
            print("[BC Debug] Current right DOF potentials:", list(zip(self._right_dofs.tolist(), coords[self._right_dofs], self.V_func.x.array[self._right_dofs])))

    def _enforce_dirichlet(self, arr: np.ndarray):
        """Hard set boundary DOF values into provided array (in-place)."""
        if hasattr(self, '_left_dofs') and hasattr(self, '_left_bc_value'):
            arr[self._left_dofs] = self._left_bc_value
        if hasattr(self, '_right_dofs') and hasattr(self, '_right_bc_value'):
            arr[self._right_dofs] = self._right_bc_value

    def _negf_density_and_derivative(self, V_array: np.ndarray, Ec: float, Efn: np.ndarray,
                                     method: str, derivative_method: str,
                                     ozaki_cutoff: int,
                                     conduction_only: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Query NEGF object for density n and derivative dn/dV (per site).

        Returns:
            n (sites,)  electron density (# / site)
            dn_dV (sites,) derivative (# / site / Volt)
        """
        # Update GF potentials (assume gf expects eV; if not adapt here)
        self.gf.V = V_array.copy()
        self.gf.Efn = Efn.copy()
        # Use fast Ozaki for density by default
        n_vec = self.gf.get_n(
            Efn=Efn,
            V=V_array,
            Ec=Ec,
            method=method,
            conduction_only=conduction_only,
            processes=1,
            ozaki_cutoff=ozaki_cutoff,
        )
        # Derivative of density wrt potential
        dn_dV_vec = self.gf.diff_rho_poisson(
            Efn=Efn,
            V=V_array,
            Ec=Ec,
            method=derivative_method,
            processes=1,
            ozaki_cutoff=ozaki_cutoff,
        )
        return n_vec.real, dn_dV_vec.real

    def _update_charge_functions(self, n_sites: np.ndarray, dn_dV_sites: np.ndarray):
        """Map per-site density & derivative to FEM functions.
        Assumes 1-1 mapping DOF <-> site ordering.
        """
        if n_sites.size != self.V_func.x.array.size:
            raise ValueError("Site density size mismatch with FEM DOF count (non-uniform mapping not implemented).")
        # Convert to concentration (#/m^3)
        n_conc = n_sites / self.site_volume_m3
        dn_dV_conc = dn_dV_sites / self.site_volume_m3
        # rho = q (N_D - n)
        self.rho_func.x.array[:] = q * (self.N_D.x.array - n_conc)
        # d rho / d V = -q * dn/dV
        self.drho_dV_func.x.array[:] = -q * dn_dV_conc
        self.rho_func.x.scatter_forward()
        self.drho_dV_func.x.scatter_forward()

    def _assemble_linear_system(self):
        V_old = self.V_func
        v = self.v_test
        u = self.u_trial
        drho = self.drho_dV_func
        rho = self.rho_func
        a_form = (self.eps * ufl.dot(ufl.grad(u), ufl.grad(v)) + drho * u * v) * ufl.dx
        L_form = (drho * V_old * v - rho * v) * ufl.dx
        A = fem.petsc.assemble_matrix(fem.form(a_form), bcs=self.bcs)
        A.assemble()
        b = fem.petsc.assemble_vector(fem.form(L_form))
        fem.petsc.apply_lifting(b, [fem.form(a_form)], bcs=[self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, self.bcs)
        return A, b

    def _anderson_update(self, V_hist: List[np.ndarray], F_hist: List[np.ndarray], m: int) -> np.ndarray:
        """Perform Anderson acceleration step.
        V_hist: list of potentials (most recent last)
        F_hist: list of residuals F(V) = V_lin - V (most recent last)
        Returns accelerated potential guess.
        """
        k = len(F_hist)
        if k < 2:
            return V_hist[-1] + F_hist[-1]
        m_use = min(m, k-1)
        # Build difference matrix
        dF = np.column_stack([F_hist[-i] - F_hist[-i-1] for i in range(1, m_use+1)])  # columns
        # Solve least squares: minimize ||F_k - dF * alpha||
        Fk = F_hist[-1]
        try:
            alpha, *_ = np.linalg.lstsq(dF, Fk, rcond=None)
            # Accelerated update
            V_new = V_hist[-1] + Fk - (dF @ alpha)
            self.diag.anderson_used.append(True)
            return V_new
        except Exception:
            self.diag.anderson_used.append(False)
            return V_hist[-1] + Fk

    def solve(self,
              initial_V: np.ndarray,
              Ec: float,
              Efn: np.ndarray,
              method: str = "ozaki_cfr",
              derivative_method: str = "ozaki_cfr",
              ozaki_cutoff: int = 60,
              max_iters: int = 50,
              tol: float = 5e-5,
              omega: float = 0.5,
              anderson_m: int = 0,
              conduction_only: bool = True,
              recompute_fermi: bool = False,
              fermi_update_interval: int = 5,
              dynamic_Ec_shift: bool = True,
              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Run coupled Gummel iterations until convergence.

        Returns (V_array, rho_array(C/m^3)).
        """
        assert 0 < omega <= 1.0
        V_loc = initial_V.copy()
        if V_loc.size != self.V_func.x.array.size:
            raise ValueError("Initial potential size mismatch with FEM DOFs")
        # Enforce BC values explicitly on initial potential
        self._enforce_dirichlet(V_loc)
        self.V_func.x.array[:] = V_loc
        self.V_func.x.scatter_forward()

        V_hist: List[np.ndarray] = []
        F_hist: List[np.ndarray] = []

        for it in range(1, max_iters+1):
            if hasattr(self.gf, 'clear_ldos_cache'):
                self.gf.clear_ldos_cache()
            if hasattr(self.gf, 'clear_ozaki_cache'):
                self.gf.clear_ozaki_cache()
            if hasattr(self.gf, 'ham') and hasattr(self.gf.ham, 'set_potential'):
                self.gf.ham.set_potential(V_loc)

            with getattr(self.gf, 'serial_mode', lambda: DummyContext())():
                Ec_eff = Ec - V_loc if (conduction_only and dynamic_Ec_shift) else Ec
                n_vec, dn_dV_vec = self._negf_density_and_derivative(V_loc, Ec_eff, Efn, method, derivative_method, ozaki_cutoff, conduction_only)
            self._update_charge_functions(n_vec, dn_dV_vec)

            A, b = self._assemble_linear_system()
            V_lin = self.V_func.copy()
            solver = PETSc.KSP().create(self.comm)
            solver.setOperators(A)
            solver.setType(PETSc.KSP.Type.PREONLY)
            pc = solver.getPC()
            pc.setType(PETSc.PC.Type.LU)
            solver.setFromOptions()
            solver.solve(b, V_lin.x.petsc_vec)
            V_lin.x.scatter_forward()
            V_lin_arr = V_lin.x.array.copy()
            try:
                solver.destroy()
            except Exception:
                pass
            try:
                A.destroy()
            except Exception:
                pass
            try:
                b.destroy()
            except Exception:
                pass
    
            residual = V_lin_arr - V_loc
            res_norm = np.linalg.norm(residual) / np.sqrt(residual.size)
            self.diag.residual_norms.append(res_norm)

    
            if anderson_m > 0:
                V_hist.append(V_loc.copy())
                F_hist.append(residual.copy())
                V_candidate = self._anderson_update(V_hist, F_hist, anderson_m)
            else:
                V_candidate = V_loc + omega * residual  
                self.diag.anderson_used.append(False)

            self._enforce_dirichlet(V_candidate)

            update_norm = np.linalg.norm(V_candidate - V_loc) / np.sqrt(V_loc.size)
            self.diag.update_norms.append(update_norm)
            self.diag.mixing_factors.append(omega)

            V_loc = V_candidate
            self.V_func.x.array[:] = V_loc
            self.V_func.x.scatter_forward()

            if verbose and self.comm.rank == 0:
                left_val = V_loc[self._left_dofs][0] if hasattr(self, '_left_dofs') else np.nan
                right_val = V_loc[self._right_dofs][0] if hasattr(self, '_right_dofs') else np.nan
                print(f"[Gummel {it:03d}] residual={res_norm:.3e} update={update_norm:.3e} V_left={left_val:.3e} V_right={right_val:.3e}")
            if res_norm < tol:
                break

            # Optional dynamic omega reduction (simple heuristic)
            if len(self.diag.residual_norms) >= 2 and self.diag.residual_norms[-1] > 0.9 * self.diag.residual_norms[-2]:
                omega = max(0.1, omega * 0.7)

            # Refresh NEGF caches and update Hamiltonian potential for next density evaluation
            if hasattr(self.gf, 'clear_ldos_cache'):
                self.gf.clear_ldos_cache()
            if hasattr(self.gf, 'clear_ozaki_cache'):
                self.gf.clear_ozaki_cache()
            if hasattr(self.gf, 'ham') and hasattr(self.gf.ham, 'set_potential'):
                self.gf.ham.set_potential(V_loc)

            # Optionally (and infrequently) recompute quasi-Fermi level; disabled by default to avoid PETSc signal issues
            if recompute_fermi and (it == 1 or (fermi_update_interval > 0 and it % fermi_update_interval == 0)):
                lower_bound = np.full_like(V_loc, -0.5)
                upper_bound = np.full_like(V_loc, 0.5)
                try:
                    Efn_new = self.gf.fermi_energy(V_loc, lower_bound, upper_bound, Ec=Ec, verbose=False,
                                                   get_n_kwargs={'method': 'ozaki_cfr'})
                    if np.any(~np.isfinite(Efn_new)):
                        if verbose and self.comm.rank == 0:
                            print("[Gummel] Warning: Non-finite Fermi levels; retaining previous Efn where invalid.")
                        mask = np.isfinite(Efn_new)
                        Efn[mask] = Efn_new[mask]
                    else:
                        Efn = Efn_new
                    self.gf.Efn = np.atleast_1d(Efn)
                except Exception as exc:
                    if verbose and self.comm.rank == 0:
                        print(f"[Gummel] Warning: fermi_energy solve skipped (error: {exc})")
                
        # Final charge density array
        rho_array = self.rho_func.x.array.copy()
        # Propagate converged potential back into underlying Hamiltonian (if available)
        if hasattr(self.gf, 'ham') and hasattr(self.gf.ham, 'set_potential'):
            try:
                # Ensure BCs strictly enforced in final array
                self._enforce_dirichlet(V_loc)
                self.gf.ham.set_potential(V_loc)
            except Exception as e:
                if verbose and self.comm.rank == 0:
                    print(f"[Gummel] Warning: failed to set Hamiltonian potential: {e}")
        return V_loc, rho_array

class DummyContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

# Convenience driver

def run_coupled_negf_poisson(gf : GreensFunction,
                             device_length: float,
                             mesh_points: int,
                             eps_abs: float,
                             Ec: float,
                             Efn: np.ndarray,
                             N_D: float = 1e21,
                             initial_V_left: float = 0.25,
                             initial_V_right: float = -0.25,
                             site_volume_m3: Optional[float] = None,
                             max_iters: int = 40,
                             tol: float = 5e-5,
                             omega: float = 0.5,
                             anderson_m: int = 5,
                             method: str = "ozaki_cfr",
                             derivative_method: str = "ozaki_cfr",
                             ozaki_cutoff: int = 60) -> Tuple[np.ndarray, np.ndarray, CoupledNEGFPoisson]:
    """High-level helper to build mesh, function space, and execute coupled solve."""
    
    gf.ham.set_voltage(initial_V_left, initial_V_right)
    
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_interval(comm, mesh_points, [0.0, device_length])
    V_space = fem.functionspace(mesh, ("Lagrange", 1))
    if site_volume_m3 is None:
        dx = device_length / mesh_points
        site_volume_m3 = dx  
    coupler = CoupledNEGFPoisson(mesh, V_space, gf, eps_abs, N_D, site_volume_m3)
    coupler.set_dirichlet_bcs(initial_V_left, initial_V_right,
                              lambda x: np.isclose(x[0], 0.0),
                              lambda x: np.isclose(x[0], device_length),
                              debug=True)
    # Initialize linear potential
    dof_x = V_space.tabulate_dof_coordinates()[:, 0]
    V_init = initial_V_left + (initial_V_right - initial_V_left) * dof_x / device_length
    V_sol, rho = coupler.solve(V_init, Ec=Ec, Efn=Efn, method=method, derivative_method=derivative_method,
                               ozaki_cutoff=ozaki_cutoff, max_iters=max_iters, tol=tol,
                               omega=omega, anderson_m=anderson_m)

    if hasattr(coupler, '_left_dofs'):
        print("[run] Enforced left BC value:", coupler._left_bc_value, " actual array value:", V_sol[coupler._left_dofs][0])
    if hasattr(coupler, '_right_dofs'):
        print("[run] Enforced right BC value:", coupler._right_bc_value, " actual array value:", V_sol[coupler._right_dofs][0])

    coupler.debug_boundary_values()
    return V_sol, rho, coupler

