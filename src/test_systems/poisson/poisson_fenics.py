"""
Modular FEniCSx based nonlinear Poisson / Poisson–Boltzmann solver
for the test_systems package. This augments (does not remove) the
finite–difference implementation contained in poisson.py.

Key features
------------
* Dimension–agnostic (works for 1D interval, 2D rectangle / generic mesh,
  and 3D meshes in principle).
* Pluggable charge density model via a callable returning (rho, drho_dV)
  given the current potential DOF array. Two helper builders are provided:
    - build_charge_model_boltzmann : classical Boltzmann approximation
    - build_charge_model_negf      : couples to existing NEGF GreensFunction
* Accepts user supplied mesh, boundary conditions, solver tolerances.
* Returns dolfinx Function objects for potential and charge density.

Minimal usage example (Boltzmann 1D)
-----------------------------------
from mpi4py import MPI
import dolfinx, ufl, numpy as np
from petsc4py import PETSc
from poisson_fenics import PoissonFenicsSolver, build_charge_model_boltzmann

L = 100e-9
mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, 200, [0.0, L])
solver = PoissonFenicsSolver(mesh, epsilon_r=11.7)
charge_model = build_charge_model_boltzmann(n_i=1e16, N_D=1e21, T=300.0)
# Dirichlet BCs (example)
V_space = solver.V
left_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim-1, lambda x: np.isclose(x[0], 0))
right_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim-1, lambda x: np.isclose(x[0], L))
left_dofs = dolfinx.fem.locate_dofs_topological(V_space, mesh.topology.dim-1, left_facets)
right_dofs = dolfinx.fem.locate_dofs_topological(V_space, mesh.topology.dim-1, right_facets)
V_left = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.25))
V_right = dolfinx.fem.Constant(mesh, PETSc.ScalarType(-0.25))
bc_left = dolfinx.fem.dirichletbc(V_left, left_dofs, V_space)
bc_right = dolfinx.fem.dirichletbc(V_right, right_dofs, V_space)

Vh, rho = solver.solve(charge_model, bcs=[bc_left, bc_right])

NEGF coupling (1D) example sketch
---------------------------------
from poisson_fenics import build_charge_model_negf
charge_model = build_charge_model_negf(ham, gf, Efn_array, Ec=-2.0,
                                       cross_section_area=getattr(ham,'cross_section_area',1.0),
                                       mapping_fun=None)
Vh, rho = solver.solve(charge_model, bcs=[bc_left, bc_right])

Notes
-----
* The NEGF coupling presently assumes a 1D system: number of device sites == number of 
  mesh vertices along the transport direction (or a mapping function is supplied).
* For multidimensional meshes, you can still provide a custom charge model which reduces
  the potential field to a 1D profile before calling NEGF density routines.
* All arrays are in SI units; potential in Volts. Charge density returned should be in C/m^3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np

try:
    import dolfinx
    import ufl
    from petsc4py import PETSc
except ImportError as e:  # pragma: no cover - environment guard
    raise RuntimeError("dolfinx and petsc4py must be installed to use poisson_fenics module") from e

# Physical constants (SI)
Q = 1.602176634e-19
KB = 1.380649e-23
EPS0 = 8.8541878128e-12

ChargeModel = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]

@dataclass
class SolverOptions:
    rtol: float = 1e-8
    atol: float = 1e-12
    max_it: int = 50
    ksp_type: str = "preonly"
    pc_type: str = "lu"
    pc_factor_mat_solver_type: Optional[str] = None  # e.g. "mumps"
    newton_convergence: str = "incremental"  # or "residual"
    linesearch: Optional[str] = None  # e.g. "basic"

class _NonlinearChargeProblem(dolfinx.fem.petsc.NonlinearProblem):
    """Internal NonlinearProblem wrapper injecting charge density & its derivative.

    F = eps * grad(V)·grad(v) * dx + rho(V) * v * dx (sign chosen so that rhs=0).
    Jacobian uses d rho / d V provided each Newton iteration.
    """
    def __init__(self, Vh, rho_fun, drho_dv_fun, eps_expr, bcs):
        self._Vh = Vh
        self._rho = dolfinx.fem.Function(Vh.function_space, name="ChargeDensity")
        self._drho_dv = dolfinx.fem.Function(Vh.function_space, name="dCharge_dV")
        self._rho_fun = rho_fun
        self._drho_dv_fun = drho_dv_fun

        V = Vh
        v = ufl.TestFunction(Vh.function_space)
        du = ufl.TrialFunction(Vh.function_space)
        # Store epsilon in a Function / Constant expression
        if np.isscalar(eps_expr):
            eps_u = dolfinx.fem.Constant(Vh.function_space.mesh, PETSc.ScalarType(eps_expr))
        else:
            eps_u = eps_expr  # assumed proper expression

        # Residual form: F(V; v) = eps ∇V·∇v dx + rho(V) v dx
        self._F_form = eps_u * ufl.dot(ufl.grad(V), ufl.grad(v)) * ufl.dx + self._rho * v * ufl.dx
        # Jacobian form using supplied numeric derivative drho/dV
        # J(V; du, v) = eps ∇du·∇v dx + (drho/dV) du v dx
        self._J_form_symbolic = eps_u * ufl.dot(ufl.grad(du), ufl.grad(v)) * ufl.dx + self._drho_dv * du * v * ufl.dx
        super().__init__(self._F_form, Vh, bcs=bcs, J=self._J_form_symbolic)

    def F(self, x: PETSc.Vec, b: PETSc.Vec):  # noqa: N802 (F mandated by interface)
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self._Vh.x.petsc_vec)
        self._Vh.x.scatter_forward()

        pot_vals = self._Vh.x.array
        rho_vals, _ = self._rho_fun(pot_vals)
        self._rho.x.array[:] = rho_vals
        self._rho.x.scatter_forward()

        with b.localForm() as b_loc:
            b_loc.set(0)
        dolfinx.fem.petsc.assemble_vector(b, dolfinx.fem.form(self._F_form))
        dolfinx.fem.apply_lifting(b, [dolfinx.fem.form(self._J_form_symbolic)], [self.bcs], [x], -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, self.bcs, x, -1.0)

    def J(self, x: PETSc.Vec, A: PETSc.Mat):  # noqa: N802
        pot_vals = self._Vh.x.array
        _, drho_dv_vals = self._drho_dv_fun(pot_vals)
        self._drho_dv.x.array[:] = drho_dv_vals
        self._drho_dv.x.scatter_forward()
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, dolfinx.fem.form(self._J_form_symbolic), self.bcs)
        A.assemble()

class PoissonFenicsSolver:
    """High-level solver orchestrating nonlinear Poisson solve.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The computational mesh.
    element_deg : int
        Polynomial degree (default 1).
    epsilon_r : float | np.ndarray | ufl.Expr
        Relative permittivity (scalar or field). If array of length = number of cells
        a DG-0 projection is performed.
    eps0 : float
        Vacuum permittivity (defaults to physical constant).
    options : SolverOptions
        PETSc / Newton solver configuration.
    """
    def __init__(self, mesh, element_deg: int = 1, epsilon_r: float | np.ndarray | object = 11.7,
                 eps0: float = EPS0, options: SolverOptions | None = None):
        self.mesh = mesh
        self.dim = mesh.topology.dim
        self.V = dolfinx.fem.functionspace(mesh, ("Lagrange", element_deg))
        self.options = options or SolverOptions()

        # Build epsilon expression
        if np.isscalar(epsilon_r):
            self.epsilon_expr = eps0 * float(epsilon_r)
        else:
            # Assume array-like per cell or a UFL object
            if isinstance(epsilon_r, np.ndarray):
                # Create DG0 Function to store per-cell values
                Vdg = dolfinx.fem.functionspace(mesh, ("DG", 0))
                eps_fun = dolfinx.fem.Function(Vdg)
                if len(epsilon_r) != mesh.topology.index_map(self.dim).size_local:
                    raise ValueError("epsilon_r array size must match number of local cells")
                eps_fun.x.array[:] = epsilon_r * eps0
                self.epsilon_expr = eps_fun
            else:
                self.epsilon_expr = epsilon_r  # assumed scaled already

    def solve(self, charge_model: ChargeModel, bcs: list, initial_guess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
              V_initial: Optional[np.ndarray] = None) -> Tuple[dolfinx.fem.Function, dolfinx.fem.Function]:
        """Run Newton solve.

        Parameters
        ----------
        charge_model : callable(V_array) -> (rho, drho_dV)
            Returns charge density (C/m^3) and its derivative wrt potential.
        bcs : list of DirichletBC
            Boundary conditions.
        initial_guess : callable(x_coords) -> potential array, optional
            To set non-linear starting point.
        V_initial : ndarray, optional
            Explicit potential DOF array to copy in (overrides initial_guess if both given).
        """
        Vh = dolfinx.fem.Function(self.V, name="Potential")
        if V_initial is not None:
            if V_initial.shape[0] != Vh.x.array.shape[0]:
                raise ValueError("V_initial size mismatch with DOFs")
            Vh.x.array[:] = V_initial
        elif initial_guess is not None:
            coords = self.V.tabulate_dof_coordinates()
            guess = initial_guess(coords.T)
            if guess.shape[0] != Vh.x.array.shape[0]:
                raise ValueError("Initial guess shape mismatch")
            Vh.x.array[:] = guess
        else:
            # default linear interpolation across x if coordinate exists
            coords = self.V.tabulate_dof_coordinates()
            if self.dim >= 1:
                x = coords[:, 0]
                Vh.x.array[:] = np.interp(x, [x.min(), x.max()], [0.0, 0.0])  # flat zero

        problem = _NonlinearChargeProblem(Vh, lambda v: charge_model(v), lambda v: charge_model(v), self.epsilon_expr, bcs)
        newton = dolfinx.nls.petsc.NewtonSolver(self.mesh.comm, problem)
        newton.convergence_criterion = self.options.newton_convergence
        newton.rtol = self.options.rtol
        newton.atol = self.options.atol
        newton.max_it = self.options.max_it

        # Configure PETSc KSP
        ksp = newton.krylov_solver
        opts = PETSc.Options()
        prefix = ksp.getOptionsPrefix()
        opts[f"{prefix}ksp_type"] = self.options.ksp_type
        opts[f"{prefix}pc_type"] = self.options.pc_type
        if self.options.pc_factor_mat_solver_type:
            opts[f"{prefix}pc_factor_mat_solver_type"] = self.options.pc_factor_mat_solver_type
        if self.options.linesearch:
            opts["snes_linesearch_type"] = self.options.linesearch
        ksp.setFromOptions()

        its, converged = newton.solve(Vh)
        if self.mesh.comm.rank == 0:
            if converged:
                print(f"Poisson solve converged in {its} Newton iterations")
            else:
                print(f"WARNING: Poisson solve NOT converged after {its} iterations")

        # Recompute final charge density using model
        rho_vals, _ = charge_model(Vh.x.array)
        rho_fun_dolfin = dolfinx.fem.Function(self.V, name="ChargeDensity")
        rho_fun_dolfin.x.array[:] = rho_vals
        return Vh, rho_fun_dolfin

# ---------------------------------------------------------------------------
# Charge model helper builders
# ---------------------------------------------------------------------------

def build_charge_model_boltzmann(n_i: float, N_D: float, T: float = 300.0, q: float = Q) -> ChargeModel:
    """Classical Boltzmann charge model (n-type) for Poisson-Boltzmann.

    rho(V) = q (N_D - n_i exp(V / V_T))
    drho/dV = -q n_i / V_T exp(V / V_T)

    Returns callable returning (rho, drho_dV) arrays.
    """
    V_T = KB * T / q
    def model(V_array: np.ndarray):
        exp_term = np.exp(V_array / V_T)
        rho = q * (N_D - n_i * exp_term)
        drho = -q * (n_i / V_T) * exp_term
        return rho, drho
    return model


def build_charge_model_negf(ham, gf, Efn_array: np.ndarray, Ec: float = -2.0,
                            cross_section_area: Optional[float] = None,
                            mapping_fun: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                            q: float = Q,
                            include_doping: bool = False,
                            donor_attr: str = 'N_donor',
                            acceptor_attr: str = 'N_acceptor',
                            doping_array_attr: str = 'one_d_doping') -> ChargeModel:
    """Construct NEGF–coupled charge model for 1D devices.

    Parameters
    ----------
    ham : Hamiltonian
        Existing Hamiltonian instance (assumed 1D for now).
    gf : GreensFunction
        Instance providing get_n and diff_rho_poisson / diff_rho_poisson style derivative.
    Efn_array : ndarray
        Quasi-Fermi level array input for NEGF routines.
    Ec : float
        Conduction band edge reference passed to NEGF density routine.
    cross_section_area : float
        Area used to convert electrons per site to volumetric density (m^2). Defaults to ham.cross_section_area or 1.
    mapping_fun : callable(potential_dofs) -> site_potential_array
        Optional mapping if mesh DOFs != number of device sites. If None, assert lengths equal.
    """
    dx = getattr(ham, 'one_d_dx', None)
    if dx is None:
        raise ValueError("Hamiltonian must define one_d_dx for NEGF coupling")
    A_cs = cross_section_area or getattr(ham, 'cross_section_area', 1.0)
    volume_per_site = dx * A_cs

    # Determine derivative method availability
    diff_attr = None
    if hasattr(gf, 'diff_rho_poisson'):
        diff_attr = 'diff_rho_poisson'
    elif hasattr(gf, 'get_diff_rho'):
        diff_attr = 'get_diff_rho'
    has_diff = diff_attr is not None

    # Extract doping if requested
    N_D = getattr(ham, donor_attr, None)
    N_A = getattr(ham, acceptor_attr, None)
    doping_array = getattr(ham, doping_array_attr, None)
    if include_doping and doping_array is None and (N_D is None and N_A is None):
        raise ValueError("include_doping=True but no doping information found on Hamiltonian")

    def model(V_array: np.ndarray):
        # Map potentials to site potentials
        if mapping_fun is not None:
            site_potentials = mapping_fun(V_array)
        else:
            if V_array.shape[0] != ham.N:
                raise ValueError("Potential DOFs mismatch device site count; provide mapping_fun")
            site_potentials = V_array

        # Electron density per site (dimensionless count)
        n_site = gf.get_n(V=site_potentials, Efn=Efn_array, Ec=Ec, use_rgf=True)
        if has_diff:
            if diff_attr == 'diff_rho_poisson':
                dn_dV_site = getattr(gf, diff_attr)(Efn=Efn_array, V=site_potentials, Ec=Ec, use_rgf=True)
            else:
                # expected signature get_diff_rho(V, Efn, Ec, ...)
                dn_dV_site = getattr(gf, diff_attr)(V=site_potentials, Efn=Efn_array, Ec=Ec, use_rgf=True)
        else:  # fallback finite difference
            dV = 1e-5  # small potential perturbation
            n_plus = gf.get_n(V=site_potentials + dV, Efn=Efn_array, Ec=Ec, use_rgf=True)
            n_minus = gf.get_n(V=site_potentials - dV, Efn=Efn_array, Ec=Ec, use_rgf=True)
            dn_dV_site = (n_plus - n_minus) / (2 * dV)

        # Convert to volumetric charge density (C/m^3)
        rho = -q * n_site / volume_per_site
        drho = -q * dn_dV_site / volume_per_site
        if include_doping:
            # Simple net charge: q (N_D - N_A - n) / volume; assume doping_array already has q * N(x)
            if doping_array is not None and hasattr(doping_array, '__len__') and len(doping_array) == n_site.shape[0]:
                # doping_array assumed in units of charge density per site: convert to volumetric by /volume_per_site
                rho += doping_array / volume_per_site
                # derivative unaffected by static doping
            else:
                ND_eff = float(N_D) if N_D is not None else 0.0
                NA_eff = float(N_A) if N_A is not None else 0.0
                rho += q * (ND_eff - NA_eff) / volume_per_site
        return rho, drho
    return model

__all__ = [
    "PoissonFenicsSolver",
    "SolverOptions",
    "build_charge_model_boltzmann",
    "build_charge_model_negf",
]
