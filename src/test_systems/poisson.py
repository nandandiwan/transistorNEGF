import numpy as np
import scipy.constants as spc
from hamiltonian import Hamiltonian

import numpy as np
kbT = spc.Boltzmann * 300
q = spc.elementary_charge
VT = kbT / q
def solve_laplace_initial_one_d(epsilon, ham: Hamiltonian):
    """
    Solves the 1D Laplace equation d/dx(epsilon * dV/dx) = 0 with gate coupling.
    This provides a good initial guess for the non-linear solver.
    """
    N = ham.N
    dx = ham.dx

    A = np.zeros((N, N))
    b = np.zeros(N)

    # Dirichlet boundary conditions
    A[0, 0] = 1.0
    b[0] = ham.Vs
    A[N-1, N-1] = 1.0
    b[N-1] = ham.Vd

    # Internal nodes
    for i in range(1, N - 1):
        eps_minus = (epsilon[i - 1] + epsilon[i]) / 2.0
        eps_plus = (epsilon[i] + epsilon[i + 1]) / 2.0
        A[i, i - 1] = eps_minus
        A[i, i]     = -(eps_plus + eps_minus)
        A[i, i + 1] = eps_plus
    
    # Gated region modification
    if ham.gate_factor > 0 and ham.C_ox > 0:
        gate_width_pts = int(ham.gate_factor * N)
        if gate_width_pts > 0:
            start_gate = (N - gate_width_pts) // 2
            end_gate = start_gate + gate_width_pts
            for i in range(start_gate, end_gate):
                if 1 <= i < N - 1:
                    A[i, i] -= dx * ham.C_ox
                    b[i] -= dx * ham.C_ox * ham.gate
    try:
        V = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Warning: Laplace matrix is singular. Returning linear potential.")
        V = np.linspace(ham.Vs, ham.Vd, N)
    return V

def solve_poisson_delta_one_d(V, diff_rho, rho, epsilon, doping, ham: Hamiltonian):
    """
    Solves one Newton-Raphson step for the non-linear Poisson equation.
    
    This function calculates the potential update, delta_V, by solving the linearized system:
    J * delta_V = -F(V)
    where J is the Jacobian matrix and F(V) is the residual of the discretized Poisson equation.
    """
    N = ham.N
    dx = ham.dx
    
    A = np.zeros((N, N))
    b = np.zeros(N)

    # Boundary Conditions for delta_V are 0 because V is fixed at the boundaries.
    A[0, 0] = 1.0
    b[0] = 0
    A[N-1, N-1] = 1.0
    b[N-1] = 0

    for i in range(1, N - 1):
        eps_minus = (epsilon[i - 1] + epsilon[i]) / 2.0
        eps_plus = (epsilon[i] + epsilon[i + 1]) / 2.0

        A[i, i - 1] = eps_minus
        A[i, i]     = -(eps_plus + eps_minus) + diff_rho[i]
        A[i, i + 1] = eps_plus
        
        laplace_term = eps_plus * (V[i+1] - V[i]) - eps_minus * (V[i] - V[i-1])
        charge_term = dx**2 * (rho[i] + doping[i])
        
        residual = laplace_term  + charge_term
        b[i] = -residual

    # --- Gated Region Modification ---
    if ham.gate_factor > 0 and ham.C_ox > 0:
        gate_width_pts = int(ham.gate_factor * N)
        if gate_width_pts > 0:
            start_gate = (N - gate_width_pts) // 2
            end_gate = start_gate + gate_width_pts
            for i in range(start_gate, end_gate):
                if 1 <= i < N - 1:
                    # Modify Jacobian (A): Add derivative of the gate term.
                    A[i, i] -= dx * ham.C_ox
                    # Modify Residual (b): Add the gate term's own residual.
                    gate_residual = -dx * ham.C_ox * (V[i] - ham.gate)
                    b[i] -= gate_residual
    
    try:
        delta_V = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Warning: The matrix A is singular. Returning a zero update.")
        delta_V = np.zeros_like(V)

    return delta_V

def solve_poisson_nonlinear(ham: Hamiltonian, doping_profile: np.ndarray, epsilon_profile: np.ndarray):
    """
    Iteratively solves the non-linear Poisson equation using the Newton-Raphson method.
    """

    V = solve_laplace_initial_one_d(epsilon_profile, ham)
    
    print("Starting non-linear Poisson solver...")
    for iteration in range(1000): # Limit iterations
        # --- Calculate Charge Density based on current V ---
        n_i = 1e16 
        charge_density = -q * n_i * np.exp(V / VT)
        
        diff_charge_density_unscaled = charge_density / VT
        diff_charge_density_scaled = ham.one_d_dx**2 * diff_charge_density_unscaled

        delta_V = solve_poisson_delta_one_d(
            V, diff_charge_density_scaled, charge_density, epsilon_profile, doping_profile, ham
        )
        
        V = V + delta_V
        
        update_norm = np.linalg.norm(delta_V)
        print(f"Iteration {iteration+1}: Update norm = {update_norm:.4e}")
        if update_norm < 1e-6:
            print(f"\nConverged after {iteration+1} iterations.")
            break
    else:
        print("\nWarning: Solver did not converge within the iteration limit.")
        
    return V
