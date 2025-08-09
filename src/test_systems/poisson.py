from hamiltonian import Hamiltonian
import numpy as np
from rgf import GreensFunction

E_0 = 8.854e-12  
Q = 1.602e-19    
K_B = 1.380649e-23 
T = 300        
VT = K_B * T / Q 
def solve_laplace_initial_one_d(epsilon, ham: Hamiltonian):
    """initial solution without the non linear term """
    N = ham.N
    dx = ham.one_d_dx

    A = np.zeros((N, N))
    b = np.zeros(N)

    # BC
    A[0, 0] = 1.0
    b[0] = ham.Vs
    A[N-1, N-1] = 1.0
    b[N-1] = ham.Vd

    # mat
    for i in range(1, N - 1):
        eps_minus = (epsilon[i - 1] + epsilon[i]) / 2.0
        eps_plus = (epsilon[i] + epsilon[i + 1]) / 2.0
        A[i, i - 1] = eps_minus
        A[i, i]     = -(eps_plus + eps_minus)
        A[i, i + 1] = eps_plus
    
    # gate
    if ham.gate_factor > 0 and ham.C_ox > 0:
        gate_width_pts = int(ham.gate_factor * N)
        if gate_width_pts > 0:
            start_gate = (N - gate_width_pts) // 2
            end_gate = start_gate + gate_width_pts
            for i in range(start_gate, end_gate):
                if 1 <= i < N - 1:
                    A[i, i] -= dx * ham.C_ox
                    b[i] -= dx * ham.C_ox * ham.Vg
    try:
        V = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Singular matrix")
        V = np.linspace(ham.Vs, ham.Vd, N)
    return V

def solve_poisson_delta_one_d(V, diff_rho, rho, epsilon, doping, ham: Hamiltonian):
    N = ham.N
    dx = ham.one_d_dx
    
    A = np.zeros((N, N))
    b = np.zeros(N)

    # BC
    A[0, 0] = 1.0
    b[0] = 0
    A[N-1, N-1] = 1.0
    b[N-1] = 0

    # build jacobina
    for i in range(1, N - 1):
        eps_minus = (epsilon[i - 1] + epsilon[i]) / 2.0
        eps_plus = (epsilon[i] + epsilon[i + 1]) / 2.0

        A[i, i - 1] = eps_minus
        A[i, i]     = -(eps_plus + eps_minus) + diff_rho[i]
        A[i, i + 1] = eps_plus
        
        laplace_term = eps_plus * (V[i+1] - V[i]) - eps_minus * (V[i] - V[i-1])
        charge_term = dx**2 * (rho[i] - doping[i])
        
        residual = laplace_term  + charge_term
        b[i] = -residual

    # gate region 
    if ham.gate_factor > 0 and ham.C_ox > 0:
        gate_width_pts = int(ham.gate_factor * N)
        if gate_width_pts > 0:
            start_gate = (N - gate_width_pts) // 2
            end_gate = start_gate + gate_width_pts
            for i in range(start_gate, end_gate):
                if 1 <= i < N - 1:
                    A[i, i] -= dx * ham.C_ox
                    gate_residual = -dx * ham.C_ox * (V[i] - ham.Vg)
                    b[i] -= gate_residual
    
    try:
        delta_V = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Warning: The matrix A is singular. Returning a zero update.")
        delta_V = np.zeros_like(V)

    return delta_V

def solve_poisson_nonlinear(ham: Hamiltonian, gf : GreensFunction):
    """
    Iteratively solves the non-linear Poisson equation using newtons method 
    """
    epsilon_profile = ham.one_d_epsilon
    doping_profile = ham.one_d_doping

    V = solve_laplace_initial_one_d(epsilon_profile, ham)
    
    print("Starting non-linear Poisson solver...")
    for iteration in range(1000): 
        n_i = 1e16 
        
        if ham.poisson_testing:
            charge_density = -ham.q * ham.n_i * np.exp(V /VT)
            diff_charge_density_unscaled = charge_density /VT
        else:
            A_cs = getattr(ham, 'cross_section_area', 1.0)  # m^2
            volume_per_site = ham.one_d_dx * A_cs            # m^3

            # Provide required arguments to NEGF density routines.
            # Use Efn = 0 reference (array) and Ec = 0 unless model supplies band edge separately.
            Efn_array = np.zeros_like(V)
            Ec = 0.0

            n_site = gf.get_n(V=V, Efn=Efn_array, Ec=Ec, use_rgf=True)  # electrons per site
            dn_dV_site = gf.diff_rho_poisson(Efn=Efn_array, V=V, Ec=Ec, use_rgf=True)

            charge_density = -ham.q * n_site / volume_per_site               
            diff_charge_density_unscaled = -ham.q * dn_dV_site / volume_per_site 
        

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
        
    return V, charge_density