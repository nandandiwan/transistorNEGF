
import os
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
from itertools import product
import multiprocessing
import numpy as np
import scipy.sparse as sp
from scipy.linalg import inv 
import scipy.constants as spc
from scipy.integrate import quad_vec

import warnings


from hamiltonian import Hamiltonian
from lead_self_energy import LeadSelfEnergy
from utils import smart_inverse  
from utils import sparse_diag_product


class GreensFunction:
    """
    Recursive Green's Function (RGF) implementation optimized with a smart inversion utility.
    """

    def __init__(self, hamiltonian: Hamiltonian, self_energy_method="sancho_rubio", energy_grid = np.linspace(-5,5, 300)):
        """
        Initialize the Green's function calculator.
        """
        
        self.ham = hamiltonian
        
        if (self.ham.periodic):
            self.k_space = np.linspace(0,1,32)
        else:
            self.k_space = np.array([0])
        self.lead_self_energy = LeadSelfEnergy(hamiltonian)
        self.self_energy_method = self_energy_method
        self.eta = 1e-6
        self.energy_grid = energy_grid
        self.dE = self.energy_grid[1] - self.energy_grid[0]
        # The sparse_threshold is now handled by the smart_inverse function
        # self.sparse_threshold = 0.1

    def fermi_distribution(self, E, mu):
        """Fermi-Dirac distribution function."""
        kT = self.ham.kbT_eV
        x = (E - mu) / kT
        with np.errstate(over='ignore'):
            return 1.0 / (1.0 + np.exp(x))

    def add_eta(self, E):
        """Add small imaginary part for numerical stability."""
        if np.iscomplexobj(E):
            return E
        return E + 1j * self.eta

    def compute_central_greens_function(self, E, ky=0,compute_lesser=True,
                                        use_rgf=True, self_energy_method=None, equilibrium=False):
        """
        Compute central region Green's function.
        """
        E = self.add_eta(E)

        if self_energy_method is None:
            self_energy_method = self.self_energy_method

        if use_rgf:
            return self._compute_rgf_greens_function(E, ky,compute_lesser, self_energy_method, equilibrium=equilibrium)
        else:
            return self._compute_direct_greens_function(E, ky,compute_lesser, self_energy_method, equilibrium=equilibrium)

    def _compute_direct_greens_function(self, E, ky,compute_lesser, self_energy_method, equilibrium=False):
        """
        Direct matrix inversion for smaller systems using the smart_inverse utility.
        Uses general Hamiltonian.create_hamiltonian interface.
        """
        if equilibrium and compute_lesser:
            raise ValueError("Cannot compute lesser Green's function in equilibrium.")

        # Get full device Hamiltonian for current type
        
        H = self.ham.create_hamiltonian(blocks=False, ky=ky)
        n = H.shape[0]
        E = self.add_eta(E)
        sigma_L = self.lead_self_energy.self_energy("left", E, ky, self_energy_method)
        sigma_R = self.lead_self_energy.self_energy("right", E,ky, self_energy_method)

        if equilibrium:
            sigma_L *= 0
            sigma_R *= 0

        block_size = sigma_L.shape[0]

        Sigma_L_full = sp.lil_matrix((n, n), dtype=complex)
        Sigma_R_full = sp.lil_matrix((n, n), dtype=complex)
        Sigma_L_full[:block_size, :block_size] = sigma_L
        Sigma_R_full[-block_size:, -block_size:] = sigma_R
        Sigma_L_full = Sigma_L_full.tocsc()
        Sigma_R_full = Sigma_R_full.tocsc()
        
        H_eff = H + Sigma_L_full + Sigma_R_full
        A = E * sp.identity(n, dtype=complex, format='csc') - H_eff

        # Use smart_inverse for efficient computation
        G_R = smart_inverse(A)

        if not sp.issparse(G_R):
            G_R = sp.csc_matrix(G_R)

        Gamma_L = 1j * (Sigma_L_full - Sigma_L_full.conj().T)
        Gamma_R = 1j * (Sigma_R_full - Sigma_R_full.conj().T)

        if not compute_lesser:
            return G_R, Gamma_L, Gamma_R

        f_L = self.fermi_distribution(np.real(E), self.ham.mu1)
        f_R = self.fermi_distribution(np.real(E), self.ham.mu2)

        Sigma_lesser = sp.lil_matrix((n, n), dtype=complex)
        Sigma_lesser +=  Gamma_L * f_L
        Sigma_lesser += Gamma_R * f_R
        Sigma_lesser *= 1j
        
        Sigma_lesser = Sigma_lesser.tocsc()

        G_A = G_R.conj().T
        G_lesser = G_R @ Sigma_lesser @ G_A
        G_lesser_diag = G_lesser.diagonal()

        return G_R, G_lesser_diag, Gamma_L, Gamma_R

    def _compute_rgf_greens_function(self, E, ky,compute_lesser, self_energy_method, equilibrium=False):
        """
        Generalized RGF computation using smart_inverse for all block inversions.
        Uses Hamiltonian.create_hamiltonian for block construction.
        """
        if equilibrium and compute_lesser:
            raise ValueError("Cannot compute lesser Green's function in equilibrium.")
        E = self.add_eta(E)
        dagger = lambda A: np.conjugate(A.T)

        # Get blocks for current device type
        H_ii, H_ij = self.ham.create_hamiltonian(blocks=True, ky=ky)
        num_blocks = len(H_ii)
        block_size = H_ii[0].shape[0]

        H_ii = [block.toarray() if sp.issparse(block) else block for block in H_ii]
        H_ij = [block.toarray() if sp.issparse(block) else block for block in H_ij]

        sigma_L = self.lead_self_energy.self_energy("left", E, ky,self_energy_method)
        sigma_R = self.lead_self_energy.self_energy("right", E, ky,self_energy_method)
        if equilibrium:
            sigma_L *= 0
            sigma_R *= 0

        Gamma_L = 1j * (sigma_L - dagger(sigma_L))
        Gamma_R = 1j * (sigma_R - dagger(sigma_R))

        f_L = self.fermi_distribution(np.real(E), self.ham.mu1)
        f_R = self.fermi_distribution(np.real(E), self.ham.mu2)

        sigma_L_lesser = Gamma_L * f_L *1j
        sigma_R_lesser = Gamma_R * f_R * 1j

        g_R = []
        g_lesser = []

        # Forward sweep - using smart_inverse
        H00_eff = H_ii[0] + sigma_L
        g0_R = smart_inverse(E * np.eye(block_size) - H00_eff)
        g_R.append(g0_R)

        if compute_lesser:
            g0_lesser = g0_R @ sigma_L_lesser @ dagger(g0_R)
            g_lesser.append(g0_lesser)

        for i in range(1, num_blocks):
            H_i_im1 = dagger(H_ij[i-1])
            sigma_recursive_R = H_i_im1 @ g_R[i - 1] @ H_ij[i-1]
            g_i_R = smart_inverse(E * np.eye(block_size) - H_ii[i] - sigma_recursive_R)
            g_R.append(g_i_R)
            if compute_lesser:
                sigma_recursive_lesser = H_i_im1 @ g_lesser[i-1] @ H_ij[i-1]
                g_i_lesser = g_R[i] @ sigma_recursive_lesser @ dagger(g_R[i])
                g_lesser.append(g_i_lesser)

        G_R = [None] * num_blocks
        G_lesser = [None] * num_blocks

        # Backward sweep - using smart_inverse
        H_N_Nm1 = dagger(H_ij[-1])
        sigma_eff_R = H_N_Nm1 @ g_R[-2] @ H_ij[-1]
        GN_R = smart_inverse(E * np.eye(block_size) - H_ii[-1] - sigma_R - sigma_eff_R)
        G_R[-1] = GN_R

        if compute_lesser:
            sigma_eff_lesser = H_N_Nm1 @ g_lesser[-2] @ H_ij[-1]
            total_sigma_lesser = sigma_R_lesser + sigma_eff_lesser
            GN_lesser = G_R[-1] @ total_sigma_lesser @ dagger(G_R[-1])
            G_lesser[-1] = GN_lesser

        for i in range(num_blocks - 2, -1, -1):
            H_i_ip1 = H_ij[i]
            H_ip1_i = dagger(H_i_ip1)
            propagator = g_R[i] @ H_i_ip1 @ G_R[i + 1] @ H_ip1_i
            G_R[i] = g_R[i] + propagator @ g_R[i]
            if compute_lesser:
                g_R_dag = dagger(g_R[i])
                term1 = g_lesser[i]
                term2 = (propagator @ g_lesser[i])
                term3 = (g_lesser[i] @ dagger(propagator))
                term4 = (g_R[i] @ H_i_ip1 @ G_lesser[i + 1] @ H_ip1_i @ g_R_dag)
                G_lesser[i] = term1 + term2 + term3 + term4

        G_R_diag = np.concatenate([np.diag(block) for block in G_R])

        if not compute_lesser:
            return G_R_diag, Gamma_L, Gamma_R

        G_lesser_diag = np.concatenate([np.diag(block) for block in G_lesser])
        return G_R_diag, G_lesser_diag, Gamma_L, Gamma_R
    


    def compute_density_of_states(self, E,ky=0,self_energy_method=None, use_rgf=True, equilibrium=False):
        G_R_diag, _, _ = self.compute_central_greens_function(
            E, ky=ky,compute_lesser=False, use_rgf=use_rgf,
            self_energy_method=self_energy_method, equilibrium=equilibrium
        )
        dos = -np.imag(G_R_diag) / np.pi
        return np.maximum(dos, 0.0)
    def _dos_non_periodic_helper(self, params):
        E, self_energy_method,use_rgf, equilibrium = params
        return np.sum(self.compute_density_of_states(E,ky=0,self_energy_method=self_energy_method, use_rgf=use_rgf, equilibrium=equilibrium))
    
    def gf_calculations_k_space(self, E, self_energy_method = "sancho_rubio",use_rgf=True, equilibrium=False) -> list:
        """Uses multiprocessing to cache GF for E,ky """
        
        # Handle both single energy values and lists
        if np.isscalar(E):
            E_list = [E]
        else:
            E_list = E
            
        param_grid = list(product(E_list, self.k_space, [self_energy_method],[use_rgf], [equilibrium]))

        print(f"Starting DOS calculations for {len(param_grid)} (E, ky) pairs...")
        print(f"ky range: {self.k_space[0]:.2f} to {self.k_space[-1]:.2f}")

        start_time = time.time()

        with multiprocessing.Pool(processes=32) as pool:
            results = pool.map(self._calculate_gf_simple, param_grid)
        end_time = time.time()
        return results
    def calculate_DOS(self, self_energy_method = "sancho_rubio",use_rgf=True, equilibrium=False):
        
        if (not self.ham.periodic): # do this for non perioidic
            param_grid = list(product(self.energy_grid, [self_energy_method],[use_rgf], [equilibrium]))
            start_time = time.time()
            with multiprocessing.Pool(processes=32) as pool:
                results = pool.map(self._dos_non_periodic_helper, param_grid)
            end_time = time.time()
            return results
        
        
        param_grid = list(product(self.energy_grid, self.k_space, [self_energy_method],[use_rgf], [equilibrium]))
        start_time = time.time()

        with multiprocessing.Pool(processes=32) as pool:
            results = pool.map(self._calculate_gf_simple, param_grid)
        end_time = time.time()
        
        # Sort results by (energy, ky) to ensure correct reshape
        param_grid_sorted = sorted(
            zip(param_grid, results),
            key=lambda x: (self.energy_grid.tolist().index(x[0][0]), self.k_space.tolist().index(x[0][1]))
        )
        results_sorted = [r for (_, r) in param_grid_sorted]

        # Each result is a list (G_R_diag), so flatten each and stack
        results_flat = [np.array(r).flatten() for r in results_sorted]
        results_matrix = np.array(results_flat).reshape(len(self.energy_grid), len(self.k_space), -1)

        # Compute DOS for each energy by summing over ky and all diagonal elements
        dos = -np.imag(results_matrix) / np.pi

        dos_sum = np.sum(dos, axis=(1, 2))  # sum over ky and diagonal elements
        return np.array(dos_sum)

    
    
    def _calculate_gf_simple(self, param):
        """Simple Green's function calculation"""
        energy, ky, self_energy_method, use_rgf,equilibrium= param
        G_R_diag, _, _ = self.compute_central_greens_function(
            energy, ky=ky,compute_lesser=False, use_rgf=use_rgf,
            self_energy_method=self_energy_method, equilibrium=equilibrium
        )
        return G_R_diag

    
    def compute_transmission(self, E, ky=0, self_energy_method=None):
        """
        Compute the transmission coefficient T(E) using the Caroli formula.
        T(E) = Tr[Γ_L * G_R * Γ_R * G_A]
        """
        # Get Green's function and broadening matrices
        G_R, Gamma_L, Gamma_R = self.compute_central_greens_function(
            E,ky=ky, use_rgf=False, self_energy_method=self_energy_method, compute_lesser=False
        )

        # Convert to dense arrays for matrix multiplication
        G_R_dense = G_R.toarray()
        G_A_dense = G_R_dense.conj().T
        Gamma_L_dense = Gamma_L.toarray()
        Gamma_R_dense = Gamma_R.toarray()

        T_matrix = Gamma_L_dense @ G_R_dense @ Gamma_R_dense @ G_A_dense
        
        T = np.real(np.trace(T_matrix))
        
        return max(T, 0) 

    def compute_conductance(self, E_F=0.0, self_energy_method=None):
        """
        Compute the zero-temperature conductance G = (2e^2/h) * T(E_F).
        """
        # Conductance in units of G0 = 2e^2/h is simply the transmission at the Fermi energy.
        T = self.compute_transmission(E=E_F, self_energy_method=self_energy_method)
        return T    
    
    def _current_worker(self, param):
        """Worker for multiprocessing: computes current for (E, ky)."""
        E, ky, self_energy_method, use_rgf = param
        
        if not use_rgf:
            transmission = self.compute_transmission(E, ky, self_energy_method)

            f_s = self.fermi_distribution(E, self.ham.mu1)
            f_d = self.fermi_distribution(E, self.ham.mu2)
            IL_contrib = transmission * f_s
            IR_contrib = transmission * f_d
            current_contribution = self.dE * (self.ham.q**2 / (2 * np.pi * spc.hbar)) * (IL_contrib - IR_contrib)
            
            return current_contribution
        
        G_R, G_lesser_diag, Gamma_L, Gamma_R = self.compute_central_greens_function(
            E, ky=ky, use_rgf=True, self_energy_method=self_energy_method, compute_lesser=True
        )
        f_s = self.fermi_distribution(E, self.ham.mu1)
        f_d = self.fermi_distribution(E, self.ham.mu2)
        sigma_less_left = Gamma_L * f_s * 1j
        sigma_less_right = Gamma_R * f_d * 1j
        A_matrix = np.diag(1j * (G_R - G_R.conj()))
        term1 = np.sum(Gamma_L.diagonal() * G_lesser_diag)
        term2 = f_s * np.sum(Gamma_L.diagonal() * A_matrix)
        # dE should be defined (energy step)
        current_contribution = self.dE * (self.ham.q**2 / (2 * np.pi * spc.hbar)) * (term1 - term2)
        return current_contribution

    def compute_total_current(self, self_energy_method="sancho_rubio", use_rgf = False):
        """
        Compute total current by summing over all (E, ky) pairs using multiprocessing.
        """
        # Define energy and ky grids
        E_list = self.energy_grid
        ky_list = self.k_space
        param_grid = list(product(E_list, ky_list, [self_energy_method], [use_rgf]))

        print(f"Starting current calculations for {len(param_grid)} (E, ky) pairs...")

        with multiprocessing.Pool(processes=32) as pool:
            results = pool.map(self._current_worker, param_grid)

        # Sum over all energy and ky points
        total_current = np.sum(results)
        return total_current

    def compute_charge_density(self, self_energy_method="sancho_rubio", use_rgf=True):
        E_list = self.energy_grid
        ky_list = self.k_space
        param_grid = list(product(E_list, ky_list, [self_energy_method], [use_rgf]))
        
        with multiprocessing.Pool(processes=32) as pool:
            results = pool.map(self._charge_worker, param_grid)
        if not results:
            print("Warning: No results returned from multiprocessing.")
            n_sites = self.ham.create_hamiltonian(blocks=False).shape[0]
            return np.zeros(n_sites)
        total_density = np.sum(results, axis=0)
        if ky_list.size > 0:
            total_density /= ky_list.size
        return total_density.real
    
    
    def _charge_worker(self, param):
        E, ky, self_energy_method, use_rgf = param
        G_R, G_lesser_diag, Gamma_L, Gamma_R = self.compute_central_greens_function(
            E, ky=ky, use_rgf=use_rgf, self_energy_method=self_energy_method, compute_lesser=True
        )
        G_n_diag = -1j * G_lesser_diag
        return self.dE * G_n_diag * 1 / (2 * np.pi)

    def diff_rho_poisson(self, Efn=None, V=None, Ef=None, num_points=51, boltzmann=False, use_rgf=True, self_energy_method="sancho_rubio"):
        """This finds the gradient of charge density wrt potential, note that all inputs are arrays here"""
        E_max = Ef + 10 * self.ham.kbT_eV 
        E_list, dE = np.linspace(Ef, E_max, num_points, retstep=True) 
        self.V = V
        self.Efn = Efn
        self.boltzmann = boltzmann

        ky_list = self.k_space
        param_grid = list(product(E_list, ky_list, [self_energy_method], [use_rgf], [dE]))

      
        with multiprocessing.Pool(processes=32) as pool:
            results = pool.map(self._diff_rho_poisson_worker, param_grid)

        if not results:
            print("Warning: No results returned from multiprocessing.")
            n_sites = self.ham.create_hamiltonian(blocks=False).shape[0]
            return np.zeros(n_sites)


        diff_rho = np.sum(results, axis=0)

        if ky_list.size > 0:
            diff_rho /= ky_list.size

        return diff_rho.real
    def _diff_rho_poisson_worker(self, param):
        E, ky, self_energy_method, use_rgf, dE = param 
        gf_param = (E, ky, self_energy_method, use_rgf, False)
        gf_matrix = self._calculate_gf_simple(gf_param) 
        ldos_vector = -1 / np.pi * np.imag(gf_matrix)
        
        exp_arg = (E - self.V - self.Efn) / self.ham.kbT_eV
        exp_arg = np.clip(exp_arg, -100, 100)
        boltzmann_part = np.exp(exp_arg)

        if self.boltzmann:
            return ldos_vector * (1 / boltzmann_part) / self.ham.kbT_eV * dE
        else:
            
            fermi_derivative_part = boltzmann_part / ( (1 + boltzmann_part)**2 )
            return ldos_vector * fermi_derivative_part / self.ham.kbT_eV * dE
        

          
          
            
    def get_n(self, V, Efn, Ec, num_points=51, boltzmann=False, use_rgf=True, self_energy_method="sancho_rubio", 
              method='adaptive', rtol=1e-6, atol=1e-12, processes=4):
        """
        Compute carrier density with improved parallel and vectorized integration methods.
        
        Parameters:
        -----------
        processes : int
            Number of parallel processes to use for k-point calculations.
        ... (rest of your docstring)
        """
        self.V = np.atleast_1d(V)
        self.Efn = np.atleast_1d(Efn)
        self.boltzmann = boltzmann

        # Store parameters that workers will need, avoiding passing them repeatedly.
        self._worker_params = {
            'use_rgf': use_rgf,
            'self_energy_method': self_energy_method
        }
        
        # Fallback to serial execution if only one k-point or one process
        if processes <= 1:
            print("Running in serial mode (1 k-point or 1 process).")
            # Create a mock pool for a unified code path
            pool = multiprocessing.Pool.ThreadPool(1) 
        else:
            print(f"Running in parallel mode with {processes} processes.")
            pool = multiprocessing.Pool(processes=processes)

        with pool:
            if method == 'adaptive':
                results = pool.map(self._adaptive_worker, self.k_space)
            elif method == 'gauss_fermi':
                self._worker_params['Ec'] = Ec
                results = pool.map(self._gauss_fermi_worker, self.k_space)
            else: # uniform
                # Create the parameter grid for the uniform method
                E_max = Ec + 10 
                E_list = np.linspace(Ec, E_max, num_points)
                if num_points > 1:
                    dE = E_list[1] - E_list[0]
                else:
                    dE = 1.0
                param_grid = list(product(E_list, self.k_space, [dE]))
                results = pool.map(self._uniform_worker, param_grid)
        
        # The reduction step is different for the uniform grid
        if not results:
            print("Warning: No results returned from processing.")
            n_sites = self.ham.get_num_sites()
            return np.zeros(n_sites)

        if method == 'uniform':
            # For uniform, results are already multiplied by dE and need to be summed
            total_density = np.sum(results, axis=0)
            if len(self.k_space) > 0:
                total_density /= len(self.k_space)
        else:
            # For adaptive/gauss, results are integrated densities per k-point
            total_density = np.mean(results, axis=0)

        return total_density.real

    # --- Worker for Adaptive Integration ---
    def _adaptive_worker(self, ky):
        """Worker function that performs adaptive integration for a single k-point."""
        kT = self.ham.kbT_eV
        n_sites = self.ham.get_num_sites()

        mu_min = np.min(self.V + self.Efn)
        mu_max = np.max(self.V + self.Efn)
        E_min = mu_min - 2
        E_max = mu_max + 2
        # In a real scenario, Ec could be k-dependent, but we assume it's constant here
        # E_min = max(Ec, E_min) # Ec should be passed if needed

        def integrand_per_k(E):
            G_R, _, _ = self.compute_central_greens_function(
                E, ky=ky,compute_lesser=False
            )
            if sp.issparse(G_R): 
                diag_G_R = G_R.diagonal()
            else:
                diag_G_R = G_R
                
                
            
            ldos_vector = -1.0 / np.pi * np.imag(diag_G_R)
            
            exp_arg = np.clip((E - self.V - self.Efn) / kT, -700, 700)
            distribution = np.exp(-exp_arg) if self.boltzmann else 1.0 / (1.0 + np.exp(exp_arg))
            
            return ldos_vector * distribution

        integral_vector, _ = quad_vec(integrand_per_k, E_min, E_max, epsrel=1e-6, epsabs=1e-12)
        return integral_vector

    # --- Worker for Gauss-Fermi Integration ---
    def _gauss_fermi_worker(self, ky):
        """Worker function for Gauss-Fermi integration for a single k-point."""
        kT = self.ham.kbT_eV
        n_sites = self.ham.get_num_sites()
        Ec = self._worker_params['Ec']
        
        mu_avg = np.mean(self.V + self.Efn)
        x_min = (Ec - mu_avg) / kT
        x_max = 20.0
        
        n_quad = 32
        x_points, weights = np.polynomial.legendre.leggauss(n_quad)
        x_scaled = 0.5 * (x_max - x_min) * x_points + 0.5 * (x_max + x_min)
        weights_scaled = weights * 0.5 * (x_max - x_min)
        
        ky_density = np.zeros(n_sites)
        for x, w in zip(x_scaled, weights_scaled):
            E_ref = mu_avg + x * kT
            G_R, _, _ = self.compute_central_greens_function(
                E_ref, ky=ky, compute_lesser=False
            )
            if sp.issparse(G_R): 
                diag_G_R = G_R.diagonal()
            else:
                diag_G_R = G_R
            
            ldos_vector = -1.0 / np.pi * np.imag(diag_G_R)
            
            exp_arg = (E_ref - self.V - self.Efn) / kT
            fermi_vector = np.exp(-exp_arg) if self.boltzmann else 1.0 / (1.0 + np.exp(exp_arg))
            
            ky_density += w * ldos_vector * fermi_vector * kT
            
        return ky_density

    # --- Worker for Uniform Grid Integration ---
    def _uniform_worker(self, params):
        """Worker for the original uniform grid method, now better vectorized."""
        E, ky, dE = params
        kT = self.ham.kbT_eV

        G_R, _, _ = self.compute_central_greens_function(
            E, ky=ky, compute_lesser=False
        )

        if sp.issparse(G_R): 
            diag_G_R = G_R.diagonal()
        else:
            diag_G_R = G_R
        
        ldos_vector = -1.0 / np.pi * np.imag(diag_G_R)
        
        exp_arg = np.clip((E - self.V - self.Efn) / kT, -700, 700)
        distribution = np.exp(-exp_arg) if self.boltzmann else 1.0 / (1.0 + np.exp(exp_arg))
        
        return ldos_vector * distribution * dE        
    def compute_ldos_matrix(self, self_energy_method="sancho_rubio", use_rgf=True):
        num_sites = self.ham.get_num_sites() 
        num_energies = len(self.energy_grid)
        ldos_matrix = np.zeros((num_energies, num_sites))

        for i, E in enumerate(self.energy_grid):
            # This needs to be parallelized for efficiency, similar to your other methods
            dos_at_E = self.compute_density_of_states(E, use_rgf=use_rgf, self_energy_method=self_energy_method)
            ldos_matrix[i, :] = dos_at_E

        return ldos_matrix