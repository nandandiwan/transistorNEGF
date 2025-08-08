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
        
        self.additional_self_energies = False
        # Büttiker probe parameters
        self.buttiker_probe_enabled = False
        self.buttiker_probe_strength = 0.00025  # Default strength from MATLAB script
        self.buttiker_probe_position = None  # Will be set to middle by default
        # The sparse_threshold is now handled by the smart_inverse function
        # self.sparse_threshold = 0.1
        
        # cache for LDOS
        self.LDOS_cache = {}

    # --- LDOS cache helpers ---
    def _ldos_cache_key(self, E, ky):
        # Round to avoid floating key mismatches
        E_key = float(np.real(E))
        ky_key = float(ky)
        return (round(E_key, 12), round(ky_key, 12))

    def clear_ldos_cache(self):
        """Clear the LDOS cache (works for plain dict or Manager proxy)."""
        try:
            self.LDOS_cache.clear()
        except Exception:
            # Fall back to replacing with a new dict
            self.LDOS_cache = {}

    def _ensure_shared_cache(self, processes: int | None):
        """
        Promote LDOS_cache to a multiprocessing.Manager dict for the duration of
        a multiprocessing section. Returns the Manager instance; caller must
        finalize via _finalize_shared_cache(manager) after the Pool work.
        """
        if processes is not None and processes > 1 and not hasattr(self.LDOS_cache, '_callmethod'):
            manager = multiprocessing.Manager()
            shared = manager.dict()
            # seed with current cache
            for k, v in self.LDOS_cache.items():
                shared[k] = v
            self.LDOS_cache = shared
            return manager
        return None

    def _finalize_shared_cache(self, manager):
        """Copy proxy cache back to a plain dict and shutdown the manager."""
        if manager is None:
            return
        try:
            # Copy back to a local dict so it remains usable after manager shutdown
            self.LDOS_cache = dict(self.LDOS_cache)
        finally:
            try:
                manager.shutdown()
            except Exception:
                pass

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

    def enable_buttiker_probe(self, strength=0.00025, position=None):
        """
        Enable Büttiker probe for broadening resonances.
        
        Args:
            strength (float): Imaginary self-energy strength (positive)
            position (int): Position index to place probe (None for middle)
        """
        self.buttiker_probe_enabled = True
        self.buttiker_probe_strength = strength
        self.buttiker_probe_position = position
        self.additional_self_energies = True

    def disable_buttiker_probe(self):
        """Disable Büttiker probe."""
        self.buttiker_probe_enabled = False
        self.additional_self_energies = False

    def _compute_buttiker_probe_self_energy(self, E):
        """
        Compute Büttiker probe self-energy matrix.
        
        Returns:
            scipy.sparse matrix: Büttiker probe self-energy
        """
        if not self.buttiker_probe_enabled:
            return None
            
        n = self.ham.get_num_sites()
        position = self.buttiker_probe_position
        if position is None:
            position = n // 2  # Middle of device
            
        # Create sparse matrix with imaginary self-energy at probe position
        Sigma_bp = sp.lil_matrix((n, n), dtype=complex)
        Sigma_bp[position, position] = -1j * self.buttiker_probe_strength
        
        return Sigma_bp.tocsc()

    def _compute_buttiker_probe_transmission_correction(self, E, G_R, Gamma_L, Gamma_R):
        """
        Compute transmission correction for Büttiker probe using the formula:
        T_corrected = T12 + (T13 * T23) / (T12 + T23)
        
        where:
        T12 = trace(Gamma_L @ G_R @ Gamma_R @ G_A)  # Direct transmission
        T13 = trace(Gamma_L @ G_R @ Gamma_bp @ G_A)  # Left to probe
        T23 = trace(Gamma_R @ G_R @ Gamma_bp @ G_A)  # Right to probe
        """
        if not self.buttiker_probe_enabled:
            return None
            
        # Büttiker probe broadening function
        Sigma_bp = self._compute_buttiker_probe_self_energy(E)
        Gamma_bp = 1j * (Sigma_bp - Sigma_bp.conj().T)
        
        G_A = G_R.conj().T
        
        # Calculate transmission coefficients
        T12 = np.real(np.trace(Gamma_L @ G_R @ Gamma_R @ G_A))  # Direct
        T13 = np.real(np.trace(Gamma_L @ G_R @ Gamma_bp @ G_A))  # Left to probe
        T23 = np.real(np.trace(Gamma_R @ G_R @ Gamma_bp @ G_A))  # Right to probe
        
        # Corrected transmission using Büttiker probe formula
        if T12 + T23 != 0:
            T_corrected = T12 + (T13 * T23) / (T12 + T23)
        else:
            T_corrected = T12
            
        return T_corrected

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
        
        # Add Büttiker probe self-energy if enabled
        H_eff = H + Sigma_L_full + Sigma_R_full
        if self.additional_self_energies and self.buttiker_probe_enabled:
            Sigma_bp = self._compute_buttiker_probe_self_energy(E)
            H_eff += Sigma_bp
        
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
        
        # self_energy_shape = sigma_L.shape[0]
        # SE_H_factor = self_energy_shape / block_size
        # for imat in range(SE_H_factor):
        #     start = imat * block_size
        #     end = (imat + 1) * block_size
        #     after = (imat + 2) * block_size 
            
        #     piece_one
            
        
        
        
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
        if use_rgf==False:
            G_R_diag = np.diag(G_R_diag.toarray())
        return G_R_diag

    
    def compute_transmission(self, E, ky=0, self_energy_method=None):
        """

        find transmission 
        """
        # Get Green's function and broadening matrices
        G_R, Gamma_L, Gamma_R = self.compute_central_greens_function(
            E,ky=ky, use_rgf=False, self_energy_method=self_energy_method, compute_lesser=False
        )

        if self.buttiker_probe_enabled:
            G_R_dense = G_R.toarray()
            G_A_dense = G_R_dense.conj().T
            Gamma_L_dense = Gamma_L.toarray()
            Gamma_R_dense = Gamma_R.toarray()
            # Use Büttiker probe corrected transmission
            T = self._compute_buttiker_probe_transmission_correction(E, G_R_dense, Gamma_L_dense, Gamma_R_dense)
        else:
            # Standard transmission calculation
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

    def compute_current_landauer(self, V_list, E_range=(-0.2, 0.8), N_E=101, self_energy_method=None):
        """
        Compute current using Landauer-Büttiker formula for a voltage range.
        
        
        Args:
            V_list: List of bias voltages to calculate
            E_range: Energy integration range (E_min, E_max)
            N_E: Number of energy points
            self_energy_method: Method for self-energy calculation
            
        Returns:
            list: Current values for each voltage
        """
        # Physical constants (from MATLAB script)
        hbar = 1.0545718e-34  # J⋅s
        q = 1.60217662e-19    # C
        IE = q**2 / (np.pi * hbar)  # Conductance quantum factor
        
        Ef = self.ham.Ef  # Fermi energy in eV
        kT = self.ham.kbT_eV  # Thermal energy in eV
        
        # Energy grid
        E_min, E_max = E_range
        E = np.linspace(E_min, E_max, N_E)
        dE = E[1] - E[0]
        
        current_list = []
        
        for V in V_list:
            # Chemical potentials
            mu1 = Ef + V/2  # Source
            mu2 = Ef - V/2  # Drain
            
            # Set the voltages in the Hamiltonian
            self.ham.set_voltage(Vs=V/2, Vd=-V/2)
            
            # Fermi distributions
            f1 = 1.0 / (1.0 + np.exp((E - mu1) / kT))
            f2 = 1.0 / (1.0 + np.exp((E - mu2) / kT))
            
            # Integrate current
            I = 0.0
            for k, Ek in enumerate(E):
                T_k = self.compute_transmission(Ek, self_energy_method=self_energy_method)
                I += dE * IE * T_k * (f1[k] - f2[k])
            
            current_list.append(I)
            print(f"V = {V:.3f} V, I = {I*1e6:.4f} μA")
        
        return current_list    
    
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

    def diff_rho_poisson(self, Efn=None, V=None, Ec=None, num_points=51, boltzmann=False, use_rgf=True, self_energy_method="sancho_rubio", method='gauss_fermi', processes=32):
        """This finds the gradient of charge density wrt potential, note that all inputs are arrays here
        Supports 'uniform' (legacy) and 'gauss_fermi' (faster) integration methods.
        """
        self.V = np.atleast_1d(V)
        self.Efn = np.atleast_1d(Efn)
        self.boltzmann = boltzmann
        
        ky_list = self.k_space

        # Create shared cache only for the duration of multiprocessing
        _mgr = self._ensure_shared_cache(processes)
        try:
            if method == 'uniform':
                E_max = Ec + 10 * self.ham.kbT_eV 
                E_list, dE = np.linspace(Ec, E_max, num_points, retstep=True)
                param_grid = list(product(E_list, ky_list, [self_energy_method], [use_rgf], [dE]))
                with multiprocessing.Pool(processes=processes) as pool:
                    results = pool.map(self._diff_rho_poisson_worker, param_grid)
            else:  # gauss_fermi
                # store params needed by the worker
                self._diff_params = {
                    'Ef': Ec,
                    'use_rgf': use_rgf,
                    'self_energy_method': self_energy_method
                }
                with multiprocessing.Pool(processes=processes) as pool:
                    results = pool.map(self._diff_rho_gauss_fermi_worker, ky_list)
        finally:
            self._finalize_shared_cache(_mgr)

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
        # Use LDOS cache
        ldos_vector = self._get_ldos_cached(E, ky, self_energy_method=self_energy_method, use_rgf=use_rgf)
        
        exp_arg = (E - self.V - self.Efn) / self.ham.kbT_eV
        exp_arg = np.clip(exp_arg, -100, 100)
        boltzmann_part = np.exp(exp_arg)

        if self.boltzmann:
            return ldos_vector * (1 / boltzmann_part) / self.ham.kbT_eV * dE
        else:
            fermi_derivative_part = boltzmann_part / ( (1 + boltzmann_part)**2 )
            return ldos_vector * fermi_derivative_part / self.ham.kbT_eV * dE

    def _diff_rho_gauss_fermi_worker(self, ky):
        """Gauss-Fermi quadrature for d rho / d V at a single k-point."""
        kT = self.ham.kbT_eV
        n_sites = self.ham.get_num_sites()
        Ef = self._diff_params['Ef']

        mu_avg = np.mean(self.V + self.Efn)
        x_min = (Ef - mu_avg) / kT
        x_max = 20.0

        n_quad = 32
        x_points, weights = np.polynomial.legendre.leggauss(n_quad)
        x_scaled = 0.5 * (x_max - x_min) * x_points + 0.5 * (x_max + x_min)
        weights_scaled = weights * 0.5 * (x_max - x_min)

        ky_diff = np.zeros(n_sites)
        for x, w in zip(x_scaled, weights_scaled):
            E_ref = mu_avg + x * kT
            ldos_vector = self._get_ldos_cached(E_ref, ky, self_energy_method=self._diff_params['self_energy_method'], use_rgf=self._diff_params['use_rgf'])
            exp_arg = np.clip((E_ref - self.V - self.Efn) / kT, -700, 700)
            if self.boltzmann:
                factor = np.exp(-exp_arg)
            else:
                expx = np.exp(exp_arg)
                factor = expx / (1.0 + expx)**2  # df/dV factor (1/kT) times dE=kT dx cancels
            ky_diff += w * ldos_vector * factor
        return ky_diff
    


    def get_n(self, V, Efn, Ec, num_points=51, boltzmann=False, use_rgf=True, self_energy_method="sancho_rubio", 
              method='gauss_fermi', rtol=1e-6, atol=1e-12, processes=32):
        """
        Compute carrier density with improved parallel and vectorized integration methods.
        
        Parameters:
        -----------
        processes : int
            Number of parallel processes to use for k-point calculations.
    
        """
        self.V = np.atleast_1d(V)
        self.Efn = np.atleast_1d(Efn)
        self.boltzmann = boltzmann

        # Store parameters that workers will need, avoiding passing them repeatedly.
        self._worker_params = {
            'use_rgf': use_rgf,
            'self_energy_method': self_energy_method
        }
        
        # Promote cache to shared proxy for multiprocessing duration
        _mgr = self._ensure_shared_cache(processes)
        try:
            # Fallback to serial execution if only one k-point or one process
            if processes <= 1:
                print("Running in serial mode (1 k-point or 1 process).")
                pool = multiprocessing.pool.ThreadPool(1)
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
        finally:
            self._finalize_shared_cache(_mgr)
        
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
            # Use LDOS cache with current solver settings
            ldos_vector = self._get_ldos_cached(
                E_ref, ky,
                self_energy_method=self._worker_params['self_energy_method'],
                use_rgf=self._worker_params['use_rgf']
            )
            
            exp_arg = (E_ref - self.V - self.Efn) / kT
            fermi_vector = np.exp(-exp_arg) if self.boltzmann else 1.0 / (1.0 + np.exp(exp_arg))
            
            ky_density += w * ldos_vector * fermi_vector * kT
            
        return ky_density

    # --- Worker for Uniform Grid Integration ---
    def _uniform_worker(self, params):
        """Worker for the original uniform grid method, now better vectorized."""
        E, ky, dE = params
        kT = self.ham.kbT_eV

        # Use LDOS cache with current solver settings
        method = getattr(self, '_worker_params', {}).get('self_energy_method', None)
        use_rgf = getattr(self, '_worker_params', {}).get('use_rgf', True)
        ldos_vector = self._get_ldos_cached(E, ky, self_energy_method=method, use_rgf=use_rgf)
        
        exp_arg = np.clip((E - self.V - self.Efn) / kT, -700, 700)
        distribution = np.exp(-exp_arg) if self.boltzmann else 1.0 / (1.0 + np.exp(exp_arg))
        
        return ldos_vector * distribution * dE        

    def _get_ldos_cached(self, E, ky, self_energy_method=None, use_rgf=True):
        key = self._ldos_cache_key(E, ky)
        try:
            if key in self.LDOS_cache:
                return np.array(self.LDOS_cache[key])
        except Exception:
            # If cache is not a mapping proxy yet
            pass
        # Compute LDOS via Green's function
        G_R, _, _ = self.compute_central_greens_function(
            E, ky=ky, compute_lesser=False, use_rgf=use_rgf,
            self_energy_method=self_energy_method
        )
        if sp.issparse(G_R):
            diag_G_R = G_R.diagonal()
        else:
            diag_G_R = G_R
        ldos_vector = -1.0 / np.pi * np.imag(diag_G_R)
        
        # Store in cache
        try:
            self.LDOS_cache[key] = ldos_vector
        except Exception:
            # If cache not shareable or some multiprocessing error, just skip caching
            pass
        return ldos_vector

    # Public alias (for external callers expecting this name)
    def get_ldos_cache(self, E, ky, self_energy_method=None, use_rgf=True):
        return self._get_ldos_cached(E, ky, self_energy_method=self_energy_method, use_rgf=use_rgf)