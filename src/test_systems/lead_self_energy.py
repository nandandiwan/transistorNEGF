
import numpy as np
from scipy import linalg
import scipy.sparse as spa
from hamiltonian import Hamiltonian
import numpy as np
import scipy.sparse as sp

class LeadSelfEnergy():
    """
    Lead self-energy calculation using surface Green's functions.
    Based on the robust implementations from OpenMX TRAN_Calc_SurfGreen.c
    """
    
    def __init__(self, hamiltonian: Hamiltonian):
        #self.ds = device
        self.ham = hamiltonian
        self.eta = 1e-12  # Small imaginary part for numerical stability
        
    def _add_eta(self, E):
        """Add small imaginary part if energy is real for numerical stability"""
        if np.imag(E) == 0:
            return E + 1j * self.eta
        return E
        
    def surface_greens_function(self, E, H00, H01, method="sancho_rubio", 
                               iteration_max=1000, tolerance=1e-6):
        """
        Calculate surface Green's function using specified method.
        
        Args:
            E: Energy (complex)
            H00: On-site Hamiltonian matrix
            H01: Hopping matrix (coupling to next layer)
            method: "sancho_rubio", "iterative", or "transfer"
            iteration_max: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Surface Green's function matrix
        """
        E = self._add_eta(E)
        
        if method == "sancho_rubio":
            return self._sancho_rubio_surface_gf(E, H00, H01, tolerance, iteration_max)
        elif method == "iterative":
            return self._iterative_surface_gf(E, H00, H01, tolerance, iteration_max)
        elif method == "transfer":
            return self._transfer_surface_gf(E, H00, H01, tolerance, iteration_max)
        elif method == "recursive":
            return self._recursive_self_energy_mixed(E, H00, H01, max_iter=iteration_max, tol = tolerance)
        else:
            raise ValueError(f"Unknown method: {method}")
    def _sancho_rubio_surface_gf(self, E, H00, H01, S00=None, iter_max=100, TOL=1e-10):
        """
        Jiezi surface_gf algorithm translated to use numpy arrays
        """
    
        n = H00.shape[0]
        I = np.eye(n, dtype=complex)
        
        # Handle overlap matrix
        if S00 is None:
            S00 = I
            
        # Convert to dense if needed
        if hasattr(H00, 'toarray'):
            H00 = H00.toarray()
        if hasattr(H01, 'toarray'):
            H01 = H01.toarray()
        if hasattr(S00, 'toarray'):
            S00 = S00.toarray()
        
        iter_c = 0
        H10 = H01.conj().T
        alpha = H10.copy()
        beta = H10.conj().T  # H10.dagger()
        epsilon = H00.copy()
        epsilon_s = H00.copy()
        E = I * E
        
        while iter_c < iter_max:
            iter_c += 1
            
            # inv_term = (w - epsilon)^(-1)
            inv_term = np.linalg.solve(E - epsilon, I)
            
            # alpha_new = alpha * inv_term * alpha
            alpha_new = alpha @ inv_term @ alpha
            
            # beta_new = beta * inv_term * beta  
            beta_new = beta @ inv_term @ beta
            
            # epsilon_new = epsilon + alpha*inv_term*beta + beta*inv_term*alpha
            epsilon_new = epsilon + alpha @ inv_term @ beta + beta @ inv_term @ alpha
            
            # epsilon_s_new = epsilon_s + alpha*inv_term*beta
            epsilon_s_new = epsilon_s + alpha @ inv_term @ beta
            
            # Check convergence using Frobenius norm
            convergence_check = np.linalg.norm(alpha_new, ord='fro')
            
            if convergence_check < TOL:
                G00 = np.linalg.solve(E - epsilon_s_new, I)
                GBB = np.linalg.solve(E - epsilon_new, I)
                break
            else:
                alpha = alpha_new.copy()
                beta = beta_new.copy() 
                epsilon = epsilon_new.copy()
                epsilon_s = epsilon_s_new.copy()
        
        if iter_c >= iter_max:
            print(f"Warning: Jiezi Surface GF did not converge after {iter_max} iterations")
            return self._recursive_self_energy_mixed(E, H00,H01)
        return G00
    def _recursive_self_energy_mixed(self, E, H00, H01, max_iter=500, tol=1e-8, mixing_beta=0.1):
        """
        Calculates the lead self-energy using a stable RGF method with
        linear mixing to ensure convergence.

        Args:
            E (float): Energy.
            eta (float): Infinitesimal broadening term.
            H00 (np.ndarray): On-site Hamiltonian of a principal layer.
            H01 (np.ndarray): Coupling from layer 0 to layer 1.
            H10 (np.ndarray): Coupling from layer 1 to layer 0.
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence tolerance.
            mixing_beta (float): Damping parameter for the iteration (0 < beta <= 1).
                                Smaller values are more stable but converge slower.

        Returns:
            np.ndarray: The converged self-energy matrix, Sigma.
        """
        # Load and prepare matrices
    
        if sp.issparse(H00): H00 = H00.toarray()
        if sp.issparse(H01): H01 = H01.toarray()
        H10 = H01.conj().T
            
        w = E
        identity = np.eye(H00.shape[0])

        # Initial guess for the surface Green's function (g_s)
        try:
            g_s = np.linalg.inv(w * identity - H00)
        except np.linalg.LinAlgError:
            print("ERROR: Failed on the very first inversion!")
            return None
        
        for i in range(max_iter):
            g_s_old = g_s.copy()

            # Calculate the self-energy based on the current g_s
            sigma = H10 @ g_s @ H01
            
            # Calculate the "next guess" for g_s
            mat_to_invert = w * identity - H00 - sigma
            try:
                g_s_new = np.linalg.inv(mat_to_invert)
            except np.linalg.LinAlgError as e:
                print(f"ERROR: Matrix inversion failed at iteration {i}. {e}")
                return None # Or handle as appropriate

            # *** MIXING STEP ***
            # Instead of g_s = g_s_new, we mix the old and new solutions
            g_s = (1 - mixing_beta) * g_s_old + mixing_beta * g_s_new

            # Check for convergence
            diff = np.linalg.norm(g_s - g_s_old) / np.linalg.norm(g_s)
            if diff < tol:
                # print(f"\n--- Convergence Reached in {i+1} iterations ---")
                final_sigma = H10 @ g_s @ H01
                return final_sigma

        print(f"Warning: RGF self-energy with mixing did not converge after {max_iter} iterations. Diff = {diff}")
        final_sigma = H10 @ g_s @ H01
        return final_sigma


    
    def _analytical_1d_surface_gf(self, E):
        """
        Analytical surface Green's function for 1D chain to match MATLAB formula.
        For H00=0, H01=-1: Uses the same branch choice as arccos method.
        """
        # MATLAB approach: k = arccos(1 - E/2t) with t=1
        # This gives the same result as -exp(ik) where k comes from arccos
        ck = 1 - E / (2 * 1.0)  # cos(k)
        ck = np.clip(ck, -1, 1)  # Ensure valid range for arccos
        k = np.arccos(ck)
        
        # Surface Green's function: G_s = -exp(ik) / t = -exp(ik)
        G_surface = -np.exp(1j * k)
        
        return np.array([[G_surface]], dtype=complex)
    
    def _iterative_surface_gf(self, E, H00, H01, tolerance=1e-6, iteration_max=1000):
        """
        Alternative iterative method for surface Green's function.
        Based on TRAN_Calc_SurfGreen_Multiple_Inverse from OpenMX.
        """
        raise Exception("this is broken")
        n = H00.shape[0]
        I = np.eye(n, dtype=complex)
        
        # Convert to dense arrays
        if hasattr(H00, 'toarray'):
            H00 = H00.toarray()
        if hasattr(H01, 'toarray'):
            H01 = H01.toarray()
        
        # h0 = E*I - H00
        h0 = E * I - H00
        
        # hl = H01, hr = H01^dagger
        hl = H01.copy()
        hr = H01.conj().T
        
        # Initial Green's function
        try:
            g0 = linalg.solve(h0, I)
        except linalg.LinAlgError:
            g0 = linalg.pinv(h0)
        
        for iteration in range(1, iteration_max):
            # Calculate hl*g0*hr
            temp1 = hl @ g0
            temp2 = temp1 @ hr
            
            # New denominator: h0 - hl*g0*hr
            h_new = h0 - temp2
            
            try:
                g_new = linalg.solve(h_new, I)
            except linalg.LinAlgError:
                g_new = linalg.pinv(h_new)
            
            # Check convergence
            diff = g_new - g0
            rms = np.sqrt(np.max(np.abs(diff)**2))
            
            if rms < tolerance:
                break
                
            g0 = g_new.copy()
        
        if iteration >= iteration_max - 1:
            print(f"Warning: Iterative Surface GF did not converge after {iteration_max} iterations")
            
        return g0
    
    def _transfer_surface_gf(self, E, H00, H01, tolerance=1e-6, iteration_max=1000):
        """
        Transfer matrix method for surface Green's function.
        Based on TRAN_Calc_SurfGreen_transfer from OpenMX.
        """
        n = H00.shape[0]
        I = np.eye(n, dtype=complex)
        
        # Convert to dense arrays
        if hasattr(H00, 'toarray'):
            H00 = H00.toarray()
        if hasattr(H01, 'toarray'):
            H01 = H01.toarray()
        
        H10 = H01.conj().T
        
        # Initial inverse: (E*I - H00)^-1
        try:
            gr00_inv = linalg.solve(E * I - H00, I)
        except linalg.LinAlgError:
            gr00_inv = linalg.pinv(E * I - H00)
        
        # Initial transfer matrices
        t_i = gr00_inv @ H10
        bar_t_i = gr00_inv @ H01
        
        T_i = t_i.copy()
        bar_T_i = bar_t_i.copy()
        
        T_i_old = T_i.copy()
        
        for iteration in range(1, iteration_max):
            # Calculate (I - t*bar_t - bar_t*t)^-1
            temp1 = t_i @ bar_t_i
            temp2 = bar_t_i @ t_i
            denominator = I - temp1 - temp2
            
            try:
                inv_denom = linalg.solve(denominator, I)
            except linalg.LinAlgError:
                inv_denom = linalg.pinv(denominator)
            
            # Update transfer matrices
            t_i_new = inv_denom @ (t_i @ t_i)
            bar_t_i_new = inv_denom @ (bar_t_i @ bar_t_i)
            
            # Update accumulated transfer matrices
            bar_T_i_new = bar_T_i @ bar_t_i_new
            T_i_new = T_i + bar_T_i @ t_i_new
            
            # Check convergence
            diff = T_i_new - T_i_old
            rms = np.sqrt(np.max(np.abs(diff)**2))
            
            if rms < tolerance:
                break
            
            # Update for next iteration
            t_i = t_i_new
            bar_t_i = bar_t_i_new
            T_i_old = T_i.copy()
            T_i = T_i_new
            bar_T_i = bar_T_i_new
        
        if iteration >= iteration_max - 1:
            print(f"Warning: Transfer Surface GF did not converge after {iteration_max} iterations")
        
        # Final surface Green's function
        final_matrix = E * I - H00 - H01 @ T_i
        try:
            return linalg.solve(final_matrix, I)
        except linalg.LinAlgError:
            return linalg.pinv(final_matrix)
    
    def self_energy(self, side, E, ky=0, method="sancho_rubio"):
        """
        Calculate lead self-energy using surface Green's functions for general device types.
        Args:
            side: "left" or "right" lead
            E: Energy
            method: Surface GF method ("sancho_rubio", "iterative", "transfer")
        Returns:
            Self-energy matrix
        """
        # Get lead Hamiltonian matrices for the device type
        H00, H01, H10 = self.ham.get_H00_H01_H10(ky=ky, side = side)
        # print(H00.toarray())
        # print("=============")
        # print(H01.toarray())
        # print("=============")
        # print(H10.toarray())
        # Handle large energies
        if np.abs(E) > 5e5:
            return np.zeros((H00.shape[0], H00.shape[0]), dtype=complex)
        Vsd = (self.ham.Vs + self.ham.Vd)

        if side == "left":
            E_lead = E - self.ham.mu1#- self.ham.Vs - (self.ham.Ef+Vsd/2)
            
        else:  # right
            E_lead = E - self.ham.mu2 #- self.ham.Vd - (self.ham.Ef-Vsd/2)

        # Calculate surface Green's function
        try:
            G_surface = self.surface_greens_function(E_lead, H00, H01, method=method)
        except Exception as e:
            print(f"Warning: Surface GF calculation failed for {side} lead: {e}")
            n = H00.shape[0]
            eta = 1e-3
            G_surface = linalg.pinv(self._add_eta(E_lead) * np.eye(n) - H00)

        # Calculate self-energy
        if side == "left":
            self_energy = H10 @ G_surface @ H01
            return self_energy
        else:
            self_energy = H01 @ G_surface @ H10
            return self_energy
