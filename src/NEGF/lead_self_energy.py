

from device import Device
import scipy.sparse as spa 
import scipy as sp
from scipy import linalg
import numpy as np
from scipy.sparse import bmat, identity, random, csc_matrix
from scipy.sparse.linalg import eigsh, eigs, spsolve

from hamiltonian import Hamiltonian
class LeadSelfEnergy():
    def __init__(self, device : Device, hamiltonian : Hamiltonian):
        self.ds = device
        
        self.ham = hamiltonian
    
        self.E = 0.01
        self.ky = 0
        # for silicon 100
        self.P = 4
        # cache
        self.layerHamiltonianCache = {}
        self.layerHamiltonianCache[self.ky] = self.ham.getLayersHamiltonian(self.ky)
        
        self.eta = 1e-6j
    
    def set_inputs(self, E, ky):
        self.E = E
        self.ky = ky
        
    @staticmethod
    def GzerozeroH_W_sparse(wmH: spa.spmatrix, t: spa.spmatrix) -> np.ndarray:
        """
        Surface Green's function calculation optimized for sparse matrices.
        """
        N = wmH.shape[0]
        

        wmH_dense = wmH.toarray()
        t_dense = t.toarray()

        A = np.zeros((2*N, 2*N), dtype=complex)
        B = np.zeros((2*N, 2*N), dtype=complex)
        
        A[:N, N:2*N] = np.eye(N)
        A[N:2*N, :N] = -t_dense.conj().T
        A[N:2*N, N:2*N] = wmH_dense
        
        B[:N, :N] = np.eye(N)
        B[N:2*N, N:2*N] = t_dense
        

        try:
            eigenvalues, eigenvectors = linalg.eig(A, B)
        except linalg.LinAlgError:
            print("Warning: Using pseudo-inverse for eigenvalue problem")
            B_pinv = linalg.pinv(B)
            eigenvalues, eigenvectors = linalg.eig(B_pinv @ A)
        

        magnitudes = np.abs(eigenvalues)

        valid_mask = np.isfinite(magnitudes) & (magnitudes > 1e-12)
        valid_eigenvalues = eigenvalues[valid_mask]
        valid_eigenvectors = eigenvectors[:, valid_mask]
        valid_magnitudes = magnitudes[valid_mask]
        
        if len(valid_eigenvalues) == 0:
            raise ValueError("No valid eigenvalues found")
        

        real_parts = np.real(valid_eigenvalues)
        sorted_indices = np.lexsort((valid_magnitudes, real_parts))
        
    
        sorted_eigenvectors = valid_eigenvectors[:, sorted_indices]
        
        Z11 = sorted_eigenvectors[:N, :N]
        Z21 = sorted_eigenvectors[N:2*N, :N]
        

        Z11_inv = linalg.pinv(Z11, rtol=1e-12)
        Gzeta = Z21 @ Z11_inv

        
        return Gzeta
    
    def self_energy(self, side, E, ky) -> np.ndarray:
        """
        Corrected self-energy calculation.
        """
        ham = self.ham
        H00, H01, H10 = ham.get_H00_H01_H10(ky, side=side, sparse=True)
        
        N = H00.shape[0]
        eta = 1e-6
            
        if side == "left":
            wmH = (E - self.ds.Vs) * spa.eye(N, dtype=complex) - H00 + 1j * eta * spa.eye(N, dtype=complex)
            Gzeta = LeadSelfEnergy.GzerozeroH_W_sparse(wmH, H10)

            selfenergy = H10.toarray() @ Gzeta @ H10.conj().T.toarray()
            return selfenergy[:ham.Nz * 10,:ham.Nz * 10]
        else:
            wmH = (E - self.ds.Vd) * spa.eye(N, dtype=complex) - H00 + 1j * eta * spa.eye(N, dtype=complex)
            Gzeta = LeadSelfEnergy.GzerozeroH_W_sparse(wmH, H01)
            selfenergy = H01.toarray() @ Gzeta @ H01.conj().T.toarray()
            return selfenergy[-ham.Nz * 10:,-ham.Nz * 10:]
        
    
    def iterative_self_energy(self, E, ky, side = "left"):
        """
        Implements Algorithm I (Iterative method) from the paper:
        "Methods for fast evaluation of self-energy matrices in tight-binding modeling"
        
        doesn't work i think
        """
        dagger = lambda A: np.conjugate(A.T)
        self.set_inputs(E, ky)
        XIs, XI, PI, h12 = self.decomposition_algorithm(side)
        
        # Convert h12 to array if it's sparse
        if hasattr(h12, 'toarray'):
            h12 = h12.toarray()
        
        # Set up matrices for iteration - use the energy with small imaginary part
        E_complex = self.E + self.eta
        I = np.eye(XI.shape[0], dtype=complex)
        
        A = E_complex * I - XI
        As = E_complex * I - XIs
        delta = 1e-5
        
        PI_dagger = PI.conj().T

        iteration_count = 0
        max_iterations = 100  
        
        while (np.max(np.abs(PI)) > delta or np.max(np.abs(PI_dagger)) > delta) and iteration_count < max_iterations:
            try:
                X_PI = np.linalg.solve(A, PI)
                X_PI_dagger = np.linalg.solve(A, PI_dagger)
                
            except np.linalg.LinAlgError:
                print(f"Error: Matrix A is singular at iteration {iteration_count}. Iteration cannot continue.")
                return np.full_like(A, np.nan)
            
            A = A - PI @ X_PI_dagger - PI_dagger @ X_PI
            As = As - PI @ X_PI_dagger
            PI = PI @ X_PI
            PI_dagger = PI_dagger @ X_PI_dagger
            iteration_count += 1

        if iteration_count >= max_iterations:
            print(f"Warning: Iteration did not converge after {max_iterations} steps")
    
        h12_dagger = h12.conj().T
        
        try:
            Y = np.linalg.solve(As, h12_dagger)
        except np.linalg.LinAlgError:
            print("Error: Matrix As is singular for the final solve step.")
            return np.full_like(As, np.nan)

        Sigma = h12 @ Y

        return Sigma
    
    def get_layer_hamiltonian(self, p, side="left") -> tuple:
        if not self.ky in self.layerHamiltonianCache:
            self.layerHamiltonianCache[self.ky] = self.ham.getLayersHamiltonian(self.ky)
        layerHamiltonians = self.layerHamiltonianCache[self.ky]
        
        if p > self.P or p < 1:
            print(f"Error: p={p}, self.P={self.P}")
            raise ValueError("layers are indexed 1,2,...self.P")
   
        if side == "left":
            layer_index = 3 - (p - 1) 
        elif side == "right":
            layer_index = (self.ham.layer_right_lead + p - 1) % 4
        else:
            raise ValueError("side must be 'left' or 'right'") 
        if layer_index not in layerHamiltonians:
            raise ValueError(f"Layer index {layer_index} not found in layer Hamiltonians")
            
        H_pp, H_p_p1 = layerHamiltonians[layer_index]

        if hasattr(H_pp, 'tocsc'):
            H_pp = H_pp.tocsc()
        if H_p_p1 is not None and hasattr(H_p_p1, 'tocsc'):
            H_p_p1 = H_p_p1.tocsc()
            
        return H_pp, H_p_p1   
        
    
    def decomposition_algorithm(self, side="left"):
        """broken"""
        dagger = lambda A: np.conjugate(A.T)

        H_tilde_matrices = [None] * self.P  
        H_cross_matrices = [None] * self.P  
        

        h_PP, h_P_P1 = self.get_layer_hamiltonian(self.P, side)

        if hasattr(h_PP, 'toarray'):
            h_PP_dense = h_PP.toarray()
        else:
            h_PP_dense = h_PP
            
        E_matrix = self.E * np.eye(h_PP_dense.shape[0], dtype=complex)
        H_tilde_PP = np.linalg.inv(E_matrix - h_PP_dense)
        H_tilde_matrices[self.P - 1] = H_tilde_PP
        H_cross_matrices[self.P - 1] = H_tilde_PP
   
        for p in range(self.P - 1, 1, -1):
            h_pp, h_p_p1 = self.get_layer_hamiltonian(p, side)
            
            if h_p_p1 is None:
                raise ValueError(f"Missing coupling H_{p},{p+1} for layer {p}")

            if hasattr(h_pp, 'toarray'):
                h_pp_dense = h_pp.toarray()
            else:
                h_pp_dense = h_pp
                
            if hasattr(h_p_p1, 'toarray'):
                h_p_p1_dense = h_p_p1.toarray()
            else:
                h_p_p1_dense = h_p_p1

            E_matrix_p = self.E * np.eye(h_pp_dense.shape[0], dtype=complex)
            coupling_term = h_p_p1_dense @ H_tilde_matrices[p] @ dagger(h_p_p1_dense)
            H_tilde_pp = np.linalg.inv(E_matrix_p - h_pp_dense - coupling_term)
            H_tilde_matrices[p - 1] = H_tilde_pp

            H_cross_pp = H_tilde_matrices[p - 1] @ h_p_p1_dense @ H_cross_matrices[p]
            H_cross_matrices[p - 1] = H_cross_pp

        C_tilde_22 = H_tilde_matrices[1]  
        C_tilde_2P = H_cross_matrices[1]  

        C_tilde_matrices = [None] * (self.P + 1)
        C_tilde_matrices[2] = C_tilde_22

        for p in range(3, self.P + 1):
            _, h_p_minus_1_p = self.get_layer_hamiltonian(p - 1, side)
            
            if h_p_minus_1_p is None:
                raise ValueError(f"Missing coupling H_{p-1},{p} for layer {p-1}")
        
            if hasattr(h_p_minus_1_p, 'toarray'):
                h_p_minus_1_p_dense = h_p_minus_1_p.toarray()
            else:
                h_p_minus_1_p_dense = h_p_minus_1_p
                
            h_p_p_minus_1 = dagger(h_p_minus_1_p_dense)
            
            H_tilde_pp = H_tilde_matrices[p - 1]
            C_tilde_prev = C_tilde_matrices[p - 1]
            
            inner_term = h_p_p_minus_1 @ C_tilde_prev @ dagger(h_p_p_minus_1)
            C_tilde_matrices[p] = H_tilde_pp + H_tilde_pp @ inner_term @ H_tilde_pp
        
        h11, h12 = self.get_layer_hamiltonian(1, side)
        if h12 is None:
            raise ValueError("Missing coupling H_1,2")
        
        if hasattr(h11, 'toarray'):
            h11_dense = h11.toarray()
        else:
            h11_dense = h11
            
        if hasattr(h12, 'toarray'):
            h12_dense = h12.toarray()
        else:
            h12_dense = h12
            
        XI_s = h11_dense + h12_dense @ C_tilde_matrices[2] @ dagger(h12_dense)
        
        _, h_P_P1 = self.get_layer_hamiltonian(self.P, side)
        if h_P_P1 is None:

            XI = XI_s
        else:
            if hasattr(h_P_P1, 'toarray'):
                h_P_P1_dense = h_P_P1.toarray()
            else:
                h_P_P1_dense = h_P_P1
                
            C_tilde_PP = C_tilde_matrices[self.P]
            XI = XI_s + dagger(h_P_P1_dense) @ C_tilde_PP @ h_P_P1_dense
        
        if h_P_P1 is None:
            PI = np.zeros((h12_dense.shape[0], h12_dense.shape[1]), dtype=complex)
        else:
            PI = h12_dense @ C_tilde_2P @ h_P_P1_dense
        
        return XI_s, XI, PI, h12_dense

    def construct_U_plus_and_Lambda_plus(eigenvalues, eigenvectors, n_dim, epsilon=0.1):
        """broken"""
        abs_vals = np.abs(eigenvalues)
        
        is_propagating = np.isclose(abs_vals, 1.0)
        is_evanescent = (abs_vals < 1.0) & (abs_vals > epsilon)
        
        selected_indices = np.where(is_propagating | is_evanescent)[0]
        
        if len(selected_indices) == 0:
            return np.array([], dtype=complex), np.array([],dtype=complex)
            
        filtered_eigenvalues = eigenvalues[selected_indices]
        filtered_eigenvectors = eigenvectors[:, selected_indices]

        Lambda_plus = np.diag(filtered_eigenvalues)
        U_plus = filtered_eigenvectors[:n_dim, :]

        return U_plus, Lambda_plus
    
    def mod_eigen_self_energy(self, E, ky, side = "left"):
        """broken"""
        dagger = lambda A: np.conjugate(A.T)
        self.set_inputs(E, ky)
        XIs, XI, PI, h12 = self.decomposition_algorithm(side)
        XIs = spa.csc_matrix(XIs)
        XI = spa.csc_matrix(XI)
        PI = spa.csc_matrix(PI)
        I = np.eye(XI.shape[0], dtype=XI)
        Z = I * 0
        D = E * I - XI
        T = -PI
        A = bmat([
            [Z, I],
            [-T.conj().T, -D]
        ], format='csc')

        B = bmat([
            [I, Z],
            [Z, T]
        ], format='csc')

        eigenvalues, eigenvectors = eigs(A, M=B, sigma=1.0, which='LM')

        U_plus, Lambda = LeadSelfEnergy.construct_U_plus_and_Lambda_plus(eigenvalues, eigenvectors, T.shape[0], epsilon=0.1)
        print(U_plus.shape)
        U_pseudo = np.linalg.pinv(U_plus)
        F = U_plus @ Lambda @ U_pseudo

        Y = np.linalg.solve(E * I - XIs.toarray() - PI.toarray() @ F, dagger(h12.toarray()))
        self_energy = h12 @ Y
        
        return self_energy
                
    def sancho_rubio_surface_gf(self, Energy, H00, H10, tol=1e-6): 
        """ 
        This iteratively calculates the surface green's function for the lead based. 
        Although it is tested for 1D, it should be good for 2D surfaces. 
        """

        Energy = Energy
        dagger = lambda A: np.conjugate(A.T)
        
        I = np.eye(H00.shape[0], dtype=complex)
 
        H01 = dagger(H10)

        epsilon_s = H00.copy()
        epsilon = H00.copy()
        alpha = H01.copy()
        beta = dagger(H10).copy()
        err = 1.0

        while err > tol:
            inv_E = np.linalg.solve(Energy * I - epsilon, I)
            epsilon_s_new = epsilon_s + alpha @ inv_E @ beta
            epsilon_new = epsilon + beta @ inv_E @ alpha + alpha @ inv_E @ beta
            alpha_new = alpha @ inv_E @ alpha
            beta_new = beta @ inv_E @ beta

            err = np.linalg.norm(alpha_new, ord='fro')

            epsilon_s, epsilon, alpha, beta = epsilon_s_new, epsilon_new, alpha_new, beta_new

        return  np.linalg.solve(Energy * I - epsilon_s, I)