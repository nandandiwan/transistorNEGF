

from device import Device
import scipy.sparse as spa 
import scipy as sp
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
    
    
    def iterative_self_energy(self, E, ky, side = "left"):
        """
        Implements Algorithm I (Iterative method) from the paper:
        "Methods for fast evaluation of self-energy matrices in tight-binding modeling"
        
        This follows the decimation algorithm applied to the condensed Hamiltonian.
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
        
        # Following Algorithm I from the paper:
        # Let E* = E + iη, A = (E*I - Ξ), As = (E*I - Ξs)
        A = E_complex * I - XI
        As = E_complex * I - XIs
        delta = 1e-5
        
        # Initialize Π† (PI_dagger) as the conjugate transpose of Π
        PI_dagger = PI.conj().T
        
        # Iterative loop following Algorithm I
        # While max(|Π|, |Π†|) > δ, do:
        iteration_count = 0
        max_iterations = 100  # Prevent infinite loops
        
        while (np.max(np.abs(PI)) > delta or np.max(np.abs(PI_dagger)) > delta) and iteration_count < max_iterations:
            try:
                # Step 3: Solve AX_Π = Π for X_Π 
                X_PI = np.linalg.solve(A, PI)
                # Step 4: Solve AX_Π† = Π† for X_Π†
                X_PI_dagger = np.linalg.solve(A, PI_dagger)
                
            except np.linalg.LinAlgError:
                print(f"Error: Matrix A is singular at iteration {iteration_count}. Iteration cannot continue.")
                return np.full_like(A, np.nan)
            
            # Step 5: Update A = A - ΠX_Π† - Π†X_Π
            
            A = A - PI @ X_PI_dagger - PI_dagger @ X_PI
        
        # Step 6: Update As = As - ΠX_Π†
            As = As - PI @ X_PI_dagger
            
            # Step 7: Update Π = ΠX_Π
            PI = PI @ X_PI
            PI_dagger = PI_dagger @ X_PI_dagger
                
                # Step 8: Update Π† = Π†X_Π†
  
            
            iteration_count += 1

        if iteration_count >= max_iterations:
            print(f"Warning: Iteration did not converge after {max_iterations} steps")
        
        # Step 9: Solve AsY = H₁₂† for Y (final step)
        h12_dagger = h12.conj().T
        
        try:
            Y = np.linalg.solve(As, h12_dagger)
        except np.linalg.LinAlgError:
            print("Error: Matrix As is singular for the final solve step.")
            return np.full_like(As, np.nan)
        
        # Step 10: Obtain the self energy Σ = H₁₂Y
        Sigma = h12 @ Y

        return Sigma
    
    def get_layer_hamiltonian(self, p, side="left") -> tuple:
        """
        Get the layer Hamiltonian blocks H_pp and H_p,p+1 for layer p.
        
        Args:
            p: Layer index (1, 2, ..., self.P)
            side: "left" or "right" lead
            
        Returns:
            tuple: (H_pp, H_p_p1) where H_pp is the on-site Hamiltonian for layer p
                   and H_p_p1 is the coupling to layer p+1
        """
        if not self.ky in self.layerHamiltonianCache:
            self.layerHamiltonianCache[self.ky] = self.ham.getLayersHamiltonian(self.ky)
        layerHamiltonians = self.layerHamiltonianCache[self.ky]
        
        if p > self.P or p < 1:
            print(f"Error: p={p}, self.P={self.P}")
            raise ValueError("layers are indexed 1,2,...self.P")
        
        # Map from 1-based indexing to 0-based indexing used in the dictionary
        if side == "left":
            # For left lead, reverse the order: layer 1 -> index 3, layer 2 -> index 2, etc.
            layer_index = 3 - (p - 1) 
        elif side == "right":
            # For right lead, use cyclical mapping based on the right lead starting layer
            layer_index = (self.ham.layer_right_lead + p - 1) % 4
        else:
            raise ValueError("side must be 'left' or 'right'") 
        
        # Ensure the layer index is valid
        if layer_index not in layerHamiltonians:
            raise ValueError(f"Layer index {layer_index} not found in layer Hamiltonians")
            
        H_pp, H_p_p1 = layerHamiltonians[layer_index]
        
        # Convert sparse matrices to CSC format for consistency
        if hasattr(H_pp, 'tocsc'):
            H_pp = H_pp.tocsc()
        if H_p_p1 is not None and hasattr(H_p_p1, 'tocsc'):
            H_p_p1 = H_p_p1.tocsc()
            
        return H_pp, H_p_p1   
        
    
    def decomposition_algorithm(self, side="left"):
        """
        Implements Algorithm 0 (Recursive Condensation of the Hamiltonian Matrix)
        from the paper to compute the condensed Hamiltonian parameters.
        
        Returns:
            tuple: (Ξs, Ξ, Π, H12) where:
                   Ξs = condensed on-site Hamiltonian for layer 1
                   Ξ = condensed on-site Hamiltonian for layers p=nP+1 (n=1,2,...)
                   Π = condensed coupling Hamiltonian 
                   H12 = coupling from layer 1 to layer 2
        """
        dagger = lambda A: np.conjugate(A.T)
        
        # Initialize matrices for recursive calculation
        H_tilde_matrices = [None] * self.P  # Ĥ_p,p matrices
        H_cross_matrices = [None] * self.P  # Ĥ_p,P matrices
        
        # Step 1: H̃_P,P = (EI_P,P - H_P,P)^(-1)
        h_PP, h_P_P1 = self.get_layer_hamiltonian(self.P, side)
        
        # Convert to dense for easier manipulation
        if hasattr(h_PP, 'toarray'):
            h_PP_dense = h_PP.toarray()
        else:
            h_PP_dense = h_PP
            
        E_matrix = self.E * np.eye(h_PP_dense.shape[0], dtype=complex)
        H_tilde_PP = np.linalg.inv(E_matrix - h_PP_dense)
        H_tilde_matrices[self.P - 1] = H_tilde_PP
        H_cross_matrices[self.P - 1] = H_tilde_PP
        
        # Step 2: For p = P-1, P-2, ..., 2 (in this order), do:
        for p in range(self.P - 1, 1, -1):  # p goes from P-1 down to 2
            h_pp, h_p_p1 = self.get_layer_hamiltonian(p, side)
            
            if h_p_p1 is None:
                raise ValueError(f"Missing coupling H_{p},{p+1} for layer {p}")
            
            # Convert to dense
            if hasattr(h_pp, 'toarray'):
                h_pp_dense = h_pp.toarray()
            else:
                h_pp_dense = h_pp
                
            if hasattr(h_p_p1, 'toarray'):
                h_p_p1_dense = h_p_p1.toarray()
            else:
                h_p_p1_dense = h_p_p1
            
            # Step 3: H̃_p,p = (EI_p,p - H_p,p - H_p,p+1 * H̃_p+1,p+1 * H†_p,p+1)^(-1)
            E_matrix_p = self.E * np.eye(h_pp_dense.shape[0], dtype=complex)
            coupling_term = h_p_p1_dense @ H_tilde_matrices[p] @ dagger(h_p_p1_dense)
            H_tilde_pp = np.linalg.inv(E_matrix_p - h_pp_dense - coupling_term)
            H_tilde_matrices[p - 1] = H_tilde_pp
            
            # Step 4: H̃_p,P = H̃_p,p * H_p,p+1 * H̃_p+1,P
            H_cross_pp = H_tilde_matrices[p - 1] @ h_p_p1_dense @ H_cross_matrices[p]
            H_cross_matrices[p - 1] = H_cross_pp
        
        # Step 5: C̃_2,2 = H̃_2,2, C̃_2,P = H̃_2,P
        C_tilde_22 = H_tilde_matrices[1]  # H̃_2,2
        C_tilde_2P = H_cross_matrices[1]  # H̃_2,P
        
        # Initialize C matrices for step 6
        C_tilde_matrices = [None] * (self.P + 1)
        C_tilde_matrices[2] = C_tilde_22
        
        # Step 6: For p = 3, ..., P (in this order), do:
        for p in range(3, self.P + 1):
            # Step 7: C̃_p,p = H̃_p,p + H̃_p,p * (H_p,p-1 * C̃_p-1,p-1 * H†_p,p-1) * H̃_p,p
            _, h_p_minus_1_p = self.get_layer_hamiltonian(p - 1, side)
            
            if h_p_minus_1_p is None:
                raise ValueError(f"Missing coupling H_{p-1},{p} for layer {p-1}")
            
            # Convert to dense
            if hasattr(h_p_minus_1_p, 'toarray'):
                h_p_minus_1_p_dense = h_p_minus_1_p.toarray()
            else:
                h_p_minus_1_p_dense = h_p_minus_1_p
                
            h_p_p_minus_1 = dagger(h_p_minus_1_p_dense)
            
            H_tilde_pp = H_tilde_matrices[p - 1]
            C_tilde_prev = C_tilde_matrices[p - 1]
            
            inner_term = h_p_p_minus_1 @ C_tilde_prev @ dagger(h_p_p_minus_1)
            C_tilde_matrices[p] = H_tilde_pp + H_tilde_pp @ inner_term @ H_tilde_pp
        
        # Step 8: Obtain Ξ_s = H_1,1 + H_1,2 * C̃_2,2 * H†_1,2
        h11, h12 = self.get_layer_hamiltonian(1, side)
        if h12 is None:
            raise ValueError("Missing coupling H_1,2")
        
        # Convert to dense
        if hasattr(h11, 'toarray'):
            h11_dense = h11.toarray()
        else:
            h11_dense = h11
            
        if hasattr(h12, 'toarray'):
            h12_dense = h12.toarray()
        else:
            h12_dense = h12
            
        XI_s = h11_dense + h12_dense @ C_tilde_matrices[2] @ dagger(h12_dense)
        
        # Step 9: Obtain Ξ = Ξ_s + H†_P,P+1 * C̃_P,P * H_P,P+1
        _, h_P_P1 = self.get_layer_hamiltonian(self.P, side)
        if h_P_P1 is None:
            # For the boundary layer, there might be no coupling to the next unit cell
            # In this case, Ξ = Ξ_s
            XI = XI_s
        else:
            if hasattr(h_P_P1, 'toarray'):
                h_P_P1_dense = h_P_P1.toarray()
            else:
                h_P_P1_dense = h_P_P1
                
            C_tilde_PP = C_tilde_matrices[self.P]
            XI = XI_s + dagger(h_P_P1_dense) @ C_tilde_PP @ h_P_P1_dense
        
        # Step 10: Obtain Π = H_1,2 * C̃_2,P * H_P,P+1
        if h_P_P1 is None:
            # If there's no coupling from P to P+1, set Π to zero
            PI = np.zeros((h12_dense.shape[0], h12_dense.shape[1]), dtype=complex)
        else:
            PI = h12_dense @ C_tilde_2P @ h_P_P1_dense
        
        # Return dense arrays
        return XI_s, XI, PI, h12_dense

    def construct_U_plus_and_Lambda_plus(eigenvalues, eigenvectors, n_dim, epsilon=0.1):
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
    
    def get_self_energy(self, E, ky, side = "left"):
       
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
                
        