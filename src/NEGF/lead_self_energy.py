import hamiltonian
from device import Device
import scipy.sparse as spa 
import scipy as sp
import numpy as np
from helper import Helper_functions
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
        
        self.eta = 1e-8j
    
    def set_inputs(self, E, ky):
        self.E = E
        self.ky = ky
    
    
    def iterative_self_energy(self, E, ky, side = "left"):
       
        dagger = lambda A: np.conjugate(A.T)
        self.set_inputs(E, ky)
        XIs, XI, PI,h12 = self.decomposition_algorithm(side)
        E = self.E + self.eta
        I = np.eye(XI.shape[0], dtype = complex)
        A = E * I - XI
        As = E * I - XIs
        delta = 1e-5

        # Initialize Π† (PIdagger) as the conjugate transpose of Π
        PIdagger = PI.conj().T
        
        while np.max(np.abs(PI)) > delta or np.max(np.abs(PIdagger)) > delta:
        
            try:
                X_PI = np.linalg.solve(A, PI)
                X_PIdagger = np.linalg.solve(A, PIdagger)
    
            except np.linalg.LinAlgError:
                print("Error: Matrix A is singular. Iteration cannot continue.")
                return np.full_like(A, np.nan)
            A = A - PI @ X_PIdagger - PIdagger @ X_PI
            As = As - PI @ X_PIdagger
            PI = PI @ X_PI
            PIdagger = PIdagger @ X_PIdagger


        h12_dagger = h12.conj().T

        try:

            Y = spsolve(csc_matrix(As), h12_dagger)
        except np.linalg.LinAlgError:
            print("Error: Matrix As is singular for the final solve step.")
            return np.full_like(As, np.nan)
        Sigma = h12 @ Y

        return Sigma
    
    def get_layer_hamiltonian(self, p, side = "left") -> spa.csc_matrix:
        if not self.ky in self.layerHamiltonianCache:
            self.layerHamiltonianCache[self.ky] = self.ham.getLayersHamiltonian(self.ky)
        layerHamiltonians = self.layerHamiltonianCache[self.ky]
        
        if p > self.P or p < 1:
            print(p, self.P)
            raise ValueError("layers are indexed 1,2,...self.P")
        
        if side == "left":
            p = 3 - (p-1) 
        elif side == "right":
            p = (self.ham.layer_right_lead + p - 1)  % 4
        else:
            raise ValueError("there is only left and right sides") 
        
        return layerHamiltonians[p]   
        
    
    def decomposition_algorithm(self, side="left"):
        dagger = lambda A: np.conjugate(A.T)
        Hpp_matrices = [None] * self.P
        HpP_matrices = [None] * self.P
        hPP, hPP1 = self.get_layer_hamiltonian(self.P, side)
        HPP = spsolve(spa.csc_matrix(self.E * np.eye(hPP.shape[0], dtype=complex) - hPP, dtype = complex), csc_matrix(np.eye(hPP.shape[0], dtype=complex)))
  
        Hpp_matrices[-1], HpP_matrices[-1] = HPP, HPP
        for i in range(self.P - 1, 0, -1):

            hpp, hpp1 = self.get_layer_hamiltonian(i, side)
            Hpp = spsolve(spa.csc_matrix(self.E * np.eye(hPP.shape[0], dtype = complex) - \
                hpp - hpp1 @ Hpp_matrices[i] @ dagger(hpp1),dtype=complex) \
                    , csc_matrix(np.eye(hPP.shape[0]), dtype=complex))
            Hpp_matrices[i - 1] = Hpp
            
            
            HpP = Hpp_matrices[i - 1] @ hpp1 @ HpP_matrices[i]
            HpP_matrices[i - 1] = HpP
            
        C22_tilde = Hpp_matrices[1]  
        C2P_tilde = HpP_matrices[1]  

        C_matrices = [None] * (self.P + 1)
        C_matrices[2] = C22_tilde


        for p in range(3, self.P + 1):
            _, h_p_minus_1_p = self.get_layer_hamiltonian(p - 1, side)
            h_p_p_minus_1 = dagger(h_p_minus_1_p)

            Hpp_tilde = Hpp_matrices[p - 1]
            C_prev_tilde = C_matrices[p - 1]

            inner_term = h_p_p_minus_1 @ C_prev_tilde @ dagger(h_p_p_minus_1)
            C_matrices[p] = Hpp_tilde + Hpp_tilde @ inner_term @ Hpp_tilde

 
        h11, h12 = self.get_layer_hamiltonian(1, side)
        _  , hP_P1 = self.get_layer_hamiltonian(self.P, side) 

        XIs = h11 + h12 @ C_matrices[2] @ dagger(h12)
        CpP_tilde = C_matrices[self.P]
        XI = XIs + dagger(hP_P1) @ CpP_tilde @ hP_P1
        
        PI = h12 @ C2P_tilde @ hP_P1
        
        return XIs.toarray(), XI.toarray(), PI.toarray(), h12

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
                
        