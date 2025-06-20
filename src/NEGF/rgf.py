from device import Device
import hamiltonian
import numpy as np
import numba
from lead_self_energy import LeadSelfEnergy
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csc_matrix, csr_matrix
from helper import Helper_functions
from time import time
from scipy.sparse.linalg import spsolve
from hamiltonian import Hamiltonian


class GreensFunction:
    def __init__(self, device_state : Device, ham : Hamiltonian):
        self.ds = device_state
        self.ham = ham
        self.eta = 1e-12j
        self.lead_self_energy = LeadSelfEnergy(device_state, ham)
        

    def self_energy(self, E, ky, tol=1e-6):
        """
        This method creates the self energy terms based on the coupling and surface green's function 
        """
        dagger = lambda A: A.conj().T
        
        # Coupling matrices
        H00, H01 = self.ham.get_H00_H01(ky, sparse=True)
        H10 = dagger(H01)

        # Surface Green's functions at left and right leads
        G_surf_left = self.surface_gf(E - self.ds.Vs, H00, H10, tol) # units are in ev
        G_surf_right = self.surface_gf(E - self.ds.Vd, H00, H10, tol)

        # Self-energy calculation (Σ = τ g τ†)
        sigma_left = H01 @ G_surf_left @ H10
        sigma_right = H10 @ G_surf_right @ H01

        return sigma_left, sigma_right
    def surface_gf(self, Energy, H00, H10, tol=1e-6): 
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
        first_time = True

        while err > tol:
            if first_time:
                inv_E = Helper_functions.sparse_inverse(csr_matrix(Energy * I) - csr_matrix(epsilon))
                first_time = False
            else:

                inv_E = np.linalg.solve(Energy * I - epsilon, I)
        
            epsilon_s_new = epsilon_s + alpha @ inv_E @ beta
            epsilon_new = epsilon + beta @ inv_E @ alpha + alpha @ inv_E @ beta
            alpha_new = alpha @ inv_E @ alpha
            beta_new = beta @ inv_E @ beta

            err = np.linalg.norm(alpha_new, ord='fro')

            epsilon_s, epsilon, alpha, beta = epsilon_s_new, epsilon_new, alpha_new, beta_new

        return  np.linalg.solve(Energy * I - epsilon_s, I)
    def fermi(x):
        return 1 / (1 + np.exp(x))
    
    
    def rgf(self, E,ky : float, self_energy_tol = 1e-6): 
        """
        This recursively calcuates the green's function (retarded and lesser) as well
        as matrix elements 1 off from diagonal. Output is a 1d array corresponding to diagonal 
        elements of matrix. 
        """
        E = E + self.eta
        ds = self.ds
        H = ds.hamiltonian.create_tight_binding(ky)
        dagger = lambda A: np.conjugate(A.T)
        f_s = GreensFunction.fermi(-ds.q * (E - ds.Vs) / (ds.kbT))
        f_d = GreensFunction.fermi(-ds.q * (E - ds.Vd) / (ds.kbT))
        fermi = lambda x,y: 1 / (1 + np.exp((x-y) / (ds.kbT  /ds.q)))
        
        sigmaL,sigmaR = self.self_energy(E,ky,self_energy_tol)
        self_energy_right = np.zeros_like(H, dtype=complex)
        self_energy_left = np.zeros_like(H, dtype=complex)
        
        self_energy_size = sigmaR.shape[0]
        self_energy_right[-self_energy_size:,-self_energy_size:] = sigmaR
        self_energy_left[0:self_energy_size,0:self_energy_size] = sigmaL
        
        
        gamma1 = 1j * (self_energy_left - dagger(self_energy_left))
        gamma2 = 1j * (self_energy_right - dagger(self_energy_right))
        self_energy_lesser = gamma1 * f_s +  gamma2 * f_d
        
        sigma_less_left = gamma1 * f_s
        sigma_less_right = gamma2 * f_d
        block_size = sigmaL.shape[0]
        gamma1 = 1j * (self_energy_left  - dagger(self_energy_left))
        gamma2 = 1j * (self_energy_right - dagger(self_energy_right))
        self_energy_lesser = gamma1 * f_s + gamma2 * f_d
        N = H.shape[0]
        num_blocks  = N // block_size
        E_matrix    = np.eye(N, dtype=complex) * E
        A           = E_matrix - H - self_energy_left - self_energy_right
        I_blk       = np.eye(block_size, dtype=complex)

        # 1) contiguous storage instead of Python lists
        g_R_blocks       = np.empty((num_blocks, block_size, block_size), dtype=complex)
        g_lesser_blocks  = np.empty_like(g_R_blocks)

        G_R      = [None] * num_blocks
        G_R_1    = [None] * (num_blocks - 1)
        G_lesser = [None] * num_blocks
        G_lesser_1 = [None] * (num_blocks - 1)

        for i in range(num_blocks):
            start, end   = i * block_size, (i + 1) * block_size
            prev         = (i - 1) * block_size

            if i == 0:                                   # first block
                g_0_r = np.linalg.solve(A[start:end, start:end], I_blk)
                g_R_blocks[0] = g_0_r
                g_lesser_blocks[0] = g_0_r @ self_energy_lesser[start:end, start:end] @ dagger(g_0_r)
            else:
                H_eff = (
                    A[start:end, start:end]
                    - A[start:end, prev:start]
                    @ g_R_blocks[i - 1]
                    @ A[prev:start, start:end]
                )
                g_i_r = np.linalg.solve(H_eff, I_blk)
                g_R_blocks[i] = g_i_r

                sigma_lesser = (
                    A[start:end, prev:start]
                    @ g_lesser_blocks[i - 1]
                    @ dagger(A[prev:start, start:end])
                )
                g_i_lesser = g_i_r @ (
                    self_energy_lesser[start:end, start:end]
                    + sigma_lesser
                    - self_energy_lesser[start:end, prev:start]
                    @ dagger(g_R_blocks[i - 1])
                    @ dagger(A[prev:start, start:end])
                    - A[start:end, prev:start]
                    @ g_R_blocks[i - 1]
                    @ self_energy_lesser[prev:start, start:end]
                ) @ dagger(g_i_r)

                g_lesser_blocks[i] = g_i_lesser

        G_R[-1]      = g_R_blocks[-1]
        G_lesser[-1] = g_lesser_blocks[-1]


        for i in reversed(range(num_blocks - 1)):
            start, end   = i * block_size, (i + 1) * block_size
            after        = (i + 2) * block_size

            # retarded
            G_R[i] = g_R_blocks[i] @ (
                I_blk
                + A[start:end, end:after] @ G_R[i + 1] @ A[after - block_size:after, start:end] @ g_R_blocks[i]
            )
            G_R_1[i] = -G_R[i + 1] @ A[after - block_size:after, start:end] @ g_R_blocks[i]

            gr0 = g_R_blocks[i]            
            ga0 = dagger(gr0)
            gr1 = g_R_blocks[i + 1]       
            ga1 = dagger(gr1)

            gqq1 = gr0 @ self_energy_lesser[start:end, end:after]   @ ga1
            gq1q = gr1 @ self_energy_lesser[end:after, start:end]   @ ga0

            G_i_lesser = (
                g_lesser_blocks[i]
                + g_R_blocks[i]
                @ (A[start:end, end:after] @ G_lesser[i + 1] @ dagger(A[end:after, start:end]))
                @ dagger(g_R_blocks[i])
                - (g_lesser_blocks[i] @ A[end:after, start:end] @ dagger(G_R_1[i].T)
                + G_R_1[i].T @ A[end:after, start:end] @ g_lesser_blocks[i])
                - (gqq1 @ dagger(A[end:after, start:end]) @ dagger(G_R[i])
                + G_R[i] @ A[start:end, end:after]     @ gq1q)
            )
            G_lesser[i] = G_i_lesser

            G_i_lesser_1 = (
                gq1q
                - G_R_1[i] @ A[start:end, end:after] @ gq1q
                - G_R[i + 1] @ A[end:after, start:end] @ g_lesser_blocks[i]
                - G_lesser[i + 1] @ dagger(A[end:after, start:end]) @ dagger(g_R_blocks[i])
            )
            G_lesser_1[i] = G_i_lesser_1[0]

        G_R_diag     = np.concatenate([np.diag(b) for b in G_R], dtype=complex)
        G_lesser_diag = np.concatenate([np.diag(b) for b in G_lesser], dtype=complex)
        
        return G_R_diag, G_lesser_diag, gamma1, gamma2, sigma_less_left, sigma_less_right
    
    
    def sparse_rgf(self, E, ky : float, self_energy_tol = 1e-6): 
        """
        This recursively calcuates the green's function (retarded and lesser) as well
        as matrix elements 1 off from diagonal. Output is a 1d array corresponding to diagonal 
        elements of matrix. 
        """
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import spsolve
        E = E + self.eta
        ds = self.ds
        dagger = lambda A: np.conjugate(A.T)
        # Get sparse channel Hamiltonian blocks (diagonal and off‑diagonal)
        diagonal_blocks, off_diagonal_blocks = self.ham.create_sparse_channel_hamlitonian(ky)
        print("finished hamiltonian construction")
        num_blocks = len(diagonal_blocks)
        
        # Compute lead self energies (dense)
        sigmaL, sigmaR = self.self_energy(E, ky, tol=self_energy_tol)
        print("finished self energy")
        block_size = sigmaL.shape[0]
        
        f_s = GreensFunction.fermi(-ds.q * (E - ds.Vs) / (ds.kbT))
        f_d = GreensFunction.fermi(-ds.q * (E - ds.Vd) / (ds.kbT))
        
        gamma1 = 1j * (sigmaL - dagger(sigmaL))
        gamma2 = 1j * (sigmaR - dagger(sigmaR))
        
        # Build self-energy lesser per block: only first and last blocks receive lead contributions
        self_energy_lesser_blocks = [np.zeros((block_size, block_size), dtype=complex) for _ in range(num_blocks)]
        self_energy_lesser_blocks[0] = gamma1 * f_s
        self_energy_lesser_blocks[-1] = gamma2 * f_d
        
        # Build effective onsite matrices per block: A_i = E·I - H_dense (subtract lead self energies on boundaries)
        A_blocks = []
        for i in range(num_blocks):
            # Convert sparse diagonal block to dense
            H_dense = diagonal_blocks[i].toarray()
            A_i = E * np.eye(H_dense.shape[0], dtype=complex) - H_dense
            if i == 0:
                A_i -= sigmaL
            if i == num_blocks - 1:
                A_i -= sigmaR
            A_blocks.append(A_i)
        
        # Forward propagation: compute g_R and g_lesser for each block
        g_R_blocks = []
        g_lesser_blocks = []
        I_blk = np.eye(block_size, dtype=complex)
        # Block 0 - use spsolve for inversion column‐by‐column
        g_r = np.linalg.solve(A_blocks[0], I_blk)
        g_R_blocks.append(g_r)
        g_lesser = g_r @ self_energy_lesser_blocks[0] @ dagger(g_r)
        g_lesser_blocks.append(g_lesser)
        
        # For blocks 1 ... (num_blocks-1)
        for i in range(1, num_blocks):
            B = off_diagonal_blocks[i-1].toarray()  # coupling from block (i-1) to i
            A_eff = A_blocks[i] - B @ g_R_blocks[i-1] @ dagger(B)
            # Use spsolve for inversion
            if i == num_blocks - 1:
                g_r = np.column_stack([spsolve(csc_matrix(A_eff), I_blk[:, j]) for j in range(block_size)])
            else:
                g_r = np.linalg.solve(A_eff, I_blk)
                
            g_R_blocks.append(g_r)
            sigma_less = self_energy_lesser_blocks[i] + B @ g_lesser_blocks[i-1] @ dagger(B)
            g_lesser = g_r @ sigma_less @ dagger(g_r)
            g_lesser_blocks.append(g_lesser)
        print("finished forward")
        # Backward propagation: incorporate non-local corrections (reversed loop kept as before)
        G_R = [None] * num_blocks
        
        
        G_R_1 = [None] * (num_blocks - 1)
        G_lesser = [None] * num_blocks
        G_lesser_1 = [None] * (num_blocks - 1)
        G_R[-1] = g_R_blocks[-1]
        G_lesser[-1] = g_lesser_blocks[-1]
        
        for i in reversed(range(num_blocks - 1)):

            G_R[i] = g_R_blocks[i] @ (
                I_blk
                + A_blocks[i][0:block_size, 0:block_size] @ G_R[i + 1] @ A_blocks[i][0:block_size, 0:block_size] @ g_R_blocks[i]
            )
            G_R_1[i] = -G_R[i + 1] @ A_blocks[i][0:block_size, 0:block_size] @ g_R_blocks[i]

            gr0 = g_R_blocks[i]            
            ga0 = dagger(gr0)
            gr1 = g_R_blocks[i + 1]       
            ga1 = dagger(gr1)

            # For simplicity, using the onsite self-energy terms from forward propagation
            gqq1 = gr0 @ self_energy_lesser_blocks[i] @ ga1
            gq1q = gr1 @ self_energy_lesser_blocks[i] @ ga0

            G_i_lesser = (
                g_lesser_blocks[i]
                + g_R_blocks[i]
                @ (A_blocks[i][0:block_size, 0:block_size] @ G_lesser[i + 1] @ dagger(A_blocks[i][0:block_size, 0:block_size]))
                @ dagger(g_R_blocks[i])
                - (g_lesser_blocks[i] @ A_blocks[i][0:block_size, 0:block_size] @ dagger(G_R_1[i].T)
                + G_R_1[i].T @ A_blocks[i][0:block_size, 0:block_size] @ g_lesser_blocks[i])
                - (gqq1 @ dagger(A_blocks[i][0:block_size, 0:block_size]) @ dagger(G_R[i])
                + G_R[i] @ A_blocks[i][0:block_size, 0:block_size]     @ gq1q)
            )
            G_lesser[i] = G_i_lesser

            G_i_lesser_1 = (
                gq1q
                - G_R_1[i] @ A_blocks[i][0:block_size, 0:block_size] @ gq1q
                - G_R[i + 1] @ A_blocks[i][0:block_size, 0:block_size] @ g_lesser_blocks[i]
                - G_lesser[i + 1] @ dagger(A_blocks[i][0:block_size, 0:block_size]) @ dagger(g_R_blocks[i])
            )
            G_lesser_1[i] = G_i_lesser_1[0]
        
        G_R_diag     = np.concatenate([np.diag(block) for block in G_R])
        G_lesser_diag = np.concatenate([np.diag(block) for block in G_lesser])
        print("finished backward")
        return G_R_diag, G_lesser_diag, gamma1, gamma2, sigmaL, sigmaR
    
    def sparse_rgf_G_R(self, E, ky : float, self_energy_tol=1e-4, self_energy_iterative=True):
        
        """
        This recursively calcuates the green's function (retarded and lesser) as well
        as matrix elements 1 off from diagonal. Output is a 1d array corresponding to diagonal 
        elements of matrix. 
        """
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import spsolve
        E = E + self.eta
        ds = self.ds
        dagger = lambda A: np.conjugate(A.T)
        # Get sparse channel Hamiltonian blocks (diagonal and off‑diagonal)
        hamiltonian_start = time()
        diagonal_blocks, off_diagonal_blocks = self.ham.create_sparse_channel_hamlitonian(ky)
        
        hamiltonian_end = time()
        num_blocks = len(diagonal_blocks)
        
        # Compute lead self energies (dense)
        #sigmaL, sigmaR = self.self_energy(E, ky, tol=self_energy_tol)
        
        self_energy_start = time() 
        if self_energy_iterative == True:
            sigmaL, sigmaR = self.lead_self_energy.iterative_self_energy(E, ky, side="left"), self.lead_self_energy.iterative_self_energy(E, ky, side="right")
        else:
            raise Exception("this has not been implemented")
        self_energy_end = time()
        block_size = sigmaL.shape[0]
        gamma1 = 1j * (sigmaL - dagger(sigmaL))
        gamma2 = 1j * (sigmaR - dagger(sigmaR))
        
        I = csc_matrix(np.eye(sigmaL.shape[0]), dtype = complex)
        

        # Build effective onsite matrices per block: A_i = E·I - H_dense (subtract lead self energies on boundaries)
        forward_start = time()
        A_blocks = []
        for i in range(num_blocks):
            A_i = E * I - diagonal_blocks[i]
            if i == 0:
                A_i -= sigmaL
            if i == num_blocks - 1:
                A_i -= sigmaR
            A_blocks.append(A_i)
        
        # Forward propagation: compute g_R and g_lesser for each block
        g_R_blocks = []

        #I_blk = np.eye(block_size, dtype=complex)
        # Block 0 - use spsolve for inversion column‐by‐column
        g_r = spsolve(A_blocks[0], I)
        g_R_blocks.append(g_r)
        
        # For blocks 1 ... (num_blocks-1)
        for i in range(1, num_blocks):
            B = off_diagonal_blocks[i-1] 
            A_eff = A_blocks[i] - B @ g_R_blocks[i-1] @ dagger(B)
            # Use spsolve for inversion
            g_r = spsolve(A_eff, I)
                
            g_R_blocks.append(g_r)

        forward_end = time()
        # Backward propagation: incorporate non-local corrections (reversed loop kept as before)
        backward_start = time()
        G_R = [None] * num_blocks

        G_R[-1] = g_R_blocks[-1]

        
        for i in reversed(range(num_blocks - 1)):

            G_R[i] = g_R_blocks[i] @ (
                I
                + A_blocks[i][0:block_size, 0:block_size] @ G_R[i + 1] @ A_blocks[i][0:block_size, 0:block_size] @ g_R_blocks[i]
            )
        backward_end = time()
        
        
        print(f"Hamiltonian construction: {hamiltonian_end - hamiltonian_start} \n \
              Self Energy construction: {self_energy_end - self_energy_start} \n \
                  Forward iteration: {forward_end - forward_start} \n \
                      Backward iteration: {backward_end - backward_start}")
        return G_R, gamma1, gamma2, sigmaL, sigmaR
    
    
        