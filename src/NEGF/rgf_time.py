from device import Device
import hamiltonian
import numpy as np
import numba
from scipy.linalg import lu_factor, lu_solve
import time


class GreensFunction:
    def __init__(self, device_state : Device):
        self.ds = device_state
        self.eta = 1e-12j
    
    def self_energy(self, E, ky):
        """
        This method creates the self energy terms based on the coupling and surface green's function 
        """
        
        t1 = time.time()
        dagger = lambda A: A.conj().T
        
        # Coupling matrices
        H00, H01 = self.ds.hamiltonian.get_H00_H01(ky)
        H10 = dagger(H01)

        # Surface Green's functions at left and right leads
        G_surf_left = self.surface_gf(E - self.ds.Vs, H00, H10) # units are in ev
        G_surf_right = self.surface_gf(E - self.ds.Vd, H00, H10)

        # Self-energy calculation (Σ = τ g τ†)
        sigma_left = H01 @ G_surf_left @ H10
        sigma_right = H10 @ G_surf_right @ H01
        t2 = time.time()
        print(f"self energy method takes: {t2 - t1}")
        return sigma_left, sigma_right
    def surface_gf(self, Energy, H00: np.ndarray, H10: np.ndarray, tol=1e-6): 
        """ 
        This iteratively calculates the surface green's function for the lead based. 
        Although it is tested for 1D, it should be good for 2D surfaces. 
        """
        print("starting SGF")
        ts1 = time.time()
        dagger = lambda A: np.conjugate(A.T)
        I = np.eye(H00.shape[0], dtype=complex)
        H01 = dagger(H10)

        epsilon_s = H00.copy()
        epsilon   = H00.copy()
        alpha     = H01.copy()
        beta      = dagger(H10).copy()
        err = 1.0

        iteration = 0
        total_iter_time = 0.0
        total_inv_time = 0.0
        total_mult_time = 0.0

        while err > tol:
            iter_start = time.time()
            iteration += 1

            t_inv_start = time.time()
            inv_E = np.linalg.inv(Energy * I - epsilon)
            t_inv_end = time.time()
            inv_time = t_inv_end - t_inv_start
            total_inv_time += inv_time

            t_mult_start = time.time()
            epsilon_s_new = epsilon_s + alpha @ inv_E @ beta
            epsilon_new   = epsilon + beta @ inv_E @ alpha + alpha @ inv_E @ beta
            alpha_new     = alpha @ inv_E @ alpha
            beta_new      = beta @ inv_E @ beta
            t_mult_end = time.time()
            mult_time = t_mult_end - t_mult_start
            total_mult_time += mult_time

            err = np.linalg.norm(alpha_new, ord='fro')
            epsilon_s, epsilon, alpha, beta = epsilon_s_new, epsilon_new, alpha_new, beta_new

            iter_end = time.time()
            iter_time = iter_end - iter_start
            total_iter_time += iter_time

            print(f"Iteration {iteration}: error={err:.2e}, inversion={inv_time:.6f}s, multiplication={mult_time:.6f}s, iter total={iter_time:.6f}s")

        final_mat = Energy * I - epsilon_s
        nonzeros = np.count_nonzero(final_mat)
        total_elements = final_mat.size
        sparsity = 1 - nonzeros / total_elements

        ts2 = time.time()
        print(f"Surface GF completed in {ts2 - ts1:.6f}s over {iteration} iterations")
        print(f"Total inversion time: {total_inv_time:.6f}s, total multiplication: {total_mult_time:.6f}s, average iteration time: {total_iter_time/iteration:.6f}s")
        print(f"Final matrix sparsity: {sparsity:.2%} (percentage of zeros)")

        return np.linalg.inv(final_mat)
    def fermi(x):
        return 1 / (1 + np.exp(x))
    def rgf(self, E, ky: float): 
        """
        This recursively calcuates the green's function (retarded and lesser) as well
        as matrix elements 1 off from diagonal. Output is a 1d array corresponding to diagonal 
        elements of matrix. 
        """
        t_total_start = time.time()
        
        E = E + self.eta
        ds = self.ds
        
        t_H_start = time.time()
        H = ds.hamiltonian.create_tight_binding(ky)
        t_H_end = time.time()
        print(f"[rgf] Hamiltonian creation: {t_H_end - t_H_start:.6f} s")
        
        dagger = lambda A: np.conjugate(A.T)
        f_s = GreensFunction.fermi(-ds.q * (E - ds.Vs) / (ds.kbT))
        f_d = GreensFunction.fermi(-ds.q * (E - ds.Vd) / (ds.kbT))
        fermi = lambda x,y: 1 / (1 + np.exp((x-y) / (ds.kbT  /ds.q)))
            
        t_se_start = time.time()
        sigmaL, sigmaR = self.self_energy(E,ky)
        t_se_end = time.time()
        print(f"[rgf] self_energy call: {t_se_end - t_se_start:.6f} s")
            
        self_energy_right = np.zeros_like(H, dtype=complex)
        self_energy_left = np.zeros_like(H, dtype=complex)
        self_energy_size = sigmaR.shape[0]
        self_energy_right[-self_energy_size:,-self_energy_size:] = sigmaR
        self_energy_left[0:self_energy_size,0:self_energy_size] = sigmaL
        
        gamma1 = 1j * (self_energy_left - dagger(self_energy_left))
        gamma2 = 1j * (self_energy_right - dagger(self_energy_right))
        self_energy_lesser = gamma1 * f_s + gamma2 * f_d
        
        sigma_less_left = gamma1 * f_s
        sigma_less_right = gamma2 * f_d
        block_size = 10
        N = H.shape[0]
        num_blocks  = N // block_size
        E_matrix    = np.eye(N, dtype=complex) * E
        A           = E_matrix - H - self_energy_left - self_energy_right
        I_blk       = np.eye(block_size, dtype=complex)

        # 1) contiguous storage instead of Python lists
        g_R_blocks      = np.empty((num_blocks, block_size, block_size), dtype=complex)
        g_lesser_blocks = np.empty_like(g_R_blocks)
        G_R      = [None] * num_blocks
        G_R_1    = [None] * (num_blocks - 1)
        G_lesser = [None] * num_blocks
        G_lesser_1 = [None] * (num_blocks - 1)

        # Forward loop timing
        t_forward_start = time.time()
        for i in range(num_blocks):
            iter_start = time.time()
            start, end = i * block_size, (i + 1) * block_size
            prev = (i - 1) * block_size

            if i == 0:  # first block
                t_inv = time.time()
                g_0_r = np.linalg.inv(A[start:end, start:end])
                t_inv_end = time.time()
                print(f"[rgf forward] Block {i} inversion: {t_inv_end - t_inv:.6f} s")
                
                t_mult = time.time()
                g_R_blocks[0] = g_0_r
                g_lesser_blocks[0] = g_0_r @ self_energy_lesser[start:end, start:end] @ dagger(g_0_r)
                t_mult_end = time.time()
                print(f"[rgf forward] Block {i} multiplication: {t_mult_end - t_mult:.6f} s")
            else:
                t_mult = time.time()
                H_eff = (A[start:end, start:end] 
                    - A[start:end, prev:start] @ g_R_blocks[i - 1] @ A[prev:start, start:end])
                t_mult_end = time.time()
                print(f"[rgf forward] Block {i} H_eff multiplications: {t_mult_end - t_mult:.6f} s")
                
                t_inv = time.time()
                g_i_r = np.linalg.inv(H_eff)
                t_inv_end = time.time()
                print(f"[rgf forward] Block {i} inversion: {t_inv_end - t_inv:.6f} s")
                
                t_mult = time.time()
                g_R_blocks[i] = g_i_r
                sigma_lesser = A[start:end, prev:start] @ g_lesser_blocks[i - 1] @ dagger(A[prev:start, start:end])
                g_i_lesser = g_i_r @ ( self_energy_lesser[start:end, start:end] + sigma_lesser
                    - self_energy_lesser[start:end, prev:start] @ dagger(g_R_blocks[i - 1]) @ dagger(A[prev:start, start:end])
                    - A[start:end, prev:start] @ g_R_blocks[i - 1] @ self_energy_lesser[prev:start, start:end] ) @ dagger(g_i_r)
                g_lesser_blocks[i] = g_i_lesser
                t_mult_end = time.time()
                print(f"[rgf forward] Block {i} additional multiplications: {t_mult_end - t_mult:.6f} s")
            iter_end = time.time()
            print(f"[rgf forward] Block {i} total: {iter_end - iter_start:.6f} s")
        t_forward_end = time.time()
        print(f"[rgf forward] Total forward loop time: {t_forward_end - t_forward_start:.6f} s")
        
        
        G_R[-1] = g_R_blocks[-1]
        G_lesser[-1] = g_lesser_blocks[-1]
        # Backward loop timing
        t_backward_start = time.time()
        for i in reversed(range(num_blocks - 1)):
            iter_start = time.time()
            start, end = i * block_size, (i + 1) * block_size
            after = (i + 2) * block_size

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
            iter_end = time.time()
            print(f"[rgf backward] Block {i} total: {iter_end - iter_start:.6f} s")
        t_backward_end = time.time()
        print(f"[rgf backward] Total backward loop time: {t_backward_end - t_backward_start:.6f} s")
        
        t_final_start = time.time()
        G_R_diag = np.concatenate([np.diag(b) for b in G_R], dtype=complex)
        G_lesser_diag = np.concatenate([np.diag(b) for b in G_lesser], dtype=complex)
        t_final_end = time.time()
        print(f"[rgf final assembly] {t_final_end - t_final_start:.6f} s")
        
        t_total_end = time.time()
        print(f"[rgf total] {t_total_end - t_total_start:.6f} s")
        return G_R_diag, G_lesser_diag, gamma1, gamma2, sigma_less_left, sigma_less_right