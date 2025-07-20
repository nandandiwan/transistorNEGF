import numpy as np
import scipy.sparse as sp

class Hamiltonian:
    """
    Constructs tight-binding Hamiltonians for various device structures.
    """
    def __init__(self, name):
        self.name = name
        self.t = 1   # Hopping energy
        self.o = 0   # Base on-site energy
        self.Vs = 0  # Source voltage
        self.Vd = 0  # Drain voltage
        self.Vg = 0  # Gate voltage applied to the device region

        self.N = 10
        # QPC dimensions
        self.W = 5   # Width of the QPC
        self.L = 10  # Length of the QPC
        
        # Physical constants
        self.kbT_eV = 8.617333e-5 # Boltzmann constant in eV/K

    def one_d_wire(self, blocks=True):
        """Return blocks or full matrix for 1D wire."""
        t, o, N = self.t, self.o, self.N
        if blocks:
            return ([sp.eye(1) * o] * N), ([sp.eye(1) * t] * (N - 1))
        else:
            A = np.zeros((N, N))
            for i in range(N):
                if i < N - 1:
                    A[i, i + 1] = t
                    A[i + 1, i] = t
                A[i, i] = o
            return sp.csc_matrix(A) 
    def create_1d_hamiltonian(self, t, o, N, blocks=True):
        """Return blocks or full matrix for 1D wire."""
        if blocks:
            return ([sp.eye(1) * o] * N), ([sp.eye(1) * t] * (N - 1))
        else:
            A = np.zeros((N, N))
            for i in range(N):
                if i < N - 1:
                    A[i, i + 1] = t
                    A[i + 1, i] = t
                A[i, i] = o
            return sp.csc_matrix(A)
    def quantum_point_contact(self, blocks=True):
        """
        Return blocks or full matrix for a quantum point contact.
        Models a saddle-point potential using a transverse Gaussian confinement
        to create a constriction, which is the correct physical model for a QPC.

        Args:
            Vg (float): The gate voltage, which controls the height of the confining barrier.
            blocks (bool): If True, returns Hamiltonian as onsite and hopping blocks for RGF.
                           If False, returns the full Hamiltonian matrix for direct inversion.
        """
        W, L, t, o = self.W, self.L, self.t, self.o
        N = W * L

        # --- Define the QPC potential geometry ---
        # Center of the channel in the transverse direction
        y_center = (W - 1) / 2.0
        
        sigma_y = W / 4.0 
        
        V_barrier = self.Vg

        U = np.zeros((L, W))
        for l in range(L):     
            for w in range(W): 
                y_coord = w
                U[l, w] = V_barrier * np.exp(-(y_coord - y_center)**2 / (2 * sigma_y**2))

        total_onsite_potential = o + U
        if blocks:
            onsite_blocks = []
            
            for l in range(L):
                h_ii = np.zeros((W, W), dtype=float)
                for w in range(W):

                    h_ii[w, w] = total_onsite_potential[l, w]
                    if w < W - 1:
                        h_ii[w, w + 1] = t
                        h_ii[w + 1, w] = t
                onsite_blocks.append(sp.csc_matrix(h_ii))

            hopping_blocks = [sp.eye(W, k=1)*t for i in range(L)]
            
            return onsite_blocks, hopping_blocks
        else:
            # --- Full matrix representation for direct inversion ---
            A = np.zeros((N, N), dtype=float)
            for l in range(L):
                for w in range(W):
                    idx =(int) (l * W + w)
                    # On-site energy from the saddle-point potential
                    
                    A[int(idx), int(idx)] = total_onsite_potential[l, w]
                    
                    # Hopping in width (transverse)
                    if w < W - 1:
                        A[idx, idx + 1] = t
                        A[idx + 1, idx] = t
                        
     
                        
            return sp.csc_matrix(A)

    def create_hamiltonian(self, blocks=True):
        """
        General interface to get the Hamiltonian for the specified device type.
        """
        if self.name ==  "one_d_wire":
            return self.one_d_wire(blocks=blocks)
        if self.name == "quantum_point_contact" or self.name == "qpc":
            return self.quantum_point_contact(blocks=blocks)
        else:
            # You can add other device types like "one_d_wire" here.
            raise ValueError(f"Unknown device type: {self.name}")

    def get_H00_H01_H10(self):
        """
        Get the principal layer Hamiltonian (H00) and coupling matrices (H01, H10)
        for the semi-infinite leads.
        """
        if self.name == "one_d_wire" or self.name == "chain":

            H00 = sp.eye(1) * self.o
            H01 = sp.eye(1) * self.t
            H10 = sp.eye(1) * self.t

            return H00, H01, H10 
        if self.name == "quantum_point_contact" or self.name == "qpc":

            H00 = self.create_1d_hamiltonian(self.t, self.o, self.W, blocks=False)
            
            # The coupling between principal layers in the lead.
            H01 = sp.eye(self.W, format='csc',k=1) * self.t
            H10 = H01.T # Assuming real hopping t
            
            return H00, H01, H10
        else:
            raise ValueError(f"Lead definition not found for device: {self.name}")
