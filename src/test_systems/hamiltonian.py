import numpy as np
import scipy.sparse as sp
import scipy.constants as spc
from unit_cell_generation import GrapeheneZigZagCell
class Hamiltonian:
    """
    Constructs tight-binding Hamiltonians for various device structures.
    """
    def __init__(self, name, periodic = False, relevant_parameters = {}):
        if((periodic and name != "zigzag")):
            raise ValueError("periodic not available or possible")
        self.T = 300  # Use the passed temperature parameter
        self.q = spc.e
        self.kbT = spc.Boltzmann * self.T  # Keep in Joules
        self.kbT_eV = spc.Boltzmann * self.T / self.q  # Also store in eV for convenience
        self.name = name
        self.t = 1   # Hopping energy
        self.o = 0.0   # Base on-site energy
        self.Vs = 0.0  # Source voltage
        self.Vd = 0.0  # Drain voltage
        self.Vg = 0  # Gate voltage applied to the device region
        self.num_orbitals = 1
        self.N = 120
       
        self.W = 5   # Width of the QPC
        self.L = 10  # Length of the QPC
        
        # Physical constants
        self.kbT_eV = 8.617333e-5 # Boltzmann constant in eV/K
        
        # for unit cell hamiltonians
        self.unit_cell = None

        
        #zig zag
        self.Nx = 10
        self.Ny = 10
        self.periodic = periodic
        
        self.relevant_parameters = relevant_parameters
        
        #modified oned
        self.hamiltonian_registry = {}
        self.lead_registry = {}
        
        self.potential = None
    
    def get_num_sites(self):
        if (self.name == "one_d_wire" or self.name == "modified_one_d"):
            return self.N
        elif (self.name == "qpc"):
            return self.W * self.L
        else:
            return self.num_orbitals * len(self.unit_cell.ATOM_POSITIONS)
        

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
        
    def modified_one_d_wire(self, blocks=True):
        t, o, N = self.t, self.o, self.N
        if blocks:
            diag, offdiag =  ([sp.eye(1) * o] * N), ([sp.eye(1) * t] * (N - 1))
            diag[N//2] += sp.eye(1) * 5
            #diag[N//2 + 1] += sp.eye(1) * 5
            return diag, offdiag
            
        else:
            A = np.zeros((N, N))
            for i in range(N):
                if i < N - 1:
                    A[i, i + 1] = t
                    A[i + 1, i] = t
                A[i, i] = o
                
            A[N//2, N//2] = 5
            #A[N//2+1, N//2+1] = 5
            return sp.csc_matrix(A) 
    def create_1d_hamiltonian(self, t, o, N, blocks=True):
        """Return blocks or full matrix for 1D wire."""
        if blocks:
            return ([sp.eye(1) * o] * N), ([sp.eye(1) * t] * (N - 1))
        else:
            A = np.zeros((N, N), dtype=complex)
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
        V_barrier = self.Vg




        
        if blocks:
            total_onsite_potential = np.zeros((W, W), dtype=float)
            for w in range(W):
                if w < 1 * L // 3:
                    total_onsite_potential[w,w] = V_barrier
                if w > 2 * L // 3:
                    total_onsite_potential[w,w] = V_barrier
            onsite_blocks = []
            
            for l in range(L):
                h_ii = np.zeros((W, W),  dtype=complex)
                for w in range(W):
                    h_ii[w, w] = o
                    if w < W - 1:
                        h_ii[w, w + 1] = t
                        h_ii[w + 1, w] = t
                h_ii += total_onsite_potential
                onsite_blocks.append(sp.csc_matrix(h_ii))

            hopping_blocks = [sp.eye(W, k=0)*t for i in range(L)]
            
            return onsite_blocks, hopping_blocks
        else:
            total_onsite_potential = np.zeros((W, L), dtype=float)
            # Set the top third of the width (w < W // 3) to Vg
            for l in range(L):
                for w in range(W):
                    if w < W // 3:
                        total_onsite_potential[w, l] = V_barrier
                    if w > 2*W // 3:
                        total_onsite_potential[w, l] = V_barrier
            # --- Full matrix representation for direct inversion ---
            A = np.zeros((N, N), dtype=complex)
            for l in range(L):
                for w in range(W):
                    idx = l * W + w
                    # On-site energy from the saddle-point potential
                    A[idx, idx] = o + total_onsite_potential[w, l]
                    # Hopping in width (transverse)
                    if w < W - 1:
                        A[idx, idx + 1] = t
                        A[idx + 1, idx] = t
                    # Hopping in length (longitudinal)
                    if l < L - 1:
                        idx_next = (l + 1) * W + w
                        A[idx, idx_next] = t
                        A[idx_next, idx] = t
            return sp.csc_matrix(A)
    
    def zig_zag_hamiltonian(self, blocks, t=-1.0, onsite_potential=0.0, ky = 0.0):
        """
        Builds the full tight-binding Hamiltonian for the nanoribbon structure.

        Args:
            t (float): The nearest-neighbor hopping parameter.
            onsite_potential (float): The on-site energy for all atoms.

        Returns:
            scipy.sparse.csr_matrix: The full Hamiltonian matrix.
        """
        if (not self.periodic and ky != 0):
            raise ValueError("cant have a nonzero ky and a non periodic lattice")
        if self.unit_cell is None:
            self.unit_cell=  GrapeheneZigZagCell(num_layers_x=self.Nx, num_layers_y=self.Ny, periodic=self.periodic)
        unitCell = self.unit_cell        
        
        num_atoms_total = len(unitCell.structure)
        num_atoms_layer = len(unitCell.layer)
        
        # Create a mapping from atom object to its index in the full structure list
        atom_to_idx = {atom: i for i, atom in enumerate(unitCell.structure)}
        
        # --- 1. Build the Onsite Block (H0) ---
        # Describes connections within one layer (unit cell)
        H0 = np.zeros((num_atoms_layer, num_atoms_layer), dtype=complex)
        
        # Create a mapping for just the first layer
        layer_atom_to_idx = {atom: i for i, atom in enumerate(unitCell.layer)}
        
        for i, atom in enumerate(unitCell.layer):
            H0[i, i] = onsite_potential
            # Find neighbors that are also in the first layer
            for neighbor, delta, l, m, n in unitCell.neighbors[atom]:
                if neighbor in layer_atom_to_idx:
                    j = layer_atom_to_idx[neighbor]
                    H0[i, j] = t


        H1 = np.zeros((num_atoms_layer, num_atoms_layer), dtype=complex)

        layer_1_atoms = set(unitCell.structure[num_atoms_layer : 2 * num_atoms_layer])

        for i, atom_in_layer0 in enumerate(unitCell.layer):
            for neighbor, delta, l, m, n in unitCell.neighbors[atom_in_layer0]:
                
                if neighbor in layer_1_atoms:
                    shifted_neighbor_pos = (neighbor.x - (unitCell.sin60 * 2), neighbor.y, neighbor.z)
                    
                    for atom_in_l0, idx in layer_atom_to_idx.items():
                        if np.allclose(atom_in_l0.pos(), shifted_neighbor_pos, atol=1e-5):
                            j = idx
                            if delta == (0,1,0) or delta == (0,-1,0): # only those that leave the unit cell 
                                H1[i, j] = t * np.exp(2 * np.pi * ky * 1j * delta[1])
                            else:
                                H1[i, j] = t 
                            
                            break
        H0_sparse = sp.csr_matrix(H0, dtype=complex)
        H1_sparse = sp.csr_matrix(H1, dtype=complex)
        
        # Create a list of the diagonal blocks (all H0)
        diagonal_blocks = [H0_sparse] * unitCell.num_layers_x
        off_diagonal_blocks = [H1_sparse] * (unitCell.num_layers_x - 1)
        if (not blocks):
            H_main_diag = sp.block_diag(diagonal_blocks, format='csc', dtype=complex)
            num_blocks = len(diagonal_blocks)
            block_rows, block_cols = diagonal_blocks[0].shape
            full_dim = num_blocks * block_rows

            H_upper = sp.lil_matrix((full_dim, full_dim), dtype=complex)

            for i, block in enumerate(off_diagonal_blocks):
                row_start = i * block_rows
                col_start = (i + 1) * block_cols
                H_upper[row_start : row_start + block_rows, col_start : col_start + block_cols] = block
            H_upper = H_upper.tocsc()
            H_full = H_main_diag + H_upper + H_upper.conj().T
            return H_full
        return diagonal_blocks, off_diagonal_blocks
        

    def create_hamiltonian(self, blocks=True, ky= 0):
        """
        General interface to get the Hamiltonian for the specified device type.
        """
        if self.name in self.hamiltonian_registry:
            H = self.hamiltonian_registry[self.name](self, blocks, ky)
        elif self.name == "zigzag":
            H = self.zig_zag_hamiltonian(blocks, ky=ky)
        elif self.name ==  "one_d_wire":
            H = self.one_d_wire(blocks=blocks)
        elif self.name == "quantum_point_contact" or self.name == "qpc":
            H = self.quantum_point_contact(blocks=blocks)
        elif self.name == "modified_one_d":
            H = self.modified_one_d_wire(blocks)
        else:
            # You can add other device types like "one_d_wire" here.
            raise ValueError(f"Unknown device type: {self.name}")
        
        H = self.add_potential(H, blocks)
        return H

    def get_H00_H01_H10(self, ky=0):
        """
        Get the principal layer Hamiltonian (H00) and coupling matrices (H01, H10)
        for the semi-infinite leads.
        """
        if self.name in self.lead_registry:
            return self.lead_registry[self.name](self, ky)
        if self.name == "one_d_wire" or self.name == "chain" or self.name =="modified_one_d":
            
            H00 = sp.eye(1) * self.o
            H01 = sp.eye(1) * self.t
            H10 = sp.eye(1) * self.t

            return H00, H01, H10 
        if self.name == "quantum_point_contact" or self.name == "qpc":
            
            H00 = self.create_1d_hamiltonian(self.t, self.o, self.W, blocks=False)
            
            # The coupling between principal layers in the lead.
            H01 = sp.eye(self.W, format='csc', dtype=complex) * self.t
            H10 = H01.T # Assuming real hopping t
            
            return H00, H01, H10
        
        if self.name == "zigzag":
            diag, offdiag = self.create_hamiltonian(True, ky)
            H00 = diag[0]
            H01 = offdiag[0]
            H10 = H01.T.conj()
            return H00, H01, H10
            
        else:
            raise ValueError(f"Lead definition not found for device: {self.name}")
        

    def get_potential(self, blocks: bool):
        if self.potential is None:
            return None

        if blocks:
            return self.potential
        else:
            # Collect all diagonals in a list
            diag_list = []
            for block in self.potential:
                diag_list.append(np.diag(block.toarray()))
            if len(diag_list) == 0:
                return None
            x = np.concatenate(diag_list)
            return sp.csc_matrix(np.diag(x))
    
    def atom_to_potential(self, atom_to_pot : dict, unit_cell):
        pot_array = [None] * (len(unit_cell.ATOM_POSITIONS))
        
        for atom_idx,atom in enumerate(unit_cell.ATOM_POSITIONS):
            pot = atom_to_pot[atom]
            pot_array[atom_idx] = sp.eye(self.num_orbitals) * pot
        
        self.potential = pot_array
    

    def add_potential(self, hamiltonian, blocks: bool):
        if self.potential is None:
            return hamiltonian
        if blocks:
            pot_list = self.get_potential(blocks)
            # Handle tuple (diag, offdiag) or just a list
            if isinstance(hamiltonian, tuple):
                diag, *rest = hamiltonian
                # Add potential only to diagonal blocks
                diag_with_pot = [d + p for d, p in zip(diag, pot_list)]
                return (diag_with_pot, *rest)
            elif isinstance(hamiltonian, list):
                # Add potential to each block in the list
                return [h + p for h, p in zip(hamiltonian, pot_list)]
            else:
                raise TypeError("Unexpected type for hamiltonian in blocks mode.")
        else:
            pot_matrix = self.get_potential(blocks)
            return hamiltonian + pot_matrix
            
    
    def register_hamiltonian(self, name, func):
        """Register a new hamiltonian construction function."""
        self.hamiltonian_registry[name] = func

    def register_lead(self, name, func):
        """Register a new lead function."""
        self.lead_registry[name] = func
        
