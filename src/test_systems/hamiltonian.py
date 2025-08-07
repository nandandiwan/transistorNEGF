import copy
import numpy as np
import scipy.sparse as sp
import scipy.constants as spc
from unit_cell_generation import GrapehenearmchairCell
class Hamiltonian:
    """
    Constructs tight-binding Hamiltonians for various device structures.
    """
    def __init__(self, name, periodic = False, relevant_parameters = {}):
        if((periodic and name != "armchair")):
            raise ValueError("periodic not available or possible")
        self.T = 300  # Use the passed temperature parameter
        self.q = spc.elementary_charge
        self.kbT = spc.Boltzmann * self.T  # Keep in Joules
        self.kbT_eV = spc.Boltzmann * self.T / self.q  # Also store in eV for convenience
        self.name = name
        self.t = 1 #* spc.hbar**2 / (2 * spc.m_e * .25 * (3e-10)**2 * self.q)   # Hopping energy
        self.o = 2 #* spc.hbar**2 / (2 * spc.m_e * .25 * (3e-10)**2 * self.q)   # Base on-site energy
        self.Vs = 0.0  # Source voltage
        self.Vd = 0.0  # Drain voltage
        self.Vg = 0  # Gate voltage applied to the device region
        self.num_orbitals = 1
        self.N = 120
        self.mu1 = 0.0 # chemical potential at left
        self.mu2 = 0.0  # chemical potential at right 
        self.Ef = 0.0
        
       
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
        self.H0 = None
        self.T0 = None
        self.relevant_parameters = relevant_parameters
        
        #modified oned
        self.hamiltonian_registry = {}
        self.lead_registry = {}
        
        self.potential = None
        
        self.mock_potential = True # True until poisson solver is in place
        self.middle_third = True # there's a sort of 
        if (self.name != "one_d_wire"): # base linear potential for graphene is not in place
            self.mock_potential = False
        
        if (self.name == "armchair"):

            self.unit_cell = GrapehenearmchairCell(num_layers_x=self.Nx, num_layers_y=self.Ny, periodic=self.periodic)
            
    
    def get_num_sites(self):
        if (self.name == "one_d_wire" or self.name == "modified_one_d"):
            return self.N
        elif (self.name == "qpc"):
            return self.W * self.L
        elif (self.name == "armchair"):
            if self.unit_cell is None:
                self.unit_cell = GrapehenearmchairCell(num_layers_x=self.Nx, num_layers_y=self.Ny, periodic=self.periodic)
            return len(self.unit_cell.structure)
        else:
            return self.num_orbitals * len(self.unit_cell.structure)
        
    def set_voltage(self, Vs=0, Vd=0, Vg=0):
        self.Vs = Vs
        self.Vd = Vd
        self.Vg = Vg
        
        self.mu1 = self.Vs + self.Ef
        self.mu2 = self.Vd + self.Ef
    def one_d_wire(self, blocks=True):
        """Return blocks or full matrix for 1D wire."""
        t, o, N = self.t, self.o, self.N
        if blocks:
            return ([sp.eye(1) * o] * N), ([sp.eye(1) * t] * (N - 1))
        else:
            A = np.zeros((N, N))
            for i in range(N):
                if i < N - 1:
                    A[i, i + 1] = -t
                    A[i + 1, i] = -t
                A[i, i] = o
            return sp.csc_matrix(A) 
        
    def modified_one_d_wire(self, blocks=True):
        t, o, N = self.t, self.o, self.N
        if blocks:
            diag, offdiag =  ([sp.eye(1) * o] * N), ([sp.eye(1) * -t] * (N - 1))
            diag[N//2] += sp.eye(1) * 5
            #diag[N//2 + 1] += sp.eye(1) * 5
            return diag, offdiag
            
        else:
            A = np.zeros((N, N))
            for i in range(N):
                if i < N - 1:
                    A[i, i + 1] = -t
                    A[i + 1, i] = -t
                A[i, i] = o
                
            A[N//2, N//2] = 5
            #A[N//2+1, N//2+1] = 5
            return sp.csc_matrix(A) 
    def create_1d_hamiltonian(self, t, o, N, blocks=True):
        """Return blocks or full matrix for 1D wire."""
        if blocks:
            return ([sp.eye(1) * o] * N), ([sp.eye(1) * -t] * (N - 1))
        else:
            A = np.zeros((N, N), dtype=complex)
            for i in range(N):
                if i < N - 1:
                    A[i, i + 1] = -t
                    A[i + 1, i] = -t
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
    
    def armchair_hamiltonian(self, blocks, t=-2.7, onsite_potential=0.0, ky=0.0):
        """
        Builds the tight-binding Hamiltonian for the armchair nanoribbon structure.

        Args:
            blocks (bool): Whether to return block format or full matrix
            t (float): The nearest-neighbor hopping parameter.
            onsite_potential (float): The on-site energy for all atoms.
            ky (float): Bloch momentum for periodic case (along x-direction)

        Returns:
            Hamiltonian in block format or full matrix
        """
        if (not self.periodic and ky != 0):
            raise ValueError("cant have a nonzero ky and a non periodic lattice")
        if self.unit_cell is None:
            self.unit_cell = GrapehenearmchairCell(num_layers_x=self.Nx, num_layers_y=self.Ny, periodic=self.periodic)
        
        unitCell = self.unit_cell        
        
        if self.periodic:
            H, T0 = self._create_armchair_hamiltonian_periodic(t=t)
            H_eff = H + T0 * np.exp(np.pi * ky * 1j) + T0.conj().T * np.exp(-np.pi * ky * 1j)
            if (not blocks):
                return H_eff
            else:
                return self._convert_to_blocks(H_eff, 4)
            
        else:
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
                                    H1[i, j] = t * np.exp(2 * np.pi * ky * 1j * delta[1] / 3)
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
    def _create_armchair_hamiltonian_periodic(self, t=-2.7):
        """
        Constructs the on-site (H) and coupling (T0) Hamiltonian matrices
        for an armchair graphene nanoribbon unit cell directly.
        
        Returns:
            (sp.csc_matrix, sp.csc_matrix): A tuple containing H and T0.
        """
        n = self.Nx
        num_atoms = 4 * n 
        

        H_rows = []
        H_cols = []

        for i in range(num_atoms):
            neighbors = []
            if (i != 0 and i != 1 and i != 4 * n -1 and i != (4*n - 2) and ((i-2) % 4) != 0 and (i - 3) % 4 != 0):
                if (i % 2 == 0):
                    neighbors = [i - 1, i + 1, i + 3]
                else:
                    neighbors = [i - 3, i + 1, i - 1]
            else:
                if (i == 0):
                    neighbors = [1, 3]
                elif (i == 1):
                    neighbors = [0, 2]
                elif (i == 4 * n - 1):
                    neighbors = [4*n -4]
                elif (i == 4 * n - 2):
                    neighbors = [ 4 * n - 3]
                elif ((i - 2) % 4 == 0):
                    neighbors = [i - 1, i + 3]
                elif ((i - 3) % 4 == 0):
                    neighbors = [i - 3, i + 1]
                else:
                    print(f"Unhandled case for atom index: {i}")
                    raise ValueError("wrong amount")
            
            for neighbor in neighbors:
                if 0 <= neighbor < num_atoms:
                    H_rows.append(i)
                    H_cols.append(neighbor)

        H_data = np.full(len(H_rows), t, dtype=float)
        H = sp.coo_matrix((H_data, (H_rows, H_cols)), shape=(num_atoms, num_atoms)).tocsc()

        T0_rows = [4 * i - 2 for i in range(1, n + 1)]
        T0_cols = [4 * i - 1 for i in range(1, n + 1)]
        T0_data = np.full(n, t, dtype=float)
        T0 = sp.coo_matrix((T0_data, (T0_rows, T0_cols)), shape=(num_atoms, num_atoms)).tocsc()
        
        return H, T0
        
    def _build_full_matrix_from_blocks(self, H0, H1, num_blocks):
        """Build full Hamiltonian matrix from H0 and H1 blocks."""
        block_size = H0.shape[0]
        full_size = num_blocks * block_size
        H_full = sp.lil_matrix((full_size, full_size), dtype=complex)
        
        # Add diagonal blocks (H0)
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            H_full[start_idx:end_idx, start_idx:end_idx] = H0
            
        # Add off-diagonal blocks (H1 and H1†)
        for i in range(num_blocks - 1):
            start_i = i * block_size
            end_i = start_i + block_size
            start_j = (i + 1) * block_size
            end_j = start_j + block_size
            
            H_full[start_i:end_i, start_j:end_j] = H1
            H_full[start_j:end_j, start_i:end_i] = H1.conj().T
            
        return H_full.tocsr()
        
    def _convert_to_blocks(self, H_full, atoms_per_layer):
        """Convert full matrix to block format."""
        diagonal_blocks = []
        off_diagonal_blocks = []
        num_layers = (int) (H_full.shape[0] / atoms_per_layer)
        
        
        for i in range(num_layers):
            start_idx = i * atoms_per_layer
            end_idx = start_idx + atoms_per_layer
            
            # Extract diagonal block
            H_ii = H_full[start_idx:end_idx, start_idx:end_idx]
            diagonal_blocks.append(sp.csr_matrix(H_ii))
            
            # Extract off-diagonal block (if not last layer)
            if i < num_layers - 1:
                start_j = (i + 1) * atoms_per_layer
                end_j = start_j + atoms_per_layer
                H_ij = H_full[start_idx:end_idx, start_j:end_j]
                off_diagonal_blocks.append(sp.csr_matrix(H_ij))
                
        return diagonal_blocks, off_diagonal_blocks

    def create_hamiltonian(self, blocks=True, ky=0, no_pot=False):
        """
        General interface to get the Hamiltonian for the specified device type.
        """
        if self.name in self.hamiltonian_registry:
            H = self.hamiltonian_registry[self.name](self, blocks, ky)
        elif self.name == "armchair":
            H = self.armchair_hamiltonian(blocks, ky=ky)
        elif self.name ==  "one_d_wire":
            H = self.one_d_wire(blocks=blocks)
        elif self.name == "quantum_point_contact" or self.name == "qpc":
            H = self.quantum_point_contact(blocks=blocks)
        elif self.name == "modified_one_d":
            H = self.modified_one_d_wire(blocks)
        else:
            # You can add other device types like "one_d_wire" here.
            raise ValueError(f"Unknown device type: {self.name}")
        if (no_pot):
            return H
        H = self.add_potential(H, blocks)
        return H

    def get_H00_H01_H10(self, ky=0, side = "left"):
        """
        Get the principal layer Hamiltonian (H00) and coupling matrices (H01, H10)
        for the semi-infinite leads.
        
        side is irrelevant where orientation is not important 
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
        
        if self.name == "armchair":

            diag, offdiag = self.create_hamiltonian(True, ky, no_pot=True)
            H00 = diag[0]
            H01 = offdiag[0]
            H10 = H01.T.conj()
            return H00, H01, H10
            
        else:
            raise ValueError(f"Lead definition not found for device: {self.name}")
        

    def get_potential(self, blocks: bool):
        if self.potential is None:
            return None
        pot = copy.deepcopy(self.potential)
        
        if (self.mock_potential):
            N = len(pot)
            n1 = N // 3
            n2 = 2 * N // 3
            for i in range(len(pot)):
                if self.middle_third:
                    if i < n1:
                        V = self.Vs
                    elif i >= n2:
                        V = self.Vd
                    else:
                        ramp_i = i - n1
                        ramp_len = n2 - n1
                        V = self.Vs + (self.Vd - self.Vs) * ramp_i / (ramp_len - 1)
                else:
                    V = self.Vs + (self.Vd - self.Vs) * i / (len(pot) - 1)
                
                pot[i] += sp.eye(pot[i].shape[0]) * V

        if blocks:
            return pot
        else:
            # Collect all diagonals in a list
            diag_list = []
            for block in pot:
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
    
    def set_barrier_potential(self, positions, height, width=1):
        """
        Set a barrier potential at specified positions.
        
        Args:
            positions (list or int): Position(s) where to place barrier
            height (float): Height of the barrier in eV
            width (int): Width of each barrier in sites
        """
        N = self.get_num_sites()
        if self.potential is None:
            self.potential = [sp.eye(self.num_orbitals) * 0.0 for _ in range(N)]
        
        if isinstance(positions, int):
            positions = [positions]
            
        for pos in positions:
            for i in range(width):
                if 0 <= pos + i < N:
                    self.potential[pos + i] += sp.eye(self.num_orbitals) * height
    
    def set_linear_potential(self, V_start, V_end, middle_third=False):

        N = self.get_num_sites()
        self.potential = []

        if middle_third:
            n1 = N // 3
            n2 = 2 * N // 3

            for i in range(N):
                if i < n1:

                    V = V_start
                elif i >= n2:
                    V = V_end
                else:
                    ramp_i = i - n1
                    ramp_len = n2 - n1
                    V = V_start + (V_end - V_start) * ramp_i / (ramp_len - 1)

                self.potential.append(sp.eye(self.num_orbitals) * V)
        else:

            for i in range(N):
                V = V_start + (V_end - V_start) * i / (N - 1)
                self.potential.append(sp.eye(self.num_orbitals) * V)
    def set_double_barrier_potential(self, barrier1_pos, barrier2_pos, barrier_height, barrier_width=2, well_depth=0.0):
        """
        Set a double barrier potential with a quantum well.
        
        Args:
            barrier1_pos (int): Position of first barrier
            barrier2_pos (int): Position of second barrier  
            barrier_height (float): Height of barriers in eV
            barrier_width (int): Width of each barrier in sites
            well_depth (float): Depth of quantum well (negative for well)
        """
        N = self.get_num_sites()
        self.potential = [sp.eye(self.num_orbitals) * 0.0 for _ in range(N)]
        
        # Set barriers
        for i in range(barrier_width):
            if 0 <= barrier1_pos + i < N:
                self.potential[barrier1_pos + i] += sp.eye(self.num_orbitals) * barrier_height
            if 0 <= barrier2_pos + i < N:
                self.potential[barrier2_pos + i] += sp.eye(self.num_orbitals) * barrier_height
        
        # Set quantum well between barriers
        well_start = barrier1_pos + barrier_width
        well_end = barrier2_pos
        for i in range(well_start, well_end):
            if 0 <= i < N:
                self.potential[i] += sp.eye(self.num_orbitals) * well_depth
    
    def clear_potential(self):
        """Clear all potential."""
        self.potential = None
    
    def reset_voltages(self):
        """Reset all voltages to zero."""
        self.Vs = 0.0
        self.Vd = 0.0
        self.Vg = 0.0
        self.set_voltage()
        
    def set_params(self, *args):
        if (self.name == "armchair"):
            Nx, Ny = args
            self.Nx = Nx
            self.Ny = Ny 
            self.unit_cell = GrapehenearmchairCell(num_layers_x=self.Nx, num_layers_y=self.Ny, periodic=self.periodic)
            
    
    def register_hamiltonian(self, name, func):
        """Register a new hamiltonian construction function."""
        self.hamiltonian_registry[name] = func

    def register_lead(self, name, func):
        """Register a new lead function."""
        self.lead_registry[name] = func

    def setup_ballistic_device(self, NS=15, NC=30, ND=15):
        """
        Setup ballistic device (no barriers) as in MATLAB script.
        """
        self.name = "one_d_wire"
        self.N = NS + NC + ND
        self.clear_potential()  # No barriers
        print(f"Setup ballistic device: NS={NS}, NC={NC}, ND={ND}, Total={self.N}")

    def setup_tunneling_device(self, NS=23, NC=4, ND=23, barrier_height=0.4):
        """
        Setup tunneling barrier device as in MATLAB script.
        """
        self.name = "one_d_wire"
        self.N = NS + NC + ND
        
        # Set uniform barrier in channel region
        self.potential = [sp.eye(self.num_orbitals) * 0.0 for _ in range(self.N)]
        for i in range(NS, NS + NC):
            self.potential[i] = sp.eye(self.num_orbitals) * barrier_height
            
        print(f"Setup tunneling device: NS={NS}, NC={NC}, ND={ND}, barrier={barrier_height} eV")

    def setup_resonant_tunneling_device(self, NS=15, NC=16, ND=15, barrier_height=0.4, barrier_width=4):
        """
        Setup resonant tunneling device (double barrier) as in MATLAB script.
        UB=[zeros(NS,1);0.4*ones(4,1);zeros(NC-8,1);0.4*ones(4,1);zeros(ND,1)]
        """
        self.name = "one_d_wire"
        self.N = NS + NC + ND
        
        # Initialize potential
        self.potential = [sp.eye(self.num_orbitals) * 0.0 for _ in range(self.N)]
        
        # First barrier: positions NS to NS+barrier_width-1
        for i in range(NS, NS + barrier_width):
            self.potential[i] = sp.eye(self.num_orbitals) * barrier_height
            
        # Second barrier: positions NS+NC-barrier_width to NS+NC-1
        for i in range(NS + NC - barrier_width, NS + NC):
            self.potential[i] = sp.eye(self.num_orbitals) * barrier_height
            
        print(f"Setup resonant tunneling device: NS={NS}, NC={NC}, ND={ND}")
        print(f"Double barriers: width={barrier_width}, height={barrier_height} eV")

    def apply_bias_potential(self, V):
        """
        Apply bias potential as in MATLAB script:
        U1 = V * [0.5*ones(1,NS) linspace(0.5,-0.5,NC) -0.5*ones(1,ND)]
        """
        NS = 15  # Assuming standard device structure
        NC = self.N - 30  # Channel length
        ND = 15
        
        if self.potential is None:
            self.potential = [sp.eye(self.num_orbitals) * 0.0 for _ in range(self.N)]
        
        # Source region: V/2
        for i in range(NS):
            if i < len(self.potential):
                self.potential[i] += sp.eye(self.num_orbitals) * (V * 0.5)
        
        # Channel region: linear drop from V/2 to -V/2
        for i in range(NS, NS + NC):
            if i < len(self.potential):
                # Linear interpolation
                x = (i - NS) / (NC - 1) if NC > 1 else 0
                V_local = V * (0.5 - x)
                self.potential[i] += sp.eye(self.num_orbitals) * V_local
        
        # Drain region: -V/2
        for i in range(NS + NC, self.N):
            if i < len(self.potential):
                self.potential[i] += sp.eye(self.num_orbitals) * (V * (-0.5))
        
