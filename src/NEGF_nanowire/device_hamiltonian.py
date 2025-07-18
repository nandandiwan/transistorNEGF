import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from NEGF_device_generation import UnitCell 
from src.tight_binding import tight_binding_params as TBP
import numpy as np
from device import Device
import scipy.sparse as sp

class Hamiltonian:
    def __init__(self, device : Device):
        self.device = device
        self.unitCell = device.unitCell
        self.Nx = (int) (self.device.unitX)
        self.Ny = (int) (self.device.unitY)
        self.Nz = (int) (self.device.unitZ)
        
        self.U_orb_to_sp3 = 0.5*np.array([[1, 1, 1, 1],
                             [1, 1,-1,-1],
                             [1,-1, 1,-1],
                             [1,-1,-1, 1]])
        Es = TBP.E['s']
        Ep = TBP.E['px'] 
        
        H_orb = np.diag([Es, Ep, Ep, Ep])

        # change-of-basis matrices
        U_orb_to_sp3 = 0.5*np.array([[1, 1, 1, 1],
                                    [1, 1,-1,-1],
                                    [1,-1, 1,-1],
                                    [1,-1,-1, 1]])
        self.U_sp3_to_orb = U_orb_to_sp3.T  

        a = (Es + 3*Ep)/4.0
        b = (Es -   Ep)/4.0
        H_sp3_explicit = np.full((4,4), b)
        np.fill_diagonal(H_sp3_explicit, a)
        self.H_sp3_explicit = H_sp3_explicit
        
        # self energy
        """Which layer is at the end?"""
        self.layer_left_lead = 0 # 0,1,2,3 
        self.layer_right_lead = 0
        
    def create_tight_binding(self,unitCell : UnitCell = None):
        if unitCell is None:
            unitCell = self.unitCell

        unitNeighbors = unitCell.neighbors
        danglingBonds = unitCell.danglingBonds
        
        
        numSilicon = len(unitNeighbors.keys())

    
        orbitals = ['s', 'px', 'py', 'pz', 'dxy','dyz','dzx','dx2y2','dz2', 's*']
        numOrbitals = len(orbitals)
        size = numSilicon * numOrbitals 
        A = np.zeros((size, size), dtype=complex)    
        
        atomToIndex = {}
        indexToAtom = {}
        for atom_index,atom in enumerate(unitNeighbors):
            atomToIndex[atom] = atom_index
            indexToAtom[atom_index] = atom
        
    
    
        for atom_idx, atom in indexToAtom.items():
            hybridizationMatrix = self.H_sp3_explicit.copy() 
            danglingBondsList = danglingBonds[atom]
            for danglingBondAtom, position in danglingBondsList:
                hybridizationMatrix[position,position] += TBP.E['sp3']            
            
            # if there are no dangling bonds this returns the standard diag matrix with onsite energies 
            onsiteMatrix = self.U_orb_to_sp3 @ hybridizationMatrix @ self.U_orb_to_sp3.T # go back to orbital basis 
            A[atom_idx*10:atom_idx*10 + 4, atom_idx*10:atom_idx*10 + 4] = onsiteMatrix
            A[atom_idx*10 + 4:atom_idx*10 + 9, atom_idx*10 + 4:atom_idx*10 + 9] = np.eye(5) * TBP.E['dxy']
            A[atom_idx*10 + 9,atom_idx*10 + 9] = TBP.E['s*']
         
    
        for atom_index in range(numSilicon):
            atom = indexToAtom[atom_index]
            neighbors = unitNeighbors[atom]
            for orbitalIndex, orbital in enumerate(orbitals):
                index_i = atom_index * numOrbitals + orbitalIndex
                for neighbor in neighbors:
                    atom2, delta, l,m,n = neighbor
                    phase = np.exp(2 * np.pi * 1j * (0*delta[1])) # blochs theorem does not work 
                    neighbor_index = atomToIndex[atom2]      
                    for secOrbitalIndex, secondOrbital in enumerate(orbitals):
                        index_j = neighbor_index * numOrbitals + secOrbitalIndex

                        hop = TBP.SK[(orbital, secondOrbital)](l, m, n, TBP.V)
                    
                        A[index_i,index_j] += hop * phase   
        
            dagger = lambda A: np.conjugate(A.T)
        if not np.allclose(A, dagger(A)):
            print("H isnt Hermitian")
            
        return A
    

    
    
    def get_H00_H01_H10(self, side="left", sparse=False):
        """
        Get H00, H01, and H10 matrices for both leads with proper symmetry.
        For symmetric DOS at zero bias, both leads should have identical structure.
        """
        if side == 'left':
            orientation = (0, 1, 2, 3)  # Always use same orientation
        elif side == "right":
            orientation = (3, 2, 1, 0)
        
        # Create 8-layer system: two adjacent 4-layer supercells
        # This allows us to extract H01 as coupling between supercells
        newUnitCell = UnitCell(channel_length=2 * 0.5431e-9,channel_width=self.device.channel_width,channel_thickness=self.device.channel_thickness, orientation=orientation)
        HT = self.create_sparse_channel_hamlitonian(unitCell=newUnitCell, blocks=False)
        
        # Calculate proper block size: 4 layers × Nz vertical blocks × 2 atoms/layer × 10 orbitals/atom
        # The 4-layer structure represents one complete unit cell in the transport direction
        block_size = self.Nz * self.Ny * 4 * 10
        
        # Extract matrices with consistent sizing
        H00 = HT[:block_size, :block_size]
        
        # H01: coupling from current unit cell to next unit cell  
        H01 = HT[:block_size, block_size:2*block_size]
        
        # H10: coupling from next unit cell to current unit cell (should be H01†)
        H10 = HT[block_size:2*block_size, :block_size]
        
        # Verify Hermitian relationship: H10 should equal H01†
        if not sparse:
            H00_dense = H00.toarray() if hasattr(H00, 'toarray') else H00
            H01_dense = H01.toarray() if hasattr(H01, 'toarray') else H01
            H10_dense = H10.toarray() if hasattr(H10, 'toarray') else H10
            
            # Check if H10 ≈ H01†
            diff = np.max(np.abs(H10_dense - H01_dense.conj().T))
            if diff > 1e-12:
                print(f"Warning: H10 != H01† for {side} lead, difference: {diff:.2e}")
            
            return H00_dense, H01_dense, H10_dense
            
        return H00, H01, H10
    
    
    def getMatrixSize(self):
        return len(self.unitCell.ATOM_POSITIONS) * 10
    
    def create_sparse_channel_hamlitonian(self, unitCell : UnitCell = None, blocks = True):
        """
        Efficiently build the sparse channel Hamiltonian using only sparse triplet construction.
        Optionally, return block structure for further optimization.
        """
        if unitCell is None:
            unitCell = self.unitCell

        unitNeighbors = unitCell.neighbors
        danglingBonds = unitCell.danglingBonds
        numSilicon = len(unitNeighbors)

        orbitals = ['s', 'px', 'py', 'pz', 'dxy','dyz','dzx','dx2y2','dz2', 's*']
        numOrbitals = len(orbitals)
        size = numSilicon * numOrbitals

        atomToIndex = {atom: idx for idx, atom in enumerate(unitNeighbors)}
        indexToAtom = {idx: atom for atom, idx in atomToIndex.items()}

        potentialPerAtom = self.potential_correction()

        rows, cols, data = [], [], []  # sparse triplets

        def add(i, j, val):
            if val != 0.0:
                rows.append(i); cols.append(j); data.append(val)
                if i != j:  # Hermitian
                    rows.append(j); cols.append(i); data.append(np.conj(val))

        # Onsite terms
        for atom_idx, atom in indexToAtom.items():
            hybridizationMatrix = self.H_sp3_explicit.copy()
            danglingBondsList = danglingBonds[atom]
            for danglingBondAtom, position in danglingBondsList:
                hybridizationMatrix[position, position] += TBP.E['sp3']
            onsiteMatrix = self.U_orb_to_sp3 @ hybridizationMatrix @ self.U_orb_to_sp3.T
            base = atom_idx * numOrbitals
            for i in range(4):
                for j in range(i, 4):
                    add(base + i, base + j, onsiteMatrix[i, j]) #+ potentialPerAtom[atom_idx] * (i == j))
            for p in range(4, 9):
                add(base + p, base + p, TBP.E['dxy'])# + potentialPerAtom[atom_idx])
            add(base + 9, base + 9, TBP.E['s*']) #+ potentialPerAtom[atom_idx])

        # Hopping terms
        for atom_idx, atom in indexToAtom.items():
            base_i = atom_idx * numOrbitals
            for atom2, delta, l, m, n in unitNeighbors[atom]:
                j = atomToIndex[atom2]
                if j < atom_idx:
                    continue  # Only fill upper triangle, Hermitian fill handles the rest
      
                for o1, orb1 in enumerate(orbitals):
                    for o2, orb2 in enumerate(orbitals):
                        hop = TBP.SK[(orb1, orb2)](l, m, n, TBP.V) 
                        add(base_i + o1, j * numOrbitals + o2, hop)

        H = sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsc()
        if not blocks:
            return H

        # Block extraction: assumes atoms are ordered by layer (x), then y, then z
        # User must ensure this ordering in ATOM_POSITIONS
        # block_size: number of orbitals per layer (all atoms in a given x)
        # For now, use Nz*Ny*10 if each x is a layer, or let user specify
        # Here, we assume each 'layer' is a unique x value in ATOM_POSITIONS

        block_size = self.Nz * self.Ny * 2 * 10
        num_blocks = self.Nx *4
        diagonal_blocks = []
        off_diagonal_blocks = []
        for i in range(num_blocks):
            s = i * block_size
            e = (i + 1) * block_size
            diagonal_blocks.append(H[s:e, s:e])
            if i < num_blocks - 1:
                off_diagonal_blocks.append(H[s:e, e:e+block_size])
        return diagonal_blocks, off_diagonal_blocks
    
    def potential_correction(self):
        # For testing bandgap opening, return zeros to isolate dangling bond effects
        return [0.0] * len(self.unitCell.ATOM_POSITIONS) * 10
        
        # Or fix the original method:
        # voltage = self.device.Ec
        # L, H = voltage.shape
        # potential_diag = [0.0] * len(self.unitCell.ATOM_POSITIONS)  # Initialize with zeros
        
        # for atom_index, atom in enumerate(self.unitCell.neighbors):
        #     px, pz = atom.x, atom.z
        #     # Ensure proper grid mapping
        #     gx = min(max(int(px / self.Nx * L), 0), L-1)
        #     gz = min(max(int(pz / self.Nz * H), 0), H-1)
        #     potential_diag[atom_index] = voltage[gx, gz]
        
        # return potential_diag    