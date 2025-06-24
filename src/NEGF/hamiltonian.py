import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from NEGF_unit_generation import UnitCell 
from src.tight_binding import tight_binding_params as TBP
import numpy as np
from device import Device
import scipy.sparse as sp

class Hamiltonian:
    def __init__(self, device : Device):
        self.device = device
        
        self.Nx = int(device.channel_length // device.block_width)
        self.Nz = int(device.channel_thickness // device.block_height)
        self.unitCell = UnitCell(self.Nz, self.Nx)
        
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
        self.layer_right_lead = self.Nx % 4 # 0,1,2,3  
        self.tempUnitCell = UnitCell(self.Nz, 5)
        
    
    def getLayersHamiltonian(self, ky) -> dict:
        layersH00, layersH01 = self.create_sparse_channel_hamlitonian(ky, self.tempUnitCell)
        layers = {}
        for layer in range(4):
            layers[layer] = [layersH00[layer], layersH01[layer]]
        return layers        
        
        
    def create_tight_binding(self,ky, unitCell : UnitCell = None):
        if unitCell is None:
            unitCell = self.unitCell

        unitNeighbors = unitCell.neighbors
        danglingBonds = unitCell.danglingBondsZ
        
        
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
                    phase = np.exp(2 * np.pi * 1j * (ky*delta[1])) # blochs theorem does not work 
                    neighbor_index = atomToIndex[atom2]      
                    for secOrbitalIndex, secondOrbital in enumerate(orbitals):
                        index_j = neighbor_index * numOrbitals + secOrbitalIndex

                        hop = TBP.SK[(orbital, secondOrbital)](l, m, n, TBP.V)
                    
                        A[index_i,index_j] += hop * phase   
        
            dagger = lambda A: np.conjugate(A.T)
        if not np.allclose(A, dagger(A)):
            print("H isnt Hermitian")
            
        return A
    
    
    
    
    
    def get_H00_H01(self, ky, sparse=False):
        """out dated method"""
        oldUnitCell = self.unitCell
        self.unitCell = UnitCell(self.Nz, 8)

        HT = self.create_sparse_hamlitonian(ky)
   
        H00 = HT[:80 * self.Nz, :80 * self.Nz]
        H01 = HT[80 * self.Nz:, :80 * self.Nz]
        self.unitCell = oldUnitCell
        if sparse:
            return H00.toarray(), H01.toarray()
        return H00, H01
    
    
    
    def getMatrixSize(self):
        return len(self.unitCell.ATOM_POSITIONS) * 10
    
    def create_sparse_hamlitonian(self, ky, unitCell : UnitCell = None):
        
        if unitCell is None:
            unitCell = self.unitCell


        unitNeighbors = unitCell.neighbors
        danglingBonds = unitCell.danglingBondsZ
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
        
        
        
        numSilicon = len(unitNeighbors.keys())

        orbitals = ['s', 'px', 'py', 'pz', 'dxy','dyz','dzx','dx2y2','dz2', 's*']
        numOrbitals = len(orbitals)
        size = numSilicon * numOrbitals 
        
        atomToIndex = {}
        indexToAtom = {}
        for atom_index,atom in enumerate(unitNeighbors):
            atomToIndex[atom] = atom_index
            indexToAtom[atom_index] = atom

        rows, cols, data = [], [], []                   # <-- sparse triplets

        # helper -----------------------------------------------------------
        def add(i, j, val):
            if val != 0.0:
                rows.append(i); cols.append(j); data.append(val)
                if i != j:                              # Hermitian
                    rows.append(j); cols.append(i); data.append(np.conj(val))
        # ------------------------------------------------------------------

        # ---------- on‑site (Si) ----------
        
        for atom_idx, atom in indexToAtom.items():

            hybridizationMatrix = self.H_sp3_explicit.copy() 
            danglingBondsList = danglingBonds[atom]
            for danglingBondAtom, position in danglingBondsList:
                hybridizationMatrix[position,position] += TBP.E['sp3']            
            
            # if there are no dangling bonds this returns the standard diag matrix with onsite energies 
        
            onsiteMatrix = self.U_orb_to_sp3 @ hybridizationMatrix @ self.U_orb_to_sp3.T 
            base = atom_idx * numOrbitals
            for i in range(4):
                for j in range(i ,4):
                    add(base + i, base + j, onsiteMatrix[i,j])
        
            for p in range(4, 9):                       # five d’s
                add(base + p, base + p, TBP.E['dxy'] )
            add(base + 9, base + 9, TBP.E['s*'])
        
    
        for atom_index, atom in indexToAtom.items():
            base_i = atom_index * numOrbitals
            for atom2, delta, l,m,n in unitNeighbors[atom]:
                j = atomToIndex[atom2]
                if j < atom_index:             
                    continue                   

                phase = np.exp(2j*np.pi*(ky*delta[1]))

                for o1, orb1 in enumerate(orbitals):
                    for o2, orb2 in enumerate(orbitals):
                        hop = TBP.SK[(orb1, orb2)](l, m, n, TBP.V) * phase
                        add(base_i + o1, j*numOrbitals + o2, hop)

                
    
        H = sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()

        return H
    
    def create_sparse_channel_hamlitonian(self, ky, unitCell : UnitCell = None):
        if unitCell is None:
            unitCell = self.unitCell

        unitNeighbors = unitCell.neighbors
        danglingBonds = unitCell.danglingBondsZ
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
        
        potentialPerAtom = self.potential_correction()

        numSilicon = len(unitNeighbors.keys())

        orbitals = ['s', 'px', 'py', 'pz', 'dxy','dyz','dzx','dx2y2','dz2', 's*']
        numOrbitals = len(orbitals)
        size = numSilicon * numOrbitals 
        
        atomToIndex = {}
        indexToAtom = {}
        for atom_index,atom in enumerate(unitNeighbors):
            atomToIndex[atom] = atom_index
            indexToAtom[atom_index] = atom

        rows, cols, data = [], [], []                   # <-- sparse triplets

        # helper -----------------------------------------------------------
        def add(i, j, val):
            if val != 0.0:
                rows.append(i); cols.append(j); data.append(val)
                if i != j:                              # Hermitian
                    rows.append(j); cols.append(i); data.append(np.conj(val))
        # ------------------------------------------------------------------

        # ---------- on‑site (Si) ----------
        
        for atom_idx, atom in indexToAtom.items():

            hybridizationMatrix = self.H_sp3_explicit.copy() 
            danglingBondsList = danglingBonds[atom]
            for danglingBondAtom, position in danglingBondsList:
                hybridizationMatrix[position,position] += TBP.E['sp3']            
            
            # if there are no dangling bonds this returns the standard diag matrix with onsite energies 
        
            onsiteMatrix = self.U_orb_to_sp3 @ hybridizationMatrix @ self.U_orb_to_sp3.T 
            base = atom_idx * numOrbitals
            for i in range(4):
                for j in range(i ,4):
                    add(base + i, base + j, onsiteMatrix[i,j] + potentialPerAtom[atom_idx] * (i == j))
        
            for p in range(4, 9):                       # five d’s
                add(base + p, base + p, TBP.E['dxy'] + potentialPerAtom[atom_idx])
            add(base + 9, base + 9, TBP.E['s*']+ potentialPerAtom[atom_idx])
        
    
        for atom_index, atom in indexToAtom.items():
            base_i = atom_index * numOrbitals
            for atom2, delta, l,m,n in unitNeighbors[atom]:
                j = atomToIndex[atom2]
                if j < atom_index:             
                    continue                   

                phase = np.exp(2j*np.pi*(ky*delta[1]))

                for o1, orb1 in enumerate(orbitals):
                    for o2, orb2 in enumerate(orbitals):
                        hop = TBP.SK[(orb1, orb2)](l, m, n, TBP.V) * phase
                        add(base_i + o1, j*numOrbitals + o2, hop)

                
    
        H = sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
        
        total_size = H.shape[0]
        block_size = self.Nz * 20 # 2 atoms per single unit z layer 
        num_blocks = (int) (total_size // block_size)
        diagonal_blocks = [None] * num_blocks
        off_diagonal_blocks = [None] * (num_blocks - 1)
        
        for block in range(0, num_blocks - 1):
            s = block * block_size
            m = (block + 1) * block_size
            e = (block + 2) * block_size
            diagonal_blocks[block] = H[s : m, s : m] 
            off_diagonal_blocks[block] = H[s : m, m : e]
        diagonal_blocks[-1] = H[-block_size:, -block_size:]       

        return diagonal_blocks, off_diagonal_blocks
    
    def potential_correction(self):
        voltage = self.device.Ec
        
        L, H = voltage.shape
        potential_diag = [None] * len(self.unitCell.ATOM_POSITIONS) * 10
        for atom_index,atom in enumerate(self.unitCell.neighbors):
            px, pz = atom.x, atom.z
            gx = (int) (px / self.Nx * L)
            gz = (int) (pz / self.Nz * H)
            
            pot = voltage[gx, gz]
            potential_diag[atom_index] = pot
        
        return potential_diag       