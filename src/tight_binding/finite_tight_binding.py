import unit_cell_generation as unit_cell
import tight_binding_params as tbp
import numpy as np
import numpy as np
import scipy.constants as spc
from itertools import product
from multiprocessing import Pool, cpu_count
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class TightBindingHamiltonian:
    def __init__(self, N):
        self.H = None
        self.N = None
        self.unitCell = unit_cell.UnitCellGeneration(N) 
        self.U_orb_to_sp3 = 0.5*np.array([[1, 1, 1, 1],
                             [1, 1,-1,-1],
                             [1,-1, 1,-1],
                             [1,-1,-1, 1]])
        Es = tbp.E['s']
        Ep = tbp.E['px'] 
        
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
            
    def create_tight_binding(self, k, N=1, potentialProfile = None):
    
        kx,ky = k
        
        unitCell = self.unitCell
        if potentialProfile is None:
            potentialProfile = unitCell.create_linear_potential(0)
        unitNeighbors = unitCell.neighborTable()
        hydrogens = unitCell.hydrogens
        
        
        
        numSilicon = len(unitNeighbors.keys())
        numHydrogen = len(hydrogens.keys()) 
    
        orbitals = ['s', 'px', 'py', 'pz', 'dxy','dyz','dzx','dx2y2','dz2', 's*']
        numOrbitals = len(orbitals)
        size = numSilicon * numOrbitals + numHydrogen * 0
        A = np.zeros((size, size), dtype=complex)    
        
        atomToIndex = {}
        indexToAtom = {}
        for atom_index,atom in enumerate(unitNeighbors):
            atomToIndex[atom] = atom_index
            indexToAtom[atom_index] = atom
        
        
        """
        for atom_index in range(numSilicon):
            for orbIndex, orbital in enumerate(orbitals):
                index = atom_index * 10 + orbIndex
                #print(orbital)
                A[index, index] += E[orbital]

        
        
        for hydrogen in hydrogens.keys():
            information = hydrogens[hydrogen]
            silicon, delta, hIndex, l,m,n = information
            
            siliconIndex = atomToIndex[silicon] #what atom is silicon 
            hydrogenIndex = numSilicon *numOrbitals + hIndex -1
            
            # first update onsite energies of the two 
            for i in range(siliconIndex * numOrbitals, (siliconIndex + 1) * numOrbitals):
                A[i,i] += E['delta_Si']
            
        
            A[hydrogenIndex,hydrogenIndex] += E['HS']
            
            for orbIndex, orb in enumerate(orbitals):
                index_i = hydrogenIndex 
                index_j = siliconIndex * numOrbitals + orbIndex
                hop = H_SK[('s', orb)](l, m, n, V)
                A[index_i,index_j] += hop
                
                A[index_j, index_i] += hop"""
                
                

        # old code with sp3 hybridization 
        for atom_idx, atom in indexToAtom.items():
            hybridizationMatrix = self.H_sp3_explicit.copy() 

            for delta in unitCell.dangling_bonds(atom):
                signs = np.array([1/4] + list(delta)) * 4
                h = unit_cell.UnitCellGeneration.determine_hybridization(signs)
                hybridizationMatrix[h, h] += tbp.E['sp3']          # increase the energy of dangling states 
            
            # if there are no dangling bonds this returns the standard diag matrix with onsite energies 
            onsiteMatrix = self.U_orb_to_sp3 @ hybridizationMatrix @ self.U_orb_to_sp3.T # go back to orbital basis 
            
            A[atom_idx*10:atom_idx*10 + 4, atom_idx*10:atom_idx*10 + 4] = onsiteMatrix
            A[atom_idx*10 + 4:atom_idx*10 + 9, atom_idx*10 + 4:atom_idx*10 + 9] = np.eye(5) * tbp.E['dxy']
            A[atom_idx*10 + 9,atom_idx*10 + 9] = tbp.E['s*']
        
        for atom_index in range(numSilicon):
            atom = indexToAtom[atom_index]
            neighbors = unitNeighbors[atom]
            for orbitalIndex, orbital in enumerate(orbitals):
                index_i = atom_index * numOrbitals + orbitalIndex
                #effectiveZinPotential = int(atom.z * 4)
                
                #print(effectiveZinPotential)
                #print(potentialProfile)
                
                #A[index_i,index_i] += potentialProfile[effectiveZinPotential]
                
                
                for neighbor in neighbors.keys():
                    delta = neighbors[neighbor][0]
                    l,m,n = neighbors[neighbor][1:]
                    phase = np.exp(2 * np.pi * 1j * (kx*delta[0] + ky*delta[1])) # blochs theorem does not work 
                    
                    neighbor_index = atomToIndex[neighbor]       
                    
                    for secOrbitalIndex, secondOrbital in enumerate(orbitals):
                        index_j = neighbor_index * numOrbitals + secOrbitalIndex
                        
                        hop = tbp.SK[(orbital, secondOrbital)](l, m, n, tbp.V)
                            
                        A[index_i,index_j] += hop * phase   
                
                    
            dagger = lambda A: np.conjugate(A.T)
        if not np.allclose(A, dagger(A)):
            print("H isnt Hermitian")

        eigvals,eigv = np.linalg.eigh(A)
        return eigvals, A
    

    def create_tight_binding_sparse(self, k, N=1, potentialProfile=None, sigma=0.5, eigRange=10):
        kx,ky = k
        
        #print(N)
        
        unitCell = self.unitCell
        if potentialProfile is None:
            potentialProfile = unitCell.create_linear_potential(0)
        unitNeighbors = unitCell.neighborTable()
        hydrogens = unitCell.hydrogens
        
        
        
        numSilicon = len(unitNeighbors.keys())
        numHydrogen = len(hydrogens.keys()) 
    
        orbitals = ['s', 'px', 'py', 'pz', 'dxy','dyz','dzx','dx2y2','dz2', 's*']
        numOrbitals = len(orbitals)
        size = numSilicon * numOrbitals + numHydrogen * 0
        A = np.zeros((size, size), dtype=complex)    
        
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

            for delta in unitCell.dangling_bonds(atom):
                signs = np.array([1/4] + list(delta)) * 4
                h = unit_cell.UnitCellGeneration.determine_hybridization(signs)
                hybridizationMatrix[h, h] += tbp.E['sp3']          # increase the energy of dangling states 
        
            # if there are no dangling bonds this returns the standard diag matrix with onsite energies 
            onsiteMatrix = self.U_orb_to_sp3 @ hybridizationMatrix @ self.U_orb_to_sp3.T # go back to orbital basis 
            base = atom_idx * numOrbitals
            for i in range(4):
                for j in range(i ,4):
                    add(base + i, base + j, onsiteMatrix[i,j])
            for p in range(4, 9):                       # five d’s
                add(base + p, base + p, tbp.E['dxy'])
            add(base + 9, base + 9, tbp.E['s*'])
            
        for atom_index, atom in indexToAtom.items():
            base_i = atom_index * numOrbitals
            for neighbor, (delta, l, m, n) in unitNeighbors[atom].items():

                j = atomToIndex[neighbor]
                if j < atom_index:              # upper‑triangle filter
                    continue                    # let add() mirror it

                phase = np.exp(2j*np.pi*(kx*delta[0] + ky*delta[1]))

                for o1, orb1 in enumerate(orbitals):
                    for o2, orb2 in enumerate(orbitals):
                        hop = tbp.SK[(orb1, orb2)](l, m, n, tbp.V) * phase
                        add(base_i + o1, j*numOrbitals + o2, hop)

                
    
        H = sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
        eigvals, eigvecs = spla.eigsh(H, k=eigRange,
                                sigma=sigma, which='LM', tol=1e-6)


        return eigvals, H

    # create the k grid
    def make_mp_grid(self,Nk):
        """Return an (Nk3, 3) array of fractional k-vectors (0 … 1) in the 1st BZ."""
        shifts = np.linspace(0, 1, Nk, endpoint=False) + 0.5/Nk   
        klist  = np.array(list(product(shifts, repeat=2)))        

        return klist                                             



    # helper method 
    def frac_shift(self, k_frac, delta):
        return (k_frac + delta) % 1.0

    #  effective-mass tensor around the CBM
    def find_effective_mass(self, k_min_frac, Nk_coarse, band_idx,
                            resolution_factor=4, a=5.431e-10):


        delta_frac = 1.0 / (Nk_coarse * resolution_factor)        # we want a finer mesh size
        dk = (2*np.pi / a) * delta_frac                       


        k0 = np.asarray(k_min_frac, float)

        # get the good energy
        def E(k_frac):
            evs, _ = self.create_tight_binding(k_frac, N=self.N)
            return evs[band_idx]


        #shift 
        ei = np.eye(2)

        # Hessian 
        H = np.zeros((2,2))
        for i in range(2):
            # second derivative along axis i
            kp = self.frac_shift(k0,  +delta_frac * ei[i])
            km = self.frac_shift(k0,  -delta_frac * ei[i])
            H[i,i] = (E(kp) + E(km) - 2*E(k0)) / dk**2

            # mixed derivatives
            for j in range(i+1, 2):
                kpp = self.frac_shift(k0, +delta_frac*ei[i] + delta_frac*ei[j])
                kmm = self.frac_shift(k0, -delta_frac*ei[i] - delta_frac*ei[j])
                kpm = self.frac_shift(k0, +delta_frac*ei[i] - delta_frac*ei[j])
                kmp = self.frac_shift(k0, -delta_frac*ei[i] + delta_frac*ei[j])
                H[i,j] = H[j,i] = (E(kpp)+E(kmm)-E(kpm)-E(kmp)) / (4*dk**2)

        # convert eV → J
        H_J = H * spc.e

        # m*  = hbar^2 *
        mstar_SI = spc.hbar**2 * np.linalg.inv(H_J)           # kg
        mstar_me = mstar_SI / spc.m_e                         # in m_e

        prin_m, prin_axes = np.linalg.eigh(mstar_me)
        return mstar_me, prin_m, prin_axes

    def eval_k(self, k_frac):
        """return the good eigenvalues"""
        eigvals, _ = self.create_tight_binding(k_frac, self.N)    
        vbm = eigvals[eigvals <=  0.0].max()     
        cbm = eigvals[eigvals >=  0.0].min()
        return vbm, cbm, eigvals
    def scan_full_BZ(self,Nk=20, store_all=True, n_jobs=None, a=5.431e-10,
                    res_factor=4):
        """
        Nk       : number of k-points per reciprocal-lattice axis (Nk³ total)
        store_all: if True, return the entire E(k) array (size Nk³ × Nb)
        n_jobs   : cores to use; default = all available
        """
        
    
        klist = self.make_mp_grid(Nk)
        nbands = len(self.create_tight_binding(np.zeros(2), self.N)[0])   # quick probe
        
        dk = 1 / Nk
        # ----- parallel diagonalisation -----
        n_jobs = n_jobs or cpu_count()
        with Pool(processes=n_jobs) as pool:
            results = pool.map(self.eval_k, klist, chunksize=len(klist)//n_jobs)

        # collect extrema
        vbm_E, cbm_E = -np.inf, np.inf
        vbm_data = cbm_data = None
        if store_all:
            all_E = np.empty((len(klist), nbands))
            print(all_E.shape)

        for idx, (v, c, eigs) in enumerate(results):
            if v > vbm_E:
                vbm_E      = v
                vbm_data   = (v, klist[idx], int(np.where(eigs==v)[0][0]))
            if c < cbm_E:
                
                cbm_E      = c
                cbm_data   = (c, klist[idx], int(np.where(eigs==c)[0][0]))
            if store_all:
                all_E[idx] = eigs

        
        
        
        Egap = cbm_E - vbm_E
        print(f"Fundamental gap = {Egap:.4f} eV")
        print("VBM : E = {:.4f} eV  at k_frac = {}".format(*vbm_data[:2]))
        print("CBM : E = {:.4f} eV  at k_frac = {}".format(*cbm_data[:2]))
        print("Direct gap" if np.allclose(vbm_data[1], cbm_data[1]) else "Indirect gap")
        mstar, prin_m, prin_ax = self.find_effective_mass(cbm_data[1], Nk,
                                                    cbm_data[2],
                                                    resolution_factor=res_factor,
                                                    a=a)
        print("\nEffective-mass tensor at CBM:\n", mstar)
        print("Principal massesₑ:\n", prin_m)

        if store_all:
            return (Egap, vbm_data, cbm_data,
                    klist, all_E,
                    mstar, prin_m, prin_ax)
        return Egap, vbm_data, cbm_data, mstar, prin_m, prin_ax


 
        
    