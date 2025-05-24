import unit_cell_generation as unitcellgeneration
import tight_binding_params as tbp
from tight_binding_params import E
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
        self.unitCell = unitcellgeneration.UnitCell(N) 
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
        
        # effective mass params
        self.Nk = 30
        self.a = 5.431e-10
       
        # Sparse settings
        self.sigma = 0.55 # start values of eigenvalues
        self.eigenRange = 10 # amount of computed eigenvalues (initially 10)
        self.cbm = {}
        self.vbm = {}
        self.cbmValue = [0,np.inf]
        self.vbmValue = [0, -np.inf]
        
    def create_tight_binding(self,k, N=1):
        kx,ky = k
        
        #print(N)
    
        unitNeighbors = self.unitCell.neighbors
        danglingBonds = self.unitCell.danglingBonds
        
        
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
                hybridizationMatrix[position,position] += E['sp3']            
            
            # if there are no dangling bonds this returns the standard diag matrix with onsite energies 
            onsiteMatrix = self.U_orb_to_sp3 @ hybridizationMatrix @ self.U_orb_to_sp3.T # go back to orbital basis 
            A[atom_idx*10:atom_idx*10 + 4, atom_idx*10:atom_idx*10 + 4] = onsiteMatrix
            A[atom_idx*10 + 4:atom_idx*10 + 9, atom_idx*10 + 4:atom_idx*10 + 9] = np.eye(5) * E['dxy']
            A[atom_idx*10 + 9,atom_idx*10 + 9] = E['s*']
    
        for atom_index in range(numSilicon):
            atom = indexToAtom[atom_index]
            neighbors = unitNeighbors[atom]
            for orbitalIndex, orbital in enumerate(orbitals):
                index_i = atom_index * numOrbitals + orbitalIndex
                for neighbor in neighbors:
                    atom2, delta, l,m,n = neighbor
                    phase = np.exp(2 * np.pi * 1j * (kx*delta[0] + ky*delta[1])) # blochs theorem does not work 
                    neighbor_index = atomToIndex[atom2]      
                    for secOrbitalIndex, secondOrbital in enumerate(orbitals):
                        index_j = neighbor_index * numOrbitals + secOrbitalIndex

                        hop = tbp.SK[(orbital, secondOrbital)](l, m, n, tbp.V)
                    
                        A[index_i,index_j] += hop * phase   
        
            dagger = lambda A: np.conjugate(A.T)
        if not np.allclose(A, dagger(A)):
            print("H isnt Hermitian")

        eigvals,eigv = np.linalg.eigh(A)
        return eigvals, eigv
 
    def create_tight_binding_sparse(self,k, N=1, potentialProfile=None, sigma=0.5, eigRange=10):
        
    
        kx, ky = k
        #print(N)
        unitNeighbors = self.unitCell.neighbors
        danglingBonds = self.unitCell.danglingBonds
        
        
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
                hybridizationMatrix[position,position] += E['sp3']            
            
            # if there are no dangling bonds this returns the standard diag matrix with onsite energies 
            onsiteMatrix = self.U_orb_to_sp3 @ hybridizationMatrix @ self.U_orb_to_sp3.T # go back to orbital basis 
        
            base = atom_idx * numOrbitals
            for i in range(4):
                for j in range(i ,4):
                    add(base + i, base + j, onsiteMatrix[i,j])
            for p in range(4, 9):                       # five d’s
                add(base + p, base + p, E['dxy'])
            add(base + 9, base + 9, E['s*'])
        
    
        for atom_index, atom in indexToAtom.items():
            base_i = atom_index * numOrbitals
            for atom2, delta, l,m,n in unitNeighbors[atom]:
                j = atomToIndex[atom2]
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


        order = np.argsort(eigvals.real)   
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        return eigvals, eigvecs

        
    def make_mp_grid(self, Nk, centred=True):
        if centred:
            # symmetric points:  -½, -(½-1/Nk), …, +(½-1/Nk)
            half_step = 0.5 / Nk
            shifts = np.arange(-0.5, 0.5, 1.0 / Nk) + half_step
        else:
            # conventional Γ‑centred grid on [0,1)
            shifts = np.arange(Nk) / Nk

        klist = np.array(list(product(shifts, repeat=2)))  # shape (Nk**2, 2)
        return klist   
                                            
    # helper method 
    def frac_shift(self, k_frac, delta):
        return (k_frac + delta) #% 1.0
    def _sparse_eval(self, k, sigma, m):
        """Return eigenvalue array (sorted) for a given sigma and range m."""
        eigenvalues, eigenvectors = self.create_tight_binding_sparse(k, self.N,
                                                 sigma=sigma,
                                                 eigRange=m)
        return np.asarray(eigenvalues), np.asarray(eigenvectors)
    
  
    def analyzeEnergyRange(self, k, energies=None, effectiveMassCalc=True,
                           max_iter=25, grow=5, σ_step=0.2):

        if energies is None:
            energies, eigenvectors = self._sparse_eval(k, self.sigma, self.eigenRange)
       
        for _ in range(max_iter):

            has_pos = np.any(energies > 0)
            has_neg = np.any(energies < 0)
            if has_pos and has_neg:
                # collect extrema
                min_pos_idx = np.argmin(energies[energies > 0])
                max_neg_idx = np.argmax(energies[energies < 0])

                min_positive = energies[energies > 0][min_pos_idx]
                max_negative = energies[energies < 0][max_neg_idx]

                # bookeeping
                if not effectiveMassCalc:
                    self.cbm[tuple(k)] = [min_positive, self.sigma, self.eigenRange]
                    if min_positive < self.cbmValue[1]:
                        print(k, min_positive)
                        self.cbmValue = [k, min_positive]

                    self.vbm[tuple(k)] = [max_negative, self.sigma, self.eigenRange]
                    if max_negative > self.vbmValue[1]:        # less negative is “larger”
                        self.vbmValue = [k, max_negative]

                # --------- prepare next k‑point search ----------------------
                self.sigma = 0.5 * (min_positive + max_negative)
                self.eigenRange = 5                           # reset to a lean window

    
                if effectiveMassCalc:
               
                    return min_positive
                return min_positive, max_negative                             
            band_min, band_max = energies.min(), energies.max()

            
            # adaptive sigma
            if not has_pos:     
                self.sigma += max(σ_step, 0.5*abs(band_min))  
            elif not has_neg:   
                self.sigma -= max(σ_step, 0.5*abs(band_max))

            # widen the window a little every time we fail
            self.eigenRange += grow
            energies = self._sparse_eval(k, self.sigma, self.eigenRange)

        # ---------- could not bracket zero within max_iter -------------------
        print(f"[warn] unable to bracket E=0 at k={k} after {max_iter} trials")
        if effectiveMassCalc:
            return None          # caller must handle this case
        
    def getMinimum(self,k_frac):
        return self.analyzeEnergyRange(k_frac, effectiveMassCalc = True)
        
    def eval_k_sparse(self, k_frac, effMass = True):
        eigenvalues, eigenvectors = self.create_tight_binding_sparse(np.array([0,0]), sigma=self.sigma, eigRange= self.eigenRange)
        self.analyzeEnergyRange(k_frac, eigenvalues, effectiveMassCalc = effMass)
    
    def determineInitialSparseSettings(self):
        k = np.array([0,0])
        eigenvalues, eigenvectors = self.create_tight_binding_sparse(np.array([0,0]), sigma=self.sigma, eigRange= self.eigenRange)
        self.analyzeEnergyRange(k, eigenvalues)
    
    def scan_full_BZ(self, Nk=51, store_all=True, n_jobs=None, a=5.431e-10, res_factor=4):
        self.Nk = Nk
        klist = self.make_mp_grid(Nk)
        #print(klist)
        for k in klist:
            self.eval_k_sparse(k, effMass=False)
    
    def calculateEffectiveMass(self, startk = np.array([0,0]), resolution=4):
        """
        This code finds the effective mass using a parabolic approximation. 
        It needs to be changed for non-parabolicity
        """
        def E(k_frac):
            evs = self.analyzeEnergyRange(k_frac,effectiveMassCalc=True)
            #print(evs, k_frac)
            return evs

        delta_frac = 1.0 / (self.Nk * resolution)
        dk = (2 * np.pi / self.a) * delta_frac

        k0 = np.asarray(startk, float)
        #print(f"this the min {E(k0)}")
        ei = np.eye(2)
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

        H_J = H * spc.e
        mstar_SI = spc.hbar**2 * np.linalg.inv(H_J)
        mstar_me = mstar_SI / spc.m_e
        eigvals, eigvecs = np.linalg.eigh(mstar_me)  
        principal_masses = eigvals       

        return principal_masses
    
    def getCbmValues(self, k, tol=1e-9): 
      
        sigma = self.sigma
       
        eigRange = self.eigenRange
        evals, evecs = self._sparse_eval(np.asarray(k, float), sigma, eigRange)

        # first eigenvalue strictly above zero (within tolerance)
        pos = np.where(evals > tol)[0]
        if pos.size == 0:
            raise RuntimeError(f"No positive eigenvalue found at k={k} "
                            f"(sigma={sigma}, eigRange={eigRange}). "
                            "Increase eigRange or adjust sigma.")
        idx = pos[0]             

        return evals[idx], evecs[:, idx]