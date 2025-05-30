import unit_cell_generation as unitcellgeneration
import tight_binding_params as tbp
from tight_binding_params import E
import numpy as np
import scipy.constants as spc
from itertools import product
from multiprocessing import Pool, cpu_count
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from math import isclose

class TightBindingHamiltonian:
    THREE_KBT_300K = 0.07719080174
    def __init__(self, N=None, thickness = None):
        
        self.H = None
        self.N = N
        self.thickness = thickness
        if self.N:
            self.unitCell = unitcellgeneration.UnitCell(N) 
        else:
            self.unitCell = unitcellgeneration.UnitCell(thickness=thickness)
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
        
    def create_tight_binding(self,k, N=1, getMatrix = False):
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
            A[atom_idx * 10: atom_idx *10 +10, atom_idx * 10: atom_idx *10 +10] += np.eye(10) * self.unitCell.ATOM_POTENTIAL[atom]
    
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
            
        if getMatrix:
            return A

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
            atomPotential = self.unitCell.ATOM_POTENTIAL[atom]
        
            onsiteMatrix = self.U_orb_to_sp3 @ hybridizationMatrix @ self.U_orb_to_sp3.T + np.eye(4) * atomPotential 
            base = atom_idx * numOrbitals
            for i in range(4):
                for j in range(i ,4):
                    add(base + i, base + j, onsiteMatrix[i,j])
        
            for p in range(4, 9):                       # five d’s
                add(base + p, base + p, E['dxy']  + atomPotential)
            add(base + 9, base + 9, E['s*'] + atomPotential)
        
    
        for atom_index, atom in indexToAtom.items():
            base_i = atom_index * numOrbitals
            for atom2, delta, l,m,n in unitNeighbors[atom]:
                j = atomToIndex[atom2]
                if j < atom_index:             
                    continue                   

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
        #print(H)
        H_J = H * spc.e
        
        mstar_SI = spc.hbar**2 * np.linalg.inv(H_J)
        mstar_me = mstar_SI / spc.m_e
        eigvals, eigvecs = np.linalg.eigh(mstar_me)  
        principal_masses = eigvals       

        return principal_masses
  
    #TODO
    def _bracket_and_bisect(self, k0, v, limit=THREE_KBT_300K,
                            initial_step=1e-4, k_max=0.5,
                            max_bisect_iter=32, tol=1e-6):
        """
        This code is supposed to pinpoint the k point at which cbm has increased by 3kbT. 
        Right now there is error finding this point. Code needs to work for arbitrary direction 
        """
        
        base_sigma = self.sigma 
        
        v = np.asarray(v, float)
        norm = np.linalg.norm(v)
        print(v, norm)
        if norm == 0.0:
            raise ValueError("direction vector v must be non‑zero")
        v /= norm                          # normalise once

        # reference CBM energy at k0
        E0, _,vbm,_ = self.getBandValues(k=k0)
        oldCBM = E0
        oldVBM = vbm
        

        def energyValues(t):
            """E0 − E(k0 + t\cdot v)."""
            cbm,_,vbm,_ = self.getBandValues(k=k0 + t * v)
            return cbm,vbm
            
        t_low, f_low = 0.0, 0.0           # ΔE = 0 at t = 0
        t = initial_step
        while t <= k_max:
            cbm,vbm = energyValues(t)
            self.sigma += 0.5 * ((cbm - oldCBM) + (vbm - oldVBM))
            oldCBM = cbm
            oldVBM = vbm
    
            f = cbm - E0
            if f >= limit:   
                print(f)          
                break
            t_low, f_low = t, f
            t *= 2                        
        else:
            return None                 

        t_high, f_high = t, f       
        # now close the range 
        for _ in range(max_bisect_iter):
            t_mid = 0.5 * (t_low + t_high)
            cbm,vbm = energyValues(t)
            self.sigma += 0.5 * ((cbm - oldCBM) + (vbm - oldVBM))
            oldCBM = cbm
            oldVBM = vbm
            f_mid = cbm - E0

            if isclose(f_mid, limit, rel_tol=0, abs_tol=tol):
                return k0 + t_mid * v

            if f_mid < limit:             # still below the target
                t_low, f_low = t_mid, f_mid
            else:
                t_high, f_high = t_mid, f_mid

        self.sigma = base_sigma
        return k0 + t_high * v
    
    def first_crossing_3kBT(self, k0=np.zeros(2), directions=[[1, 0], [0, 1]],
                        **kwargs):
        """
        For each direction in `directions` (iterable of 2‑D vectors) return the
        first k‑point where deltaE >= 3kBT.  Yields (v, k_cross) pairs; k_cross is None
        if the threshold is never reached before hitting the search limit.
        """
        k0 = np.asarray(k0, float)
        for v in directions:
            yield np.asarray(v, float), self._bracket_and_bisect(k0, v, **kwargs)
            
            
    def calculateNonParabolicEffectiveMass(self,
        k0=np.asarray([0.0, 0.0]),
        samplingVectors=[[1, 0], [-1, 0], [0, 1], [0, -1]],
        step_fraction=0.5,
        fallback_step=1e-4):
        """Finds effective mass tensor based on the sampling vectors"""
        
        E0, *_ = self.getBandValues(k=k0)
        crossings = {tuple(v): kc for v, kc in
                    self.first_crossing_3kBT(k0=k0, directions=samplingVectors)}
        
        print(crossings)
        
        points = []
        for direction, kvec in crossings.items():         
            point = np.concatenate([kvec,                
                                    [TightBindingHamiltonian.THREE_KBT_300K]])  
            points.append(point)

        
        
        points.append(np.asarray([k0[0], k0[1], 0]))      #

        points = np.asarray(points, dtype=float)

        def fit_paraboloid_and_hessian(points, include_linear=True):

            pts = np.asarray(points, dtype=float)
            kx, ky, E = pts.T

            if include_linear:
                A = np.column_stack([kx**2 + ky**2, kx, ky, np.ones_like(kx)])
                alpha, beta_x, beta_y, E0 = np.linalg.lstsq(A, E, rcond=None)[0]
            else:
                A = np.column_stack([kx**2 + ky**2, np.ones_like(kx)])
                alpha, E0 = np.linalg.lstsq(A, E, rcond=None)[0]
                beta_x = beta_y = 0.0

            H = np.array([[2*alpha, 0.0],
                        [0.0,     2*alpha]])

            return H

        H = fit_paraboloid_and_hessian(points) 
        
        
        H_J = H * spc.e / (2 * np.pi / self.a)**2
        mstar_SI = spc.hbar**2 * np.linalg.inv(H_J)
        mstar_me = mstar_SI / spc.m_e
        eigvals, eigvecs = np.linalg.eigh(mstar_me)  
        principal_masses = eigvals       

        return principal_masses
            
       
    
    def getBandValues(self, k, earlierk = np.asarray([0,0]), tol=1e-9): 
        """Redundant method (analyzeEnergyRange exists) that does not expand range or sigma - better for larger systems  
        for energy range but also gives cbm - test with PT to make it more effecient 
        
        1. test if cbm exists in range 
        2. if so return, if not half k point and do 1
        3. if so use pt theory to find expected delta E cbm and vbm from doubling k 
        4. modify sigma and go back to original k point
        5. make sigma back to original value and return valid cbm/vbm values
        """
      
        sigma     = self.sigma
        eigRange  = self.eigenRange
  
        evals, evecs = self._sparse_eval(np.asarray(k, float), sigma, eigRange)
        evals -= np.float64(sigma)

        pos = np.where(evals >  tol)[0]
        if pos.size == 0:
            
            
            raise RuntimeError(
                f"No positive eigenvalue found at k={k} "
                f"(sigma={sigma}, eigRange={eigRange}). "
                "Increase eigRange or adjust sigma."
            )
        cbm_idx  = pos[0]
        cbm_E    = float(evals[cbm_idx])
        cbm_vec  = evecs[:, cbm_idx]

        neg = np.where(evals < -tol)[0]
        if neg.size == 0:
            raise RuntimeError(
                f"No negative eigenvalue found at k={k} "
                f"(sigma={sigma}, eigRange={eigRange}). "
                "Increase eigRange or adjust sigma."
            )
        vbm_idx  = neg[-1]
        vbm_E    = float(evals[vbm_idx])
        vbm_vec  = evecs[:, vbm_idx]

        return cbm_E + np.float64(sigma), cbm_vec, vbm_E + np.float64(sigma), vbm_vec
    
    
    def getCBM(self, k = np.asarray([0,0])):
        cbm,_,_,_ = self.getBandValues(k)

    def get_potential_matrix(self):
        """This function returns the matr"""

        ORBITALS = ('s', 'px', 'py', 'pz',
                'dxy', 'dyz', 'dzx', 'dx2y2', 'dz2', 's*')
        NUM_ORBITALS = len(ORBITALS)
        
        atom_order  = list(self.unitCell.neighbors)      
        V_atom      = np.asarray(
            [self.unitCell.ATOM_POTENTIAL[a] for a in atom_order],
            dtype=float)


        V_diag = np.repeat(V_atom, NUM_ORBITALS)

        return np.diag(V_diag)
    
    def getChangeInVoltage(self):
        ORBITALS = ('s', 'px', 'py', 'pz',
        'dxy', 'dyz', 'dzx', 'dx2y2', 'dz2', 's*')
        NUM_ORBITALS = len(ORBITALS)
        
        atom_order  = list(self.unitCell.neighbors)      
        V_atom      = np.asarray(
            [self.unitCell.ATOM_POTENTIAL[a] for a in atom_order],
            dtype=float)


        V_diag = np.repeat(V_atom, NUM_ORBITALS)
    
        
        if self.unitCell.OLD_POTENTIAL is None:
            V_diag2 = np.zeros_like(V_diag)
        else:
            V_atom      = np.asarray(
            [self.unitCell.OLD_POTENTIAL[a] for a in atom_order],
            dtype=float)
            V_diag2 = np.repeat(V_atom, NUM_ORBITALS)
        self.unitCell.OLD_POTENTIAL = self.unitCell.ATOM_POTENTIAL
        A = np.diag(V_diag) - np.diag(V_diag2)
        #
        return A
    def modifySigmaForVoltage(self, cbmEv,cbmVec, vbmEv, vbmVec):
        """This function uses perturbation theory to guess the sigma for the sparse solver"""
        
        potMatrix = self.getChangeInVoltage()
        
        delta1 = np.conjugate(cbmVec) @ potMatrix @ cbmVec
        delta2 = np.conjugate(vbmVec) @ potMatrix @ vbmVec
        
        cbmEv, vbmEv = cbmEv + delta1, vbmEv + delta2
        self.sigma = 0.5 * (cbmEv + vbmEv)
        
    def modifySigmaForDeltaK(self,k0,deltak,cbmEv,cbmVec, vbmEv, vbmVec):
        Ap = self.create_tight_binding(k0 + deltak,getMatrix=True)
        A0 = self.create_tight_binding(k0)
        deltaA = Ap - A0
        delta1 = np.conjugate(cbmVec) @ deltaA @ cbmVec
        delta2 = np.conjugate(vbmVec) @ deltaA @ vbmVec
        
        cbmEv, vbmEv = cbmEv + delta1, vbmEv + delta2
        self.sigma = 0.5 * (cbmEv + vbmEv)
        
        
    
        
        
     
      