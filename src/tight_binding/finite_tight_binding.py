import unit_cell_generation as unitcellgeneration
import tight_binding_params as tbp
from tight_binding_params import E
import numpy as np
import scipy.constants as spc
from itertools import product
import os, multiprocessing as mp
from joblib import Parallel, delayed
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray
from math import isclose
import time
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
        self.base_sigma =  0.55
        self.eigenRange = 10 # amount of computed eigenvalues (initially 10)
        self.cbm = {}
        self.vbm = {}
        self.cbmValue = [0,np.inf]
        self.vbmValue = [0, -np.inf]
        
    def create_tight_binding(self,k, N=1, getMatrix = False):
        kx, ky = 4 / np.sqrt(2)* k

        
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
        
    
        kx, ky =4 / np.sqrt(2)* k

        
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
        """
        This code changes the sigma and eigenrange to find the cbm/vbm values
        It is meant to be used over large k space calculations - not single time uses (sigma changes everytime)
        Once this method is done using use setBaseSigma() to get sigma back to normal value 
        
        Single use energy calcultations use 
        """

        if energies is None:
            energies, eigenvectors = self._sparse_eval(k, self.sigma, self.eigenRange)
       
        
        for _ in range(max_iter):
            energies = np.sort(energies)
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

                # next kpt search
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
            energies, eigenvectors = self._sparse_eval(k, self.sigma, self.eigenRange)
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
    
    def setBaseSigma(self):
        self.sigma = self.base_sigma
    
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
        dk = (2 * np.pi / self.a * 4 / np.sqrt(2)) * delta_frac

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
    
    
    def vectorizedFirstCrossing(self, v, *, limit=THREE_KBT_300K,
                                n1: int = 150, chunk: int = 16,
                                parallel: bool = False,
                                E0_cbm: float | None = None,
                                E0_vbm: float | None = None):
        """
        Return the first k along +v/√2 where CBM rises by ≥limit
        and the first where VBM falls by ≤−limit.

        If E0_cbm / E0_vbm are supplied we skip an extra Γ calculation.
        """

        v = np.asarray(v, float)
        nrm = np.linalg.norm(v)
        if nrm == 0.0:
            raise ValueError("direction vector v must be non‑zero")
        v /= nrm
        k_max = v / np.sqrt(2.0)

        # --- reference energies at Γ (if not supplied) ------------------
        if E0_cbm is None or E0_vbm is None:
            E0_cbm, _, E0_vbm, _ = self.getBandValues(k=np.zeros(2))

        # -- (code identical to your latest version, but uses E0_cbm/E0_vbm) --
        # design k path
        t_samples = np.linspace(0.0, 1.0, n1, endpoint=False)
        k_samples = t_samples[:, None] * k_max

        def _solve_many(k_block):
            cbms = np.empty(len(k_block))
            vbms = np.empty(len(k_block))
            for i, kvec in enumerate(k_block):
                cbms[i], _, vbms[i], _ = self.getBandValues(kvec)
            return cbms, vbms

        if parallel:
            _runner = Parallel(n_jobs=-1, prefer="processes", batch_size=chunk)
            def _solve_many(k_block):
                vals = _runner(delayed(self.getBandValues)(k)
                            for k in k_block)
                cbms = np.fromiter((x[0] for x in vals), float, len(k_block))
                vbms = np.fromiter((x[2] for x in vals), float, len(k_block))
                return cbms, vbms

        first_c, first_v = None, None
        for start in range(0, n1, chunk):
            k_block = k_samples[start:start + chunk]
            cbm_blk, vbm_blk = _solve_many(k_block)

            up_mask   = (cbm_blk - E0_cbm) >= limit
            down_mask = (vbm_blk - E0_vbm) <= -limit

            if first_c is None and up_mask.any():
                first_c = k_block[up_mask.argmax()]
            if first_v is None and down_mask.any():
                first_v = k_block[down_mask.argmax()]
            if first_c is not None and first_v is not None:
                break

        return {"cbm_cross": first_c, "vbm_cross": first_v}
    def _fit_paraboloid_H(points):
        pts  = np.asarray(points, float)
        k2   = pts[:, 0]**2 + pts[:, 1]**2        # kx² + ky²
        α    = np.dot(k2, pts[:, 2]) / np.dot(k2, k2)
        return np.diag([2*α, 2*α])                # H = 2α I (2×2)

    def _bracket_and_bisect(self, k0, v, *,
                            limit: float = THREE_KBT_300K,
                            initial_step: float = 1e-4,
                            k_max: float = 0.5,
                            max_iter: int = 32,
                            tol: float = 1e-6):

        v = np.asarray(v, float)
        nrm = np.linalg.norm(v)
        if nrm == 0.0:
            raise ValueError("direction vector v must be non‑zero")
        v /= nrm

        
        E0_cbm, _, E0_vbm, _ = self.getBandValues(k=k0)

        # helpers
        def eval_edge(t):
            """Return (cbm, vbm) at k = k0 + t·v."""
            cbm, _, vbm, _ = self.getBandValues(k=k0 + t * v)
            return cbm, vbm

        hi = initial_step
        cbm_lo = vbm_lo = 0.0
        cbm_hi = vbm_hi = None   # hi endpoint once bracketed

        while hi <= k_max and (cbm_hi is None or vbm_hi is None):
            cbm, vbm = eval_edge(hi)

            if cbm_hi is None and cbm - E0_cbm >= limit:
                cbm_hi = hi
            if vbm_hi is None and E0_vbm - vbm >= limit:
                vbm_hi = hi

            if cbm_hi is None: cbm_lo = hi
            if vbm_hi is None: vbm_lo = hi
            hi *= 2.0

        k_cbm = k_vbm = None
        if cbm_hi is None and vbm_hi is None:
            # neither edge reached the threshold within k_max
            return {'cbm_cross': None, 'vbm_cross': None}

        if cbm_hi is not None:
            lo, hi = cbm_lo, cbm_hi
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                cbm_mid, _ = eval_edge(mid)
                err = cbm_mid - E0_cbm - limit
                if isclose(err, 0.0, abs_tol=tol):
                    break
                lo, hi = (mid, hi) if err < 0 else (lo, mid)
            k_cbm = k0 + mid * v

  
        if vbm_hi is not None:
            lo, hi = vbm_lo, vbm_hi
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                _, vbm_mid = eval_edge(mid)
                err = E0_vbm - vbm_mid - limit
                if isclose(err, 0.0, abs_tol=tol):
                    break
                lo, hi = (mid, hi) if err < 0 else (lo, mid)
            k_vbm = k0 + mid * v

        return {'cbm_cross': k_cbm, 'vbm_cross': k_vbm}
 
    def calculateNonParabolicEffectiveMass(self,
                                        k0=np.zeros(2),
                                        *, band='conduction',
                                        limit=THREE_KBT_300K,
                                        symmetry: bool = True,
                                        verbose: bool = False):
        """
        Five‑point effective mass from a ΔE paraboloid fit.

        If `symmetry=True` we exploit the xy isotropy
        giving the required 4 + 1 points with a single crossing search.
        """
        E0_cbm, _, E0_vbm, _ = self.getBandValues(k=np.zeros(2))

        pts = []                              

        if symmetry:
            # ---- single crossing along +x --------------------------------
            cross_dict = self._bracket_and_bisect(k0=np.zeros(2), v=[1,0])

            k_cross = cross_dict['cbm_cross'] if band == 'conduction' \
                    else           cross_dict['vbm_cross']

            if k_cross is None:
                raise RuntimeError(f"No {band} crossing found along (1,0)")

            dk  = k_cross - k0
            r   = abs(dk[0])                

       
            pts.extend([[+r, 0, limit],
                        [-r, 0, limit],
                        [0, +r, limit],
                        [0, -r, limit]])
            print(pts)

            if verbose:
                direc = 'CBM' if band == 'conduction' else 'VBM'
                print(f"{direc} ΔE={limit:.3g} eV at |dk| = {r:.4e} (symmetry applied)")

        else:
            # ---- explicit ±x ±y evaluation -------------------------------
            directions = ((1,0), (-1,0), (0,1), (0,-1))
            for v in directions:
                cd = self._bracket_and_bisect(k0=np.zeros(2), v=v)
                k_cross = cd['cbm_cross'] if band == 'conduction' else cd['vbm_cross']
                if k_cross is None:
                    raise RuntimeError(f"No {band} crossing found along {v}")

                dk = k_cross - k0
                pts.append([dk[0], dk[1], limit])

                if verbose:
                    print(f"{band} cross {v}: k = {k_cross}, |dk| = {np.linalg.norm(dk):.3e}")


        pts.append([0.0, 0.0, 0.0])
        H = TightBindingHamiltonian._fit_paraboloid_H(pts)

        dk_dq = (2 * np.pi / self.a) * (4 / np.sqrt(2))   
        H_J   = H * spc.e / dk_dq**2
        mstar = spc.hbar**2 * np.linalg.inv(H_J) / spc.m_e
        return np.sort(np.linalg.eigvalsh(mstar))
   
            
    def getBandValues(self, k: NDArray[np.float64], tol: float = 1e-9):
        """This method doesn't change sigma and finds the energy values using first order perturbation theory"""
        def _band_edges(evals: NDArray, vecs: NDArray):
            """Extract CBM/VBM from *shifted* eigenvalues `evals`."""
            pos_mask = evals >  tol         
            neg_mask = evals < -tol
            if not (pos_mask.any() and neg_mask.any()):
                raise RuntimeError("eigenRange must straddle the band gap")

            cbm_idx = pos_mask.argmax()               
            vbm_idx = len(evals) - 1 - neg_mask[::-1].argmax()  
            return (float(evals[cbm_idx]), vecs[:, cbm_idx],
                    float(evals[vbm_idx]), vecs[:, vbm_idx])

        base_sigma = self.sigma        
        eig_range  = self.eigenRange

        try:
            # gamma point
            evals_g, vecs_g = self._sparse_eval(np.zeros(2), base_sigma, eig_range)
            cbmE_g, cbmV_g, vbmE_g, vbmV_g = _band_edges(evals_g - base_sigma,
                                                        vecs_g)

            self.modifySigmaForDeltaK(np.zeros(2), k,
                                    cbmE_g + base_sigma, cbmV_g,
                                    vbmE_g + base_sigma, vbmV_g)

            # target k 
            sigma = self.sigma        
            evals_k, vecs_k = self._sparse_eval(k, sigma, eig_range)
            cbmE_k, cbmV_k, vbmE_k, vbmV_k = _band_edges(evals_k - sigma, vecs_k)

            return np.real(cbmE_k + sigma), cbmV_k, np.real(vbmE_k + sigma), vbmV_k

        finally:
            self.setBaseSigma()
    
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
    def modifySigmaDeltaH(self, deltaH,cbmEv,cbmVec, vbmEv, vbmVec):
        """PT method to change sigma values"""
        delta1 = np.conjugate(cbmVec) @ deltaH @ cbmVec
        delta2 = np.conjugate(vbmVec) @ deltaH @ vbmVec
        cbmEv, vbmEv = cbmEv + delta1, vbmEv + delta2
        self.sigma = 0.5 * (cbmEv + vbmEv)
        
    
    def modifySigmaForVoltage(self, cbmEv,cbmVec, vbmEv, vbmVec):

        
        potMatrix = self.getChangeInVoltage()
        self.modifySigmaDeltaH(potMatrix,cbmEv,cbmVec, vbmEv, vbmVec)
        
    def modifySigmaForDeltaK(self,k0,deltak,cbmEv,cbmVec, vbmEv, vbmVec):
        Ap = self.create_tight_binding(k0 + deltak,getMatrix=True)
        A0 = self.create_tight_binding(k0,getMatrix=True)
        deltaA = Ap - A0
        self.modifySigmaDeltaH(deltaA,cbmEv,cbmVec, vbmEv, vbmVec)
        

    def calculateFermiEnergy(self):
        m1 = self.calculateNonParabolicEffectiveMass(
             k0=[0,0], band='valence', symmetry=True)[0]
        m2 = self.calculateNonParabolicEffectiveMass(
             k0=[0,0], band='conduction', symmetry=True)[0]
        cbm,cbmVec, vbm, vbmVec = self.getBandValues(np.zeros(2))
        return 0.5 * (cbm + vbm) + 3 / 4 * TightBindingHamiltonian.THREE_KBT_300K * np.log(m1 / m2)
        
                
    
        
        
