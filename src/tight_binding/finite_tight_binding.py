import tight_binding.unit_cell_generation as unit_cell
import tight_binding.tight_binding_params as tbp
import numpy as np
import numpy as np
import scipy.constants as spc
from itertools import product
from multiprocessing import Pool, cpu_count


class TightBindingHamiltonian:
    def __init__(self, N):
        self.H = None
        self.N = None
        self.unitCell = unit_cell.UnitCellGeneration(N)
        self.potentialProfile = self.unitCell.create_linear_potential(0)
        self.U_orb_to_sp3 = 0.5*np.array([[1, 1, 1, 1],
                             [1, 1,-1,-1],
                             [1,-1, 1,-1],
                             [1,-1,-1, 1]])
        
        a = (tbp.Es + 3*tbp.Ep)/4.0
        b = (tbp.Es -   tbp.Ep)/4.0
        self.H_sp3_explicit = np.full((4,4), b)
        
        
    def create_Hamiltonian(self, potentialProfile):
        def create_TB_Hamiltonian(k):
            kx,ky = k
            unitNeighbors = self.unitCell.neighborTable()
            hydrogens = self.unitCell.hydrogens
            
                
    
            numSilicon = len(unitNeighbors.keys())
            numHydrogen = len(hydrogens.keys()) * 0
        
            orbitals = ['s', 'px', 'py', 'pz', 'dxy','dyz','dzx','dx2y2','dz2', 's*']
            numOrbitals = len(orbitals)
            size = numSilicon * numOrbitals + numHydrogen * 1
            A = np.zeros((size, size), dtype=complex)    
            
            atomToIndex = {}
            indexToAtom = {}
            for atom_index,atom in enumerate(unitNeighbors):
                atomToIndex[atom] = atom_index
                indexToAtom[atom_index] = atom
        
  
            for atom_idx, atom in indexToAtom.items():
                hybridizationMatrix = self.H_sp3_explicit.copy() 

                for delta in self.unitCell.dangling_bonds(atom):
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
            
            return A
        return create_TB_Hamiltonian
    def solve_TB_hamiltonian(self,k):
        H_func = self.create_Hamiltonian(self.potentialProfile)
        A = H_func(k)
        eigvals,eigv = np.linalg.eigh(A)
        return eigvals
    
    def setLinearPotential(self, V):
        self.potentialProfile = self.unitCell.create_linear_potential(V)
    
    def setGeneralPotential(self, newPotentialProfile):
        self.potentialProfile = newPotentialProfile

    # create the k grid 
    def make_mp_grid(self,Nk):
        """Return an (Nk3, 3) array of fractional k-vectors (0 … 1) in the 1st BZ."""
        shifts = np.linspace(0, 1, Nk, endpoint=False) + 0.5/Nk   
        klist  = np.array(list(product(shifts, repeat=2)))        

        return klist                                             

    def eval_k(self,k_frac):
        """return the good eigenvalues"""
        eigvals = self.solve_TB_hamiltonian(k_frac)    
        vbm = eigvals[eigvals <=  0.0].max()     
        cbm = eigvals[eigvals >=  0.0].min()
        return vbm, cbm, eigvals

    # helper method 
    def frac_shift(self,k_frac, delta):
        return (k_frac + delta) % 1.0

    #  effective-mass tensor around the CBM
    def effective_mass_helper(self, k_min_frac, Nk_coarse, band_idx,
                            resolution_factor=4, a=5.431e-10):


        delta_frac = 1.0 / (Nk_coarse * resolution_factor)        # we want a finer mesh size
        dk = (2*np.pi / a) * delta_frac                       


        k0 = np.asarray(k_min_frac, float)

        # get the good energy
        def E(k_frac):
            evs = self.solve_TB_hamiltonian(k_frac)
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


    def effectiveMass(self, Nk=20, store_all=True, n_jobs=None, a=5.431e-10,
                    res_factor=4):
        """
        Nk       : number of k-points per reciprocal-lattice axis (Nk³ total)
        store_all: if True, return the entire E(k) array (size Nk³ × Nb)
        n_jobs   : cores to use; default = all available
        """
        klist = self.make_mp_grid(Nk)
        nbands = len(self.solve_TB_hamiltonian(np.zeros(2)))   # quick probe
        
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
        self.gap = Egap
        mstar, prin_m, prin_ax = self.effective_mass_helper(cbm_data[1], Nk,
                                                    cbm_data[2],
                                                    resolution_factor=res_factor,
                                                    a=a)
        return mstar



            