import os

# --- Add these lines at the very top of your script ---
# This must be done BEFORE importing numpy or other scientific libraries.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import multiprocessing
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product
from lead_self_energy import LeadSelfEnergy
from device import Device
import numpy as np
import scipy as sp
import scipy.sparse as spa
import scipy.sparse.linalg as spla
from poisson import PoissonSolver
from rgf import GreensFunction
from hamiltonian import Hamiltonian
import time

class Charge():
    def __init__(self, device : Device):
        self.device = device
        
        self.poisson = PoissonSolver(device)
        self.ham = Hamiltonian(device)
        self.GF = GreensFunction(device, self.ham)
        self.lse = LeadSelfEnergy(self.device, self.ham)
        
        self.energy_range = np.linspace(-5,5, 200)
        self.k_space = np.linspace(0,1,32) # exact number of processes 
        
        self.atoms = self.ham.unitCell.ATOM_POSITIONS # the atoms         
        
        # These are lists that give quasi fermi energy and potential at location of every atom, needs to be initialized every time 
        self.unsmearedEFN = None
        self.unsmearedPhi = None
    
        self.weights = {}
    
    
    def calculate_real_GR(self, E):
        """This function returns real space greens function"""
        G_R_list = self.gf_calculations_k_space(E)
        Nk = len(G_R_list)
        G_R_REAL = np.zeros_like(G_R_list[0])
        for G_R in G_R_list:
            G_R_REAL += G_R
        G_R_REAL = G_R_REAL / Nk       
        return G_R_REAL
        

    def gf_calculations_k_space(self, E) -> list:
        """Uses multiprocessing to cache GF for E,ky """
        
        param_grid = list(product(E, self.k_space))

        print(f"Starting DOS calculations for {len(param_grid)} (E, ky) pairs...")
        print(f"ky range: {self.k_space[0]:.2f} to {self.k_space[-1]:.2f}")

        start_time = time.time()

        with multiprocessing.Pool(processes=32) as pool:
            results = pool.map(self._calculate_gf_simple, param_grid)
        end_time = time.time()
        return results
    
    def _calculate_gf_simple(self, param):
        """Simple Green's function calculation"""
        energy, ky = param
        eta = 1e-6
        H = self.ham.create_sparse_channel_hamlitonian(ky, blocks=False)
        
        # Add self-energies at contacts
        lse = self.lse
        sl = lse.self_energy(side="left",E= energy,ky= ky)
        sr = lse.self_energy(side="right", E=energy,ky= ky)
        
        # Add to boundary blocks
        H[:sl.shape[0], :sl.shape[0]] += sl
        H[-sr.shape[0]:, -sr.shape[0]:] += sr
        
        # Construct and solve for Green's function
        E_complex = energy + 1j * eta
        H_gf = spa.csc_matrix(np.eye(H.shape[0], dtype=complex) * E_complex) - H
        I = spa.csc_matrix(np.eye(H.shape[0], dtype=complex))
        
        # Solve for diagonal elements only (more memory efficient)
        G_R = spla.spsolve(H_gf, I)
        if spa.issparse(G_R):
            G_R = G_R.diagonal()
        else:
            G_R = np.diagonal(G_R)
        
        return G_R
    
    def calculate_LDOS(self, E):
        """Returned unsmeared version of LDOS"""
        G_R_REAL = self.calculate_real_GR(E)
        LDOS_points = {}
        for atom_idx, atom in enumerate(self.atoms):
            ldos_contribution =  -1 / np.pi * np.sum(G_R_REAL[atom_idx * 10, (atom_idx + 1) * 10]).imag
            LDOS_points[atom.getPos()] = ldos_contribution
        
        return np.asarray(LDOS_points.values())
    
    def fermi(self, E, mod="False"):
        EFN, Phi = self.unsmearedEFN, self.unsmearedPhi
        if mod:
            raise NotImplemented
        else:
            return 1 /(1 + np.exp((E*np.ones_like(EFN) - Phi - EFN) / self.device.kbT))
    
    def compute_n_helper(self, E):
        LDOS_points = self.calculate_LDOS(E)
        
        # all are numpy arrays
        return LDOS_points * self.fermi(E)
            