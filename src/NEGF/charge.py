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
from scipy import linalg
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
        self.smearedEFN = np.zeros((self.device.nx * self.device.nz)) # EFN has same dimension as poisson solver 
        self.smearedPhi = self.device.potential
        self.smearedLDOS = None # TODO
    
        self.weights = {}
        
        
    def unsmear_to_smear(self, A : dict):
        """performs interpolation to get values that match up with poisson matrix (DO THIS AT END OF MP FOR ALL CALCULATIONS)"""
        max_X = self.device.unitX / 4
        max_Z = self.device.unitZ 
        nx, nz = self.device.nx, self.device.nz
        
        # Create regular grid coordinates
        x_grid = np.linspace(0, max_X, nx)
        z_grid = np.linspace(0, max_Z, nz)
        X_grid, Z_grid = np.meshgrid(x_grid, z_grid, indexing='ij')
        
        # Extract coordinates and values from dictionary (ignoring y-component)
        points = []
        values = []
        for coord, value in A.items():
            x, y, z = coord  
            points.append([x, z])
            values.append(value)
        
        points = np.array(points)
        values = np.array(values)
        
        # Create target grid points for interpolation
        grid_points = np.column_stack([X_grid.ravel(), Z_grid.ravel()])
        
        from scipy.interpolate import griddata
        interpolated_values = griddata(points, values, grid_points, method='nearest', fill_value=0.0)
        smeared_array = interpolated_values.reshape((nx, nz))
        
        return smeared_array
        
        
        
    
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
        G_R_diag, Gamma_L, Gamma_R = self.GF.compute_central_greens_function(energy, ky, compute_lesser=False)
        
        return G_R_diag
    
    def calculate_LDOS(self, E) -> dict:
        """Returned unsmeared version of LDOS"""
        G_R_REAL = self.calculate_real_GR(E)
        LDOS_points = {}
        for atom_idx, atom in enumerate(self.atoms):
            ldos_contribution =  -1 / np.pi * np.sum(G_R_REAL[atom_idx * 10, (atom_idx + 1) * 10]).imag
            LDOS_points[atom.getPos()] = ldos_contribution
        
        return LDOS_points
    
    def calculate_smeared_LDOS(self, E):
        LDOS_points = self.calculate_LDOS(E)
        return self.unsmear_to_smear(LDOS_points)
        
    
    def fermi(self, E, mod="False"):
        EFN, Phi = self.smearedEFN, self.smearedPhi
        if mod:
            raise NotImplemented
        else:
            return 1 /(1 + np.exp((E*np.ones_like(EFN) - Phi - EFN) / self.device.kbT))
    
    def compute_EFN_helper(self, E):
        """This will be part for the bracket bisect way of finding the EFN
        , we take EC as 1.2 for now (note even if 'EC' is actually in the bandgap it 
        makes no diff since LDOS is zero in bandgap)
        
        this function also uses vectorization to find EFN in entire device at the same time 
        """
        LDOS_points = self.calculate_LDOS(E)
        
        # all are numpy arrays
        return LDOS_points * self.fermi(E)
    
    def calculate_EC(self):
        """Use calculate_DOS to find the EC. utilize TB code from before"""
        raise NotImplemented
    
    def calculate_DOS(self, energy_range=None, ky_range=None, method="sancho_rubio", 
                     eta=1e-6, save_data=True, filename="dos_data.txt"):
        """
        Calculate Density of States (DOS) using robust surface Green's functions.
        
        Args:
            energy_range: Energy points for DOS calculation
            ky_range: Transverse momentum points  
            method: Surface GF method ("sancho_rubio", "iterative", "transfer")
            eta: Small imaginary part for broadening
            save_data: Whether to save DOS data to file
            filename: Output filename for DOS data
            
        Returns:
            energies, total_dos: Energy points and corresponding DOS values
        """
        if energy_range is None:
            energy_range = np.linspace(-2.0, 2.0, 200)
        if ky_range is None:
            ky_range = np.linspace(0, 1, 32)
            
        print(f"Calculating DOS with {len(energy_range)} energy points and {len(ky_range)} ky points")
        print(f"Using {method} method for surface Green's functions")
        
        # Calculate DOS for each energy point
        total_dos = np.zeros(len(energy_range))
        
        start_time = time.time()
        
        for i, energy in enumerate(energy_range):
            dos_ky = []
            for ky in ky_range:
                try:
                    dos_val = self._calculate_dos_point(energy, ky, method, eta)
                    dos_ky.append(dos_val)
                except Exception as e:
                    print(f"Error at E={energy:.3f}, ky={ky:.3f}: {e}")
                    dos_ky.append(0.0)
            
            # Average over ky points
            total_dos[i] = np.mean(dos_ky)
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                eta_remaining = elapsed * (len(energy_range) - i - 1) / (i + 1)
                print(f"Progress: {i+1}/{len(energy_range)} ({100*(i+1)/len(energy_range):.1f}%), "
                      f"ETA: {eta_remaining:.1f}s")
        
        end_time = time.time()
        print(f"DOS calculation completed in {end_time - start_time:.2f} seconds")
        
        if save_data:
            # Save DOS data
            dos_data = np.column_stack((energy_range, total_dos))
            np.savetxt(filename, dos_data, header="Energy(eV)  DOS(states/eV)", 
                      fmt='%.6e', delimiter='\t')
            print(f"DOS data saved to {filename}")
        
        return energy_range, total_dos
    
    def _calculate_dos_point(self, energy, ky, method="sancho_rubio", eta=1e-6):
        """
        Calculate DOS at a single (E, ky) point using robust surface Green's functions.
        Based on OpenMX TRAN implementation for symmetry preservation.
        
        Returns:
            float: DOS value at this energy point
        """
        try:
            # Create device Hamiltonian
            H = self.ham.create_sparse_channel_hamlitonian(ky, blocks=False)
            
            # Calculate self-energies using the cleaned implementation
            lse = self.lse
            
            # Use symmetric treatment for zero bias
            if abs(self.device.Vs) < 1e-10 and abs(self.device.Vd) < 1e-10:
                # For zero bias, ensure symmetric treatment
                sl = lse.self_energy(side="left", E=energy, ky=ky, method=method)
                sr = lse.self_energy(side="right", E=energy, ky=ky, method=method)
            else:
                # For finite bias, include voltage effects
                sl = lse.self_energy(side="left", E=energy, ky=ky, method=method)
                sr = lse.self_energy(side="right", E=energy, ky=ky, method=method)
            
            # Add self-energies to Hamiltonian boundaries
            H_total = H.copy()
            
            # Left contact self-energy
            sl_size = min(sl.shape[0], H.shape[0])
            H_total[:sl_size, :sl_size] += sl[:sl_size, :sl_size]
            
            # Right contact self-energy  
            sr_size = min(sr.shape[0], H.shape[0])
            H_total[-sr_size:, -sr_size:] += sr[-sr_size:, -sr_size:]
            
            # Construct and solve for Green's function
            E_complex = energy + 1j * eta
            H_gf = spa.csc_matrix(np.eye(H_total.shape[0], dtype=complex) * E_complex) - H_total
            
            # Calculate DOS from trace of imaginary part of Green's function
            # DOS(E) = -1/π * Im[Tr(G_R(E))]
            try:
                # For large matrices, compute trace more efficiently
                G_diag = spla.spsolve(H_gf, np.ones(H_total.shape[0], dtype=complex))
                dos_value = -np.sum(np.imag(G_diag)) / np.pi
            except:
                # Fallback: solve full matrix
                I = spa.identity(H_total.shape[0], dtype=complex, format='csc')
                G_R = spla.spsolve(H_gf, I)
                if spa.issparse(G_R):
                    dos_value = -np.imag(G_R.diagonal().sum()) / np.pi
                else:
                    dos_value = -np.imag(np.trace(G_R)) / np.pi
            
            return float(dos_value.real) if np.isfinite(dos_value) else 0.0
        
        except Exception as e:
            print(f"Error in DOS calculation at (E={energy:.3f}, ky={ky:.3f}): {e}")
            return 0.0
    
    
    def calculate_n(self):
        """This finds Gn and then integrates over E and k. This entire function needs to be revamped to handle MP"""
        self.GF.compute_electron_density()
        # do multiprocessing to sum over E and k 
        # then smear 
        n_unsmeared = self.dosomething()
        self.n = n_unsmeared
        return self.unsmear_to_smear(n_unsmeared)