import os
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from itertools import product
import multiprocessing
import numpy as np
import scipy.sparse as sp
from scipy.linalg import inv 
import scipy.constants as spc
import warnings
from scipy.optimize import brentq # Import the brentq root-finder

from rgf import GreensFunction
from hamiltonian import Hamiltonian
from lead_self_energy import LeadSelfEnergy
from utils import smart_inverse  
from utils import sparse_diag_product

from scipy.optimize import brentq # Make sure this import is at the top of your file

def brent_dekker_fermi_level(gf: GreensFunction, n_NEGF: np.array, LDOS: np.array, V: np.array, Ec: float = None):
    """
    Calculates the quasi-fermi level E_Fn at each site using Brent's method.

    This implementation is OPTIMIZED to avoid re-calculating the LDOS in every
    iteration of the root-finder. It uses a pre-calculated LDOS matrix.

    Parameters:
        gf (GreensFunction): 
            The object containing NEGF parameters like the energy grid and temperature.
        n_NEGF (numpy.array): 
            The target charge density at each site from the full NEGF calculation. Shape: (num_sites,).
        LDOS (numpy.array): 
            The PRE-CALCULATED local density of states. 
            Shape: (num_energy_points, num_sites).
        V (numpy.array): 
            The electrostatic potential at each site. Shape: (num_sites,).
        Ec (float, optional): 
            The conduction band edge. Integration starts from the grid point closest to Ec.
    """
    num_sites = V.shape[0]
    EFN = np.zeros(num_sites)
    
    energy_grid = gf.energy_grid
    dE = gf.dE
    kbT = gf.ham.kbT_eV

    if Ec is not None:
        start_idx = np.abs(energy_grid - Ec).argmin()
    else:
        start_idx = 0
        
    def calculate_n_equilibrium_fast(efn_trial, site_idx):
        ldos_at_site = LDOS[start_idx:, site_idx]
        potential_at_site = V[site_idx]
        integration_energies = energy_grid[start_idx:]

        # Core logic from your n_worker
        exp_arg = (integration_energies - potential_at_site - efn_trial) / kbT
        exp_arg = np.clip(exp_arg, -100, 100)
        fermi_dist = 1.0 / (1.0 + np.exp(exp_arg))
        
        integrand = 2 * ldos_at_site * fermi_dist
        
        return np.trapezoid(integrand, dx=dE)

    print("Starting Optimized Brent's solver for Quasi-Fermi Level...")
    for i in range(num_sites):
        target_n = n_NEGF[i]

        residual_function = lambda efn: calculate_n_equilibrium_fast(efn, i) - target_n

        # Bracket the root [a, b]
        a = V[i] - 20 * kbT
        b = V[i] + 20 * kbT
        
        try:
            # Find the root for site i
            efn_root = brentq(residual_function, a, b, xtol=1e-6, rtol=1e-6)
            EFN[i] = efn_root
        except ValueError:
            warnings.warn(f"Brent's method failed for site {i}. f(a) and f(b) may have the same sign.",
                          RuntimeWarning)
            EFN[i] = V[i] # Simple fallback
    
    print("Quasi-Fermi Level calculation complete.")
    return EFN