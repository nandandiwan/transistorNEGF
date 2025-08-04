import os
import time
import numpy as np
import scipy.sparse as sp

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from hamiltonian import Hamiltonian
from rgf import GreensFunction
from plotter import Plotter

def test_single_device():
    """
    Test device with linear potential profile.
    """
    print("Testing Tunelling Device...")
    
    # Create 1D wire hamiltonian
    ham = Hamiltonian(name="one_d_wire", periodic=False)
    ham.N = 20  # Number of sites
    print(f"Created hamiltonian with {ham.get_num_sites()} sites")
    print(f"num_orbitals = {ham.num_orbitals}")
    
    # Set linear potential from source to drain (creates ballistic transport)
    ham.mu1 = 0.1
    ham.mu2 = 0.1
    ham.set_barrier_potential(positions=[10], height=0.5, width=3)  # 0.5V bias
    print("Set linear potential for ballistic transport")
    
    # Check if potential was set
    print(f"Potential is None: {ham.potential is None}")
    if ham.potential is not None:
        print(f"Potential length: {len(ham.potential)}")
        # Access sparse matrix properly
        print(f"First few potential values: {[ham.potential[i].toarray()[0,0] for i in range(min(5, len(ham.potential)))]}")
    
    gf = GreensFunction(hamiltonian=ham)  # Reduced for faster testing

    config = {
        "dos": False,
        "transmission": True,
        "Id-Vs": False,
        "Id-Vg": False,
        "potential_profile": True,
        "Vs_list": [0, 0.1, 0.2, .4,.5,1]  # Reduced for testing
    }

    Plotter.create_plots(ham, gf, config, test_name="single_device_debug")
    print("Ballistic device plots saved!")

if __name__ == "__main__":
    test_single_device()
