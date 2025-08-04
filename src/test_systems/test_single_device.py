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

def test_ballistic_device():
    """
    Test ballistic device with linear potential profile.
    """
    print("Testing Ballistic Device...")
    
    # Create 1D wire hamiltonian
    ham = Hamiltonian(name="one_d_wire", periodic=False)
    ham.N = 20  # Number of sites
    print(f"Created hamiltonian with {ham.get_num_sites()} sites")
    print(f"num_orbitals = {ham.num_orbitals}")
    
    # Set linear potential from source to drain (creates ballistic transport)
    ham.set_linear_potential(V_start=0.0, V_end=-0.5)  # 0.5V bias
    print("Set linear potential for ballistic transport")
    
    # Check if potential was set
    print(f"Potential is None: {ham.potential is None}")
    if ham.potential is not None:
        print(f"Potential length: {len(ham.potential)}")
        # Access sparse matrix properly
        print(f"First few potential values: {[ham.potential[i].toarray()[0,0] for i in range(min(5, len(ham.potential)))]}")
    
    # Create Green's function with energy range around Fermi level
    gf = GreensFunction(hamiltonian=ham, energy_grid=np.linspace(-2, 2, 50))  # Reduced for faster testing

    # Configuration for plots - focus on transmission and I-V
    config = {
        "dos": False,
        "transmission": False,
        "Id-Vs": False,
        "Id-Vg": False,
        "potential_profile": True,
        "Vs_list": [0, 0.1, 0.2]  # Reduced for testing
    }

    Plotter.create_plots(ham, gf, config, test_name="ballistic_device_debug")
    print("Ballistic device plots saved!")

if __name__ == "__main__":
    test_ballistic_device()
