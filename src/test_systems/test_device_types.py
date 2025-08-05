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
    This reproduces the ballistic device from Fig. 9.5.10(a).
    """
    print("Testing Ballistic Device...")
    
    # Create 1D wire hamiltonian
    ham = Hamiltonian(name="one_d_wire", periodic=False)
    ham.N = 20  # Number of sites
    print(f"Created hamiltonian with {ham.get_num_sites()} sites")
    
    # Set linear potential from source to drain (creates ballistic transport)
    # This creates a smooth potential drop from left to right
    
    print("Set linear potential for ballistic transport")
    
    # Create Green's function with energy range around Fermi level
    gf = GreensFunction(hamiltonian=ham, energy_grid=np.linspace(-.1, .8, 301))

    # Configuration for plots - focus on transmission and I-V
    config = {
        "dos": False,
        "transmission": True,
        "Id-Vs": True,
        "Id-Vg": False,
        "potential_profile": True,
        "Vs_list": [0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Bias voltages as in the figure
    }

    Plotter.create_plots(ham, gf, config, test_name="ballistic_device")
    print("Ballistic device plots saved!")

def test_tunneling_device():
    """
    Test tunneling device with single high barrier.
    This reproduces the tunneling device from Fig. 9.5.10(b).
    """
    print("Testing Tunneling Device...")
    
    # Create 1D wire hamiltonian
    ham = Hamiltonian(name="one_d_wire", periodic=False)
    ham.N = 46  # Number of sites
    
    # Create a single high barrier in the middle
    N = ham.get_num_sites()
    barrier_pos = N // 2  # Middle of the device
    barrier_height = 0.4  # High barrier for tunneling
    barrier_width = 4     # Width of barrier
    
    ham.set_barrier_potential(positions=barrier_pos, height=barrier_height, width=barrier_width)
    
    # Create Green's function
    gf = GreensFunction(hamiltonian=ham, energy_grid=np.linspace(-.1, 1, 301))

    # Configuration for plots
    config = {
        "dos": False,
        "transmission": True,
        "Id-Vs": True,
        "Id-Vg": False,
        "potential_profile": True,
        "Vs_list": [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

    Plotter.create_plots(ham, gf, config, test_name="tunneling_device")
    print("Tunneling device plots saved!")

def test_resonant_tunneling_device():
    """
    Test resonant tunneling device with double barrier structure.
    This reproduces the resonant tunneling device from Fig. 9.5.10(c).
    """
    print("Testing Resonant Tunneling Device...")
    
    # Create 1D wire hamiltonian
    ham = Hamiltonian(name="one_d_wire", periodic=False)
    ham.N = 46  # Longer device for double barrier structure
    
    # Create double barrier with quantum well
    N = ham.get_num_sites()
    barrier1_pos = N // 3      # First barrier position
    barrier2_pos = 2 * N // 3 -4 # Second barrier position
    barrier_height = 0.4       # Barrier height
    barrier_width = 4          # Width of each barrier
    well_depth = -0.0          # Slight well between barriers (optional)
    
    ham.set_double_barrier_potential(
        barrier1_pos=barrier1_pos,
        barrier2_pos=barrier2_pos,
        barrier_height=barrier_height,
        barrier_width=barrier_width,
        well_depth=well_depth
    )
    
    # Create Green's function
    gf = GreensFunction(hamiltonian=ham, energy_grid=np.linspace(-.1, 1, 301))

    # Configuration for plots
    config = {
        "dos": False,
        "transmission": True,
        "Id-Vs": True,
        "Id-Vg": False,
        "potential_profile": True,
        "Vs_list": [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

    Plotter.create_plots(ham, gf, config, test_name="resonant_tunneling_device")
    print("Resonant tunneling device plots saved!")

def test_custom_potential_example():
    """
    Example of creating custom potential using direct potential list manipulation.
    This shows how to use the manual method you described.
    """
    print("Testing Custom Potential Example...")
    
    ham = Hamiltonian(name="one_d_wire", periodic=False)
    ham.N = 20
    
    # Manual potential setting (your method)
    N = ham.get_num_sites()
    pot = [sp.eye(1) * 0.0] * N  # Initialize with zeros
    
    # Add barriers manually
    pot[N//3] += sp.eye(1) * 0.3       # First barrier
    pot[N//3 + 1] += sp.eye(1) * 0.3   # Extend first barrier
    
    pot[2 * N//3] += sp.eye(1) * 0.5   # Second barrier
    pot[2 * N//3 + 1] += sp.eye(1) * 0.5  # Extend second barrier
    
    ham.potential = pot  # Set directly
    ham.set_linear_potential()
    
    gf = GreensFunction(hamiltonian=ham, energy_grid=np.linspace(-2, 2, 300))

    config = {
        "dos": False,
        "transmission": True,
        "Id-Vs": True,
        "Id-Vg": False,
        "potential_profile": True,
        "Vs_list": [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

    Plotter.create_plots(ham, gf, config, test_name="custom_potential_example")
    print("Custom potential device plots saved!")

def run_all_device_tests():
    """Run all device tests."""
    print("="*50)
    print("Running Device Type Tests")
    print("="*50)
    
    try:
        # Test each device type
        test_ballistic_device()
        print("-"*30)
        
        test_tunneling_device()
        print("-"*30)
        
        test_resonant_tunneling_device()
        print("-"*30)
        
        test_custom_potential_example()
        print("-"*30)
        
        print("All device tests completed!")
        print("Check the 'plots' directory for results.")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_device_tests()
