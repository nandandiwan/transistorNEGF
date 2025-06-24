import os

# --- Add these lines at the very top of your script ---
# This must be done BEFORE importing numpy or other scientific libraries.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product
import time
from device import Device
from rgf import GreensFunction
from hamiltonian import Hamiltonian


def calculate_trace_wrapper(param):
    """
    Calculates the summed complex trace for one (E, ky) pair.
    """
    energy, ky = param

    device = Device()
    ham = Hamiltonian(device)
    rgf = GreensFunction(device_state=device, ham=ham)
    
    # sparse_rgf_G_R is expected to return a tuple, where the first element is a list of matrices
    G_R_list, _, _, _, _ = rgf.sparse_rgf_G_R(energy, ky)
    complex_trace = 0.0 + 0.0j
    for gr_matrix in G_R_list:
        try:
            complex_trace += gr_matrix.trace()
        except:
    
            raise Exception("error with gr")
    
    return (energy, complex_trace)


if __name__ == "__main__":

    energy_values = np.linspace(-10, 15, 100)
    ky_values = np.linspace(0, 1.0, 32)
    dk = ky_values[1] - ky_values[0] # The differential k_y step
    
    param_grid = list(product(energy_values, ky_values))
    
    print(f"Starting calculations for {len(param_grid)} (E, ky) pairs...")
    start_time = time.time()


    with multiprocessing.Pool(processes=32) as pool: #

        results = pool.map(calculate_trace_wrapper, param_grid)
        
    end_time = time.time()
    print(f"Calculations finished in {end_time - start_time:.2f} seconds.")

    # Aggregate traces for each energy value
    energy_traces = defaultdict(complex)
    for energy, trace in results:
        energy_traces[energy] += trace

    dos_energies = sorted(energy_traces.keys())
    
    dos_values = [(-1.0 / np.pi) * np.imag(energy_traces[E]) * (dk / (2 * np.pi)) for E in dos_energies]

    # Plotting the Density of States
    plt.figure(figsize=(10, 6))
    plt.plot(dos_energies, dos_values, label='Density of States D(E)')
    plt.xlabel('Energy (E)')
    plt.ylabel('Density of States D(E)')
    plt.title('Density of States as a function of Energy')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("dos")
    plt.show()