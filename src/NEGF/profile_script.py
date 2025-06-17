import numpy as np
import multiprocessing
from itertools import product
import time

from device import Device
from rgf import GreensFunction
from hamiltonian import Hamiltonian

def calculate_wrapper(params):
    """
    A self-contained function for each worker process.
    Initializes objects and runs the calculation for one (energy, ky) pair.
    """
    energy, ky = params

    device = Device()
    ham = Hamiltonian(device)
    rgf = GreensFunction(device_state=device, ham = ham)

    result = rgf.sparse_rgf_G_R(energy, ky)
    print(f"Done with (E={energy:.2f}, ky={ky:.2f})") 
    
    return (energy, ky, result)

if __name__ == "__main__":
    energy_values = np.linspace(-3, 5, 100)
    ky_values = np.linspace(0, 1, 25)
    

    param_grid = list(product(energy_values, ky_values))
    

    num_processes = 32

    print(f"Starting {len(param_grid)} calculations using {num_processes} processes...")
    start_time = time.time()
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(calculate_wrapper, param_grid)
        
    end_time = time.time()
    
    print(f"\nCalculation finished.")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Successfully computed {len(results)} data points.")