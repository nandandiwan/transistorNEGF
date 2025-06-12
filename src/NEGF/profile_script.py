import numpy as np
import multiprocessing
from itertools import product
import time

# Assume these classes are defined in their respective files
from device import Device
from rgf import GreensFunction

def calculate_wrapper(params):
    """
    A self-contained function for each worker process.
    Initializes objects and runs the calculation for one (energy, ky) pair.
    """
    energy, ky = params
    
    # Each process must create its own instances of the classes.
    device = Device()
    rgf = GreensFunction(device_state=device)
    
    result = rgf.sparse_rgf_G_R(energy, ky)
    print("done 1")
    
    # Return inputs with the result for easy tracking
    return (energy, ky, result)

if __name__ == "__main__":
    # 1. Define the parameter space
    energy_values = np.linspace(-3, 5, 100)
    ky_values = np.linspace(0, 1, 25)
    
    # Create a list of all (energy, ky) pairs
    param_grid = list(product(energy_values, ky_values))
    
    num_processes = 32
    
    print(f"Starting {len(param_grid)} calculations using {num_processes} processes...")
    start_time = time.time()
    
    # 2. Run the multiprocessing pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # The map function distributes the tasks and collects the results
        results = pool.map(calculate_wrapper, param_grid)
        
    end_time = time.time()
    
    print(f"\nCalculation finished.")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Successfully computed {len(results)} data points.")
    
    # The 'results' variable is a list of tuples: [(e1, ky1, res1), (e2, ky2, res2), ...]
    # You can now process or save this data as needed.