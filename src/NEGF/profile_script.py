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
    
    # === SOLUTION: INITIALIZE HEAVY OBJECTS HERE ===
    # Each process creates its own lightweight instances from scratch.
    # This avoids the massive data transfer from the main process.
    device = Device()
    rgf = GreensFunction(device_state=device)
    # ===============================================
    
    # Now run the calculation
    result = rgf.sparse_rgf_G_R(energy, ky)
    print(f"Done with (E={energy:.2f}, ky={ky:.2f})") # More informative print
    
    # Return inputs with the result for easy tracking
    return (energy, ky, result)

if __name__ == "__main__":
    # 1. Define the parameter space
    energy_values = np.linspace(-3, 5, 2)
    ky_values = np.linspace(0, 1, 5)
    
    # Create a list of all (energy, ky) pairs
    param_grid = list(product(energy_values, ky_values))
    

    num_processes = 2
    print(f"Machine has {num_processes} CPU cores.")
    # =========================================================
    
    print(f"Starting {len(param_grid)} calculations using {num_processes} processes...")
    start_time = time.time()
    
    # 2. Run the multiprocessing pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(calculate_wrapper, param_grid)
        
    end_time = time.time()
    
    print(f"\nCalculation finished.")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Successfully computed {len(results)} data points.")