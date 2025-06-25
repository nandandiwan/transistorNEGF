from device import Device
import numpy as np
from poisson import PoissonSolver
from rgf import GreensFunction
from hamiltonian import Hamiltonian
class Solve:
    """wrapper class to self consistently solve poisson equation and NEGF equations"""
    
    def __init__(self, device : Device):
        self.device = device
        
        self.poisson = PoissonSolver(device)
        self.ham = Hamiltonian(device)
        self.GF = GreensFunction(device, self.ham)
        
        
    def gf_calculations_k_space(self) -> list:
        """Uses multiprocessing to return """
        return 
    
     

    
    