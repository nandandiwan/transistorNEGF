import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root) 
from src.tight_binding import tight_binding_params as TBP
import numpy as np
import scipy.sparse as sp

class Hamiltonian:
    def __init__(self, name):
        self.name = name
        self.t = 1
        self.o = 0
        self.Vs = 0
        self.N = 1
        self.Vd = 0
        self.kbT_eV = 1.38e-23/(1.609e-19)
        

    def one_d_wire(self, t, o, blocks=True):
        if blocks:
            return ([sp.eye(1) *o]  *self.N), ([sp.eye(1) * t ]* (self.N-1))
        else:
            A = np.zeros((self.N, self.N))
            for i in range(self.N):
                if i < self.N - 1:
                    A[i, i + 1] = self.t
                    A[i + 1, i] = self.t
                A[i,i] = self.o
            
            return sp.csc_matrix(A)     
    
    
    def get_H00_H01_H10(self, t, o):
        """
        Get H00, H01, and H10 matrices for both leads with proper symmetry.
        For symmetric DOS at zero bias, both leads should have identical structure.
        """
        return sp.eye(1) *o,sp.eye(1) *t,sp.eye(1) *t
    
    