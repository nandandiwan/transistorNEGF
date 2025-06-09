import numpy as np
import scipy.constants as spc
from hamiltonian import Hamiltonian
class Device:
    def __init__(self):
        # Physical constants
        self.T = 300  # Kelvin
        self.kbT = spc.Boltzmann * self.T
        self.q = spc.e
        self.hbar = spc.hbar
        self.m = 0.45 * spc.m_e
        self.epsl = 8.854e-12 * 3.3
        self.epox = 8.854e-12 * 3.9
        self.a = 5.431e-10
        # hamiltonian params
        self.block_width = 0.75 * self.a 
        self.block_height = 0.75 * self.a
        
        # device parameters
        self.channel_length = 10e-9
        self.channel_thickness = 3e-9
        self.unitX = int(self.channel_length // self.block_width)
        self.unitZ = int(self.channel_thickness // self.block_height)
        
        # voltage 
        self.VG = 0
        self.Vs = 0
        self.Vd = 0
        
        # solver 
        self.hamiltonian = Hamiltonian(self.unitX, self.unitZ)