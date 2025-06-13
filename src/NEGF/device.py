import numpy as np
import scipy.constants as spc
from hamiltonian import Hamiltonian
class Device:
    def __init__(self, channel_length = 10e-9, channel_thickness = 3e-9):
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
        self.block_width = 0.25 * self.a 
        self.block_height = 0.75 * self.a
        
        # device parameters
        self.channel_length = channel_length
        self.channel_thickness = channel_thickness
        self.unitX = int(self.channel_length // self.block_width)
        self.unitZ = int(self.channel_thickness // self.block_height)
        
        # voltage 
        self.VG = 0
        self.Vs = 0
        self.Vd = 0
        
        # silicon parameters
        self.epsilon_FE = self.default_material_params['epsilon_FE']  # FE background permittivity 
        self.epsilon_IL = self.default_material_params['epsilon_IL']  # Interfacial Layer Dielectric permittivity 
        self.epsilon_Si = self.default_material_params['epsilon_Si']  # Silicon permittivity 
        self.epsilon_Spacer = self.default_material_params['epsilon_Spacer']  # Spacer dielectric permittivity 
        self.epsilon_Insulator = self.default_material_params['epsilon_Insulator']  # Spacer dielectric permittivity 
        
        self.N_donor = self.default_material_params['N_donor']  # Drain Donor Doping concentration in m^-3
        self.N_acceptor = self.default_material_params['N_acceptor']  # Channel Acceptor Doping concentration in m^-3
        self.N_c = self.default_material_params['N_c']
        self.N_v = self.default_material_params['N_v']
        self.n_i = self.default_material_params['n_i']
        self.E_g = self.default_material_params['E_g']

        self.delta_E_0 = 0.5 * self.k_B * self.T_0 / self.e_charge * np.log(self.N_c/ self.N_v)
        self.E_c = self.E_g/2 + self.delta_E_0
        self.E_v = -self.E_g/2 + self.delta_E_0

        self.E_d = self.E_c - self.default_material_params['E_d']
        self.E_a = self.E_v + self.default_material_params['E_a']
        
        
        # solver 
        self.hamiltonian = Hamiltonian(self.unitX, self.unitZ)
        
        # poisson solver within channel params
        self.resolution = 1
        self.Nz = (4 * self.unitZ + 1) + 4 * self.unitZ * (self.resolution - 1) 
        self.Nx = 4 * self.unitX + (4 * self.unitZ - 1)* (self.resolution - 1)
        self.dx = self.channel_length / (self.Nx - 1)
        self.dz = self.channel_thickness / (self.Nz - 1)
        
        
        
        self.Ec = np.zeros((self.Nx, self.Nz))
        self.Q0 = np.zeros_like(self.Ec)
        
        
        
        self.electron_affinity = np.zeros((self.Nx, self.Nz))
        self.n_matrix = np.zeros_like(self.Ec)
        self.p_matrix = np.zeros_like(self.Ec)
        self.Epsilon = np.zeros((self.Nx - 1, self.Nz - 1))
        
        self.NA = np.zeros((self.Nx, self.Nz))
        self.ND = np.zeros((self.Nx, self.Nz))
        
        self.Efn = np.zeros_like(self.Ec)
        
    def setVoltages(self, Vg, Vd, Vs):
        self.Vg = Vg
        self.Vd = Vd
        self.Vs = Vs
        