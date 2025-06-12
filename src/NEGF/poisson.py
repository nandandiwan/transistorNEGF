from device import Device
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import convolve
import scipy.constants as spc


class Poisson:
    def __init__(self, device : Device):
        self.ds = device
        self.dx = self.ds.dx
        self.dz = self.ds.dz
        self.Nx = self.ds.Nx
        self.Nz = self.ds.Nz
        
        n_points = self.Nx * self.Nz


        
    def init_Poisson(self):
        def index(i,j):
            return i + self.Nx
        
        Ec = self.ds.Ec
        epsilon = self.ds.Epsilon
        n = self.ds.n_matrix
        p = self.ds.p_matrix
        NA = self.ds.NA
        ND = self.ds.ND
        ratio = self.dx / self.dz
        
        i_list_for_coo_matrix = []
        j_list_for_coo_matrix = []
        data_list_for_coo_matrix = []
        n_points = self.Nx * self.Nz
        
        
        G = np.zeros(n_points)
        
        for j in range(self.Nz):
            for i in range(self.Nx):
                idx = index(i,j)
    
                if i == 0:
                    i_list_for_coo_matrix.append(idx)
                    j_list_for_coo_matrix.append(idx)
                    data_list_for_coo_matrix.append(1.0)
                    G[idx] = self.ds.VG
        
                elif i == self.Nx-1:
                    i_list_for_coo_matrix.append(idx)
                    j_list_for_coo_matrix.append(idx)
                    data_list_for_coo_matrix.append(1.0)
                    G[idx] = self.ds.Vd
                    
                elif j == 0:  # device bottom side 
          
                    ''' floating substrate (Neumaan BC):  '''               
                    i_list_for_coo_matrix.extend([idx, idx,])
                    j_list_for_coo_matrix.extend([index(i, j+1), idx,])
                    data_list_for_coo_matrix.extend(
                        [1,
                            -1]
                    ) 
                    G[idx] = 0
                elif j == self.Nz-1 :  # device top side, Neumann BC
                    i_list_for_coo_matrix.extend([idx, idx,])
                    j_list_for_coo_matrix.extend([index(i, j-1), idx,])
                    data_list_for_coo_matrix.extend(
                        [1,
                            -1]
                    )                     
                    G[idx] = 0
                else:
                    i_list_for_coo_matrix.extend([idx, idx, idx, idx, idx])
                    j_list_for_coo_matrix.extend([index(i+1, j), index(i, j+1), index(i-1, j), index(i, j-1), idx])
                    data_list_for_coo_matrix.extend(
                        [(epsilon[i, j] + epsilon[i,j-1])/2 / ratio**2, 
                         (epsilon[i-1, j] + epsilon[i,j])/2,
                         (epsilon[i-1, j] + epsilon[i-1,j-1])/2 / ratio**2,
                         (epsilon[i, j-1] + epsilon[i-1,j-1])/2, 
                         -(epsilon[i,j] + epsilon[i-1, j] + epsilon[i-1, j-1] + epsilon[i, j-1])/2 * (1+1/ratio**2)]
                    )

                    G[idx] = 0 # RHS of Poisson equation

        A = sp.coo_matrix((data_list_for_coo_matrix, (i_list_for_coo_matrix, j_list_for_coo_matrix)))

        # transform matrix to CSR representation
        A = A.tocsr()

        # solve this linear system
        Init_V = spla.spsolve(A, G)

        # reshape the solution vector to matrix
        Init_V = Init_V.reshape((self.Nz, self.Nx)).T

        return Init_V

    def delta_Poisson(self):
        def index(i,j):
            return i + self.Nx
        Ec = self.ds.Ec
        epsilon = self.ds.Epsilon
        n = self.ds.n_matrix
        p = self.ds.p_matrix
        NA = self.ds.NA
        ND = self.ds.ND
        rho = self.ds.q * (self.p - self.n + self.NA - self.ND)
        rho = rho.copy()
        epsilon = self.epsilon.copy()
        Ec = self.Ec.copy()
        
        
        ratio = self.dx / self.dz
        for j in range(self.Nz):
            for i in range(self.Nx):
                idx = index(i,j)
    
                if i == 0:
                        self.A[idx, index(i+1, j)] = 1
                        self.A[idx, idx] = -1 
                        self.G[idx] = 0
        
                elif i == self.Nx-1:
                    self.A[idx, index(i-1, j)] = 1
                    self.A[idx, idx] = -1
                    self.G[idx] = 0
                    
                elif j == 0:  # device bottom side 
                    ''' floating substrate (Neumaan BC):  '''               
                    self.A[idx, index(i, j+1)] = 1
                    self.A[idx, idx] = -1
                    self.G[idx] = 0
                elif j == self.Nz-1 :  # device top side, Neumann BC
                    self.A[idx, index(i, j-1)] = 1
                    self.A[idx, idx] = -1
                    self.G[idx] = 0
                else:
                    epsilon = self.epsilon
                    self.A[idx, index(i+1, j)] = (epsilon[i, j] + epsilon[i,j-1])/2 / ratio**2 # a1
                    self.A[idx, index(i, j+1)] = (epsilon[i-1, j] + epsilon[i,j])/2  # a2
                    self.A[idx, index(i-1, j)] = (epsilon[i-1, j] + epsilon[i-1,j-1])/2 / ratio**2  # a3
                    self.A[idx, index(i, j-1)] = (epsilon[i, j-1] + epsilon[i-1,j-1])/2  # a4
                    self.A[idx, idx] = -(epsilon[i,j] + epsilon[i-1, j] + epsilon[i-1, j-1] + epsilon[i, j-1])/2 * (1+1/ratio**2) - \
                        1/self.ds.V_therm * self.ds.e_charge * (p[i,j] + n[i,j]) * self.dz**2 / spc.epsilon_0  
                    
                    self.G[idx] = -(rho[i,j]) * self.dz**2 / spc.epsilon_0 - \
                        (
                           self.A[idx, index(i+1, j)]*Ec[i+1,j] +self.A[idx, index(i, j+1)]*Ec[i,j+1] + \
                        self.A[idx, index(i-1, j)]*Ec[i-1,j] +self.A[idx, index(i, j-1)]*Ec[i,j-1] + \
                           -(epsilon[i,j] + epsilon[i-1, j] + epsilon[i-1, j-1] + epsilon[i, j-1])/2 * (1+1/ratio**2)*Ec[i,j]
                        ) # RHS of Poisson equation

        # convert this matrix into csr representation
        F =self.A.tocsr()

        # solve this linear system
        Delta_V = spla.spsolve(F, self.G)

        # reshape the solution vector to array
        Delta_V = Delta_V.reshape((self.Nz, self.Nx)).T

        return Delta_V
    
    def solvePoisson(self):
        self.ds.Ec = self.init_Poisson()
        err = 1
        while (err > 1e-5):
            dV = self.delta_Poisson()
            err = np.max(np.abs(dV))
            self.ds.Ec += dV