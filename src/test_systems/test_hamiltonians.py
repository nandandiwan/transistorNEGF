import os
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import numpy as np
from hamiltonian import Hamiltonian
from rgf import GreensFunction
from plotter import Plotter

def plot_ham(name : str, periodic: bool, new_ham = False, lead_func = None, ham_func = None):
    ham = Hamiltonian(name=name, periodic=periodic)
    if (new_ham):
        ham.register_hamiltonian(name, ham_func)
        ham.register_lead(name, lead_func)
    ham.mu1 = 0
    ham.mu2 = 0
    ham.set_params(10, 3)
    ham.Ef = 0
    

    gf = GreensFunction(hamiltonian=ham, energy_grid=np.linspace(-3, 3, 501))

    # Configuration for plots
    config = {
        "dos": True,
        "transmission": True,
        "Id-Vs": True,
        "Id-Vg": False,
        "Vs_list": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "Vg_list": [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

    Plotter.create_plots(ham, gf, config)

    

plot_ham("zigzag", False)
