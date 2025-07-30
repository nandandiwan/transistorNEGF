import numpy as np
from hamiltonian import Hamiltonian
from rgf import GreensFunction
from plotter import Plotter

def plot_ham(name : str, new_ham = False, lead_func = None, ham_func = None):
    ham = Hamiltonian(name=name)
    if (new_ham):
        ham.register_hamiltonian(name, ham_func)
        ham.register_lead(name, lead_func)
        
    gf = GreensFunction(hamiltonian=ham, energy_grid=np.linspace(-2, 2, 101))

    # Configuration for plots
    config = {
        "dos": True,
        "transmission": True,
        "Id-Vs": True,
        "Id-Vg": True,
        "Vs_list": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "Vg_list": [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

    print("Generating plots and report for 1D wire...")
    Plotter.create_plots(ham, gf, config)
    print("Plots and report saved in ./plots/one_d_wire/")
    

plot_ham("one_d_wire")
