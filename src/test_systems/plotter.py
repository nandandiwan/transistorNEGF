import os
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from hamiltonian import Hamiltonian
from rgf import GreensFunction
import matplotlib.pyplot as plt


class Plotter:
    def create_plots(ham : Hamiltonian, gf : GreensFunction, config : dict):
        """
        Generates a report of system based on config

        Args:
            ham (Hamiltonian): Hamiltonian object preloaded with name
            gf (GreensFunction): Green's Function object preloaded with hamiltonian 

        Returns:
            None
        """
        import os
        # system information
        name = ham.name
        relevant_parameters = ham.relevant_parameters

        # Create output folder
        output_dir = os.path.join(os.getcwd(), "plots", name)
        os.makedirs(output_dir, exist_ok=True)

        def plot_dos():
            E_list = getattr(gf, 'energy_grid', None)
            dos = gf.calculate_DOS() if hasattr(gf, 'calculate_DOS') else None
            if E_list is not None and dos is not None:
                plt.figure()
                plt.plot(E_list, dos, label="DOS")
                plt.xlabel("Energy (eV)")
                plt.ylabel("Density of States")
                plt.title(f"DOS for {name}")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "dos.png"))
                plt.close()

        def plot_transmission():
            E_list = getattr(gf, 'energy_grid', None)
            transmission = [gf.compute_transmission(E) for E in E_list] if E_list is not None else None
            if E_list is not None and transmission is not None:
                plt.figure()
                plt.plot(E_list, transmission, label="Transmission")
                plt.xlabel("Energy (eV)")
                plt.ylabel("Transmission T(E)")
                plt.title(f"Transmission for {name}")
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_dir, "transmission.png"))
                plt.close()

        def plot_Id_Vs():
            Vs_list = config.get("Vs_list", [0, 0.1, 0.2, 0.3, 0.4, 0.5])
            Id_list = []
            for Vs in Vs_list:
                ham.Vs = Vs
                Id = gf.compute_total_current() if hasattr(gf, 'compute_total_current') else None
                Id_list.append(Id)
            plt.figure()
            plt.plot(Vs_list, Id_list, marker='o', label="Id-Vs")
            plt.xlabel("Source Voltage Vs (V)")
            plt.ylabel("Current Id (A)")
            plt.title(f"Id-Vs for {name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "Id_vs.png"))
            plt.close()

        def plot_Id_Vg():
            Vg_list = config.get("Vg_list", [0, 0.1, 0.2, 0.3, 0.4, 0.5])
            Id_list = []
            for Vg in Vg_list:
                ham.Vg = Vg
                Id = gf.compute_total_current() if hasattr(gf, 'compute_total_current') else None
                Id_list.append(Id)
            plt.figure()
            plt.plot(Vg_list, Id_list, marker='o', label="Id-Vg")
            plt.xlabel("Gate Voltage Vg (V)")
            plt.ylabel("Current Id (A)")
            plt.title(f"Id-Vg for {name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "Id_vg.png"))
            plt.close()

        # Add more plots as needed, e.g., band structure, eigenvalues, etc.
        if config.get("dos", True):
            plot_dos()
        if config.get("transmission", True):
            plot_transmission()
        if config.get("Id-Vs", True):
            plot_Id_Vs()
        if config.get("Id-Vg", True):
            plot_Id_Vg()

        # Save a summary report
        with open(os.path.join(output_dir, "report.txt"), "w") as f:
            f.write(f"System: {name}\n")
            f.write(f"Parameters: {relevant_parameters}\n")
            f.write(f"Plots generated: {', '.join([k for k in config if config[k]])}\n")
        
 
