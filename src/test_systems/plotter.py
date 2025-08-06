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
    def create_plots(ham : Hamiltonian, gf : GreensFunction, config : dict, test_name : str = None):
        """
        Generates a report of system based on config

        Args:
            ham (Hamiltonian): Hamiltonian object preloaded with name
            gf (GreensFunction): Green's Function object preloaded with hamiltonian 
            config (dict): Configuration dictionary for what plots to generate
            test_name (str): Custom name for the test (optional)

        Returns:
            None
        """
        # system information
        name = test_name if test_name is not None else ham.name
        relevant_parameters = ham.relevant_parameters

        # Create output folder
        output_dir = os.path.join(os.getcwd(), "plots", name + " " + str(ham.periodic))
        os.makedirs(output_dir, exist_ok=True)

        def plot_dos():
            ham.reset_voltages()
            #ham.clear_potential()
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
            ham.reset_voltages()
            #ham.clear_potential()
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
            ham.reset_voltages()
            #ham.clear_potential()
            Vs_list = config.get("Vs_list", [0, 0.1, 0.2, 0.3, 0.4, 0.5])
            Id_list = []
            for Vs in Vs_list:
                ham.set_voltage(Vs=Vs)
                print(ham.Vs, ham.Vd, ham.mu1, ham.mu2)
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
            ham.reset_voltages()
            #ham.clear_potential()
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
        
        def plot_potential_profile():
            """Plot the potential profile and band diagram."""
            ham.reset_voltages()
            if hasattr(ham, 'potential') and ham.potential is not None:
                N = ham.get_num_sites()
                positions = list(range(N))
                potentials = []
                
                for i in range(N):
                    if i < len(ham.potential):
                        # Extract the potential value (handle sparse matrices)
                        pot_element = ham.potential[i]
                        if hasattr(pot_element, 'toarray'):
                            # Sparse matrix - convert to dense
                            pot_val = pot_element.toarray()[0,0]
                        elif hasattr(pot_element, 'shape') and pot_element.shape == (1, 1):
                            pot_val = pot_element[0,0]
                        elif hasattr(pot_element, '__getitem__'):
                            pot_val = pot_element[0,0] if hasattr(pot_element[0], '__getitem__') else pot_element[0]
                        else:
                            pot_val = pot_element
                        potentials.append(float(pot_val.real if hasattr(pot_val, 'real') else pot_val))
                    else:
                        potentials.append(0.0)
                
                plt.figure(figsize=(10, 6))
                plt.plot(positions, potentials, 'b-', linewidth=2, label='Potential Profile')
                plt.xlabel('Position (site index)')
                plt.ylabel('Energy (eV)')
                plt.title(f'Potential Profile for {name}')
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join(output_dir, "potential_profile.png"))
                plt.close()
            else:
                # Create a zero potential plot as placeholder
                N = ham.get_num_sites()
                positions = list(range(N))
                potentials = [0.0] * N
                
                plt.figure(figsize=(10, 6))
                plt.plot(positions, potentials, 'b-', linewidth=2, label='Potential Profile (Zero)')
                plt.xlabel('Position (site index)')
                plt.ylabel('Energy (eV)')
                plt.title(f'Potential Profile for {name} (No Potential Set)')
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join(output_dir, "potential_profile.png"))
                plt.close()

        # Add more plots as needed, e.g., band structure, eigenvalues, etc.
        
        # Backup potential before any plots (since other plots clear it)
        potential_backup = ham.potential
        
        if config.get("dos", True):
            plot_dos()
        if config.get("transmission", True):
            plot_transmission()
        if config.get("Id-Vs", True):
            plot_Id_Vs()
        if config.get("Id-Vg", True):
            plot_Id_Vg()
        
        # Restore potential for potential profile plot
        ham.potential = potential_backup
        if config.get("potential_profile", False):
            plot_potential_profile()

        # Save a summary report
        with open(os.path.join(output_dir, "report.txt"), "w") as f:
            f.write(f"System: {name}\n")
            f.write(f"Parameters: {relevant_parameters}\n")
            f.write(f"Plots generated: {', '.join([k for k in config if config[k]])}\n")
        
 
