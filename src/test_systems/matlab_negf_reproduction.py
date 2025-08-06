#!/usr/bin/env python3
"""
MATLAB NEGF Script Reproduction

This script reproduces the quantum transport calculations from the provided MATLAB script,
including three device configurations:
1. Ballistic device (no barriers)
2. Tunneling device (single barrier)
3. Resonant tunneling device (double barrier with quantum well)

It also implements the Büttiker probe for broadening resonances and generates
I-V characteristics plots matching the MATLAB results.

Based on the MATLAB script with the following key parameters:
- Physical constants: hbar=1.06e-34, q=1.6e-19, m=0.25*9.1e-31
- Energy range: E = linspace(-0.2, 0.8, 101)
- Voltage range: V = linspace(0, 0.5, 26)
- Device dimensions: NS=15, NC varies, ND=15
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# Set environment variables for thread control
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from hamiltonian import Hamiltonian
from rgf import GreensFunction


def setup_matlab_parameters():
    """Set up physical parameters matching the MATLAB script."""
    params = {
        # Physical constants (all MKS, except energy in eV)
        'hbar': 1.0545718e-34,  # J⋅s (corrected value)
        'q': 1.60217662e-19,    # C
        'm_eff': 0.25 * 9.10938356e-31,  # kg (effective mass)
        'IE': None,  # Will be calculated as q^2/(π*hbar)
        
        # Energy parameters (eV)
        'Ef': 0.1,   # Fermi energy
        'kT': 0.025, # Thermal energy
        
        # Lattice parameters
        'a': 3e-10,  # Lattice spacing in meters
        't0': 1.0,   # Hopping parameter in eV (will be recalculated)
        
        # Device dimensions
        'NS': 15,    # Source sites
        'ND': 15,    # Drain sites
        'NC_ballistic': 30,     # Channel sites for ballistic
        'NC_tunneling': 4,      # Channel sites for tunneling
        'NC_resonant': 16,      # Channel sites for resonant tunneling
        
        # Potential parameters
        'barrier_height': 0.4,  # eV
        'barrier_width': 4,     # sites
        
        # Energy and voltage grids
        'E_min': -0.2,
        'E_max': 0.8,
        'N_E': 101,
        'V_min': 0.0,
        'V_max': 0.5,
        'N_V': 26,
        
        # Büttiker probe
        'buttiker_strength': 0.00025,
    }
    
    # Calculate derived parameters
    params['IE'] = params['q']**2 / (np.pi * params['hbar'])
    # In the MATLAB script: t0 = hbar^2 / (2*m*a^2*q)
    # But we'll use normalized units where t0 = 1 eV for simplicity
    
    return params


def test_ballistic_device(params, with_buttiker=False):
    """Test ballistic device (no barriers)."""
    print("\n" + "="*60)
    print("TESTING BALLISTIC DEVICE")
    print("="*60)
    
    # Setup Hamiltonian
    ham = Hamiltonian("one_d_wire")
    ham.setup_ballistic_device(NS=params['NS'], NC=params['NC_ballistic'], ND=params['ND'])
    
    # Set physical parameters
    ham.Ef = params['Ef']
    ham.kbT_eV = params['kT']
    ham.t = params['t0']
    
    # Setup Green's function calculator
    gf = GreensFunction(ham)
    
    # Enable Büttiker probe if requested
    if with_buttiker:
        gf.enable_buttiker_probe(strength=params['buttiker_strength'])
        print(f"Büttiker probe enabled with strength {params['buttiker_strength']}")
    
    # Test transmission spectrum
    print("Calculating transmission spectrum...")
    E_list = np.linspace(params['E_min'], params['E_max'], params['N_E'])
    transmission = []
    
    for E in E_list:
        T = gf.compute_transmission(E)
        transmission.append(T)
    
    # Test current-voltage characteristics  
    print("Calculating I-V characteristics...")
    V_list = np.linspace(params['V_min'], params['V_max'], params['N_V'])
    current = gf.compute_current_landauer(V_list, E_range=(params['E_min'], params['E_max']), 
                                        N_E=params['N_E'])
    
    return E_list, transmission, V_list, current


def test_tunneling_device(params, with_buttiker=False):
    """Test tunneling device (single barrier)."""
    print("\n" + "="*60)
    print("TESTING TUNNELING DEVICE")
    print("="*60)
    
    # Setup Hamiltonian
    ham = Hamiltonian("one_d_wire")
    ham.setup_tunneling_device(NS=params['NS'], NC=params['NC_tunneling'], ND=params['ND'],
                              barrier_height=params['barrier_height'])
    
    # Set physical parameters
    ham.Ef = params['Ef']
    ham.kbT_eV = params['kT']
    ham.t = params['t0']
    
    # Setup Green's function calculator
    gf = GreensFunction(ham)
    
    # Enable Büttiker probe if requested
    if with_buttiker:
        gf.enable_buttiker_probe(strength=params['buttiker_strength'])
        print(f"Büttiker probe enabled with strength {params['buttiker_strength']}")
    
    # Test transmission spectrum
    print("Calculating transmission spectrum...")
    E_list = np.linspace(params['E_min'], params['E_max'], params['N_E'])
    transmission = []
    
    for E in E_list:
        T = gf.compute_transmission(E)
        transmission.append(T)
    
    # Test current-voltage characteristics
    print("Calculating I-V characteristics...")
    V_list = np.linspace(params['V_min'], params['V_max'], params['N_V'])
    current = gf.compute_current_landauer(V_list, E_range=(params['E_min'], params['E_max']), 
                                        N_E=params['N_E'])
    
    return E_list, transmission, V_list, current


def test_resonant_tunneling_device(params, with_buttiker=False):
    """Test resonant tunneling device (double barrier)."""
    print("\n" + "="*60)
    print("TESTING RESONANT TUNNELING DEVICE")
    print("="*60)
    
    # Setup Hamiltonian
    ham = Hamiltonian("one_d_wire")
    ham.setup_resonant_tunneling_device(NS=params['NS'], NC=params['NC_resonant'], ND=params['ND'],
                                       barrier_height=params['barrier_height'],
                                       barrier_width=params['barrier_width'])
    
    # Set physical parameters
    ham.Ef = params['Ef']
    ham.kbT_eV = params['kT'] 
    ham.t = params['t0']
    
    # Setup Green's function calculator
    gf = GreensFunction(ham)
    
    # Enable Büttiker probe if requested
    if with_buttiker:
        gf.enable_buttiker_probe(strength=params['buttiker_strength'])
        print(f"Büttiker probe enabled with strength {params['buttiker_strength']}")
    
    # Test transmission spectrum
    print("Calculating transmission spectrum...")
    E_list = np.linspace(params['E_min'], params['E_max'], params['N_E'])
    transmission = []
    
    for E in E_list:
        T = gf.compute_transmission(E)
        transmission.append(T)
    
    # Test current-voltage characteristics
    print("Calculating I-V characteristics...")
    V_list = np.linspace(params['V_min'], params['V_max'], params['N_V'])
    current = gf.compute_current_landauer(V_list, E_range=(params['E_min'], params['E_max']), 
                                        N_E=params['N_E'])
    
    return E_list, transmission, V_list, current


def plot_potential_profile(ham, params, title):
    """Plot the potential profile for a device."""
    N = ham.get_num_sites()
    positions = np.arange(N) * params['a'] * 1e9  # Convert to nm
    
    if ham.potential is not None:
        potentials = []
        for i in range(N):
            if i < len(ham.potential):
                pot_element = ham.potential[i]
                if hasattr(pot_element, 'toarray'):
                    pot_val = pot_element.toarray()[0, 0]
                else:
                    pot_val = pot_element[0, 0] if hasattr(pot_element, '__getitem__') else pot_element
                potentials.append(float(pot_val.real))
            else:
                potentials.append(0.0)
    else:
        potentials = np.zeros(N)
    
    return positions, potentials


def create_comparison_plots(results_dict, params):
    """Create comparison plots matching the MATLAB figure style."""
    
    # Create figure with subplots (2x3 grid to match the image layout)
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    fig.suptitle('NEGF Transport Simulation Results', fontsize=16, fontweight='bold')
    
    device_names = ['Ballistic', 'Tunneling', 'Resonant Tunneling']
    
    for i, (device_type, (E_list, transmission, V_list, current)) in enumerate(results_dict.items()):
        # Left column: Potential profiles + Energy vs Position
        ax_left = axes[i, 0]
        
        # For now, let's plot the transmission spectrum in the left column
        # (potential profiles will be added when we set up the devices properly)
        ax_left.plot(E_list, transmission, 'b-', linewidth=2)
        ax_left.set_xlabel('Energy (eV)', fontsize=12)
        ax_left.set_ylabel('Transmission T(E)', fontsize=12)
        ax_left.set_title(f'{device_names[i]} Device - Transmission', fontsize=14)
        ax_left.grid(True, alpha=0.3)
        ax_left.set_xlim([params['E_min'], params['E_max']])
        
        # Right column: Current vs Voltage
        ax_right = axes[i, 1]
        current_uA = np.array(current) * 1e6  # Convert to μA
        ax_right.plot(V_list, current_uA, 'b-', linewidth=2, marker='o', markersize=4)
        ax_right.set_xlabel('Voltage (V)', fontsize=12)
        ax_right.set_ylabel('Current (μA)', fontsize=12)
        ax_right.set_title(f'{device_names[i]} Device - I-V', fontsize=14)
        ax_right.grid(True, alpha=0.3)
        ax_right.set_xlim([0, 0.5])
        
        # Set y-axis limits to match expected current scales
        if i == 0:  # Ballistic
            ax_right.set_ylim([0, 4])
        elif i == 1:  # Tunneling  
            ax_right.set_ylim([0, 1])
        else:  # Resonant tunneling
            ax_right.set_ylim([0, 3])
    
    plt.tight_layout()
    return fig


def main():
    """Main function to run all tests and generate plots."""
    print("MATLAB NEGF Script Reproduction")
    print("="*80)
    
    # Setup parameters
    params = setup_matlab_parameters()
    print("Physical parameters:")
    for key, value in params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6e}")
        else:
            print(f"  {key}: {value}")
    
    # Test all device configurations
    results = {}
    
    # Test with Büttiker probe enabled for resonance broadening
    use_buttiker = True
    
    try:
        # Ballistic device
        E_list, transmission, V_list, current = test_ballistic_device(params, with_buttiker=use_buttiker)
        results['ballistic'] = (E_list, transmission, V_list, current)
        
        # Tunneling device
        E_list, transmission, V_list, current = test_tunneling_device(params, with_buttiker=use_buttiker)
        results['tunneling'] = (E_list, transmission, V_list, current)
        
        # Resonant tunneling device
        E_list, transmission, V_list, current = test_resonant_tunneling_device(params, with_buttiker=use_buttiker)
        results['resonant'] = (E_list, transmission, V_list, current)
        
        # Create comparison plots
        print("\nGenerating comparison plots...")
        fig = create_comparison_plots(results, params)
        
        # Save the plot
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "matlab_negf_reproduction.png")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filename}")
        
        # Show the plot
        plt.show()
        
        print("\n" + "="*80)
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
