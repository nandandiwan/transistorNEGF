# Silicon Nano-Transistor NEGF Simulation

This repository contains a comprehensive implementation for modeling silicon nano-transistors using the Non-Equilibrium Green's Function (NEGF) formalism. The simulation aims to understand and predict the behavior of ultra-thin MOSFET devices (3-10 nm channel heights) where quantum effects become dominant. In particular the sim hopes to explain why 3 nm thickness tranistor has a steeper ID-VG curve than 6 nm thickness transistor. 

## Overview

The simulation consists of two main components:
1. **Tight-Binding Model with Finite Z-Direction**: Provides the fundamental electronic structure and effective mass parameters
2. **NEGF Quantum Transport Simulation**: Implements self-consistent Poisson-NEGF solver for 2D (periodic in y) silicon nano-transistors

The ultimate goal is to extend the proven 1D multi-orbital NEGF implementation (see `test_device_multiple_orbitals.ipynb`) to a full 2D periodic silicon nano-transistor simulation.

## Main References

- **OpenMX**: Open source package for Material eXplorer - provides the foundational TRAN (transport) module algorithms
- **Jiezi**: Advanced tight-binding and transport simulation framework

## Project Structure

### 1. Tight-Binding Framework (`src/tight_binding/`)

This module implements finite-size tight-binding calculations for silicon with quantum confinement in the z-direction.

#### Key Components:

**`finite_tight_binding.py`**
- Core `TightBindingHamiltonian` class for finite-size silicon structures
- Implements sp3d5s* orbital basis with proper hybridization
- Supports both layer-based (N layers) and thickness-based construction
- Calculates effective mass parameters from band structure
- Utilizes sprase solver methods to analyze valence/conduction over k-space

**`unit_cell_generation.py`**
- `UnitCell` class for generating atomic positions in finite silicon structures  
- Manages neighbor relationships and dangling bond identification
- Supports voltage profile application for transport calculations

**`tight_binding_params.py`**
- Silicon tight-binding parameters (Slater-Koster integrals)
- On-site energies for s, p, d orbitals
- Based on established silicon parameter sets

**`plotter.py`**
- Visualization tools for band structures, DOS, and atomic configurations

ns

### 2. NEGF Quantum Transport (`src/NEGF/`)

This module implements the full NEGF formalism for quantum transport in silicon nano-transistors.

#### Key Components:

**Core NEGF Engine:**
- `solve.py`: Main solver class with self-consistent Poisson-NEGF iteration
- `rgf.py`: Recursive Green's Function implementation (lesser greens function is broken)
- `hamiltonian.py`: Device Hamiltonian construction with tight-binding basis
- `device.py`: Device geometry and material parameter management
- `charge.py`: DOS and LDOS calculations via multiprocessing

**Lead Self-Energy Calculations:**
- `lead_self_energy.py`: Robust implementation with multiple algorithms
- Implements Sancho-Rubio, iterative, and transfer matrix methods
- Based on OpenMX algorithms
- See `LEAD_SELF_ENERGY_DOCUMENTATION.md` for detailed implementation notes

**Electrostatics and Transport:**
- `poisson.py`: 2D Poisson solver for electrostatic potential
- `charge.py`: Charge density calculations from Green's functions
- `helper.py`: Utility functions for matrix operations and data handling

**Device Construction:**
- `NEGF_unit_generation.py`: Interface between tight-binding and NEGF modules
- Handles unit cell to device scaling and boundary condition application

#### Current Implementation Status:

**Working Features:**
- ✅ 1D multi-orbital NEGF simulation (`test_device_multiple_orbitals.ipynb`)
- ✅ Tight binding model for finite-z with effective mass, E-k plots 

**In Development:**
- 🔄 Full 2D (periodic in y) silicon nano-transistor extension
- 🔄 Self consistent NEGF-Poisson solver for 2D
- 🔄 Scattering and incoherent transport self energy (plan to use SCBA)

#### Key Notebook:
**`test_device_multiple_orbitals.ipynb`**
- Complete working example of 1D multi-orbital NEGF transistor simulation
- Demonstrates self-consistent Poisson-NEGF solution
- Includes I-V characteristics and transmission calculations
- Serves as foundation for 2D extension

## Installation and Usage

See `src/NEGF/test_device_multiple_orbitals.ipynb` for a full working simulation.

## Technical Details

### Physics 
- **Quantum Confinement**: Finite tight-binding in z-direction captures quantum confinement effects
- **Transport**: NEGF formalism with proper lead self-energies and scattering
- **Electrostatics**: Self-consistent Poisson equation solution
- **Material Properties**: Silicon-specific tight-binding parameters and effective masses

### Computational Approach
- **Sparse Matrix Operations**: Optimized for large device simulations (used extensively in TB model)
- **Parallel Processing**: Multi-threaded Green's function calculations
- **Numerical Stability**: Robust algorithms with automatic fallback methods
- **Memory Efficiency**: Block-wise operations (broken for RGF) and selective storage

