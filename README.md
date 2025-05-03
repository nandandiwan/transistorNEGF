# Transistor Model using NEGF

This program seeks to efficiently model a thin MOSFET in hopes of explaining why at thin channel heights (3-10 nm) the I-V curve improves. 

## Basic structure 
A variety of approximations are used to in order to model the device efficiently. The crux of the program is the self-consistent Poisson-NEGF solver which gives the current and charge distribution in the device. The device hamiltonian is written using the effective mass approximation of which is found through a finite tight binding model of silicon. 

