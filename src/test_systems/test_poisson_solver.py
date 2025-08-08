import os
import sys
import numpy as np

# Allow imports from this folder
HERE = os.path.dirname(__file__)
sys.path.append(HERE)

from hamiltonian import Hamiltonian
import poisson


def make_uniform_eps(ham, eps_r=11.7):
    eps0 = 8.854187817e-12
    return np.full(ham.N, eps_r * eps0)


def test_linear_no_charge_no_gate():
    # With no carriers (n_i=0), no doping and no gate, solution should be linear.
    ham = Hamiltonian("one_d_wire")
    ham.N = 101
    ham.one_d_dx = 0.5e-9
    ham.set_voltage(Vs=0.0, Vd=0.1, Vg=0.0)
    ham.C_ox = 0.0
    ham.one_d_epsilon = make_uniform_eps(ham)
    ham.one_d_doping = np.zeros(ham.N)
    ham.n_i = 0
    V, _ = poisson.solve_poisson_nonlinear(ham)

    x = np.arange(ham.N)
    V_lin = ham.Vs + (ham.Vd - ham.Vs) * (x / (ham.N - 1))

    assert np.allclose(V, V_lin, atol=1e-9)


def test_gate_pulls_potential_up():
    # Gate present in middle should raise potential there relative to edges (Vs=Vd=0)
    ham = Hamiltonian("one_d_wire")
    ham.N = 201
    ham.one_d_dx = 0.5e-9
    ham.set_voltage(Vs=0.0, Vd=0.0, Vg=1.0)
    ham.C_ox = 2e-5  # F/m^2
    ham.gate_factor = 0.5
    ham.one_d_epsilon = make_uniform_eps(ham)
    ham.one_d_doping = np.zeros(ham.N)
    ham.n_i = 0
    V, _ = poisson.solve_poisson_nonlinear(ham)

    N = ham.N
    g0 = (N - int(ham.gate_factor * N)) // 2
    g1 = g0 + int(ham.gate_factor * N)

    V_mid = np.mean(V[g0:g1])
    V_edge = 0.5 * (np.mean(V[:g0]) + np.mean(V[g1:]))

    assert V_mid > V_edge + 1e-4


def test_uniform_positive_doping_bends_down():
    # Positive fixed charge (donors) with zero contacts and no gate should bend V negative in the interior.
    ham = Hamiltonian("one_d_wire")
    ham.N = 151
    ham.one_d_dx = 0.5e-9
    ham.set_voltage(Vs=0.0, Vd=0.0, Vg=0.0)
    ham.C_ox = 0.0
    ham.one_d_epsilon = make_uniform_eps(ham)
    Nd = 1e22  # m^-3
    q = 1.602176634e-19
    ham.one_d_doping = q * Nd * np.ones(ham.N)

    V, charge = poisson.solve_poisson_nonlinear(ham)

    # Expect interior below contacts (negative potential due to positive space charge)
    print(V)
    assert np.min(V[1:-1]) < -1e-5
    # Charge density should be negative (electrons)
    assert np.all(charge <= 0.0)

test_gate_pulls_potential_up()
test_linear_no_charge_no_gate()
test_uniform_positive_doping_bends_down()