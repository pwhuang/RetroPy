import sys
sys.path.insert(0, '../..')

import reaktoro_transport.solver as solver
from dolfin import *
import numpy as np

# Comparing the solution of the concentration transport in the Cartesian coordinates
# with respect to the transient asymptotic solution
# The asymtotic solution is derived by assuming Da is in the order of epsilon squared,
# with left boundary equals to 0 and C(x, t=0) = 0.

def diff_reac_transient_sol_fracture(Pe, Da, eps, C_b, x_space, t):
    alpha2 = 2*Da/eps**2
    n_space = np.arange(1,21,1)

    C = np.zeros_like(x_space)

    for n in n_space:
        C += (1.0-np.exp(-0.25*t*(np.pi*(2*n-1))**2 - t*alpha2 ))/((2*n-1)*((np.pi*(2*n-1))**2 + 4*alpha2))\
        *np.sin((2*n-1)/2*np.pi*x_space)

    C = 16*alpha2/np.pi*C

    return C

nx_nd = 30
ny_nd = 30
mesh_2d = UnitSquareMesh(nx_nd, ny_nd)

epsilon = 0.1
Pe = 0
Da = 0.01
C_b = 0  # Concentration boundary condition at the left boundary
initial_expr = Expression('0', degree=1)
dt_num = 1e-2
time_steps = 10
theta_num = 0.5 # The Crank-Nicolson Scheme

C = solver.concentration_transport2D_transient(mesh_2d, epsilon, Pe, Da, C_b, initial_expr\
                                              , dt_num, time_steps, theta_num)

# Picking the last time step to compare
C_vertex = C[-1].compute_vertex_values(mesh_2d).reshape(nx_nd+1, ny_nd+1)

x_space = np.linspace(0,1,nx_nd+1)
asym_sol = diff_reac_transient_sol_fracture(Pe, Da, epsilon, C_b, x_space, dt_num*time_steps)

print(np.linalg.norm(asym_sol-C_vertex[0,:]))

tolerance = epsilon**2
def test_answer():
    assert np.linalg.norm(asym_sol-C_vertex[0,:]) < tolerance
