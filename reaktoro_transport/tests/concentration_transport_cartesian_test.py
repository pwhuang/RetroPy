import sys
sys.path.insert(0, '../..')

import reaktoro_transport.solver as solver
from dolfin import *
import numpy as np

# Comparing the solution of the concentration transport in the Cartesian coordinates
# with respect to the asymptotic solution
# The asymtotic solution is derived by assuming Da is in the order of epsilon squared.


def adv_diff_reac_sol_fracture(Pe, Da, eps, C_b, x_space):
    k = 0
    alpha2 = 2*Da/eps**2
    G_plus  = 1.0/24*(Pe + (576*alpha2 + Pe**2 - 576*k*Pe)**0.5)
    G_minus = 1.0/24*(Pe - (576*alpha2 + Pe**2 - 576*k*Pe)**0.5)

    k2 = (alpha2/(alpha2 - k*Pe) - C_b)/(-1 + G_plus/G_minus*np.exp(G_plus-G_minus))
    k1 = C_b - alpha2/(alpha2-k*Pe) - k2

    C = alpha2/(alpha2-k*Pe) + k1*np.exp(x_space*G_minus) + k2*np.exp(x_space*G_plus)

    return C

nx_nd = 60
ny_nd = 60
mesh_2d = UnitSquareMesh(nx_nd, ny_nd)

epsilon = 0.1
Pe = 1
Da = 0.01
C_b = 0  #Concentration boundary condition at the left boundary
order = 1

C = solver.concentration_transport2D(mesh_2d, epsilon, Pe, Da, order, C_b, 'Cartesian')
C_vertex = C.compute_vertex_values(mesh_2d).reshape(nx_nd+1, ny_nd+1)

x_space = np.linspace(0,1,nx_nd+1)
asym_sol = adv_diff_reac_sol_fracture(Pe, Da, epsilon, C_b, x_space)

#print(np.linalg.norm(asym_sol-C_vertex[0,:]))

tolerance = epsilon**2
def test_answer():
    assert np.linalg.norm(asym_sol-C_vertex[0,:]) < tolerance
