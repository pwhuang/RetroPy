import sys
sys.path.insert(0, '../..')

import reaktoro_transport.solver as solver
import reaktoro_transport.tools as tools
from dolfin import *
import numpy as np

# Comparing the concetration transport phase field solve
# with respect to the asymptotic solution

def adv_diff_reac_sol_fracture(Pe, Da, eps, C_b, x_space):
    k = 0
    alpha2 = 2*Da/eps**2
    G_plus  = 1.0/24*(Pe + (576*alpha2 + Pe**2 - 576*k*Pe)**0.5)
    G_minus = 1.0/24*(Pe - (576*alpha2 + Pe**2 - 576*k*Pe)**0.5)

    k2 = (alpha2/(alpha2 - k*Pe) - C_b)/(-1 + G_plus/G_minus*np.exp(G_plus-G_minus))
    k1 = C_b - alpha2/(alpha2-k*Pe) - k2

    C = alpha2/(alpha2-k*Pe) + k1*np.exp(x_space*G_minus) + k2*np.exp(x_space*G_plus)

    return C

nx = 31
ny = 13

mesh_nd = RectangleMesh(Point(0,0.4), Point(1,1.6), nx, ny, 'right/left')

# This range
for i in range(1):
    CC = FunctionSpace(mesh_nd, 'CG', 1)
    DG_space = FunctionSpace(mesh_nd, 'DG', 0)

    # Define phase field
    expr = Expression('0.5*(1+ tanh((x[1] - h1)/r)) + 0.5*(1+ tanh((h2- x[1])/r))', degree=1\
                      , tol=DOLFIN_EPS, r=1e-3, h1=1.5, h2=0.5)

    phi = interpolate(expr, CC)

    # Make sure nothing is above 1
    phi.vector()[np.where(phi.vector()[:]>1)[0]] = 1.0

    # Project norm of the phase field to DG space
    phi_norm = project(sqrt(inner(grad(phi), grad(phi))), DG_space)

    # Find the points to refine, where the phi_norm > 5
    dof_coordinates = DG_space.tabulate_dof_coordinates()
    dof_x = dof_coordinates[:, 0]
    dof_y = dof_coordinates[:, 1]

    point_x = dof_x[np.where(phi_norm.vector()>5)[0]]
    point_y = dof_y[np.where(phi_norm.vector()>5)[0]]

    points = np.array([point_x, point_y]).T

    depth = 1
    mesh_nd = tools.refine_mesh_around_points(mesh_nd, points, depth)

# Refine whole mesh again
mesh_nd = refine(mesh_nd)
CC = FunctionSpace(mesh_nd, 'CG', 1)
DG_space = FunctionSpace(mesh_nd, 'DG', 0)
phi = interpolate(expr, CC)

phi.vector()[np.where(phi.vector()[:]>1)[0]] = 1.0

epsilon = 0.1
Pe = 10
Da = 0.1
c_left_bc = 0

u, p = solver.stokes_lubrication_phase_field(mesh_nd, epsilon, phi)
C = solver.concentration_transport_phase_field(mesh_nd, epsilon, Pe, Da, c_left_bc, u, phi)

# Sample some points to compare
x_space = np.linspace(0, 1, 51)

C_nd = []
for x_point in x_space:
    C_nd.append(C(Point(x_point, 0.65)))

C_nd = np.array(C_nd)
reference_solution = adv_diff_reac_sol_fracture(Pe, Da, epsilon, c_left_bc, x_space)

print(np.linalg.norm(reference_solution - C_nd.T))

# Try to improve this.
def test_answer():
    assert np.linalg.norm(reference_solution - C_nd.T) < 0.1
