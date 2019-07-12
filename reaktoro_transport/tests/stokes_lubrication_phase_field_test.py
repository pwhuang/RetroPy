import sys
sys.path.insert(0, '../..')

import reaktoro_transport.solver as solver
import reaktoro_transport.tools as tools
from dolfin import *
import numpy as np

# Comparing the stokes_lubrication_phase_field solution with -0.5*(y-1)*y

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
u, p = solver.stokes_lubrication_phase_field(mesh_nd, epsilon, phi)

# Sample some points to compare
x_space = np.linspace(0, 1, 51)
x_space_mesh = np.linspace(0.4, 1.6, 51)

u_nd = []
for x_point in x_space_mesh:
    u_nd.append(u(Point(0.5, x_point)))

u_nd = np.array(u_nd)
reference_velocity_solution = -0.5*(x_space-1)*(x_space)

# Try to improve this.
def test_answer():
    assert np.linalg.norm(reference_velocity_solution - u_nd.T[0]) < 0.3
