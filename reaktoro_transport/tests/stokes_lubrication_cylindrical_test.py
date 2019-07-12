import sys
sys.path.insert(0, '../..')

import reaktoro_transport.solver as solver
from dolfin import *
import numpy as np

# Comparing the stokes_lubrication solution in the cylindrical coordinates with 0.25*(1-y**2)

nx_nd = 10
ny_nd = 16
mesh_2d = UnitSquareMesh(nx_nd, ny_nd)

class top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0, DOLFIN_EPS)

u, p = solver.stokes_lubrication_cylindrical(mesh_2d, 0.1, top)

u_y = u.compute_vertex_values(mesh_2d).reshape(2, ny_nd+1, nx_nd+1)[0,:,5]
y_space = np.linspace(0,1,ny_nd+1)
asym_sol = 0.25*(1-y_space**2)

#This tolerance is very high, see if we can fix this?
def test_answer():
    assert np.linalg.norm(asym_sol-u_y) < 1e-2
