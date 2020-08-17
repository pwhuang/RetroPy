import numpy as np
from dolfin import *

import sys
sys.path.insert(0, '../../projects/Reaktoro-Transport')
import reaktoro_transport.solver as solver
import reaktoro_transport.tools as tools

# Reading mesh and MeshFunction that stores the fluid solid markers
mesh_2d = Mesh()

xdmf_obj = XDMFFile(MPI.comm_world, sys.argv[1])
xdmf_obj.read(mesh_2d)
phi_DG_MF = MeshFunction('size_t', mesh_2d, dim=mesh_2d.geometric_dimension())
xdmf_obj.read(phi_DG_MF)
xdmf_obj.close()

V = FunctionSpace(mesh_2d, 'DG', 0)
phi_DG = Function(V)
phi_DG.vector()[:] = phi_DG_MF.array()[:]

# Create a CR function that stores the fluid/solid boundaries
CR_space = FunctionSpace(mesh_2d, 'CR', 1)
cr_dof = CR_space.dofmap().dofs(mesh_2d, 1)
phi_CR = project(phi_DG, CR_space, solver_type='gmres', preconditioner_type='amg')

boundary_markers = MeshFunction('size_t', mesh_2d, dim=mesh_2d.geometric_dimension()-1)

# In addition to marking solid, fluid and the solid/fluid boundaries,
# we can still use fenics/dolfin functions to mark the obvious boundaries.
# For example, we want to apply a pressure boundary on the left boundary:
class left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

class right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 10.0, DOLFIN_EPS)

class top_bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0, DOLFIN_EPS) or near(x[1], 10.0, DOLFIN_EPS))

b_left = left()
b_right = right()
b_noslip = top_bottom()

# Boundary on the fluid domain is marked as 0
boundary_markers.set_all(0)

# Then Mark boundaries
b_left.mark(boundary_markers, 2)
b_right.mark(boundary_markers, 3)

# Find the solid, fluid, s/f boundary indices
index_solid = phi_CR.vector()[cr_dof] > 1.0-1e-10
index_sf_boundary = np.logical_and(phi_CR.vector()[cr_dof] <= 1.0-1e-10, phi_CR.vector()[cr_dof] >= 1e-10)

# Mark them
boundary_markers.array()[index_solid] = 1
boundary_markers.array()[index_sf_boundary] = 4

# Mark the noslip boundaries at the end
b_noslip.mark(boundary_markers, 5)

# Save them as xdmf for simulations!
xdmf_obj = XDMFFile(MPI.comm_world, 'boundary.xdmf')
xdmf_obj.write(boundary_markers)
xdmf_obj.close()
