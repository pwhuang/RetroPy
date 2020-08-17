import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from dolfin import *

import sys
sys.path.insert(0, '../../projects/Reaktoro-Transport')
import reaktoro_transport.solver as solver
import reaktoro_transport.tools as tools

# Loading image. Zeros represent pore, and ones represent solid.
im = np.load(sys.argv[1])

ny, nx = np.shape(im)

xmin = 0.0
xmax = 10.0
ymin = 0.0
ymax = 10.0

pixel_length = (xmax-xmin)/nx
init_cell_count = 19
min_cell_size = 0.1

x_space = np.linspace(xmin, xmax-pixel_length, nx) + 0.5*pixel_length
y_space = np.linspace(ymin, ymax-pixel_length, ny) + 0.5*pixel_length

# Append the end points
x_space = np.append(xmin, x_space)
x_space = np.append(x_space, xmax)

y_space = np.append(ymin, y_space)
y_space = np.append(y_space, ymax)

xx, yy = np.meshgrid(x_space, y_space)
x = xx.flatten()
y = yy.flatten()

# Create a interpolator
im_padded = np.pad(im, 1, 'edge')
im_paddedT = np.flip(np.rot90(im_padded), axis=0)

f_of_phi_dg = RegularGridInterpolator((x_space, y_space), im_paddedT, 'nearest', bounds_error=True)

# Generate a rectangle mesh using dolfin/fenics
mesh_2d = RectangleMesh.create(MPI.comm_world, [Point(xmin, ymin), Point(xmax, ymax)]\
                               , [init_cell_count, int(init_cell_count*ny/nx)]\
                               , CellType.Type.triangle, 'right/left')

mesh_2d, phi_DG = tools.refine_mesh_dg(mesh_2d, f_of_phi_dg, threshold=1e-10\
                                       , min_cell_size=min_cell_size, where='b')

num_vertices = int(MPI.sum(MPI.comm_world, mesh_2d.num_vertices()))
if MPI.rank(MPI.comm_world)==0:
    print('Vertices count: ', num_vertices)

phi_DG_filtered = phi_DG.copy()
phi_DG_filtered.vector()[np.nonzero(phi_DG.vector()[:]>1e-14)[0]] = 1

# Create a CR function that stores the fluid/solid boundaries
CR_space = FunctionSpace(mesh_2d, 'CR', 1)
cr_dof = CR_space.dofmap().dofs(mesh_2d, 1)

phi_CR = project(phi_DG, CR_space, solver_type='gmres', preconditioner_type='amg')

dof_coordinates_cr = CR_space.tabulate_dof_coordinates()
dof_x_cr = dof_coordinates_cr[:, 0]
dof_y_cr = dof_coordinates_cr[:, 1]

# Create cell_markers and boundary_markers MeshFunction
cell_markers = MeshFunction('size_t', mesh_2d, dim=2)
boundary_markers = MeshFunction('size_t', mesh_2d, dim=1)

cell_markers.array()[:] = phi_DG_filtered.vector()[:]

xdmf_obj = XDMFFile(MPI.comm_world, 'mesh.xdmf')
xdmf_obj.write(mesh_2d)
xdmf_obj.write(cell_markers)
xdmf_obj.close()
