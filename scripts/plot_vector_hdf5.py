from dolfin import *
from dolfin.common.plotting import mplot_function, mplot_mesh
from reaktoro_transport.tools import LoadOutputFile

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

filepath, keys, t_id = sys.argv[1], sys.argv[2], sys.argv[3]

f = h5py.File(filepath + '.h5', 'r')
fxdmf = LoadOutputFile(filepath)
mesh_2d = fxdmf.load_mesh()

geometry = f['Mesh']['mesh']['geometry'][:]
triangulation = f['Mesh']['mesh']['topology'][:]

p_x = geometry[:,0]
p_y = geometry[:,1]

vector = f[keys][f'{keys}_{t_id}']['vector'][:].flatten()

RT_space = FunctionSpace(mesh_2d, 'RT', 1)
CG_space = VectorFunctionSpace(mesh_2d, 'CG', 1)

vector_RT = Function(RT_space)
vector_RT.vector()[:] = vector
vector_CG = interpolate(vector_RT, CG_space)

fig, ax = plt.subplots(1, 1, figsize=(6,6))

cbar = mplot_function(ax, vector_CG)
ax.triplot(p_x, p_y, triangulation, c='w', lw=0.3, alpha=0.5)

ax.set_aspect('equal')
fig.colorbar(cbar)
plt.show()
