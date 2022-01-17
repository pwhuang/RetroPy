import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

filepath, keys, t_id = sys.argv[1], sys.argv[2], sys.argv[3]

f = h5py.File(filepath + '.h5', 'r')

print(f.keys())

geometry = f['Mesh']['mesh']['geometry'][:]
triangulation = f['Mesh']['mesh']['topology'][:]

p_x = geometry[:,0]
p_y = geometry[:,1]

scalar = f[keys][f'{keys}_{t_id}']['vector'][:].flatten()

fig, ax = plt.subplots(1, 1, figsize=(6,6))

cbar = ax.tripcolor(p_x, p_y, triangulation, scalar, cmap='viridis')
ax.triplot(p_x, p_y, triangulation, c='w', lw=0.3, alpha=0.5)

ax.set_aspect('equal')

fig.colorbar(cbar)
plt.show()
