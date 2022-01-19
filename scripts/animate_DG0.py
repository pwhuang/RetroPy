import numpy as np
from scipy.interpolate import interp1d

import h5py
import matplotlib.pyplot as plt

from matplotlib.tri import Triangulation
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys

in_path, out_path, keys = sys.argv[1], sys.argv[2], sys.argv[3]
t_start_id, t_end_id = int(sys.argv[4]), int(sys.argv[5])
playback_rate = float(sys.argv[6])

f = h5py.File(f'{in_path}.h5', 'r')
times = np.load(f'{in_path}.npy')

print(len(times))
print(f'start time = {times[t_start_id]}')
print(f'end time = {times[t_end_id]}')

times = times[t_start_id:t_end_id+1]

duration = (times[t_end_id] - times[t_start_id])/playback_rate
fps = 30
total_frames = int(fps*duration)

# Load mesh geometry
geometry = f['Mesh']['mesh']['geometry'][:]
triangulation = f['Mesh']['mesh']['topology'][:]

p_x = geometry[:,0]
p_y = geometry[:,1]

# Generate barycentric center of the triangles
triang = Triangulation(p_x, p_y, triangulation)

p_center_x = []
p_center_y = []

for triang in triangulation:
    p_center_x.append(np.mean(p_x[triang]))
    p_center_y.append(np.mean(p_y[triang]))

t_ids = np.arange(t_start_id, t_end_id+1)

scalar_list = []

for t_id in t_ids:
    scalar_list.append(f[keys][f'{keys}_{t_id}']['vector'][:].flatten())

scalar_list = np.array(scalar_list)

# interpolation
time_interp = np.linspace(times[t_start_id], times[t_end_id], total_frames)
t_id_interp = np.arange(0, time_interp.shape[0])

# print(scalar_list.shape)
# print(times.shape)

lin_interp = interp1d(times, scalar_list.T, kind='linear')

scalar_interp_list = lin_interp(time_interp).T

fig, ax = plt.subplots(1, 1, figsize=(3,5))
cbar = ax.tripcolor(p_x, p_y, triangulation, scalar_interp_list[0],
                    cmap='viridis')
#ax.triplot(p_x, p_y, triangulation, c='w', lw=0.3, alpha=0.4)
ax.set_aspect('equal')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.2)

fig.colorbar(cbar, cax=cax)
plt.tight_layout()

def init():
    return cbar,

def update(i):
    cbar.set_array(scalar_interp_list[i])
    return cbar,

#plt.show()
ani = FuncAnimation(fig, update, frames=t_id_interp, init_func=init, repeat=False, cache_frame_data=False)

movie_writer = FFMpegWriter(fps=fps, codec='h264')
movie_writer._tmpdir = out_path

ani.save(f'{out_path}.mp4', writer=movie_writer, dpi=400)
movie_writer.finish()

