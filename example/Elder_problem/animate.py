# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.tools import AnimateDG0Function
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation, LinearTriInterpolator
from matplotlib.patches import FancyArrowPatch

import numpy as np
import sys

class Animate(AnimateDG0Function):
    def init_matplotlib(self):
        self.scalar_min, self.scalar_max = 0.0, 1.0

        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 3))
        self.collection = self.ax.tripcolor(self.triang_CR, self.scalar_to_animate[0], cmap='Spectral_r',
                                            vmin=self.scalar_min, vmax=self.scalar_max, shading='flat')#, levels=11)

        ani.init_vector_plot(0)

        self.time_unit = ' (-)'
        self.ax.set_title(f'time = {self.times_to_animate[0]:.3f}' + self.time_unit)

        self.ax.set_aspect('equal')
        self.ax.set_xlim(0.0, 4.0)
        self.ax.set_ylim(0.0, 1.0)

        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='3%', pad='5%')

        self.fig.colorbar(self.collection, cax=cax)
        plt.tight_layout()

    def init_vector_plot(self, i):
        x_space = np.linspace(0, 4.0, 41)
        y_space = np.linspace(0, 1.0, 16)
        self.xx, self.yy = np.meshgrid(x_space, y_space, indexing='xy')

        self.interp_x = LinearTriInterpolator(self.triang_center, self.vector_x[i])
        self.interp_y = LinearTriInterpolator(self.triang_center, self.vector_y[i])

        vx, vy = self.interp_x(self.xx, self.yy), self.interp_y(self.xx, self.yy)
        self.stpset = self.ax.streamplot(self.xx, self.yy, vx, vy, color='w',
                                         linewidth=0.5, arrowsize=0.5, density=2.0)

    def update_animation(self, i):
        super().update_animation(i)

        interval = 3
        if i%interval==0:
            self.interp_x = LinearTriInterpolator(self.triang_center, self.vector_x[i])
            self.interp_y = LinearTriInterpolator(self.triang_center, self.vector_y[i])

            vx, vy = self.interp_x(self.xx, self.yy), self.interp_y(self.xx, self.yy)

            self.stpset.lines.remove()
            for art in self.ax.get_children():
                if isinstance(art, FancyArrowPatch):
                    art.remove()

            self.stpset = self.ax.streamplot(self.xx, self.yy, vx, vy, color='w',
                                             linewidth=0.5, arrowsize=0.5, density=2.0)

in_path, out_path, keys = 'elder_problem', 'elder_problem_temp.mp4', 'Temp'
t_start_id, t_end_id = 0, 49
playback_rate = 1e-3
is_preview = int(sys.argv[1])

ani = Animate(fps=30, playback_rate=playback_rate, file_type='hdf5')
ani.open(in_path)
ani.set_times_to_plot(t_start_id, t_end_id)
ani.load_scalar_function(keys)
ani.load_vector_function('velocity')
ani.scalar_list = ani.interpolate_over_space(ani.scalar_list)
ani.scalar_list = ani.average_over_triangles(ani.scalar_list)
ani.scalar_to_animate = ani.interpolate_over_time(ani.scalar_list)
ani.vector_x = ani.interpolate_over_time(ani.vector_x)
ani.vector_y = ani.interpolate_over_time(ani.vector_y)
ani.init_matplotlib()
ani.set_time_scale(scaling_factor=1.0, unit=' (-)')

if is_preview==1:
    plt.show()
else:
    ani.init_animation()
    ani.save_animation(out_path, dpi=250)
