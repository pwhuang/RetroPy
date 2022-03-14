from reaktoro_transport.tools import AnimateDG0Function
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation, LinearTriInterpolator

import numpy as np
import sys

class Animate(AnimateDG0Function):
    def init_matplotlib(self):
        self.scalar_min, self.scalar_max = 0.0, 1.0

        self.fig, ax = plt.subplots(1, 1, figsize=(8, 2))
        # self.triang = Triangulation(self.px_CR, self.py_CR)
        # maskedTris = self.triang.get_masked_triangles()
        # print(maskedTris)
        cbar = ax.tripcolor(self.px, self.py, self.triangulation, self.scalar_to_animate[0], cmap='Spectral_r',
                            vmin=self.scalar_min, vmax=self.scalar_max, facecolor=None)#, levels=11)

        #cbar.set_array(self.scalar_to_animate[0][maskedTris])
        #ax.quiver(self.p_center_x, self.p_center_y, self.vector_x[-1], self.vector_y[-1])
        self.time_unit = ' (-)'
        ax.set_title(f'time = {self.times_to_animate[0]:.3f}' + self.time_unit)

        ax.set_aspect('equal')
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, 1.0)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad='5%')

        self.fig.colorbar(cbar, cax=cax)
        #ax.axis('off')

        plt.tight_layout()

        return cbar, ax

in_path, out_path, keys = 'elder_problem', 'elder_problem_temp.mp4', 'Temp'
t_start_id, t_end_id = 0, 49
playback_rate = 1e-3
is_preview = int(sys.argv[1])

ani = Animate(fps=30, playback_rate=playback_rate, file_type='hdf5')
ani.open(in_path)
ani.set_times_to_plot(t_start_id, t_end_id)
ani.load_scalar_function(keys)
#ani.scalar_list = ani.interpolate_over_space(ani.scalar_list)
#ani.load_vector_function('velocity')
ani.interpolate_over_time()
ani.init_matplotlib()
ani.set_time_scale(scaling_factor=1.0, unit=' (-)')

if is_preview==1:
    plt.show()
else:
    ani.init_animation()
    ani.save_animation(out_path, dpi=200)
