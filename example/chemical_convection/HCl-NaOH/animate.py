from reaktoro_transport.tools import AnimateDG0Function
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import numpy as np
import sys

class Animate(AnimateDG0Function):
    def init_matplotlib(self):
        scalar_min = np.min(self.scalar_to_animate.flatten())
        scalar_max = np.max(self.scalar_to_animate.flatten())

        rho_min, rho_center, rho_max = 1.01, 1.013, 1.04
        color_norm = TwoSlopeNorm(vmin=rho_min, vcenter=rho_center, vmax=rho_max)
        self.scalar_to_animate *= 1e3

        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 4), 
                                    constrained_layout=True,)
                                    #gridspec_kw={'width_ratios': [1.0, 0.05]})
        self.collection = self.ax.tripcolor(self.px, self.py, self.triang,
                            self.scalar_to_animate[0], cmap='YlGnBu',
                            norm=color_norm, rasterized=True)
        #                    vmin=scalar_min, vmax=scalar_max)
        #ax.triplot(p_x, p_y, triangulation, c='w', lw=0.3, alpha=0.4)

        self.time_unit = ' min'
        self.ax.set_title(f'time = {self.times_to_animate[0]/60.0:.3f}' + self.time_unit)

        self.fig.colorbar(self.collection, shrink=0.6, ax=self.ax)
        self.ax.axis('off')

        #self.ax.set_xlim(0.0, 51.0)
        self.ax.set_ylim(15.0, 40.0)
        self.ax.set_aspect('equal')

    def generate_color_map(self):
        colors = np.load('../../pH_colormap.npy')
        x = np.linspace(0.0, 1.0, colors.shape[0])
        color_map = LinearSegmentedColormap.from_list('pH_cmap', list(zip(x, colors)))

        return color_map

in_path, out_path, keys = sys.argv[1], sys.argv[2], sys.argv[3]
t_start_id, t_end_id = int(sys.argv[4]), int(sys.argv[5])
playback_rate = float(sys.argv[6])
is_preview = int(sys.argv[7])

ani = Animate(fps=30, playback_rate=playback_rate, file_type='hdf5')
ani.open(in_path, time_type='npy')

ani.set_times_to_plot(t_start_id, t_end_id)
ani.set_time_scale(scaling_factor=1.0/60.0, unit=' minutes')

ani.load_scalar_function(keys)
ani.scalar_to_animate = ani.interpolate_over_time(ani.scalar_list)
ani.init_matplotlib()


if is_preview==1:
    plt.show()
else:
    ani.init_animation()
    ani.save_animation(out_path, dpi=300)
