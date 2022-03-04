from reaktoro_transport.tools import AnimateDG0Function
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

import numpy as np
import sys

class Animate(AnimateDG0Function):
    def init_matplotlib(self):
        scalar_min, scalar_max = 0.0, 1.0

        self.fig, ax = plt.subplots(1, 1, figsize=(8, 2))
        cbar = ax.tripcolor(self.p_x, self.p_y, self.triangulation,
                            self.scalar_to_animate[-1], cmap='Spectral_r',
                            vmin=scalar_min, vmax=scalar_max)

        #ax.triplot(p_x, p_y, triangulation, c='w', lw=0.3, alpha=0.4)
        ax.set_aspect('equal')
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, 1.0)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad='5%')

        self.fig.colorbar(cbar, cax=cax)
        #ax.axis('off')
        plt.tight_layout()

        return cbar

in_path, out_path, keys = 'elder_problem', 'elder_problem_temp.mp4', 'Temp'
t_start_id, t_end_id = 0, 49
playback_rate = 1e-3
is_preview = int(sys.argv[1])

ani = Animate(fps=30, playback_rate=playback_rate)
ani.open(in_path)
ani.load_scalar_function(keys, t_start_id, t_end_id)
ani.interpolate_over_time()
ani.init_matplotlib()

if is_preview==1:
    plt.show()
else:
    ani.init_animation()
    ani.save_animation(out_path, dpi=200)
