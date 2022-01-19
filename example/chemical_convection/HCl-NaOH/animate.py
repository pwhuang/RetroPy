from reaktoro_transport.tools import AnimateDG0Function
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys

class Animate(AnimateDG0Function):
    def init_matplotlib(self):
        self.fig, ax = plt.subplots(1, 1, figsize=(3,5))
        cbar = ax.tripcolor(p_x, p_y, triangulation, scalar_interp_list[0],
                            cmap='viridis')
        #ax.triplot(p_x, p_y, triangulation, c='w', lw=0.3, alpha=0.4)
        ax.set_aspect('equal')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.2)

        fig.colorbar(cbar, cax=cax)
        plt.tight_layout()

        return cbar

in_path, out_path, keys = sys.argv[1], sys.argv[2], sys.argv[3]
t_start_id, t_end_id = int(sys.argv[4]), int(sys.argv[5])
playback_rate = float(sys.argv[6])

ani = Animate(fps=30, playback_rate=10.0)
ani.open(in_path)
ani.load_scalar_function(keys, t_start_id, t_end_id)
ani.interpolate_over_time()
ani.init_matplotlib()
plt.show()
#ani.init_animation()
#ani.save_animation(out_path, dpi=400)
