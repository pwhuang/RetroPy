from reaktoro_transport.tools import AnimateDG0Function
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm

import numpy as np
import sys

class Animate(AnimateDG0Function):
    def init_matplotlib(self):
        self.scalar_to_animate = self.scalar_to_animate*1e3
        scalar_min = np.min(self.scalar_to_animate.flatten())
        scalar_max = np.max(self.scalar_to_animate.flatten())
        print(scalar_min, scalar_max)
        print(self.times[-1])

        offset = TwoSlopeNorm(vmin=scalar_min, vcenter=1.0125, vmax=scalar_max)

        self.fig, axes = plt.subplots(1, 1, figsize=(4,3))
        
        for i, idx in enumerate([105]):
            cbar = axes.tripcolor(self.p_x, self.p_y, self.triangulation,
                                  self.scalar_to_animate[-1], cmap='Spectral', norm=offset)
                            #vmin=scalar_min, vmax=scalar_max)
        # ax.triplot(p_x, p_y, triangulation, c='w', lw=0.3, alpha=0.4)
        axes.set_aspect('equal')
        #ax.set_xlim(12.0, 25.0)
        axes.set_ylim(13.0, 42.0)
        axes.axis('off')  

        #divider = make_axes_locatable(axes[-1])
        #cax = divider.append_axes('right', size='5%', pad='5%')
        cax = axes.inset_axes([1.05, 0.1, 0.05, 0.8], transform = axes.transAxes)

        self.fig.colorbar(cbar, cax=cax)
        plt.tight_layout(w_pad=0.01)

        return cbar

in_path, out_path, keys = sys.argv[1], sys.argv[2], sys.argv[3]
t_start_id, t_end_id = int(sys.argv[4]), int(sys.argv[5])
playback_rate = float(sys.argv[6])

ani = Animate(fps=30, playback_rate=playback_rate)
ani.open(in_path)
ani.load_scalar_function(keys, t_start_id, t_end_id)
ani.interpolate_over_time()
ani.init_matplotlib()
#plt.savefig('fig_compare.png', dpi=400)
plt.show()
#ani.init_animation()
#ani.save_animation(out_path, dpi=400)
