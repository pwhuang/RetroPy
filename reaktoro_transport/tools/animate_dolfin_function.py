from . import *
from matplotlib.tri import Triangulation
from matplotlib.animation import FuncAnimation
#import matplotlib.pyplot as plt

def animate_dolfin_function(mesh_2d, func, fig, ax):
    mesh_x = mesh_2d.coordinates()[:,0]
    mesh_y = mesh_2d.coordinates()[:,1]
    connectivity = mesh_2d.cells()

    triang = Triangulation(mesh_x, mesh_y, connectivity)

    CG_space = FunctionSpace(mesh_2d, 'CG', 1)
    v_to_d = vertex_to_dof_map(CG_space)

    level = np.linspace(0, 1, 41)

    cb = ax.tricontourf(triang, func[0].vector()[v_to_d], levels=level)
    fig.colorbar(cb)

    def init():
        #ax.set_ylim(0,1)
        #ax.set_xlim(0,1)
        return cb,

    def update(t):
        ax.set_title('timesteps = ' + str(t))
        ax.tricontourf(triang, func[t].vector()[v_to_d], levels=level)
        #ln.set_data(x_space, adv_diff_reac_transient_sol_fracture(Pe, Da, epsilon, 0, x_space, t))
        return cb,

    ani = FuncAnimation(fig, update, frames=np.arange(1,100,1), init_func=init\
                        , blit=True, interval=300, cache_frame_data=False)

    return ani
