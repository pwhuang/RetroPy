import numpy as np
from scipy.interpolate import interp1d

from matplotlib.animation import FuncAnimation, FFMpegWriter
import h5py

class AnimateDG0Function:
    def __init__(self, fps=30, playback_rate=1.0):
        self.fps = fps
        self.playback_rate = playback_rate

    def open(self, filepath):
        self.file_handle = h5py.File(f'{filepath}.h5', 'r')
        self.times = np.load(f'{filepath}.npy')

        print(self.times.shape)
        self.__init_mesh_geometry()

    def __init_mesh_geometry(self):
        geometry = self.file_handle['Mesh']['mesh']['geometry'][:]
        self.triangulation = self.file_handle['Mesh']['mesh']['topology'][:]

        self.p_x = geometry[:,0]
        self.p_y = geometry[:,1]

        # Generate barycentric center of the triangles
        self.p_center_x = []
        self.p_center_y = []

        for triang in self.triangulation:
            self.p_center_x.append(np.mean(self.p_x[triang]))
            self.p_center_y.append(np.mean(self.p_y[triang]))

    def load_scalar_function(self, keys, t_start_id, t_end_id):
        self.t_ids = np.arange(t_start_id, t_end_id+1)

        self.scalar_list = []

        for t_id in self.t_ids:
            self.scalar_list.append(self.file_handle[keys][f'{keys}_{t_id}']['vector'][:].flatten())

        self.scalar_list = np.array(self.scalar_list)

        self.times = self.times[t_start_id:t_end_id+1]
        duration = (self.times[t_end_id] - self.times[t_start_id])/self.playback_rate
        self.total_frames = int(self.fps*duration)
        self.times_to_animate = np.linspace(self.times[t_start_id], self.times[t_end_id], self.total_frames)
        self.t_id_animate = np.arange(0, self.times_to_animate.shape[0])

    def interpolate_over_time(self):
        lin_interp = interp1d(self.times, self.scalar_list.T, kind='linear')
        self.scalar_to_animate = lin_interp(self.times_to_animate).T

    def init_matplotlib(self):
        pass

    def init_animation(self):
        cbar = self.init_matplotlib()

        def init():
            return cbar,

        def update(i):
            cbar.set_array(self.scalar_to_animate[i])
            return cbar,

        self.ani = FuncAnimation(self.fig, update, frames=self.t_id_animate,
                                 init_func=init, repeat=False, cache_frame_data=False)

    def save_animation(self, filepath, dpi):
        movie_writer = FFMpegWriter(fps=self.fps, codec='h264')
        movie_writer._tmpdir = filepath

        self.ani.save(f'{filepath}', writer=movie_writer, dpi=dpi)
        movie_writer.finish()
