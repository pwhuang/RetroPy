import numpy as np
from scipy.interpolate import interp1d

from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.tri import Triangulation
import h5py
from dolfin import (FunctionSpace, Function, Mesh, HDF5File, XDMFFile, MPI,
                    project, TestFunction)

class AnimateDG0Function:
    def __init__(self, fps=30, playback_rate=1.0, file_type='xdmf'):
        self.fps = fps
        self.playback_rate = playback_rate

        if file_type not in (supported_types:=['xdmf', 'hdf5']):
            raise NameError(f'Only supports {supported_types}.')
        self.file_type = file_type

    def open(self, filepath):
        self.file_handle = h5py.File(f'{filepath}.h5', 'r')
        self.times = np.load(f'{filepath}_time.npy')

        self.hdf5_handle = HDF5File(MPI.comm_world, f'{filepath}.h5', 'r')
        self.xdmf_handle = XDMFFile(MPI.comm_world, f'{filepath}_mesh.xdmf')
        print(self.times.shape)
        self.__init_mesh_geometry()

    def __init_mesh_geometry(self):
        self.mesh = Mesh()
        self.xdmf_handle.read(self.mesh)
        self.xdmf_handle.close()

        self.px = self.mesh.coordinates()[:,0]
        self.py = self.mesh.coordinates()[:,1]
        self.triang = self.mesh.cells()

        # Generate barycentric center of the triangles
        self.px_center = []
        self.py_center = []

        for triang in self.triang:
            self.px_center.append(np.mean(self.px[triang]))
            self.py_center.append(np.mean(self.py[triang]))

        self.px_center = np.array(self.px_center).flatten()
        self.py_center = np.array(self.py_center).flatten()
        self.triang_center = Triangulation(self.px_center, self.py_center)

    def set_times_to_plot(self, t_start_id, t_end_id):
        self.t_ids = np.arange(t_start_id, t_end_id+1)

        self.times = self.times[t_start_id:t_end_id+1]
        starttime = self.times[0]
        endtime = self.times[-1]
        duration = (endtime - starttime)/self.playback_rate

        total_frames = int(np.ceil(self.fps*duration))
        self.times_to_animate = np.linspace(starttime, endtime, total_frames)
        self.frame_id = np.arange(0, total_frames)

    def load_scalar_function(self, keys):
        self.scalar_list = []

        if self.file_type == 'xdmf':
            for t_id in self.t_ids:
                self.scalar_list.append(self.file_handle[keys][f'{keys}_{t_id}']['vector'][:].flatten())

        elif self.file_type == 'hdf5':
            for t_id in self.t_ids:
                self.scalar_list.append(self.file_handle[f'{keys}'][f'vector_{t_id}'][:].flatten())

        self.scalar_list = np.array(self.scalar_list)

    def load_vector_function(self, keys):
        self.vector_list = []

        if self.file_type == 'xdmf':
            for t_id in self.t_ids:
                self.vector_list.append(self.file_handle[keys][f'{keys}_{t_id}']['vector'][:].flatten())

        elif self.file_type == 'hdf5':
            for t_id in self.t_ids:
                self.vector_list.append(self.file_handle[f'{keys}'][f'vector_{t_id}'][:])

        self.vector_list = np.array(self.vector_list)
        self.vector_x = self.vector_list.reshape(len(self.t_ids), -1, 2)[:, :, 0]
        self.vector_y = self.vector_list.reshape(len(self.t_ids), -1, 2)[:, :, 1]

    def interpolate_over_space(self, scalar_list):
        interpolated_scalar = []
        DG_space = FunctionSpace(self.mesh, 'DG', 0)
        CR_space = FunctionSpace(self.mesh, 'CR', 1)

        self.px_CR = CR_space.tabulate_dof_coordinates()[:, 0]
        self.py_CR = CR_space.tabulate_dof_coordinates()[:, 1]
        self.triang_CR = Triangulation(self.px_CR, self.py_CR).triangles

        w = TestFunction(CR_space)
        DG_func = Function(DG_space)
        CR_func = Function(CR_space)

        for t_id in self.t_ids:
            DG_func.vector()[:] = scalar_list[t_id]
            interpolated_scalar.append(project(DG_func, CR_space).vector()[:])

        return np.array(interpolated_scalar)

    def average_over_triangles(self, scalar_list):
        interpolated_scalar = []

        for t_id in self.t_ids:
            interpolated_scalar.append(scalar_list[t_id][self.triang_CR].mean(axis=1))

        return np.array(interpolated_scalar)

    def interpolate_over_time(self, scalar_list):
        lin_interp = interp1d(self.times, scalar_list.T, kind='linear')
        return lin_interp(self.times_to_animate).T

    def init_matplotlib(self):
        pass

    def set_time_scale(self, scaling_factor, unit):
        self.scaling_factor = scaling_factor
        self.time_unit = unit

    def update_animation(self, i):
        self.cbar.set_array(self.scalar_to_animate[i])
        self.ax.set_title(f'time = {self.times_to_animate[i]*self.scaling_factor:.3f}' + self.time_unit)

    def init_animation(self):
        self.ani = FuncAnimation(self.fig, self.update_animation, frames=self.frame_id,
                                 repeat=False, cache_frame_data=False, blit=False)

    def save_animation(self, filepath, dpi):
        movie_writer = FFMpegWriter(fps=self.fps, codec='h264')
        movie_writer._tmpdir = filepath

        self.ani.save(f'{filepath}', writer=movie_writer, dpi=dpi)
        movie_writer.finish()
