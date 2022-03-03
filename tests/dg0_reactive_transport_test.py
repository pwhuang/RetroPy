import os
os.environ['OMP_NUM_THREADS'] = '1'

from reaktoro_transport.physics import DG0Kernel
from reaktoro_transport.solver import TransientRK2Solver
from reaktoro_transport.manager import XDMFManager

from benchmarks import ReactingSpecies

from math import isclose
from dolfin.common.plotting import mplot_function
import matplotlib.pyplot as plt

class DG0ReactiveTransportTest(ReactingSpecies, DG0Kernel, TransientRK2Solver,
                               XDMFManager):
    def __init__(self, nx, is_output=False):
        super().__init__(*self.get_mesh_and_markers(nx, 'triangle'))

        self.set_flow_field()
        self.define_problem()
        self.set_solver_forms()
        self.generate_solver()
        self.set_solver_parameters(linear_solver='gmres', preconditioner='amg')

        if is_output==True:
            self.generate_output_instance('reacting_species')

    def solve_transport(self, dt_val, timesteps):
        self.dt.assign(dt_val)
        endtime = 0.0

        self.save_to_file(time=endtime)

        for i in range(timesteps):
            self.solve_first_step()
            endtime += dt_val*self.kappa.values()[0]
            self.t_end.assign(endtime)
            self.solve_second_step()
            endtime += dt_val*(1.0 - self.kappa.values()[0])
            self.t_end.assign(endtime)
            self.save_to_file(time=endtime)

        self.delete_output_instance()

    def mpl_output(self):
        solution = self.get_solution()
        numerical = self.get_fluid_components()
        plt_kwargs = {'vmin': 0.0, 'vmax': 1.0, 'cmap': 'coolwarm'}

        fig, ax = plt.subplots(2, 2, figsize=(9,9))
        mplot_function(ax[0,0], solution.sub(0), **plt_kwargs)
        mplot_function(ax[0,1], solution.sub(1), **plt_kwargs)
        mplot_function(ax[1,0], numerical.sub(0), **plt_kwargs)
        mplot_function(ax[1,1], numerical.sub(1), **plt_kwargs)

        ax[0,0].set_title('solution, c1')
        ax[0,1].set_title('solution, c2')
        ax[1,0].set_title('numerical, c1')
        ax[1,1].set_title('numerical, c2')
        plt.show()

nx = 20
list_of_dt = [1e-1]
timesteps = [10]
err_norms = []

for i, dt in enumerate(list_of_dt):
    problem = DG0ReactiveTransportTest(nx, is_output=False)
    problem.set_kappa(1.0)
    problem.solve_transport(dt_val=dt, timesteps=timesteps[i])
    problem.generate_solution(dt*timesteps[0])

    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)

# TODO: Figure out how to setup and benchmark this problem properly.
# By calling the mpl_output function, one can see the numerical solution
# resembles the analytical the solution. However, the error says otherwise.
# This is a non-urgent issue. Please address this when you are interested in
# this particular problem.
# problem.mpl_output()
print(err_norms)

def test_function():
    assert err_norms[-1] < 0.5
