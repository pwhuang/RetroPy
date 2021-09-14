import os
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(0, '../../')

from reaktoro_transport.physics import DG0Kernel
from reaktoro_transport.solver import TransientSolver

from reaktoro_transport.tests.benchmarks import ChargeBalancedDiffusion

from dolfin.common.plotting import mplot_function
import matplotlib.pyplot as plt

class DG0ChargeBalanceTest(ChargeBalancedDiffusion, DG0Kernel, TransientSolver):
    def __init__(self, nx, t0, is_output=False):
        super().__init__(*self.get_mesh_and_markers(nx))

        self.set_flow_field()
        self.define_problem(t0=t0)
        self.generate_solver()
        self.set_solver_parameters(linear_solver='gmres', preconditioner='amg')

        if is_output==True:
            self.generate_output_instance('charge_balance')

    def mpl_output(self):
        fig, ax = plt.subplots(1,1)
        mplot_function(ax, self.solution.sub(0), lw=3)
        mplot_function(ax, self.fluid_components.sub(0), ls=(0,(5,5)), lw=3)
        plt.show()

nx, t0 = 51, 1.0
list_of_dt = [3e-1]
timesteps = [10]
err_norms = []

for i, dt in enumerate(list_of_dt):
    problem = DG0ChargeBalanceTest(nx, t0, is_output=False)
    problem.solve_transport(dt_val=dt, timesteps=timesteps[i])

    t_end = timesteps[i]*dt + t0
    problem.get_solution(t_end)
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)

print(err_norms)

def test_function():
    assert err_norms[-1] < 1e-2
