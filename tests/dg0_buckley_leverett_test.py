# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

os.environ["OMP_NUM_THREADS"] = "1"

from retropy.physics import DG0Kernel
from retropy.solver import PETScSolver
from retropy.benchmarks import BuckleyLeverett

from utility_functions import convergence_rate

from math import isclose
import matplotlib.pyplot as plt


class DG0BuckleyLeverettTest(BuckleyLeverett, DG0Kernel, PETScSolver):
    def __init__(self, nx, s_init):
        super().__init__(self.get_mesh_and_markers(nx))

        self.s_init = s_init
        self.define_problem()
        self.set_flow_field()
        self.generate_solver()
        self.set_solver_parameters(linear_solver="gmres", preconditioner="jacobi")

    def solve_one_step(self):
        return self._problems[0].solve()

    def mpl_output(self):
        x_space = self.cell_coord.x.array
        numerical_solution = self.fluid_components.x.array
        analytical_solution = self.solution.x.array

        _, ax = plt.subplots(1, 1)
        ax.plot(x_space, analytical_solution, lw=3, c="C0")
        ax.plot(x_space, numerical_solution, ls=(0, (5, 5)), lw=2, c="C1")
        plt.show()


nx_list = [23, 46]
dt_list = [2.2e-2, 1.1e-2]
timesteps = [40, 80]
err_norms = []

for nx, dt, timestep in zip(nx_list, dt_list, timesteps):
    problem = DG0BuckleyLeverettTest(nx, s_init=0.1)
    problem.solve_transport(dt_val=dt, timesteps=timestep)

    t_end = timestep * dt
    problem.get_solution(t_end)
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)

print(err_norms)

convergence_rate_m = convergence_rate(err_norms, dt_list)
print(convergence_rate_m)

def test_function():
    assert isclose(convergence_rate_m[0], 1.0, rel_tol=0.2)
