# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

os.environ["OMP_NUM_THREADS"] = "1"

from retropy.physics import DG0Kernel
from retropy.solver import PETScSolver
from retropy.benchmarks import HeterogeneousDiffusion

from utility_functions import convergence_rate

from math import isclose

import matplotlib.pyplot as plt

class DG0HeterogeneousDiffusionTest(HeterogeneousDiffusion, DG0Kernel, PETScSolver):
    def __init__(self, nx):
        marked_mesh = self.get_mesh_and_markers(nx)
        super().__init__(marked_mesh)

        self.define_problem()
        self.set_flow_field()
        self.generate_solver()

        self.set_solver_parameters(linear_solver="gmres", preconditioner="none")

    def mpl_output(self):
        x_space = self.cell_coord.x.array
        numerical_solution = self.fluid_components.x.array
        analytical_solution = self.solution.x.array

        _, ax = plt.subplots(1, 1)
        ax.plot(x_space, analytical_solution, lw=3, c="C0")
        ax.plot(x_space, numerical_solution, ls=(0, (5, 5)), lw=2, c="C1")
        plt.show()


nx_list = [10, 20]
element_diameters = []
err_norms = []

for nx in nx_list:
    problem = DG0HeterogeneousDiffusionTest(nx)
    
    problem.solve_transport()
    
    numerical_solution = problem.get_solution()
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)
    element_diameters.append(problem.get_mesh_characterisitic_length())

print(err_norms)

convergence_rate_m = convergence_rate(err_norms, element_diameters)
print(convergence_rate_m)


def test_function():
    assert isclose(convergence_rate_m[0], 1, rel_tol=0.2)
