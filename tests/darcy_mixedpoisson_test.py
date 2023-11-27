# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'

from retropy.problem import DarcyFlowMixedPoisson
from utility_functions import convergence_rate
from benchmarks import DarcyFlowBenchmark

import matplotlib.pyplot as plt
from math import isclose

class DarcyMixedPoissonTest(DarcyFlowMixedPoisson, DarcyFlowBenchmark):
    """"""

    def __init__(self, nx):
        marked_mesh = DarcyFlowBenchmark.get_mesh_and_markers(self, nx)
        DarcyFlowMixedPoisson.__init__(self, marked_mesh)

        self.set_pressure_fe_space('DG', 0)
        self.set_velocity_fe_space('BDM', 1)
        self.get_solution()

        DarcyFlowBenchmark.set_material_properties(self)
        DarcyFlowBenchmark.set_boundary_conditions(self)
        DarcyFlowBenchmark.set_momentum_sources(self)

        self.set_additional_parameters(r_val=0.0)
        self.assemble_matrix()
        self.set_flow_solver_params(petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                   "pc_factor_mat_solver_type": "mumps"})

    def mpl_output(self):
        x_space, y_space = self.cell_coord.x.array.reshape(-1, 2).T
        numerical_solution = self.fluid_pressure.x.array
        analytical_solution = self.sol_pressure.x.array

        x, y, _ = self.pressure_func_space.tabulate_dof_coordinates().T

        fig, ax = plt.subplots(1, 2, figsize = (10, 5))
        ax[0].tricontourf(x_space, y_space, analytical_solution)
        ax[1].tricontourf(x, y, numerical_solution)
        plt.show()

# nx is the mesh element in one direction.
list_of_nx = [10, 20]
element_diameters = []
p_err_norms = []
v_err_norms = []

for nx in list_of_nx:
    problem = DarcyMixedPoissonTest(nx)
    problem.solve_flow()
    pressure_error_norm, velocity_error_norm = problem.get_error_norm()

    p_err_norms.append(pressure_error_norm)
    v_err_norms.append(velocity_error_norm)
    element_diameters.append(problem.get_mesh_characterisitic_length())

    print(problem.get_flow_residual())

    problem.mpl_output()

convergence_rate_p = convergence_rate(p_err_norms, element_diameters)
convergence_rate_v = convergence_rate(v_err_norms, element_diameters)

print(convergence_rate_p, convergence_rate_v)

def test_function():
    assert isclose(convergence_rate_p, 2, rel_tol=0.05)\
       and isclose(convergence_rate_v, 2, rel_tol=0.05)
