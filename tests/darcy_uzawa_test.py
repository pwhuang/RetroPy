# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

os.environ["OMP_NUM_THREADS"] = "1"

from retropy.problem import DarcyFlowUzawa
from utility_functions import convergence_rate
from benchmarks import DarcyFlowBenchmark

import matplotlib.pyplot as plt
from math import isclose
from dolfinx.fem import Function


class DarcyUzawaTest(DarcyFlowUzawa, DarcyFlowBenchmark):
    """"""

    def __init__(self, nx):
        marked_mesh = DarcyFlowBenchmark.get_mesh_and_markers(self, nx)
        DarcyFlowUzawa.__init__(self, marked_mesh)

        self.set_pressure_fe_space("DG", 0)
        self.set_velocity_fe_space("BDM", 1)
        self.get_solution()

        self.set_material_properties()
        self.set_boundary_conditions()
        self.set_momentum_sources()

        self.set_additional_parameters(r_val=4e2, omega_by_r=0.3)
        self.assemble_matrix()
        self.set_flow_solver_params()


# nx is the mesh element in one direction.
list_of_nx = [10, 20]
element_diameters = []
p_err_norms = []
v_err_norms = []

for nx in list_of_nx:
    problem = DarcyUzawaTest(nx)
    problem.solve_flow(target_residual=1e-8, max_steps=100)
    pressure_error_norm, velocity_error_norm = problem.get_error_norm()

    p_err_norms.append(pressure_error_norm)
    v_err_norms.append(velocity_error_norm)
    element_diameters.append(problem.get_mesh_characterisitic_length())

    print(problem.get_flow_residual())

convergence_rate_p = convergence_rate(p_err_norms, element_diameters)
convergence_rate_v = convergence_rate(v_err_norms, element_diameters)

print(convergence_rate_p, convergence_rate_v)


def test_function():
    assert isclose(convergence_rate_p, 2, rel_tol=0.05) and isclose(
        convergence_rate_v, 2, rel_tol=0.05
    )
