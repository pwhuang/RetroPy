# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

os.environ["OMP_NUM_THREADS"] = "1"

from retropy.problem import DarcyFlowUzawa
from retropy.benchmarks import DarcyBoundarySource

from utility_functions import convergence_rate
import numpy as np


class DarcyBoundaryTest(DarcyFlowUzawa, DarcyBoundarySource):
    """"""

    def __init__(self, nx):
        marked_mesh = super().get_mesh_and_markers(nx)
        DarcyFlowUzawa.__init__(self, marked_mesh)

        self.set_pressure_fe_space("DG", 0)
        self.set_velocity_fe_space("RT", 1)

        self.set_material_properties()
        self.generate_form()
        self.set_boundary_conditions(penalty_value=0.0)
        self.set_momentum_sources()

        self.set_additional_parameters(r_val=1e2, omega_by_r=1.0)
        self.assemble_matrix()

        self.set_flow_solver_params()
        self.get_solution()


# nx is the mesh element in one direction.
list_of_nx = [10, 20]
element_diameters = []
p_err_norms = []
v_err_norms = []

for nx in list_of_nx:
    problem = DarcyBoundaryTest(nx)
    problem.solve_flow(target_residual=1e-10, max_steps=100)
    pressure_error_norm, velocity_error_norm = problem.get_error_norm()

    p_err_norms.append(pressure_error_norm)
    v_err_norms.append(velocity_error_norm)
    element_diameters.append(problem.get_mesh_characterisitic_length())

    print(problem.get_flow_residual())

convergence_rate_p = convergence_rate(p_err_norms, element_diameters)
convergence_rate_v = convergence_rate(v_err_norms, element_diameters)
rates = np.append(convergence_rate_p, convergence_rate_v)
print(rates)


def test_function():
    assert np.allclose(rates, [1.5, 0.5], rtol=0.1)