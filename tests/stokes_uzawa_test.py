# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os

os.environ["OMP_NUM_THREADS"] = "1"

from retropy.mesh import MarkedRectangleMesh
from retropy.problem import StokesFlowUzawa
from utility_functions import convergence_rate
from benchmarks import StokesFlowBenchmark

from ufl import FiniteElement, VectorElement, EnrichedElement
from dolfinx import io
from dolfinx.fem import Function, FunctionSpace
import numpy as np


class StokesUzawaTest(StokesFlowUzawa, StokesFlowBenchmark):
    """"""

    def __init__(self, nx):
        marked_mesh = self.get_mesh_and_markers(nx)
        StokesFlowUzawa.__init__(self, marked_mesh)

        self.set_pressure_fe_space("DG", 0)
        self.set_velocity_vector_fe_space("CG", 2)

        self.set_pressure_ic(0.0)
        self.get_solution()

        self.set_material_properties()
        self.set_boundary_conditions()
        self.set_momentum_sources()

        self.set_additional_parameters(r_val=5e1, omega_by_r=1.0)
        self.assemble_matrix()
        self.set_flow_solver_params()

    def xdmf_output(self):
        with io.XDMFFile(self.mesh.comm, "out/u.xdmf", "w") as file:
            file.write_mesh(self.mesh)
            file.write_function(self.fluid_pressure)
            file.write_function(self.sol_pressure)

            velocity_CG1 = Function(self.Vec_CG1_space)
            velocity_CG1.name = "fluid_velocity"
            velocity_CG1.interpolate(self.fluid_velocity)
            file.write_function(velocity_CG1)

            velocity_CG1.interpolate(self.sol_velocity)
            velocity_CG1.name = "sol_velocity"
            file.write_function(velocity_CG1)


# nx is the mesh element in one direction.
list_of_nx = [10, 20]
element_diameters = []
p_err_norms = []
v_err_norms = []

for nx in list_of_nx:
    problem = StokesUzawaTest(nx)
    problem.solve_flow(target_residual=1e-4, max_steps=10)
    pressure_error_norm, velocity_error_norm = problem.get_error_norm()

    p_err_norms.append(pressure_error_norm)
    v_err_norms.append(velocity_error_norm)
    element_diameters.append(problem.get_mesh_characterisitic_length())

    print(problem.get_error_norm())

convergence_rate_p = convergence_rate(p_err_norms, element_diameters)
convergence_rate_v = convergence_rate(v_err_norms, element_diameters)
rates = np.append(convergence_rate_p, convergence_rate_v)

print(rates)
# problem.xdmf_output()


def test_function():
    assert np.allclose(rates, [2.5, 2.5], rtol=0.2)
