# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'

from retropy.physics import DG0Kernel
from retropy.solver import SteadyStateSolver
from retropy.mesh import XDMFMesh, MarkedRectangleMesh

from utility_functions import convergence_rate
from benchmarks import DiffusionBenchmark

from dolfinx.fem import Constant
from math import isclose

class XDMFRectangleMesh(MarkedRectangleMesh, XDMFMesh):
    def __init__(self):
        super().__init__()

class DiffusionBenchmark(DiffusionBenchmark):
    def get_mesh_and_markers(self, filename, meshname):
        mesh_factory = XDMFRectangleMesh()
        mesh_factory.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
        mesh_factory.set_top_right_coordinates(coord_x = 1.0, coord_y = 1.0)

        mesh = mesh_factory.read_mesh(filename, meshname)
        boundary_markers, marker_dict, facet_dict = mesh_factory.generate_boundary_markers()
        interior_markers = mesh_factory.generate_interior_markers()
        domain_markers = mesh_factory.generate_domain_markers()

        return (mesh, boundary_markers, interior_markers,
                domain_markers, marker_dict, facet_dict)

class DG0SteadyDiffusionReadMeshTest(DiffusionBenchmark, DG0Kernel, SteadyStateSolver):
    def __init__(self, *args):
        super().__init__(*self.get_mesh_and_markers(*args),
                         option='cell_centered')

        self.set_flow_field()
        self.define_problem()
        self.set_problem_bc()

        self.generate_solver()
        self.set_solver_parameters(linear_solver='gmres', preconditioner='jacobi')

    def set_problem_bc(self):
        values = DiffusionBenchmark.set_problem_bc(self)
        # When solving steady-state problems, the diffusivity of the diffusion
        # boundary is a penalty term to the variational form.
        self.add_component_diffusion_bc('solute', diffusivity=Constant(self.mesh, 1e3),
                                        kappa=Constant(self.mesh, 1.0), values=values)

list_of_filenames = ['mesh/2d_1e-1.xdmf', 'mesh/2d_5e-2.xdmf']
element_diameters = [1e-1, 5e-2]
err_norms = []

for filename in list_of_filenames:
    problem = DG0SteadyDiffusionReadMeshTest(filename, 'Grid')
    problem.solve_transport()
    numerical_solution = problem.get_solution()
    error_norm = problem.get_error_norm()
    err_norms.append(error_norm)

print(err_norms)

convergence_rate_m = convergence_rate(err_norms, element_diameters)
print(convergence_rate_m)

def test_function():
    assert isclose(convergence_rate_m, 1, rel_tol=0.5)
