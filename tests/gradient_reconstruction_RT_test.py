# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
os.environ['OMP_NUM_THREADS'] = '1'

from retropy.problem import TransportProblemBase
from retropy.mesh import MarkedRectangleMesh
from retropy.solver import GradientSolver

from dolfin import (FunctionSpace, Function, interpolate,
                    Expression, norm, FacetNormal)
from numpy import array

from utility_functions import convergence_rate
from math import isclose

class GradientReconstructionRTTest(TransportProblemBase, GradientSolver):
    def __init__(self, projection_space):
        self.num_component = 1
        GradientSolver.set_projection_space(self, projection_space)

    def set_mesh(self, mesh):
        self.mesh = mesh
        self.n = FacetNormal(self.mesh)

expr = Expression('cos(M_PI*x[0])*cos(M_PI*x[1])', degree=1)
sol_expr = Expression(['-M_PI*sin(M_PI*x[0])*cos(M_PI*x[1])',
                       '-M_PI*cos(M_PI*x[0])*sin(M_PI*x[1])'], degree=1)
nx_list = [10, 20]
element_diameters = 1.0/array(nx_list)
error_norm = []

for nx in nx_list:
    mesh_factory = MarkedRectangleMesh()

    mesh_factory.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
    mesh_factory.set_top_right_coordinates(coord_x = 1.0, coord_y = 1.0)
    mesh_factory.set_number_of_elements(nx, nx)
    mesh_factory.set_mesh_type('triangle')
    mesh = mesh_factory.generate_mesh()
    domain_markers = mesh_factory.generate_domain_markers()
    boundary_markers, marker_dict = mesh_factory.generate_boundary_markers()

    DG_space = FunctionSpace(mesh, 'DG', 0)
    RT_space = FunctionSpace(mesh, 'RT', 1)

    func_to_project = interpolate(expr, DG_space)
    solution = interpolate(sol_expr, RT_space)

    func_to_assign = Function(RT_space)
    error_func = Function(RT_space)

    problem = GradientReconstructionRTTest(RT_space)
    problem.set_mesh(mesh)
    problem.set_boundary_markers(boundary_markers)
    problem.set_domain_markers(domain_markers)
    problem.set_projection_form([func_to_project])
    problem.generate_projection_solver(func_to_assign, [1,2,3,4])
    problem.set_projection_solver_params(preconditioner='none')
    problem.solve_projection()

    error_func.assign(solution - func_to_assign)
    error_norm.append(norm(error_func, 'l2'))

conv_rate = convergence_rate(error_norm, element_diameters)
print(error_norm, conv_rate)

def test_function():
    assert isclose(conv_rate, 2, rel_tol=0.1)
