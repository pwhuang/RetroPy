import os
os.environ['OMP_NUM_THREADS'] = '1'

from reaktoro_transport.problem import TransportProblemBase
from reaktoro_transport.mesh import MarkedRectangleMesh
from reaktoro_transport.solver import ProjectionSolver

from dolfin import FunctionSpace, Function, interpolate, Expression, norm
from numpy import array

from utility_functions import convergence_rate
from math import isclose

class ProjectionCG1DG0Test(TransportProblemBase, ProjectionSolver):
    def __init__(self, projection_space):
        ProjectionSolver.set_projection_space(self, projection_space)

    def set_mesh(self, mesh):
        self.mesh = mesh

expr = Expression('cos(M_PI*x[0])*cos(M_PI*x[1])', degree=1)
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

    CG_space = FunctionSpace(mesh, 'CG', 1)
    DG_space = FunctionSpace(mesh, 'DG', 0)

    func_to_project = interpolate(expr, CG_space)
    solution = interpolate(expr, DG_space)

    func_to_assign = Function(DG_space)
    error_func = Function(DG_space)

    problem = ProjectionCG1DG0Test(DG_space)
    problem.set_mesh(mesh)
    problem.set_domain_markers(domain_markers)
    problem.set_projection_form(func_to_project)
    problem.generate_projection_solver(func_to_assign)
    problem.set_projection_solver_params(preconditioner='none')
    problem.solve_projection()

    error_func.assign(solution - func_to_assign)
    error_norm.append(norm(error_func, 'l2'))

print(error_norm)
conv_rate = convergence_rate(error_norm, element_diameters)

def test_function():
    assert isclose(conv_rate, 2, rel_tol=0.1)
