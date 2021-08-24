import sys
sys.path.insert(0, '../../')

from reaktoro_transport.mesh import MarkedRectangleMesh
from reaktoro_transport.physics import DG0Kernel
from reaktoro_transport.problem import TracerTransportProblem

from dolfin import Expression, TestFunction, dx, assemble, inner, CellVolume
from numpy import array, dot
from ufl import Index

nx = 2
mesh_type = 'triangle'

class MixedFunctionIndexTest(TracerTransportProblem, DG0Kernel):
    def __init__(self, mesh):
        self.mesh = mesh

mesh_factory = MarkedRectangleMesh()
mesh_factory.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
mesh_factory.set_top_right_coordinates(coord_x = 1.0, coord_y = 1.0)
mesh_factory.set_number_of_elements(nx, nx)
mesh_factory.set_mesh_type(mesh_type)

mesh = mesh_factory.generate_mesh()
boundary_markers, marker_dict = mesh_factory.generate_boundary_markers()
domain_markers = mesh_factory.generate_domain_markers()

problem = MixedFunctionIndexTest(mesh)#, boundary_markers, domain_markers)
problem.set_components('A', 'B', 'C', 'D')
problem.set_component_fe_space()
problem.set_component_ics(Expression(['1', '2', '3', '4'], degree=0))
mixed_function = problem.get_fluid_components()

print(mixed_function.vector()[:].reshape(-1, problem.num_component).sum(axis=1))

n = int(mixed_function.vector()[:].size/problem.num_component)

vector_diff = mixed_function.vector()[:] - array([1,2,3,4]*n)
error = dot(vector_diff, vector_diff)

print(error)

def test_function():
    assert error < 1e-15
