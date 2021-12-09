import os
os.environ['OMP_NUM_THREADS'] = '1'

from dolfin import FunctionSpace, DOLFIN_EPS
from numpy import all
from reaktoro_transport.mesh import MarkedRectangleMesh

class MarkedRectangleMeshTest(MarkedRectangleMesh):

    def __init__(self):
        super().__init__()

    def get_boundary_markers_values(self):
        cr_space = FunctionSpace(self.mesh, 'CR', 1)
        cr_dof = cr_space.dofmap().dofs(self.mesh, 1)

        coord_x = cr_space.tabulate_dof_coordinates()[cr_dof,0]
        coord_y = cr_space.tabulate_dof_coordinates()[cr_dof,1]

        idx_left   = coord_x < self.xmin + DOLFIN_EPS
        idx_bottom = coord_y < self.ymin + DOLFIN_EPS
        idx_right  = coord_x > self.xmax - DOLFIN_EPS
        idx_top    = coord_y > self.ymax - DOLFIN_EPS

        marker_value_left = self.boundary_markers.array()[idx_left]
        marker_value_bottom = self.boundary_markers.array()[idx_bottom]
        marker_value_right = self.boundary_markers.array()[idx_right]
        marker_value_top = self.boundary_markers.array()[idx_top]

        return (marker_value_left, marker_value_bottom,
                marker_value_right, marker_value_top)


mesh_factory = MarkedRectangleMeshTest()

mesh_factory.set_bottom_left_coordinates(coord_x = -1.0, coord_y = -1.0)
mesh_factory.set_top_right_coordinates(coord_x = 1.0, coord_y = 1.0)
mesh_factory.set_number_of_elements(10, 10)
mesh_factory.set_mesh_type('triangle')
mesh = mesh_factory.generate_mesh()
boundary_markers, marker_dict = mesh_factory.generate_boundary_markers()

left, bottom, right, top = mesh_factory.get_boundary_markers_values()

def test_function():
    assert all(right==1) and all(top==2) and all(left==3) and all(bottom==4)
