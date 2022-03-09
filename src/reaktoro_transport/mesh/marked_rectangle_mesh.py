from dolfin import (RectangleMesh, MeshFunction, Point, DOLFIN_EPS,
                    FunctionSpace, MPI)
from dolfin.cpp.mesh import CellType

from . import MarkerCollection

class MarkedRectangleMesh(MarkerCollection):
    """"""

    def __init__(self):
        super().__init__()

    def set_bottom_left_coordinates(self, coord_x, coord_y):
        self.bottom_left_point = Point(coord_x, coord_y)
        self.xmin = coord_x
        self.ymin = coord_y

    def set_top_right_coordinates(self, coord_x, coord_y):
        self.top_right_point = Point(coord_x, coord_y)
        self.xmax = coord_x
        self.ymax = coord_y

    def set_number_of_elements(self, num_elements_x, num_elements_y):
        self.num_elements_x = num_elements_x
        self.num_elements_y = num_elements_y

    def set_mesh_type(self, mesh_type):
        if mesh_type == 'triangle':
            self.mesh_type = CellType.Type.triangle
        elif mesh_type == 'quadrilateral':
            self.mesh_type = CellType.Type.quadrilateral
        else:
            raise Exception("This class supports 'triangle' and 'quadrilateral' mesh.")

    def generate_mesh(self, mesh_shape='right'):
        self.mesh = RectangleMesh.create(MPI.comm_world,
                                         [self.bottom_left_point, self.top_right_point],
                                         [self.num_elements_x, self.num_elements_y],
                                         self.mesh_type,
                                         mesh_shape)

        return self.mesh

    def generate_boundary_markers(self, boundary_eps=1e-8):
        self.boundary_markers = MeshFunction('size_t', self.mesh,
                                             dim=self.mesh.geometric_dimension()-1)

        self.boundary_markers.set_all(0)

        right_marker = self.RightBoundary(self.xmax, boundary_eps)
        top_marker = self.TopBoundary(self.ymax, boundary_eps)
        left_marker = self.LeftBoundary(self.xmin, boundary_eps)
        bottom_marker = self.BottomBoundary(self.ymin, boundary_eps)

        right_marker.mark(self.boundary_markers, 1)
        top_marker.mark(self.boundary_markers, 2)
        left_marker.mark(self.boundary_markers, 3)
        bottom_marker.mark(self.boundary_markers, 4)

        marker_dict = {'right': 1, 'top': 2, 'left': 3, 'bottom': 4}

        return self.boundary_markers, marker_dict

    def generate_domain_markers(self):
        self.domain_markers = MeshFunction('size_t', self.mesh,
                                           dim=self.mesh.geometric_dimension())

        self.domain_markers.set_all(0)

        return self.domain_markers

    def plot_boundary_markers(self, ax, s=20, colormap='Blues'):
        """Plots boundary markers given matplotlib AxesSubpot instance."""

        cr_space = FunctionSpace(self.mesh, 'CR', 1)
        cr_dof = cr_space.dofmap().dofs(self.mesh, 1)

        coord_x = cr_space.tabulate_dof_coordinates()[cr_dof, 0]
        coord_y = cr_space.tabulate_dof_coordinates()[cr_dof, 1]

        cb = ax.scatter(coord_x, coord_y,
                        c=self.boundary_markers.array(), cmap=colormap, s=s)

        return cb
