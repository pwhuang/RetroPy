from dolfin import IntervalMesh, MeshFunction, Point, DOLFIN_EPS, FunctionSpace
from dolfin.cpp.mesh import CellType

from . import *

class MarkedLineMesh(MarkerCollection):
    """"""

    def __init__(self):
        """Initialize default values."""

        super().__init__()
        self.xmin = 0.0
        self.xmax = 1.0
        self.num_elements_x = 10

    def set_left_coordinates(self, coord_x):
        self.xmin = coord_x

    def set_right_coordinates(self, coord_x):
        self.xmax = coord_x

    def set_number_of_elements(self, num_elements_x):
        self.num_elements_x = num_elements_x

    def generate_mesh(self):
        self.mesh = IntervalMesh(self.num_elements_x, self.xmin, self.xmax)

        return self.mesh

    def set_boundary_markers(self):
        self.boundary_markers = MeshFunction('size_t', self.mesh,
                                             dim=self.mesh.geometric_dimension()-1)

        self.boundary_markers.set_all(0)

        right_marker = self.RightBoundary(self.xmax)
        left_marker = self.LeftBoundary(self.xmin)

        right_marker.mark(self.boundary_markers, 2)
        left_marker.mark(self.boundary_markers, 1)

        marker_dict = {'right': 2, 'left': 1}

        return self.boundary_markers, marker_dict

    def set_domain_markers(self):
        self.domain_markers = MeshFunction('size_t', self.mesh,
                                           dim=self.mesh.geometric_dimension())

        self.domain_markers.set_all(0)

        return self.domain_markers
