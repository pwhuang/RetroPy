# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from dolfinx.mesh import create_rectangle, locate_entities, meshtags
from dolfinx.mesh import CellType
from mpi4py import MPI
import numpy as np

from . import MarkerCollection

class MarkedRectangleMesh(MarkerCollection):
    """"""

    def __init__(self):
        super().__init__()

    def set_bottom_left_coordinates(self, coord_x, coord_y):
        self.bottom_left_point = (coord_x, coord_y)
        self.xmin = coord_x
        self.ymin = coord_y

    def set_top_right_coordinates(self, coord_x, coord_y):
        self.top_right_point = (coord_x, coord_y)
        self.xmax = coord_x
        self.ymax = coord_y

    def set_number_of_elements(self, num_elements_x, num_elements_y):
        self.num_elements_x = num_elements_x
        self.num_elements_y = num_elements_y

    def set_mesh_type(self, mesh_type):
        if mesh_type == 'triangle':
            self.mesh_type = CellType.triangle
        elif mesh_type == 'quadrilateral':
            self.mesh_type = CellType.quadrilateral
        else:
            raise Exception("This class supports 'triangle' and 'quadrilateral' mesh.")

    def generate_mesh(self, mesh_shape='right'):
        self.mesh = create_rectangle(MPI.COMM_WORLD,
                                     [self.bottom_left_point, self.top_right_point],
                                     [self.num_elements_x, self.num_elements_y],
                                     self.mesh_type, diagonal=mesh_shape)

        return self.mesh

    def generate_boundary_markers(self, boundary_eps=1e-8):
        right_marker = lambda x: np.isclose(x[0], self.xmax)
        top_marker = lambda x: np.isclose(x[1], self.ymax)
        left_marker = lambda x: np.isclose(x[0], self.xmin)
        bottom_marker = lambda x: np.isclose(x[1], self.ymin)

        marker_dict = {'right': 1, 'top': 2, 'left': 3, 'bottom': 4}

        boundaries = [(marker_dict['right'], right_marker),
                      (marker_dict['top'], top_marker),
                      (marker_dict['left'], left_marker),
                      (marker_dict['bottom'], bottom_marker)]

        facet_indices, facet_markers = [], []
        fdim = self.mesh.topology.dim - 1

        for (marker, locator) in boundaries:
            facets = locate_entities(self.mesh, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))

        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        self.boundary_markers = meshtags(self.mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

        return self.boundary_markers, marker_dict

    def generate_domain_markers(self):
        cell_indices = locate_entities(self.mesh, self.mesh.topology.dim, lambda x: x[0] < np.inf)
        cell_value = np.zeros_like(cell_indices, dtype=np.int32)
        self.domain_markers = meshtags(self.mesh, self.mesh.topology.dim, cell_indices, cell_value)

        return self.domain_markers

    def plot_boundary_markers(self, ax, s=20, colormap='Blues'):
        """Plots boundary markers given matplotlib AxesSubpot instance."""

        pass

        # TODO: Fix this method.
        # cr_space = FunctionSpace(self.mesh, 'CR', 1)
        # cr_dof = cr_space.dofmap().dofs(self.mesh, 1)

        # coord_x = cr_space.tabulate_dof_coordinates()[cr_dof, 0]
        # coord_y = cr_space.tabulate_dof_coordinates()[cr_dof, 1]

        # cb = ax.scatter(coord_x, coord_y,
        #                 c=self.boundary_markers.array(), cmap=colormap, s=s)

        # return cb
