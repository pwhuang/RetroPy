# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from dolfinx.mesh import create_interval, locate_entities, meshtags
from mpi4py import MPI
import numpy as np
from . import MarkerCollection

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
        self.mesh = create_interval(MPI.COMM_WORLD, self.num_elements_x,
                                    [self.xmin, self.xmax])

        return self.mesh

    def generate_boundary_markers(self):
        right_marker = lambda x: np.isclose(x[0], self.xmax)
        left_marker = lambda x: np.isclose(x[0], self.xmin)

        marker_dict = {'right': 2, 'left': 1}

        boundaries = [(marker_dict['right'], right_marker),
                      (marker_dict['left'], left_marker)]

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
