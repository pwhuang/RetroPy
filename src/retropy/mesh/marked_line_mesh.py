# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from dolfinx.mesh import (
    create_interval,
    locate_entities,
    meshtags,
    locate_entities_boundary,
)

from mpi4py import MPI
import numpy as np


class MarkedLineMesh:
    """"""

    def __init__(self, xmin, xmax, num_elements):
        self.set_left_coordinates(xmin)
        self.set_right_coordinates(xmax)
        self.set_number_of_elements(num_elements)
        self.locate_and_mark_boundaries()

        self.generate_mesh()
        self.generate_boundary_markers()
        self.generate_interior_markers()
        self.generate_domain_markers()

    def set_left_coordinates(self, coord_x):
        self.xmin = coord_x

    def set_right_coordinates(self, coord_x):
        self.xmax = coord_x

    def set_number_of_elements(self, num_elements):
        self.num_elements = num_elements

    def locate_and_mark_boundaries(self, boundary_eps=1e-8):
        right_marker = lambda x: np.isclose(x[0], self.xmax, atol=boundary_eps)
        left_marker = lambda x: np.isclose(x[0], self.xmin, atol=boundary_eps)

        self.marker_dict = {"right": 2, "left": 1}
        self.locator_dict = {"right": right_marker, "left": left_marker}

        return self.marker_dict, self.locator_dict

    def generate_mesh(self):
        self.mesh = create_interval(
            MPI.COMM_WORLD, self.num_elements, [self.xmin, self.xmax]
        )

        return self.mesh

    def generate_boundary_markers(self):
        self.facet_dict = {}
        face_dim = self.mesh.topology.dim - 1

        for key, locator in self.locator_dict.items():
            self.facet_dict[key] = locate_entities_boundary(
                self.mesh, face_dim, locator
            )

        self.boundary_markers = self.generate_facet_tags(
            self.mesh, self.locator_dict, self.marker_dict
        )

        return self.boundary_markers

    def generate_interior_markers(self):
        locator_dict = {"interior": lambda x: x[0] < np.inf}
        marker_dict = {"interior": 0}

        self.interior_markers = self.generate_facet_tags(
            self.mesh, locator_dict, marker_dict
        )

        return self.interior_markers

    def generate_domain_markers(self):
        volume_dim = self.mesh.topology.dim

        cell_indices = locate_entities(self.mesh, volume_dim, lambda x: x[0] < np.inf)
        cell_value = np.zeros_like(cell_indices, dtype=np.int32)
        self.domain_markers = meshtags(self.mesh, volume_dim, cell_indices, cell_value)

        return self.domain_markers

    @staticmethod
    def generate_facet_tags(mesh, locator_dict, marker_dict):
        facet_indices, facet_markers = [], []
        fdim = mesh.topology.dim - 1

        for key, locator in locator_dict.items():
            facets = locate_entities(mesh, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker_dict[key], dtype=np.int32))

        facet_indices = np.hstack(facet_indices)
        facet_markers = np.hstack(facet_markers)
        sorted_facets = np.argsort(facet_indices)

        return meshtags(
            mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
        )
