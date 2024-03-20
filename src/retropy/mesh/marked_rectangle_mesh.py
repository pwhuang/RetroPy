# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from dolfinx.mesh import (
    create_rectangle,
    locate_entities,
    locate_entities_boundary,
    meshtags,
    CellType,
    DiagonalType,
    GhostMode,
)
from mpi4py import MPI
import numpy as np


class MarkedRectangleMesh:
    """"""

    def __init__(self):
        pass

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
        if mesh_type == "triangle":
            self.mesh_type = CellType.triangle
        elif mesh_type == "quadrilateral":
            self.mesh_type = CellType.quadrilateral
        else:
            raise Exception("This class supports 'triangle' and 'quadrilateral' mesh.")

    def locate_and_mark_boundaries(self, boundary_eps=1e-8):
        right_marker = lambda x: np.isclose(x[0], self.xmax, atol=boundary_eps)
        top_marker = lambda x: np.isclose(x[1], self.ymax, atol=boundary_eps)
        left_marker = lambda x: np.isclose(x[0], self.xmin, atol=boundary_eps)
        bottom_marker = lambda x: np.isclose(x[1], self.ymin, atol=boundary_eps)

        self.marker_dict = {"right": 1, "top": 2, "left": 3, "bottom": 4}

        self.locator_dict = {
            "right": right_marker,
            "top": top_marker,
            "left": left_marker,
            "bottom": bottom_marker,
        }

        return self.marker_dict, self.locator_dict

    def generate_mesh(self, mesh_shape="right"):
        shape_dict = {
            "right": DiagonalType.right,
            "left": DiagonalType.left,
            "left/right": DiagonalType.left_right,
            "right/left": DiagonalType.right_left,
            "crossed": DiagonalType.crossed,
        }

        self.mesh = create_rectangle(
            MPI.COMM_WORLD,
            [self.bottom_left_point, self.top_right_point],
            [self.num_elements_x, self.num_elements_y],
            self.mesh_type,
            diagonal=shape_dict[mesh_shape],
            ghost_mode=GhostMode.shared_facet,
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
