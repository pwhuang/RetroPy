# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedRectangleMesh

class MeshFactory(MarkedRectangleMesh):
    def __init__(self, nx, ny, mesh_type='quadrilateral', mesh_shape='left/right'):
        self.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
        self.set_top_right_coordinates(coord_x = 25.0, coord_y = 90.0)
        self.set_number_of_elements(nx, ny)
        self.set_mesh_type(mesh_type)
        self.locate_and_mark_boundaries()

        self.generate_mesh(mesh_shape)

        self.generate_boundary_markers()
        self.generate_interior_markers()
        self.generate_domain_markers()
