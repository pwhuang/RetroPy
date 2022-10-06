# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from reaktoro_transport.mesh import MarkedRectangleMesh
from dolfin import SubDomain, DOLFIN_EPS, refine, MeshFunction, plot

import matplotlib.pyplot as plt

class MeshFactory(MarkedRectangleMesh):
    def __init__(self):
        super().__init__()

    def get_mesh_and_markers(self, nx, ny, mesh_type='triangle'):
        self.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
        self.set_top_right_coordinates(coord_x = 25.0, coord_y = 90.0)
        self.set_number_of_elements(nx, ny)
        self.set_mesh_type(mesh_type)

        self.generate_mesh('left/right')
        #self.refine_mesh()

        boundary_markers, self.marker_dict = self.generate_boundary_markers()
        domain_markers = self.generate_domain_markers()

        return self.mesh, boundary_markers, domain_markers

    def refine_mesh(self):
        class middle(SubDomain):
            def inside(self, x, on_boundary):
                return x[1]>20.0

        cell_markers = MeshFunction('bool', self.mesh, dim=self.mesh.geometric_dimension())

        cell_middle = middle()

        cell_markers.set_all(0)
        cell_middle.mark(cell_markers, 1)

        self.mesh = refine(self.mesh, cell_markers)
