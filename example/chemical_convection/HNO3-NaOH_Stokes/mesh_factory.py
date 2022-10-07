# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from retropy.mesh import MarkedRectangleMesh, XDMFMesh
from dolfin import SubDomain, DOLFIN_EPS, refine, MeshFunction, plot

import matplotlib.pyplot as plt

class MeshFactory(MarkedRectangleMesh, XDMFMesh):
    def __init__(self):
        super().__init__()

    def get_mesh_and_markers(self):
        try:
            self.read_mesh('pore_mesh.xdmf')
        except:
            raise Exception('filepath does not contain mesh.')

        self.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
        self.set_top_right_coordinates(coord_x = 25.0, coord_y = 90.0)

        #self.mesh.scale(1.0)
        #self.refine_mesh()

        boundary_markers, self.marker_dict = self.generate_boundary_markers()
        domain_markers = self.generate_domain_markers()

        return self.mesh, boundary_markers, domain_markers

    def generate_boundary_markers(self, boundary_eps=1e-8):
        self.boundary_markers = MeshFunction('size_t', self.mesh,
                                             dim=self.mesh.geometric_dimension()-1)

        self.boundary_markers.set_all(0)

        all_marker = self.AllBoundary()

        # This index is resevered for all the not explicitly marked cells
        all_marker.mark(self.boundary_markers, 555)
        marker_dict = {'noslip': 555}

        return self.boundary_markers, marker_dict

    def refine_mesh(self):
        cell_markers = MeshFunction('bool', self.mesh, dim=self.mesh.geometric_dimension())
        cell_markers.set_all(1)
        self.mesh = refine(self.mesh, cell_markers)
