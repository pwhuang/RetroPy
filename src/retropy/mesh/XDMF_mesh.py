# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from dolfinx.mesh import meshtags
from dolfinx.io.utils import XDMFFile
from mpi4py import MPI

class XDMFMesh:
    """This class reads mesh in the XDMF format."""

    def read_mesh(self, filepath: str):
        xdmf_obj = XDMFFile(MPI.COMM_WORLD, filepath)
        self.mesh = xdmf_obj.read_mesh()
        xdmf_obj.close()

        return self.mesh

    def read_boundary_markers(self, filepath: str):
        xdmf_obj = XDMFFile(MPI.COMM_WORLD, filepath)
        self.boundary_markers = xdmf_obj.read_meshtags()
        xdmf_obj.close()

        return self.boundary_markers
