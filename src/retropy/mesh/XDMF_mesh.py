# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from mpi4py import MPI
from dolfinx.io.utils import XDMFFile

class XDMFMesh:
    """This class reads mesh in the XDMF format."""

    def read_mesh(self, filepath: str, meshname: str):
        xdmf_obj = XDMFFile(MPI.COMM_WORLD, filepath, 'r')
        self.mesh = xdmf_obj.read_mesh(name=meshname)
        xdmf_obj.close()

        return self.mesh

    def read_boundary_markers(self, filepath: str):
        pass
        # TODO: Figure out how to load boundary markers using new dolfinx IO.
