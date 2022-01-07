from dolfin import Mesh, XDMFFile
from . import *

class XDMFMesh:
    """This class reads mesh in the XDMF format."""

    def read_mesh(self, filepath: str):
        self.mesh = Mesh()

        xdmf_obj = XDMFFile(MPI.comm_world, filepath)
        xdmf_obj.read(self.mesh)
        xdmf_obj.close()

        return self.mesh

    def read_boundary_markers(self, filepath: str):
        self.boundary_markers = MeshFunction('size_t', self.mesh,
                                             dim=self.mesh.geometric_dimension()-1)

        xdmf_obj = XDMFFile(MPI.comm_world, filepath)
        xdmf_obj.read(boundary_markers)
        xdmf_obj.close()

        return self.boundary_markers
