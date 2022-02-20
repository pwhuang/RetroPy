from reaktoro_transport.mesh import XDMFMesh, MarkedRectangleMesh
from dolfin import (SubDomain, DOLFIN_EPS, refine, MeshFunction, assemble,
                    Constant, Function, MPI)

from dolfin import FunctionSpace, TestFunction, FacetArea, Measure
import numpy as np

class MeshFactory(MarkedRectangleMesh):
    def __init__(self):
        super().__init__()

    def get_mesh_and_markers(self, nx, ny):
        # try:
        #     self.read_mesh(filepath)
        # except:
        #     raise Exception('filepath does not contain any mesh.'
        #                     'Please run generate_mesh.py ')

        self.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
        self.set_top_right_coordinates(coord_x = 100.0, coord_y = 50.0)
        self.set_number_of_elements(nx, ny)
        self.set_mesh_type('triangle')

        self.generate_mesh('crossed')
        #self.refine_mesh()
        self.boundary_markers, self.marker_dict = self.generate_boundary_markers()
        #self.set_boundary_markers(self.boundary_markers)
        domain_markers = self.generate_domain_markers()

        return self.mesh, self.boundary_markers, domain_markers

    def generate_domain_markers(self):
        return super().generate_domain_markers()

    def refine_mesh(self):
        class where_to_refine(SubDomain):
            def inside(self, x, on_boundary):
                return x[1]>0.0

        cell_markers = MeshFunction('bool', self.mesh, dim=self.mesh.geometric_dimension())
        cell_markers.set_all(1)
        #where_to_refine().mark(cell_markers, 1)

        self.mesh = refine(self.mesh, cell_markers)

    def mark_inflow_boundary_cells(self):
        ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundary_markers)

        DG0_space = FunctionSpace(self.mesh, 'DG', 0)
        w = TestFunction(DG0_space)

        one = Constant(1.0)
        facet_area = FacetArea(self.mesh)

        cell_marker = assemble(w*one/facet_area*ds(self.marker_dict['top'])).get_local()[:]
        cell_marker = np.array(cell_marker, dtype=float)

        idx = np.arange(0, cell_marker.size)

        boundary_cell_idx = idx[cell_marker > 1e-10]
        domain_cell_idx = idx[cell_marker < 1e-10]

        return boundary_cell_idx, domain_cell_idx
