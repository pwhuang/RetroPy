import sys
sys.path.insert(0, '../../')

from reaktoro_transport.mesh import MarkedRectangleMesh
from reaktoro_transport.problem import TracerTransportProblem
from dolfin import Expression, inner, interpolate, assemble, Constant
from dolfin import Function, norm

class EllipticTransportBenchmark(TracerTransportProblem):
    """This benchmark problem is based on the following work: Optimal order
    convergence of a modified BDM1 mixed finite element scheme for reactive
    transport in porous media by Fabian Brunner et. al., 2012, published in
    Advances in Water Resources. doi: 10.1016/j.advwatres.2011.10.001
    """

    def get_mesh_and_markers(self, nx):
        mesh_factory = MarkedRectangleMesh()
        mesh_factory.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
        mesh_factory.set_top_right_coordinates(coord_x = 1.0, coord_y = 1.0)
        mesh_factory.set_number_of_elements(nx, nx)
        mesh_factory.set_mesh_type('triangle')

        mesh = mesh_factory.generate_mesh()
        boundary_markers, self.marker_dict = mesh_factory.set_boundary_markers()
        domain_markers = mesh_factory.set_domain_markers()

        self.mesh_characteristic_length = 2.0/nx

        return mesh, boundary_markers, domain_markers

    def get_mesh_characterisitic_length(self):
        return self.mesh_characteristic_length

    def set_flow_field(self):
        V = VectorFunctionSpace(self.mesh, "Crouzeix-Raviart", 1)
        self.fluid_velocity = interpolate(Expression(('0.9', '0.9'), degree=1), V)

    def define_problem(self):
        self.set_components('solute')
        self.set_component_fe_space()
        self.initialize_form()

        self.add_advection(marker=0)
        self.add_diffusion('solute', diffusivity=1.0, marker=0)

        self.mark_component_boundary(**{'solute': self.marker_dict.values()})
