import sys
sys.path.insert(0, '../../')

from reaktoro_transport.mesh import MarkedRectangleMesh
from dolfin import Expression, inner, interpolate, assemble, Constant
from dolfin import Function, norm

class DarcyFlowBenchmark:
    """This benchmark problem is based on Ada Johanne Ellingsrud's masters
    thesis: Preconditioning unified mixed discretization of coupled Darcy-
    Stokes flow. https://www.duo.uio.no/bitstream/handle/10852/45338/paper.pdf
    """

    def get_mesh_and_markers(self, nx):
        mesh_factory = MarkedRectangleMesh()
        mesh_factory.set_bottom_left_coordinates(coord_x = -1.0, coord_y = -1.0)
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

    def set_material_properties(self):
        self.set_permeability(Constant(1.0))
        self.set_fluid_density(1.0)
        self.set_fluid_viscosity(1.0)
        self.set_gravity((0.0, 0.0))

    def set_boundary_conditions(self):
        self.mark_flow_boundary(pressure = [self.marker_dict['left'], self.marker_dict['right']],
                                velocity = [self.marker_dict['top'], self.marker_dict['bottom']])

        self.set_form_and_pressure_bc([Expression(('exp(x[1])*sin(M_PI*x[0])'), degree=1)]*2)
        self.set_velocity_bc([Expression(('sin(M_PI*x[1])', 'cos(M_PI*x[0])'), degree=1)]*2)

    def set_momentum_sources(self):
        momentum_sources = [self._mu/self._k*Expression(('sin(M_PI*x[1])', 'cos(M_PI*x[0])'), degree=1),
                            Expression(('M_PI*cos(M_PI*x[0])*exp(x[1])',
                                        'sin(M_PI*x[0])*exp(x[1])'), degree=1)]

        self.add_momentum_source(momentum_sources)

    def get_solution(self):
        self.sol_pressure = interpolate(Expression('exp(x[1])*sin(M_PI*x[0])', degree=1), self.pressure_func_space)
        self.sol_velocity = interpolate(Expression(('sin(M_PI*x[1])', 'cos(M_PI*x[0])'), degree=1),
                                        self.velocity_func_space)

        return self.sol_pressure, self.sol_velocity

    def get_error_norm(self):
        pressure_error = Function(self.pressure_func_space)
        velocity_error = Function(self.velocity_func_space)

        pressure_error.assign(self.fluid_pressure-self.sol_pressure)
        velocity_error.assign(self.fluid_velocity-self.sol_velocity)

        pressure_error_norm = norm(pressure_error, 'l2')
        velocity_error_norm = norm(velocity_error, 'l2')

        return pressure_error_norm, velocity_error_norm
