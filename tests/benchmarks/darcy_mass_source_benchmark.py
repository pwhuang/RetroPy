from reaktoro_transport.mesh import MarkedRectangleMesh
from dolfin import Expression, inner, interpolate, assemble, Constant
from dolfin import Function, norm

import matplotlib.pyplot as plt
from dolfin.common.plotting import mplot_function

class DarcyMassSourceBenchmark:
    """This benchmark problem studies the effect of mass source (e.g., density
    change over time) on Darcy flow.
    """

    def get_mesh_and_markers(self, nx):
        mesh_factory = MarkedRectangleMesh()
        mesh_factory.set_bottom_left_coordinates(coord_x = -1.0, coord_y = -1.0)
        mesh_factory.set_top_right_coordinates(coord_x = 1.0, coord_y = 1.0)
        mesh_factory.set_number_of_elements(nx, nx)
        mesh_factory.set_mesh_type('triangle')

        mesh = mesh_factory.generate_mesh()
        boundary_markers, self.marker_dict = mesh_factory.generate_boundary_markers()
        domain_markers = mesh_factory.generate_domain_markers()

        self.mesh_characteristic_length = 1.0/nx

        return mesh, boundary_markers, domain_markers

    def get_mesh_characterisitic_length(self):
        return self.mesh_characteristic_length

    def set_material_properties(self):
        self.set_porosity(Constant(1.0))
        self.set_permeability(Constant(1.0))
        self.set_fluid_density(Constant(1.0))
        self.set_fluid_viscosity(1.0)
        self.set_gravity((0.0, 0.0))

    def set_boundary_conditions(self):
        self.mark_flow_boundary(pressure = [],
                                velocity = [self.marker_dict['left'], self.marker_dict['right'],
                                            self.marker_dict['top'], self.marker_dict['bottom']])

        self.set_pressure_bc([])
        self.generate_form()
        self.generate_residual_form()
        self.set_velocity_bc([Constant([0.0, 0.0])]*4)

    def set_mass_sources(self):
        mass_sources = [Expression(('2*M_PI*M_PI*cos(M_PI*x[0])*cos(M_PI*x[1])'), degree=1)]

        self.add_mass_source(mass_sources)
        self.add_mass_source_to_residual_form(mass_sources)

    def get_solution(self):
        self.sol_pressure = interpolate(Expression('cos(M_PI*x[0])*cos(M_PI*x[1])', degree=1),
                                        self.pressure_func_space)
        self.sol_velocity = interpolate(Expression(['M_PI*sin(M_PI*x[0])*cos(M_PI*x[1])',
                                                    'M_PI*cos(M_PI*x[0])*sin(M_PI*x[1])'], degree=1),
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

    def plot_pv_solution(self):
        fig, ax = plt.subplots(1, 2, figsize=(8,4))
        cb = mplot_function(ax[0], self.get_fluid_pressure())
        mplot_function(ax[1], self.get_fluid_velocity())
        fig.colorbar(cb)
        plt.show()
