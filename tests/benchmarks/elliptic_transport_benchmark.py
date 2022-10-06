# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from reaktoro_transport.mesh import MarkedRectangleMesh
from reaktoro_transport.problem import TracerTransportProblem

from dolfin import Expression, inner, interpolate, assemble, Constant
from dolfin import VectorFunctionSpace, Function, norm

class EllipticTransportBenchmark(TracerTransportProblem):
    """
    This benchmark problem is based on the following work: Optimal order
    convergence of a modified BDM1 mixed finite element scheme for reactive
    transport in porous media by Fabian Brunner et. al., 2012, published in
    Advances in Water Resources. doi: 10.1016/j.advwatres.2011.10.001
    """

    def get_mesh_and_markers(self, nx, mesh_type):
        mesh_factory = MarkedRectangleMesh()
        mesh_factory.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
        mesh_factory.set_top_right_coordinates(coord_x = 1.0, coord_y = 1.0)
        mesh_factory.set_number_of_elements(nx, nx)
        mesh_factory.set_mesh_type(mesh_type)

        mesh = mesh_factory.generate_mesh()
        boundary_markers, self.marker_dict = mesh_factory.generate_boundary_markers()
        domain_markers = mesh_factory.generate_domain_markers()

        self.mesh_characteristic_length = 1.0/nx

        return mesh, boundary_markers, domain_markers

    def get_mesh_characterisitic_length(self):
        return self.mesh_characteristic_length

    def set_flow_field(self):
        V = VectorFunctionSpace(self.mesh, "CG", 1)
        self.fluid_velocity = interpolate(Expression(('0.9', '0.9'), degree=1), V)

    def define_problem(self):
        self.set_components('solute')
        self.set_component_fe_space()
        self.set_advection_velocity()
        self.initialize_form()

        self.set_molecular_diffusivity([1.0])
        self.add_implicit_advection(marker=0)
        self.add_implicit_diffusion('solute', marker=0)

        mass_source = '(2.9-1.8*x[0])*x[1]*(1.0-x[1]) + ' + \
                      '(2.9-1.8*x[1])*x[0]*(1.0-x[0])'
        self.add_mass_source(['solute'], [Expression(mass_source, degree=1)])

        self.mark_component_boundary(**{'solute': self.marker_dict.values()})

        # When solving steady-state problems, the diffusivity of the diffusion
        # boundary is a penalty term to the variational form.
        self.add_component_diffusion_bc('solute', diffusivity=Constant(100.0),
                                        values=[Constant(0.0)]*len(self.marker_dict))

    def get_solution(self):
        # To match the rank in mixed spaces,
        # one should supply a list of expressions to the Expression Function.
        expr = Expression(['x[0]*(1.0-x[0])*x[1]*(1.0-x[1])'], degree=1)

        self.solution = Function(self.comp_func_spaces)
        self.solution.assign(interpolate(expr, self.comp_func_spaces))

        return self.solution.copy()

    def get_error_norm(self):
        mass_error = Function(self.comp_func_spaces)

        mass_error.assign(self.fluid_components - self.solution)

        mass_error_norm = norm(mass_error, 'l2')

        return mass_error_norm
