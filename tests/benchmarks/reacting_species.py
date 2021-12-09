from reaktoro_transport.mesh import MarkedRectangleMesh
from reaktoro_transport.problem import TracerTransportProblem

from dolfin import Expression, inner, interpolate, assemble, Constant
from dolfin import VectorFunctionSpace, FunctionSpace, Function, norm
from dolfin import exp, as_vector

class ReactingSpecies(TracerTransportProblem):
    """
    This benchmark problem is based on the following work: Optimal order
    convergence of a modified BDM1 mixed finite element scheme for reactive
    transport in porous media by Fabian Brunner et. al., 2012, published in
    Advances in Water Resources. doi: 10.1016/j.advwatres.2011.10.001
    """

    solution_list = ['x[0]*(2.0-x[0])*pow(x[1], 3)*exp(-0.1*t)/27.0',
                     'pow(x[0]-1.0, 2)*x[1]*x[1]*exp(-0.1*t)/9.0']

    def get_mesh_and_markers(self, nx, mesh_type):
        mesh_factory = MarkedRectangleMesh()
        mesh_factory.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
        mesh_factory.set_top_right_coordinates(coord_x = 2.0, coord_y = 3.0)
        mesh_factory.set_number_of_elements(nx, int(1.5*nx))
        mesh_factory.set_mesh_type(mesh_type)

        mesh = mesh_factory.generate_mesh()
        boundary_markers, self.marker_dict = mesh_factory.generate_boundary_markers()
        domain_markers = mesh_factory.generate_domain_markers()

        self.mesh_characteristic_length = 2.0/nx

        return mesh, boundary_markers, domain_markers

    def get_mesh_characterisitic_length(self):
        return self.mesh_characteristic_length

    def set_flow_field(self):
        V = VectorFunctionSpace(self.mesh, 'CG', 1)
        self.fluid_velocity = interpolate(Expression(('0.0', '-1.0'), degree=1), V)

    def define_problem(self):
        self.set_components('c1', 'c2')
        self.set_component_fe_space()
        self.set_advection_velocity()
        self.initialize_form()

        self.set_molecular_diffusivity([0.1, 0.1])

        sol_expr = Expression(self.solution_list, degree=1, t=0.0)
        self.set_component_ics(sol_expr)

        flux_boundaries = [self.marker_dict['top'],
                           self.marker_dict['left'],
                           self.marker_dict['right']]

        self.mark_component_boundary(**{'c1': flux_boundaries,
                                        'c2': flux_boundaries,
                                        'outlet': [self.marker_dict['bottom']]})

        self.t_end = Constant(0.0)

    def add_physics_to_form(self, u, kappa=Constant(1.0), f_id=0):

        self.add_explicit_advection(u, kappa, marker=0, f_id=f_id)
        self.add_outflow_bc(f_id)

        self.add_implicit_diffusion('c1', kappa, marker=0, f_id=f_id)
        self.add_implicit_diffusion('c2', kappa, marker=0, f_id=f_id)

        source_c1 = 'x[1]*(x[0]-2.0)*x[0]*(0.1*pow(x[1], 2) + 3.0*x[1]'+\
                    '+ 0.6) + 0.2*pow(x[1], 3)'

        source_c2 = '-pow(x[0]-1.0, 2)*(0.1*x[1]*x[1] + 2.0*x[1] + 0.2)'+\
                    '-0.2*x[1]*x[1]'

        f_of_t = exp(Constant(-0.1)*self.t_end)

        self.add_mass_source(['c1'], [Expression(source_c1, degree=1)
                                      *f_of_t/Constant(27)],
                                      kappa, f_id)
        self.add_mass_source(['c2'], [Expression(source_c2, degree=1)
                                      *f_of_t/Constant(9)],
                                      kappa, f_id)

        boundary_source_c1 = 'x[0]*(2-x[0])'
        boundary_source_c2 = 'pow(x[0]-1, 2)'
        zero = Constant(0.0)

        self.add_component_advection_bc('c1', [Expression(boundary_source_c1, degree=1)
                                               *f_of_t,
                                               zero, zero],
                                               kappa, f_id)

        self.add_component_advection_bc('c2', [Expression(boundary_source_c2, degree=1)
                                               *f_of_t,
                                               zero, zero],
                                               kappa, f_id)

        diff_expr_c1 = [Expression(boundary_source_c1, degree=1)*f_of_t,
                        zero, zero]

        self.add_component_diffusion_bc('c1', self._D[0], diff_expr_c1, kappa, f_id)

        diff_expr_c2 = [(-2.0/3)*self._D[1]*Expression(boundary_source_c2, degree=1)*f_of_t,
                        -self._D[1]*Expression('x[1]*x[1]*2/9', degree=1)*f_of_t,
                        -self._D[1]*Expression('x[1]*x[1]*2/9', degree=1)*f_of_t]
        self.add_component_flux_bc('c2', diff_expr_c2, kappa, f_id)

        self.add_sources(as_vector([-Constant(1.0)*u[0]*u[1]*u[1],
                                    -Constant(2.0)*u[0]*u[1]*u[1]]),
                         kappa, f_id)

    def generate_solution(self, t):
        expr = Expression(self.solution_list, degree=1, t=t)

        self.solution = Function(self.comp_func_spaces)
        self.solution.assign(interpolate(expr, self.comp_func_spaces))

    def get_solution(self):
        return self.solution

    def get_error_norm(self):
        mass_error = Function(self.comp_func_spaces)

        mass_error.assign(self.fluid_components - self.solution)

        mass_error_norm = norm(mass_error, 'l2')

        return mass_error_norm
