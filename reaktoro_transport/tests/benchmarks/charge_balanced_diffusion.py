import sys
sys.path.insert(0, '../../')

from reaktoro_transport.mesh import MarkedLineMesh

from dolfin import Expression, inner, interpolate, Constant
from dolfin import VectorFunctionSpace, Function, norm

class ChargeBalancedDiffusion:
    """
    This benchmark problem tests whether two species diffusing with different
    molecular diffusivities diffuse at the same rate when charge balance of the
    species are involved. Particularly, it demonstrates using the Crank-Nicolson
     timestepping to solve diffusion problems.
    """

    expression_string = ['exp(-pow(x[0]-x0, 2)/(4*D*t))/sqrt(4*M_PI*D*t)',
                         'exp(-pow(x[0]-x0, 2)/(4*D*t))/sqrt(4*M_PI*D*t)']

    center_of_mass = 0.5

    def get_mesh_and_markers(self, nx):
        mesh_factory = MarkedLineMesh()
        mesh_factory.set_left_coordinates(coord_x = 0.0)
        mesh_factory.set_right_coordinates(coord_x = 1.0)
        mesh_factory.set_number_of_elements(nx)

        mesh = mesh_factory.generate_mesh()
        boundary_markers, self.marker_dict = mesh_factory.generate_boundary_markers()
        domain_markers = mesh_factory.generate_domain_markers()

        self.mesh_characteristic_length = 1.0/nx

        return mesh, boundary_markers, domain_markers

    def get_mesh_characterisitic_length(self):
        return self.mesh_characteristic_length

    def set_flow_field(self):
        self.velocity_func_space = VectorFunctionSpace(self.mesh, "CG", 1)
        V = self.velocity_func_space

        self.fluid_velocity = interpolate(Expression(['0.0', ], degree=1), V)

    def define_problem(self, t0):
        self.set_components('Na+', 'Cl-')
        self.set_component_fe_space()
        self.initialize_form()

        self.D_Na = 1.33e-3
        self.D_Cl = 2.03e-3
        self.Z_Na = 1
        self.Z_Cl = -1

        self.avg_D = (self.Z_Na + abs(self.Z_Cl))/\
                     (self.Z_Na/self.D_Na + abs(self.Z_Cl)/self.D_Cl)

        self.set_molecular_diffusivity([self.D_Na, self.D_Cl])
        self.set_charge([self.Z_Na, self.Z_Cl])
        self.set_molar_mass([1.0, 1.0])

        self.mark_component_boundary(**{'Na+': self.marker_dict.values(),
                                        'Cl-': self.marker_dict.values()})

        expr = Expression(self.expression_string,
                          D=self.avg_D, x0=self.center_of_mass, t=t0, degree=1)
        self.set_component_ics(expr)

    def add_physics_to_form(self, u):
        theta = Constant(0.5)
        one = Constant(1.0)

        self.add_implicit_diffusion('Na+', kappa=theta, marker=0)
        self.add_explicit_diffusion('Na+', u, kappa=one-theta, marker=0)
        self.add_implicit_diffusion('Cl-', kappa=theta, marker=0)
        self.add_explicit_diffusion('Cl-', u, kappa=one-theta, marker=0)

        self.add_explicit_charge_balanced_diffusion(u, kappa=theta, marker=0)
        self.add_semi_implicit_charge_balanced_diffusion(u, kappa=one-theta, marker=0)

    def get_solution(self, t_end):
        expr = Expression(self.expression_string,
                          D=self.avg_D, x0=self.center_of_mass, t=t_end, degree=1)

        self.solution = Function(self.comp_func_spaces)
        self.solution.assign(interpolate(expr, self.comp_func_spaces))

        return self.solution.copy()

    def get_error_norm(self):
        mass_error = Function(self.comp_func_spaces)

        mass_error.assign(self.fluid_components - self.solution)

        mass_error_norm = norm(mass_error, 'l2')

        return mass_error_norm
