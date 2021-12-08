import sys
sys.path.insert(0, '../../')

from reaktoro_transport.mesh import MarkedRectangleMesh
from reaktoro_transport.problem import TracerTransportProblemExp

from dolfin import Expression, inner, interpolate, assemble, Constant
from dolfin import VectorFunctionSpace, Function, norm

from numpy import exp

class DiffusionBenchmark:
    """"""

    def get_mesh_and_markers(self, nx, mesh_type):
        mesh_factory = MarkedRectangleMesh()
        mesh_factory.set_bottom_left_coordinates(coord_x = 0.0, coord_y = 0.0)
        mesh_factory.set_top_right_coordinates(coord_x = 1.0, coord_y = 1.0)
        mesh_factory.set_number_of_elements(nx, nx)
        mesh_factory.set_mesh_type(mesh_type)

        mesh = mesh_factory.generate_mesh(mesh_shape='crossed')
        boundary_markers, self.marker_dict = mesh_factory.generate_boundary_markers()
        domain_markers = mesh_factory.generate_domain_markers()

        self.mesh_characteristic_length = 1.0/nx

        return mesh, boundary_markers, domain_markers

    def get_mesh_characterisitic_length(self):
        return self.mesh_characteristic_length

    def set_flow_field(self):
        V = VectorFunctionSpace(self.mesh, "CG", 1)
        self.fluid_velocity = interpolate(Expression(('0.0', '0.0'), degree=1), V)

    def define_problem(self):
        self.set_components('solute')
        self.set_component_fe_space()
        self.initialize_form()

        self.set_molecular_diffusivity([1.0])
        self.add_implicit_diffusion('solute', marker=0)

        mass_source = '2.0*M_PI*M_PI*sin(M_PI*x[0])*sin(M_PI*x[1])'
        self.add_mass_source(['solute'], [Expression(mass_source, degree=1)])

        self.mark_component_boundary(**{'solute': self.marker_dict.values()})

    def add_physics_to_form(self, u0):
        pass

    def add_time_derivatives(self, u0):
        pass

    def set_problem_bc(self):
        """
        This problem requires 1.0 Dirichlet bc on all boundaries.
        Since the implementation of Dirichlet bcs depends on the solving scheme,
         this method should be defined in tests.
        """
        num_marked_boundaries = len(self.marker_dict)
        values = [Constant(0.0)]*num_marked_boundaries

        return values

    def get_solution(self):
        # To match the rank in mixed spaces,
        # one should supply a list of expressions to the Expression Function.
        expr = Expression(['sin(M_PI*x[0])*sin(M_PI*x[1])'], degree=1)

        self.solution = Function(self.comp_func_spaces)
        self.solution.assign(interpolate(expr, self.comp_func_spaces))

        return self.solution

    def get_error_norm(self):
        mass_error = Function(self.comp_func_spaces)
        mass_error.assign(self.fluid_components - self.solution)

        mass_error_norm = norm(mass_error, 'l2')

        return mass_error_norm
