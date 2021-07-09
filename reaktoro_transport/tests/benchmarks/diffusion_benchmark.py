import sys
sys.path.insert(0, '../../')

from reaktoro_transport.tests.benchmarks import EllipticTransportBenchmark

from dolfin import Expression, inner, interpolate, assemble, Constant
from dolfin import VectorFunctionSpace, Function, norm

class DiffusionBenchmark(EllipticTransportBenchmark):
    """"""

    def set_flow_field(self):
        V = VectorFunctionSpace(self.mesh, "CG", 1)
        self.fluid_velocity = interpolate(Expression(('0.0', '0.0'), degree=1), V)

    def define_problem(self, problem_type):
        self.set_components('solute')
        self.set_component_fe_space()
        self.initialize_form(problem_type)

        self.add_implicit_diffusion('solute', diffusivity=1.0, marker=0)

        mass_source = '2.0*M_PI*M_PI*sin(M_PI*x[0])*sin(M_PI*x[1])'
        self.add_mass_source([Expression(mass_source, degree=1)])

        self.mark_component_boundary(**{'solute': self.marker_dict.values()})

    def set_problem_bc(self):
        """
        This problem requires 0 Dirichlet bc on all boundaries.
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

        return self.solution.copy()
