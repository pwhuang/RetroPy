import sys
sys.path.insert(0, '../')

from benchmarks import EllipticTransportBenchmark

from dolfin import Expression, inner, interpolate, assemble, Constant
from dolfin import VectorFunctionSpace, FunctionSpace, Function, norm, cos, pi
from dolfin import DOLFIN_EPS

class RotatingCone(EllipticTransportBenchmark):
    """This benchmark problem is inspired by Randall J. Leveque's work:
    High-resolution conservative algorithms for advection in incompressible
    flow, published in SIAM Journal on Numerical Analysis.
    doi: doi.org/10.1137/0733033
    """

    sol_expr = Expression(['(pow((x[0]-x0)/sigma, 2) + pow((x[1]-x1)/sigma, 2) < 0.1)' +
                          ' ? cos(M_PI*(x[0]-x0)/sigma) * cos(M_PI*(x[1]-x1)/sigma) : eps', ],
                          degree=1, x0=0.5, x1=0.8, sigma=0.2, eps=DOLFIN_EPS)

    def set_flow_field(self):
        V = VectorFunctionSpace(self.mesh, 'CG', 1)
        expr = Expression(['sin(M_PI*x[0])*sin(M_PI*x[0])*sin(2*M_PI*x[1])',
                           '-sin(M_PI*x[1])*sin(M_PI*x[1])*sin(2*M_PI*x[0])'],
                           degree = 1)

        self.t_end = Constant(0.0)
        self.fluid_velocity = interpolate(expr, V)*cos(Constant(pi)*self.t_end)
        self.fluid_pressure = Function(self.DG0_space)

    def define_problem(self):
        self.set_components('solute')
        self.set_component_fe_space()
        self.set_advection_velocity()

        self.initialize_form()

        self.mark_component_boundary(**{'solute': self.marker_dict.values()})

        self.set_component_ics(self.sol_expr)

    def add_physics_to_form(self, u, kappa=Constant(1.0), f_id=0):
        self.add_explicit_advection(u, kappa, marker=0, f_id=f_id)

    def get_solution(self):
        self.solution = Function(self.comp_func_spaces)
        self.solution.assign(interpolate(self.sol_expr, self.comp_func_spaces))

        return self.solution.copy()

    def get_total_mass(self):
        self.total_mass = assemble(self.fluid_components[0]*self.dx)
        return self.total_mass

    def get_center_of_mass(self):
        center_x = assemble(self.fluid_components[0]*self.cell_coord[0]*self.dx)
        center_y = assemble(self.fluid_components[0]*self.cell_coord[1]*self.dx)

        return center_x/self.total_mass, center_y/self.total_mass
