from . import *

class SteadyStateSolver:
    """A solver class that is used as a mixin for problem classes."""

    num_forms = 1

    def generate_solver(self):
        """"""

        self.__forms = self.get_forms()
        self.__form = self.__forms[0]

        self.__func_space = self.get_function_space()
        self.__u1 = Function(self.__func_space)

        a, L = lhs(self.__form), rhs(self.__form)

        problem = LinearVariationalProblem(a, L, self.__u1, self.get_dirichlet_bcs())
        self.__solver = LinearVariationalSolver(problem)

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='amg'):
        prm = self.__solver.parameters
        prm['linear_solver'] = linear_solver
        prm['preconditioner'] = preconditioner

        set_default_solver_parameters(prm['krylov_solver'])

    def solve_transport(self):
        self.__solver.solve()
        self.fluid_components.assign(self.__u1)
