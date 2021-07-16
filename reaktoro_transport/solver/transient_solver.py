from . import *

class TransientSolver:
    """A solver class that is used as a mixin for problem classes."""

    num_forms = 1

    def generate_solver(self):
        """"""

        self.__func_space = self.get_function_space()

        self.__u0 = self.fluid_components
        self.__u1 = Function(self.comp_func_spaces)

        self.add_time_derivatives(self.__u0)
        self.__forms = self.get_forms()
        self.__form = self.__forms[0]

        a, L = lhs(self.__form), rhs(self.__form)

        problem = LinearVariationalProblem(a, L, self.__u1, self.get_dirichlet_bcs())
        self.__solver = LinearVariationalSolver(problem)

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='amg'):
        prm = self.__solver.parameters
        prm['linear_solver'] = linear_solver
        prm['preconditioner'] = preconditioner

        set_default_solver_parameters(prm['krylov_solver'])

    def set_dt(self, dt_val):
        self.dt.assign(dt_val)

    def solve_one_step(self):
        self.__solver.solve()
        self.__u0.assign(self.__u1)

    def solve_transport(self, dt_val=1.0, timesteps=1):
        """"""

        self.set_dt(dt_val)
        self.save_to_file(time=0.0)

        for i in range(timesteps):
            self.solve_one_step()
            self.save_to_file(time=(i+1)*dt_val)
