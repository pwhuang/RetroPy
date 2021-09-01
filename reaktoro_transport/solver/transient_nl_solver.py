from . import *

class TransientNLSolver(TransientSolver):
    """A solver class that is used as a mixin for problem classes."""

    num_forms = 1

    def generate_solver(self):
        """"""

        self.__func_space = self.get_function_space()

        self.__u0 = self.fluid_components
        self.__u1 = Function(self.comp_func_spaces)

        self.add_physics_to_form(self.__u0)
        self.add_time_derivatives(self.__u0)
        self.__forms = self.get_forms()
        self.__form = self.__forms[0]

        du = self.get_trial_function()
        self.__form = action(self.__form, self.__u1)
        J = derivative(self.__form, self.__u1, du)
        bcs = self.get_dirichlet_bcs()

        problem = NonlinearVariationalProblem(self.__form, self.__u1, bcs, J)
        self.__solver = NonlinearVariationalSolver(problem)

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='amg'):
        prm = self.__solver.parameters

        prm['newton_solver']['absolute_tolerance'] = 1e-10
        prm['newton_solver']['relative_tolerance'] = 1e-10
        prm['newton_solver']['maximum_iterations'] = 50
        prm['newton_solver']['relaxation_parameter'] = 1.0
        prm['newton_solver']['linear_solver'] = linear_solver
        prm['newton_solver']['preconditioner'] = preconditioner

        set_default_solver_parameters(prm['newton_solver']['krylov_solver'])

        return prm

    def solve_one_step(self):
        self.__solver.solve()

    def get_solution(self):
        return self.__u1

    def assign_u1_to_u0(self):
        self.__u0.assign(self.__u1)
