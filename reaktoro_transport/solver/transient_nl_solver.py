from . import *

class TransientNLSolver(TransientSolver):
    """A solver class that is used as a mixin for problem classes."""

    num_forms = 1

    def generate_solver(self):
        """"""

        self.__func_space = self.get_function_space()

        self.__u0 = as_vector([exp(u) for u in self.fluid_components])
        self.__u1 = Function(self.comp_func_spaces)

        self.add_physics_to_form(self.__u0)
        self.add_time_derivatives(self.__u0)
        self.__forms = self.get_forms()
        self.__form = self.__forms[0]

        du = TrialFunction(self.__func_space)
        self.__form = action(self.__form, self.__u1)
        J = derivative(self.__form, self.__u1, du)
        bcs = self.get_dirichlet_bcs()

        problem = NonlinearVariationalProblem(self.__form, self.__u1, bcs, J)
        self.__solver = NonlinearVariationalSolver(problem)

        # Link to super class
        self._TransientSolver__solver = self.__solver
        self._TransientSolver__u1 = self.__u1

    def get_solver_parameters(self):
        return self.__solver.parameters

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='jacobi'):
        prm = self.get_solver_parameters()

        prm['nonlinear_solver'] = 'snes'

        nl_solver_type = 'snes_solver'

        prm[nl_solver_type]['absolute_tolerance'] = 1e-10
        prm[nl_solver_type]['relative_tolerance'] = 1e-14
        prm[nl_solver_type]['maximum_iterations'] = 50
        prm['snes_solver']['method'] = 'newtonls'
        prm['snes_solver']['line_search'] = 'bt'
        prm[nl_solver_type]['linear_solver'] = linear_solver
        prm[nl_solver_type]['preconditioner'] = preconditioner

        set_default_solver_parameters(prm[nl_solver_type]['krylov_solver'])

    def assign_u1_to_u0(self):
        self.fluid_components.assign(self.__u1)

    def assign_u0_to_u1(self):
        self.__u1.assign(self.fluid_components)
