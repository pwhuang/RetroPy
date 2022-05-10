from . import *
from petsc4py import PETSc
import numpy as np

class CustomNLSolver(TransientSolver):
    """A solver class that is used as a mixin for problem classes."""

    num_forms = 1

    def generate_solver(self, eval_jacobian=True):
        """"""

        self.__func_space = self.get_function_space()

        self.__u0 = self.get_fluid_components()
        self.__u1 = Function(self.__func_space)
        self.__du = TrialFunction(self.__func_space)


        self.add_time_derivatives(self.__u0)
        self.add_physics_to_form(self.__u0)

        self.__forms = self.get_forms()
        self.__form = self.__forms[0]

        self.__form = action(self.__form, self.__u1)

        if eval_jacobian==True:
            self.evaluate_jacobian(self.jacobian_forms[0])

        self._TransientSolver__u1 = self.__u1

        b = PETScVector()
        J_mat = PETScMatrix()

        self._dummy_u = self.__u1.copy(deepcopy=True)

        self.snes = PETSc.SNES().create(MPI.comm_world)
        self.snes.setFunction(self.__F, b.vec())
        self.snes.setJacobian(self.__J, J_mat.mat())
        #self.snes.setVariableBounds((self.__u1.vector()-40).vec(), (self.__u1.vector()+30).vec())

        # bcs = self.get_dirichlet_bcs()
        # self.__problem = NonlinearVariationalProblem(self.__form, self.__u1, bcs, J)
        # self.__solver = PETScSNESSolver(MPI.comm_world, nls_type='default')
        # self._TransientSolver__solver = self.__solver

    def __F(self, snes, x, F):
        F = PETScVector(F)
        u = self.__u1
        #self._dummy_u.vector()[:] = np.exp(u.vector()[:])
        self._dummy_u.vector()[:] = u.vector()[:]

        x.copy(self._dummy_u.vector().vec())
        u.vector().apply("")

        assemble(self.__form, tensor=F)
        for bc in self.get_dirichlet_bcs():
            bc.apply(F, x)
            bc.apply(F, u.vector())

    def __J(self, snes, x, J, P):
        J = PETScMatrix(J)
        u = self.__u1
        #self._dummy_u.vector()[:] = np.exp(u.vector()[:])
        self._dummy_u.vector()[:] = u.vector()[:]

        x.copy(self._dummy_u.vector().vec())
        u.vector().apply("")

        assemble(self.jacobian, tensor=J)

        for bc in self.get_dirichlet_bcs():
            bc.apply(J)

    def solve_one_step(self):
        self.snes.solve(None, self.__u1.vector().vec())

    # def solve_one_step(self):
    #     self.__solver.solve(self.__problem, self.__u1.vector())

    def evaluate_jacobian(self, form):
        self.jacobian = derivative(action(form, self.__u1), self.__u1, self.__du)

    def get_solver_parameters(self):
        return PETSc.Options()
        #return self.__solver.parameters

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='jacobi'):
        opts = self.get_solver_parameters()

        opts['snes_monitor'] = None
        opts['snes_linesearch_monitor'] = None
        opts['snes_converged_reason'] = None
        opts['snes_linesearch'] = 'bt'
        #opts['snes_fd'] = None

        #opts['ksp_type'] = 'bcgs'
        #opts['ksp_monitor_true_residual'] = None
        opts['ksp_max_it'] = 500
        opts['ksp_rtol'] = 1e-12
        opts['ksp_atol'] = 1e-14
        #opts['ksp_pc_type'] = 'lu'
        opts['ksp_converged_reason'] = None

        opts['pc_hypre_boomeramg_strong_threshold'] = 0.4
        opts['pc_hypre_boomeramg_truncfactor'] = 0.0
        opts['pc_hypre_boomeramg_print_statistics'] = 0

        self.snes.setFromOptions()

        self.snes.setTolerances(rtol=1e-10, atol=1e-12, max_it=5)
        self.snes.getKSP().setType('bcgs')
        self.snes.getKSP().getPC().setType('hypre')
        self.snes.getKSP().getPC().setHYPREType('boomeramg')
        #self.snes.getKSP().getPC().setGAMGType('agg')


        # prm['nonlinear_solver'] = 'snes'
        #
        # nl_solver_type = 'snes_solver'
        #
        # prm[nl_solver_type]['absolute_tolerance'] = 1e-10
        # prm[nl_solver_type]['relative_tolerance'] = 1e-14
        # prm[nl_solver_type]['maximum_iterations'] = 50
        # prm['snes_solver']['method'] = 'newtonls'
        # prm['snes_solver']['line_search'] = 'bt'
        # prm[nl_solver_type]['linear_solver'] = linear_solver
        # prm[nl_solver_type]['preconditioner'] = preconditioner
        #
        # set_default_solver_parameters(prm[nl_solver_type]['krylov_solver'])
