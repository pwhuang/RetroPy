# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *
from mpi4py import MPI
from petsc4py import PETSc

class TransientNLSolver(TransientSolver):
    """A solver class that is used as a mixin for problem classes."""

    num_forms = 1

    def generate_solver(self, eval_jacobian=True):
        """"""

        self.__func_space = self.get_function_space()

        self.__u0 = self.get_fluid_components()
        self.__u1 = Function(self.__func_space)
        
        self.__du = TrialFunction(self.__func_space)
        self._TransientSolver__u1 = self.__u1

        self.initial_guess = 0.0

        self.add_time_derivatives(self.__u0)
        self.add_physics_to_form(self.__u0, kappa=Constant(self.mesh, 1.0), f_id=0)

        self.__forms = self.get_forms()
        self.__form = self.__forms[0]

        self.__form = action(self.__form, self.__u1)

        if eval_jacobian==True:
            J = derivative(self.__form, self.__u1, self.__du)
        else:
            J = self.jacobian

        bcs = self.get_dirichlet_bcs()

        problem = NonlinearProblem(self.__form, self.__u1, bcs, J)
        self.__solver = NewtonSolver(MPI.COMM_WORLD, problem)

    def evaluate_jacobian(self, form):
        self.jacobian = derivative(action(form, self.__u1), self.__u1, self.__du)

    def solve_one_step(self):
        self.guess_solution()
        num_iterations, converged = self.__solver.solve(self.__u1)

        return num_iterations, converged

    def guess_solution(self):
        self.__u1.x.array[:] = self.initial_guess

    def get_solver(self):
        return self.__solver

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='jacobi'):
        ksp = self.__solver.krylov_solver
        opts = PETSc.Options()
        option_prefix = ksp.getOptionsPrefix()

        opts[f"{option_prefix}ksp_type"] = linear_solver
        opts[f"{option_prefix}pc_type"] = preconditioner
        # opts[f"{option_prefix}pc_factor_mat_solver_type"] = 'mumps'
        ksp.setFromOptions()

        self.__solver.convergence_criterion = 'residual'
        self.__solver.atol = 1e-12
        self.__solver.rtol = 1e-14
        self.__solver.max_it = 1000
        self.__solver.nonzero_initial_guess = True
        self.__solver.report = True