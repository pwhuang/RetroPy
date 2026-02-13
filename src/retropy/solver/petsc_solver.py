# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *
from dolfinx.fem.petsc import LinearProblem

class PETScSolver:
    """A solver class that is used as a mixin for problem classes."""

    def generate_solver(self, **kwargs):
        """"""

        self.__func_space = self.get_function_space()

        self.__u0 = self.get_fluid_components()
        self.__u1 = Function(self.__func_space)

        one = Constant(self.mesh, 1.0)

        self.add_physics_to_form(self.__u0, kappa=one, f_id=0)

        if self.num_forms > 1:
            self.add_corrector_to_form(self.__u1, f_id=1)
        
        self._problems = []

        for form in self.get_forms():
            a, L = lhs(form), rhs(form)
            self._problems.append(LinearProblem(a, L, petsc_options_prefix="basic_linear_problem", 
                                                bcs=self.get_dirichlet_bcs(), u=self.__u1))

    def get_solver(self):
        return self._problems[0].solver

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='jacobi', id=0):
        prm = self._problems[id].solver
        prm.setType(linear_solver)
        prm.getPC().setType(preconditioner)

        set_default_solver_parameters(prm)

        return prm
    
    def solve_one_step(self):
        return [problem.solve() for problem in self._problems]

    def get_solver_u1(self):
        return self.__u1

    def assign_u1_to_u0(self):
        self.fluid_components.x.array[:] = self.__u1.x.array

    def assign_u0_to_u1(self):
        self.__u1.x.array[:] = self.fluid_components.x.array

    def solve_transport(self, dt_val=1.0, timesteps=1):
        """"""

        self.dt.value = dt_val

        for _ in range(timesteps):
            self.solve_one_step()
            self.assign_u1_to_u0()

            self.current_time.value += dt_val
            self.save_to_file(time=self.current_time.value)
