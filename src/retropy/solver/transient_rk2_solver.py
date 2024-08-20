# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *

class TransientRK2Solver:
    """A solver class that is used as a mixin for problem classes."""

    num_forms = 2

    def set_solver_forms(self):
        self.__func_space = self.get_function_space()

        self.__u0 = self.get_fluid_components()
        self.__u1 = Function(self.__func_space)
        self.kappa = Constant(self.mesh, 0.5)

        one, two = Constant(self.mesh, 1.0), Constant(self.mesh, 2.0)

        self.add_time_derivatives(self.__u0, kappa=one, f_id=0)
        self.add_physics_to_form(self.__u0, kappa=one, f_id=0)

        self.add_time_derivatives(self.__u1, kappa=two, f_id=1)
        self.add_corrector_to_form(self.__u0, self.__u1, f_id=1)

    def add_corrector_to_form(self, u0, u1, f_id):
        one = Constant(self.mesh, 1.0)
        self.add_physics_to_form(u0, kappa=one, f_id=f_id)

    def generate_solver(self):
        self.__forms = self.get_forms()
        __u0 = self.fluid_components
        bcs = self.get_dirichlet_bcs()

        a1, L1 = lhs(self.__forms[0]), rhs(self.__forms[0])
        a2, L2 = lhs(self.__forms[1]), rhs(self.__forms[1])

        self.__problem1 = LinearProblem(a1, L1, bcs, self.__u1)
        self.__problem2 = LinearProblem(a2, L2, bcs, __u0)

    def get_solver_functions(self):
        return self.fluid_components, self.__u1

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='jacobi'):
        prm = self.__problem1.solver
        prm.setType(linear_solver)
        prm.getPC().setType(preconditioner)

        set_default_solver_parameters(prm)

        prm = self.__problem2.solver
        prm.setType(linear_solver)
        prm.getPC().setType(preconditioner)

        set_default_solver_parameters(prm)

    def solve_first_step(self):
        self.__problem1.solve()

    def solve_second_step(self):
        self.__problem2.solve()

    def assign_u1_to_u0(self):
        self.fluid_components.x.array[:] = self.__u1.x.array

    def solve_transport(self, dt_val=1.0, timesteps=1):
        """"""

        self.dt.value = dt_val
        self.save_to_file(time=self.current_time.value)

        for _ in range(timesteps):
            self.solve_first_step()
            self.solve_second_step()

            self.current_time.value += dt_val
            self.save_to_file(time=self.current_time.value)
