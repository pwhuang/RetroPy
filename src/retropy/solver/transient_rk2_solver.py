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

        one = Constant(self.mesh, 1.0)
        half = Constant(self.mesh, 0.5)

        self.add_physics_to_form(self.__u0, kappa=one, f_id=0)
        self.add_time_derivatives(self.__u0, kappa=one, f_id=0)

        self.add_physics_to_form(self.__u1, kappa=half/self.kappa, f_id=1)
        self.add_sources((self.__u1 - self.__u0)/self.dt,
                         (one - half/self.kappa)/self.kappa, f_id=1)
        self.add_time_derivatives(self.__u0, f_id=1)

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

    def set_kappa(self, kappa):
        """
        When kappa=0.5, the RK2 method results in the midpoint scheme.
        When kappa=1.0, the RK2 method results in Heun's method. If both steps
        are TVD (Total Variation Diminishing), such Heun's method is also known
        as SSP (Strong Stability Preserving) methods.
        """

        self.kappa.value = kappa

    def solve_first_step(self):
        self.__problem1.solve()

    def solve_second_step(self):
        self.__problem2.solve()

    def solve_transport(self, dt_val=1.0, timesteps=1):
        """"""

        self.dt.value = dt_val
        self.save_to_file(time=self.current_time.value)

        for _ in range(timesteps):
            self.solve_first_step()
            self.solve_second_step()

            self.current_time.value += dt_val
            self.save_to_file(time=self.current_time.value)
