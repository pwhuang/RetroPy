# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

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

        self.__problem = LinearProblem(a, L, self.get_dirichlet_bcs(), self.__u1)

    def set_solver_parameters(self, linear_solver='gmres', preconditioner='jacobi'):
        prm = self.__problem.solver
        prm.setType(linear_solver)
        prm.getPC().setType(preconditioner)

        set_default_solver_parameters(prm)

    def solve_transport(self):
        self.__problem.solve()
        self.fluid_components.vector.array_w = self.__u1.vector.array_r
