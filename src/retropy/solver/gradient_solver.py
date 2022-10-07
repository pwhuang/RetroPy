# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *

class GradientSolver(ProjectionSolver):
    def set_projection_form(self, func_to_project: Function):
        V = self.projection_space

        w = TestFunctions(V)
        u = TrialFunctions(V)

        for i in range(self.num_component):
            self.projection_form = inner(w[i], u[i])*self.dx
            self.projection_form += inner(div(w[i]), func_to_project[i])*self.dx
            self.projection_form += -inner(self.n, w[i])*func_to_project[i]*self.ds

    def generate_projection_solver(self, func_to_assign: Function, markers: list):
        a, L = lhs(self.projection_form), rhs(self.projection_form)
        bc_list = []
        for marker in markers:
            bc_list.append(DirichletBC(self.projection_space,
                                       Constant([0.0, 0.0]*self.num_component),
                                       self.boundary_markers, marker))

        problem = LinearVariationalProblem(a, L, func_to_assign, bcs=bc_list)
        self.projection_solver = LinearVariationalSolver(problem)
