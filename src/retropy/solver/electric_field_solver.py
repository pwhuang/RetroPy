# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *

class ElectricFieldSolver:
    def set_electric_field_form(self):
        self.__func_space = self.velocity_func_space
        V = self.__func_space

        w = TestFunction(V)
        u = TrialFunction(V)
        self.electric_field = Function(V)

        c0 = self.get_fluid_components()
        z = self.charge
        D = self.molecular_diffusivity

        zDC = Constant(0.0)
        z2DC = Constant(0.0)

        for i in range(self.num_component):
            zDC += z[i]*D[i]*c0[i]
            z2DC += z[i]**2*D[i]*c0[i]

        self.__projection_form = inner(w, z2DC*u)*self.dx
        self.__projection_form += inner(div(w), zDC)*self.dx
        self.__projection_form += -inner(self.n, w)*zDC*self.ds

    def generate_electric_field_solver(self, markers: list):
        a, L = lhs(self.__projection_form), rhs(self.__projection_form)
        bc_list = []

        if self.mesh.geometric_dimension()==1:
            electric_field_boundary = Constant([0.0, ])
        elif self.mesh.geometric_dimension()==2:
            electric_field_boundary = Constant([0.0, 0.0])
        elif self.mesh.geometric_dimension()==3:
            electric_field_boundary = Constant([0.0, 0.0, 0.0])

        for marker in markers:
            bc_list.append(DirichletBC(self.__func_space, electric_field_boundary,
                                       self.boundary_markers, marker))

        problem = LinearVariationalProblem(a, L, self.electric_field, bcs=bc_list)
        self.__solver = LinearVariationalSolver(problem)

    def set_electric_field_solver_params(self, linear_solver='gmres', preconditioner='jacobi'):
        prm = self.__solver.parameters
        prm['linear_solver'] = linear_solver
        prm['preconditioner'] = preconditioner

        prm_k = prm['krylov_solver']
        prm_k['absolute_tolerance'] = 1e-16
        prm_k['relative_tolerance'] = 1e-14
        prm_k['maximum_iterations'] = 5000
        prm_k['error_on_nonconvergence'] = True
        prm_k['monitor_convergence'] = False
        prm_k['nonzero_initial_guess'] = True

    def solve_electric_field(self):
        info('Solving electric_field.')
        self.__solver.solve()
