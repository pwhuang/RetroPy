# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *


class StokesFlowMixedPoisson(TransportProblemBase, StokesFlowBase):
    """This class utilizes the mixed Poisson method to solve
    the pressure and velocity of Stokes's flow.
    """

    def generate_form(self):
        """Sets up the FeNiCs form of Stokes flow"""
        self.generate_residual_form()

        func_space_list = [self.velocity_finite_element, self.pressure_finite_element]

        self.mixed_func_space = FunctionSpace(self.mesh, MixedElement(func_space_list))
        W = self.mixed_func_space

        (self.__u, self.__p) = TrialFunctions(W)
        (self.__v, self.__q) = TestFunctions(W)

        u, p = self.__u, self.__p
        v, q = self.__v, self.__q

        mu, rho, g = self._mu, self._rho, self._g
        dx = self.dx

        self.__r = Constant(self.mesh, ScalarType(0.0))
        r = self.__r

        self.mixed_form = (
            mu * inner(grad(v), grad(u)) * dx
            - inner(div(v), p) * dx
            + r * inner(div(v), div(rho * u)) * dx
            - inner(v, rho * g) * dx
            + q * div(rho * u) * dx
        )

        self.mixed_bc = []

    def add_mass_source(self, sources):
        super().add_mass_source_to_residual_form(sources)
        q, v, r = self.__q, self.__v, self.__r
        dx = self.dx

        for source in sources:
            self.mixed_form -= q * source * dx + r * inner(div(v), source) * dx

    def add_momentum_source(self, sources: list):
        super().add_momentum_source_to_residual_form(sources)
        v = self.__v

        for source in sources:
            self.mixed_form -= inner(v, source) * self.dx

    def set_pressure_bc(self, bc: dict):
        # TODO: This method needs to be tested.
        super().set_pressure_bc(bc)

        v, n, ds = self.__v, self.n, self.ds
        mu = self._mu
        u = self.__u

        for key, pressure_bc in self.pressure_bc.items():
            marker = self.marker_dict[key]
            self.mixed_form += +inner(pressure_bc * n, v) * ds(marker) - mu * inner(
                dot(grad(u), n), v
            ) * ds(marker)

    def set_pressure_dirichlet_bc(self, bc: dict):
        self.pressure_zero_bc = []

        for key, pressure_bc in bc.items():
            dofs = locate_dofs_topological(
                V=(self.mixed_func_space.sub(1), self.pressure_func_space),
                entity_dim=self.mesh.topology.dim - 1,
                entities=self.facet_dict[key],
            )
            self.mixed_bc.append(
                dirichletbc(pressure_bc, dofs, self.mixed_func_space.sub(1))
            )

    def set_velocity_bc(self, bc: dict):
        """"""

        super().set_velocity_bc(bc)

        for key, velocity_bc in bc.items():
            dofs = locate_dofs_topological(
                V=(self.mixed_func_space.sub(0), self.velocity_func_space),
                entity_dim=self.mesh.topology.dim - 1,
                entities=self.facet_dict[key],
            )
            self.mixed_bc.append(
                dirichletbc(velocity_bc, dofs, self.mixed_func_space.sub(0))
            )

    def set_additional_parameters(self, r_val: float, **kwargs):
        self.__r.value = r_val

    def assemble_matrix(self):
        self.__a, self.__L = lhs(self.mixed_form), rhs(self.mixed_form)

    def set_flow_solver_params(self, petsc_options):
        self.problem = LinearProblem(
            self.__a, self.__L, self.mixed_bc, petsc_options=petsc_options
        )

    def solve_flow(self, **kwargs):
        U = self.problem.solve()

        u, p = U.sub(0).collapse(), U.sub(1).collapse()

        self.fluid_velocity.interpolate(u)
        self.fluid_pressure.x.array[:] = p.x.array
        self.fluid_pressure.x.scatter_forward()
