# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *


class DarcyFlowUzawa(TransportProblemBase, DarcyFlowBase):
    """This class utilizes the Augmented Lagrangian Uzawa method to solve
    the pressure and velocity of Darcy flow.
    """

    def generate_form(self):
        """Sets up the FeNiCs form of Darcy flow."""
        self.generate_residual_form()

        V = self.velocity_func_space
        Q = self.pressure_func_space

        self.__u, self.__p = TrialFunction(V), TrialFunction(Q)
        self.__v, self.__q = TestFunction(V), TestFunction(Q)

        u, p = self.__u, self.__p
        v, q = self.__v, self.__q

        self.__u0 = self.fluid_velocity
        self.__p0 = self.fluid_pressure

        u0, p0 = self.__u0, self.__p0

        mu, k, rho, g = self._mu, self._k, self._rho, self._g

        self.__r = Constant(self.mesh, ScalarType(1.0))
        self.omega = Constant(self.mesh, ScalarType(1.0))
        r, omega = self.__r, self.omega

        dx = self.dx

        self.form_update_velocity = (
            mu / k * inner(v, u) * dx
            + r * inner(div(v), div(rho * u)) * dx
            - inner(p0, div(v)) * dx
            - inner(v, rho * g) * dx
        )

        self.form_update_pressure = (
            q * (p - p0) * dx + omega * q * (div(rho * u0)) * dx
        )

        self.functions_to_save = [self.fluid_pressure, self.fluid_velocity]

    def add_mass_source(self, sources: list):
        super().add_mass_source_to_residual_form(sources)
        q, v, r, omega = self.__q, self.__v, self.__r, self.omega
        dx = self.dx

        for source in sources:
            self.form_update_velocity -= r * inner(div(v), source) * dx
            self.form_update_pressure -= q * omega * source * dx

    def add_momentum_source(self, sources: list):
        super().add_momentum_source_to_residual_form(sources)
        v = self.__v

        for source in sources:
            self.form_update_velocity -= inner(v, source) * self.dx

    def set_pressure_bc(self, bc: dict):
        super().set_pressure_bc(bc)
        v, n, ds = self.__v, self.n, self.ds

        for key, pressure_bc in self.pressure_bc.items():
            marker = self.marker_dict[key]
            self.form_update_velocity += pressure_bc * inner(n, v) * ds(marker)

    def add_weak_pressure_bc(self, penalty_value=0.0):
        super().add_weak_pressure_bc(penalty_value)
        v, n, ds = self.__v, self.n, self.ds
        p, u = self.fluid_pressure, self.__u
        alpha = Constant(self.mesh, penalty_value)
        h = Circumradius(self.mesh)
        mu, k, rho, g = self._mu, self._k, self.fluid_density, self._g
        q = self.__q

        for key, pressure_bc in self.pressure_bc.items():
            marker = self.marker_dict[key]
            # TODO: Develop an augmented Langrangian Uzawa's method that utilizes
            # Robin boundary conditions.
            self.form_update_velocity += (
                alpha
                * k
                / mu
                * ((pressure_bc - p) / h - rho * dot(g, n))
                * dot(n, v)
                * ds(marker)
            )
            self.form_update_velocity += alpha * dot(u, n) * dot(n, v) * ds(marker)

    def set_additional_parameters(
        self, r_val: float, omega_by_r: float, *args, **kwargs
    ):
        """For 0 < omega/r < 2, the augmented system converges."""

        self.__r.value = r_val
        self.omega.value = r_val * omega_by_r

    def assemble_matrix(self):
        self.assemble_residual_vector()

        F_velocity = self.form_update_velocity
        F_pressure = self.form_update_pressure

        self.__a_v, self.__L_v = lhs(F_velocity), rhs(F_velocity)
        self.__a_p, self.__L_p = lhs(F_pressure), rhs(F_pressure)

    def set_flow_solver_params(self, petsc_options_v, petsc_options_p, *args, **kwargs):
        self.problem_v = LinearProblem(
            self.__a_v,
            self.__L_v,
            u=self.__u0,
            petsc_options_prefix="update_velocity",
            bcs=self.velocity_bc,
            petsc_options=petsc_options_v,
        )

        self.problem_p = LinearProblem(
            self.__a_p,
            self.__L_p,
            u=self.__p0,
            petsc_options_prefix="update_pressure",
            bcs=[],
            petsc_options=petsc_options_p,
        )

    def solve_flow(self, target_residual: float, max_steps: int):
        steps = 0

        residual = self.get_flow_residual()
        while residual > target_residual and steps < max_steps:
            if MPI.COMM_WORLD.rank == 0:
                print(f"Darcy flow residual = {str(residual)}")

            self.problem_v.solve()
            self.problem_p.solve()

            steps += 1
            residual = self.get_flow_residual()

        if MPI.COMM_WORLD.rank == 0:
            print(f"Steps used: {str(steps)}")
