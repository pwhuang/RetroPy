# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *


class DarcyFlowUzawa(TransportProblemBase, DarcyFlowBase):
    """This class utilizes the Augmented Lagrangian Uzawa method to solve
    the pressure and velocity of Darcy flow.
    """

    def generate_form(self):
        """Sets up the FeNiCs form of Darcy flow."""

        V = self.velocity_func_space
        Q = self.pressure_func_space

        self.__u, self.__p = TrialFunction(V), TrialFunction(Q)
        self.__v, self.__q = TestFunction(V), TestFunction(Q)

        u, p = self.__u, self.__p
        v, q = self.__v, self.__q

        self.__u0 = self.fluid_velocity
        self.__p0 = self.fluid_pressure

        u0, p0 = self.__u0, self.__p0

        mu, k, rho, g, phi = self._mu, self._k, self._rho, self._g, self._phi

        self.__r = Constant(self.mesh, 1.0)
        self.omega = Constant(self.mesh, 1.0)
        r, omega = self.__r, self.omega

        n, dx, ds = self.n, self.dx, self.ds

        self.form_update_velocity = (
            mu / k * inner(v, u) * dx
            + r * inner(div(v), div(rho * phi * u)) * dx
            - inner(p0, div(v)) * dx
            - inner(v, rho * g) * dx
        )

        self.form_update_pressure = (
            q * (p - p0) * dx + omega * q * (div(rho * phi * u0)) * dx
        )

        for key, pressure_bc in self.pressure_bc.items():
            marker = self.marker_dict[key]
            self.form_update_velocity += pressure_bc * inner(n, v) * ds(marker)

        self.functions_to_save = [self.fluid_pressure, self.fluid_velocity]

    def add_mass_source(self, sources: list):
        q, v, r, omega = self.__q, self.__v, self.__r, self.omega
        dx = self.dx

        for source in sources:
            self.form_update_velocity -= r * inner(div(v), source) * dx
            self.form_update_pressure -= q * omega * source * dx

    def add_momentum_source(self, sources: list):
        v = self.__v

        for source in sources:
            self.form_update_velocity -= inner(v, source) * self.dx

    def set_additional_parameters(self, r_val: float, omega_by_r: float):
        """For 0 < omega/r < 2, the augmented system converges."""

        self.__r.value = r_val
        self.omega.value = r_val * omega_by_r

    def assemble_matrix(self):
        F_velocity = self.form_update_velocity
        F_pressure = self.form_update_pressure

        self.a_v, self.L_v = form(lhs(F_velocity)), form(rhs(F_velocity))
        a_p, self.L_p = form(lhs(F_pressure)), form(rhs(F_pressure))

        self.A_v = assemble_matrix(self.a_v, bcs=self.velocity_bc)
        self.b_v = assemble_vector(self.L_v)
        self.A_v.assemble()
        apply_lifting(self.b_v, [self.a_v], bcs=[self.velocity_bc])
        self.b_v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b_v, self.velocity_bc)

        self.A_p = assemble_matrix(a_p, bcs=[])
        self.A_p.assemble()
        self.b_p = assemble_vector(self.L_p)

    def set_flow_solver_params(self, **kwargs):
        # Users can override this method.
        # Or, TODO: make this method more user friendly.

        self.solver_v = PETSc.KSP().create(self.mesh.comm)
        self.solver_v.setOperators(self.A_v)
        self.solver_v.setType("preonly")
        self.solver_v.setTolerances(rtol=1e-12, atol=1e-14)

        pc = self.solver_v.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        pc.setFactorSetUpSolverType()

        self.solver_p = PETSc.KSP().create(self.mesh.comm)
        self.solver_p.setOperators(self.A_p)
        self.solver_p.setType("gmres")
        self.solver_p.setTolerances(rtol=1e-12, atol=1e-14)

        pc = self.solver_p.getPC()
        pc.setType("none")

    def solve_flow(self, target_residual: float, max_steps: int):
        # TODO: Tidy up the logic of this section.
        steps = 0

        residual = self.get_flow_residual()
        while residual > target_residual and steps < max_steps:
            if MPI.COMM_WORLD.rank == 0:
                print(f"Darcy flow residual = {str(residual)}")

            self.solver_v.solve(self.b_v, self.__u0.vector)
            self.solver_p.solve(self.b_p, self.__p0.vector)

            # TODO: figure out why scattering of p0 is not necessary here.
            self.__u0.x.scatter_forward()

            self.b_v = assemble_vector(self.L_v)
            apply_lifting(self.b_v, [self.a_v], bcs=[self.velocity_bc])
            self.b_v.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            set_bc(self.b_v, self.velocity_bc)

            self.b_p = assemble_vector(self.L_p)

            steps += 1

            residual = self.get_flow_residual()

        if MPI.COMM_WORLD.rank == 0:
            print(f"Steps used: {str(steps)}")
