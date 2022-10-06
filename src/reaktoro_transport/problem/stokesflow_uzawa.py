# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *
import ufl

class StokesFlowUzawa(TransportProblemBase, StokesFlowBase):
    """This class utilizes the Augmented Lagrangian Uzawa method to solve
    the pressure and velocity of the Stokes equation.
    """

    def generate_form(self):
        """Sets up the FeNiCs Form of Stokes flow."""

        self.generate_residual_form()

        V = self.velocity_func_space
        Q = self.pressure_func_space

        self.__u, self.__p = TrialFunction(V), TrialFunction(Q)
        self.__v, self.__q = TestFunction(V), TestFunction(Q)

        u, p = self.__u, self.__p
        v, q = self.__v, self.__q

        self.__u0 = self.fluid_velocity
        self.__p0 = self.fluid_pressure
        self.__u1 ,self.__p1 = Function(V), Function(Q)

        u0, u1, p0, p1 = self.__u0, self.__u1, self.__p0, self.__p1
        mu, rho, g = self._mu, self._rho, self._g

        self.__r = Constant(0.0)
        self.omega = Constant(1.0)
        r, omega = self.__r, self.omega

        n = self.n
        dx, ds, dS = self.dx, self.ds, self.dS(0)

        # alpha = Constant(0.0)
        # a_stab = Constant(0.0)
        #
        # h = CellDiameter(self.mesh)
        # h2 = ufl.Min(h("+"), h("-"))
        #
        # def tensor_jump(v, n):
        #     return ufl.outer(v, n)("+") + ufl.outer(v, n)("-")
        #
        # stab = (
        #     mu * (a_stab / h2) * inner(tensor_jump(u, n), tensor_jump(v, n)) * dS
        #     - mu * inner(avg(grad(u)), tensor_jump(v, n)) * dS
        #     - mu * inner(avg(grad(v)), tensor_jump(u, n)) * dS
        # )

        self.form_update_velocity = mu*inner(grad(v), grad(u))*dx \
                                    + r*inner(div(v), div(rho*u))*dx \
                                    - inner(p0, div(v))*dx \
                                    - inner(v, rho*g)*dx
        #                             + stab #+ inner(v, (u - u0))*dx
        #
        # for marker in self.stokes_boundary_dict['noslip']:
        #     self.form_update_velocity += (alpha/h * inner(outer(v, n), outer(u, n)) * ds(marker)
        #      - inner(grad(u), outer(v, n)) * ds(marker)
        #      - inner(grad(v), outer(u, n)) * ds(marker))

        self.form_update_pressure = q*(p-p0)*dx + omega*q*div(rho*u1)*dx

    def set_pressure_bc(self, pressure_bc):
        super().set_pressure_bc(pressure_bc)

        v = self.__v
        n = self.n
        ds = self.ds
        mu = self._mu
        u0 = self.__u0

        for i, marker in enumerate(self.stokes_boundary_dict['inlet']):
            self.form_update_velocity += \
            inner(self.pressure_bc[i]*n, v)*ds(marker) \
            - mu*inner(dot(grad(u0), n), v)*ds(marker)

    def add_mass_source(self, sources: list):
        q, v, r, omega = self.__q, self.__v, self.__r, self.omega

        for source in sources:
            self.form_update_velocity -= r*inner(div(v), source)*self.dx
            self.form_update_pressure -= q*omega*source*self.dx

    def add_momentum_source(self, sources: list):
        v = self.__v

        for source in sources:
            self.form_update_velocity -= inner(v, source)*self.dx

    def set_additional_parameters(self, r_val: float, omega_by_r: float):
        """For 0 < omega/r < 2, the augmented system converges."""

        self.__r.assign(r_val)
        self.omega.assign(r_val*omega_by_r)

    def assemble_matrix(self):
        """"""
        F_velocity = self.form_update_velocity
        F_pressure = self.form_update_pressure

        a_v, self.L_v = lhs(F_velocity), rhs(F_velocity)
        a_p, self.L_p = lhs(F_pressure), rhs(F_pressure)

        self.A_v = assemble(a_v)
        self.A_p = assemble(a_p)
        self.b_v, self.b_p = PETScVector(), PETScVector()

    def set_flow_solver_params(self):
        # Users can override this method.
        # Or, TODO: make this method more user friendly.

        self.solver_v = PETScLUSolver('mumps')
        self.solver_p = PETScKrylovSolver('gmres', 'none')

        prm_v = self.solver_v.parameters
        prm_p = self.solver_p.parameters

        TransportProblemBase.set_default_solver_parameters(prm_p)

    def solve_flow(self, target_residual: float, max_steps: int):
        """"""
        steps = 0

        residual = 1.0 + self.get_flow_residual()
        while(residual > target_residual and steps < max_steps):
            if (MPI.rank(MPI.comm_world)==0):
                info('Stokes flow residual = ' + str(residual))

            assemble(self.L_v, tensor=self.b_v)

            for bc in self.velocity_bc:
                bc.apply(self.A_v, self.b_v)

            self.solver_v.solve(self.A_v, self.__u1.vector(), self.b_v)

            assemble(self.L_p, tensor=self.b_p)
            self.solver_p.solve(self.A_p, self.__p1.vector(), self.b_p)

            self.__u0.assign(self.__u1)
            self.__p0.assign(self.__p1)

            residual = self.get_flow_residual()

            steps+=1

        if (MPI.rank(MPI.comm_world)==0):
            info('Stokes flow residual = ' + str(residual))

        if (MPI.rank(MPI.comm_world)==0):
            info('Steps used: ' + str(steps))

        return self.__u0, self.__p0

    def get_flow_rate(self, surface_id):
        return abs(assemble(dot(self.fluid_velocity, self.n)*self.ds(surface_id)))

    def get_surface_area(self, surface_id):
        return assemble(Constant(1.0)*self.ds(surface_id))
