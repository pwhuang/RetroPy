# SPDX-FileCopyrightText: 2022 Po-Wei Huang geopwhuang@gmail.com
# SPDX-License-Identifier: LGPL-2.1-or-later

from . import *

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
        
        u0, p0 = self.__u0, self.__p0
        mu, rho, g = self._mu, self._rho, self._g

        self.__r = Constant(self.mesh, ScalarType(0.0))
        self.omega = Constant(self.mesh, ScalarType(1.0))
        r, omega = self.__r, self.omega

        dx = self.dx

        self.form_update_velocity = mu*inner(grad(v), grad(u))*dx \
                                    + r*inner(div(v), div(rho*u))*dx \
                                    - inner(p0, div(v))*dx \
                                    - inner(v, rho*g)*dx

        self.form_update_pressure = q*(p-p0)*dx + omega*q*div(rho*u0)*dx

    def set_pressure_bc(self, bc):
        super().set_pressure_bc(bc)

        v, n, ds = self.__v, self.n, self.ds
        mu = self._mu
        u0 = self.__u0

        for key, pressure_bc in bc.items():
            marker = self.marker_dict[key]
            self.form_update_velocity += pressure_bc*inner(n, v)*ds(marker)
            self.form_update_velocity -= mu*inner(dot(grad(u0), n), v)*ds(marker)

    def add_mass_source(self, sources: list):
        super().add_mass_source_to_residual_form(sources)
        q, v, r, omega = self.__q, self.__v, self.__r, self.omega

        for source in sources:
            self.form_update_velocity -= r*inner(div(v), source)*self.dx
            self.form_update_pressure -= q*omega*source*self.dx

    def add_momentum_source(self, sources: list):
        super().add_momentum_source_to_residual_form(sources)
        v = self.__v

        for source in sources:
            self.form_update_velocity -= inner(v, source)*self.dx

    def set_additional_parameters(self, r_val: float, omega_by_r: float):
        """For 0 < omega/r < 2, the augmented system converges."""

        self.__r.value = r_val
        self.omega.value = r_val * omega_by_r

    def assemble_matrix(self):
        """"""
        F_velocity = self.form_update_velocity
        F_pressure = self.form_update_pressure

        self.a_v, self.L_v = form(lhs(F_velocity)), form(rhs(F_velocity))
        self.a_p, self.L_p = form(lhs(F_pressure)), form(rhs(F_pressure))

        self.A_v = assemble_matrix(self.a_v, bcs=self.velocity_bc)
        self.A_v.assemble()
        
        self.b_v = assemble_vector(self.L_v)
        apply_lifting(self.b_v, [self.a_v], bcs=[self.velocity_bc])
        self.b_v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b_v, self.velocity_bc)

        self.A_p = assemble_matrix(self.a_p, bcs=self.pressure_bc)
        self.A_p.assemble()

    def set_flow_solver_params(self):
        # Users can override this method.
        # Or, TODO: make this method more user friendly.

        self.solver_v = PETSc.KSP().create(self.mesh.comm)
        self.solver_v.setOperators(self.A_v)
        self.solver_v.setType("preonly")
        self.solver_v.setTolerances(rtol=1e-12, atol=1e-14)

        pc = self.solver_v.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("superlu_dist")
        pc.setFactorSetUpSolverType()

        self.solver_p = PETSc.KSP().create(self.mesh.comm)
        self.solver_p.setOperators(self.A_p)
        self.solver_p.setType("gmres")
        self.solver_p.setTolerances(rtol=1e-12, atol=1e-14)

        pc = self.solver_p.getPC()
        pc.setType("none")

    def solve_flow(self, target_residual: float, max_steps: int):
        """"""
        steps = 0

        residual = self.get_flow_residual()
        while residual > target_residual and steps < max_steps:
            if MPI.COMM_WORLD.rank == 0:
                print(f"Stokes flow residual = {str(residual)}")

            self.solver_v.solve(self.b_v, self.__u0.vector)
            self.__u0.x.scatter_forward()

            self.b_p = assemble_vector(self.L_p)
            apply_lifting(self.b_p, [self.a_p], bcs=[self.pressure_bc])
            self.b_p.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            set_bc(self.b_p, self.pressure_bc)
            
            self.solver_p.solve(self.b_p, self.__p0.vector)
            self.__p0.x.scatter_forward()

            self.b_v = assemble_vector(self.L_v)
            apply_lifting(self.b_v, [self.a_v], bcs=[self.velocity_bc])
            self.b_v.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )
            set_bc(self.b_v, self.velocity_bc)

            steps += 1
            residual = self.get_flow_residual()

        if MPI.COMM_WORLD.rank == 0:
            print(f"Steps used: {str(steps)}")