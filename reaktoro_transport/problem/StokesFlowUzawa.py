from . import *

class StokesFlowUzawa(TransportProblemBase):
    """The Augmented Lagrangian Uzawa method for solving the pressure
    and velocity of the Stokes equation.
    """

    def set_pressure_ic(self, init_cond_pressure: Expression):
        self.init_cond_pressure = init_cond_pressure

    def mark_physical_boundary(self, **kwargs):
        """This method gives boundary markers physical meaning.

        Arguments
        ---------
        inlet : Sets the boundary flow rate.
        noslip : Sets the boundary to no-slip boundary condition.
        velocity_bc : User defined velocity boundary condition.
        """

        self.boundary_dict = kwargs

    def set_form_and_pressure_bc(self, pressure_bc_val: list):
        """Sets up the FeNiCs Form of Stokes flow"""

        self.velocity_func_space = FunctionSpace(self.mesh, self.velocity_finite_element)
        self.pressure_func_space = FunctionSpace(self.mesh, self.pressure_finite_element)

        V = self.velocity_func_space
        Q = self.pressure_func_space

        u, p = TrialFunction(V), TrialFunction(Q)
        v, q = TestFunction(V), TestFunction(Q)

        self.u0, self.u1 = Function(V), Function(V)
        self.p0 = interpolate(self.init_cond_pressure, Q)
        self.p1 = Function(Q)

        u0, u1, p0, p1 = self.u0, self.u1, self.p0, self.p1

        n = FacetNormal(self.mesh)

        self.r = Constant(0.0)
        self.omega = Constant(1.0)

        # Shorthand for domain integrals
        dx = self.dx
        ds = self.ds

        self.form_update_velocity = inner(grad(v), grad(u))*dx \
                                    - inner(p0, div(v))*dx \
                                    + self.r*inner(div(v), div(u))*dx

        self.form_update_pressure = q*p*dx - q*p0*dx \
                                    + self.omega*q*div(u1)*dx

        self.residual_momentum_form = (inner(grad(u0), grad(v)) - div(v)*p0)*dx
        self.residual_mass_form = q*div(u0)*dx

        for i, marker in enumerate(self.boundary_dict['inlet']):
             self.form_update_velocity += \
             Constant(pressure_bc_val[i])*inner(n, v)*ds(marker) \
             - dot(n, dot(grad(u), v))*ds(marker)

             self.residual_momentum_form += \
             Constant(pressure_bc_val[i])*inner(n, v)*ds(marker) \
             - dot(n, dot(grad(u0), v))*ds(marker)

    def set_uzawa_parameters(self, r_val: float, omega_val: float):
        """When r = 0, it converges for omega < 2. Try 1.5 first.
        For 0 < omega < 2r, the augmented system converges.
        One can choose r >> 1.
        """

        self.r.assign(r_val)
        self.omega.assign(omega_val)

    def set_velocity_bc(self, velocity_bc_val=[]):
        """
        Arguments
        ---------
        velocity_bc_val : list of Constants,
                          e.g., [Constant((1.0, -1.0)), Constant((0.0, -2.0))]
        """

        if self.mesh.geometric_dimension()==2:
            noslip = Constant((0.0, 0.0))
        elif self.mesh.geometric_dimension()==3:
            noslip = Constant((0.0, 0.0, 0.0))

        self.velocity_bc = []

        for marker in self.boundary_dict['noslip']:
            self.velocity_bc.append(DirichletBC(self.velocity_func_space,
                                                noslip,
                                                self.boundary_markers, marker))

        for i, marker in enumerate(self.boundary_dict['velocity_bc']):
            self.velocity_bc.append(DirichletBC(self.velocity_func_space,
                                                velocity_bc_val[i],
                                                self.boundary_markers, marker))

    def assemble_matrix(self):
        """"""
        F_velocity = self.form_update_velocity
        F_pressure = self.form_update_pressure

        a_v, self.L_v = lhs(F_velocity), rhs(F_velocity)
        a_p, self.L_p = lhs(F_pressure), rhs(F_pressure)

        self.A_v = assemble(a_v)
        self.A_p = assemble(a_p)
        self.b_v, self.b_p = PETScVector(), PETScVector()

    def set_solver(self):
        # The user can override this method.
        # Or, TODO: make this method more user friendly.

        self.solver_v = PETScLUSolver('mumps')
        self.solver_p = PETScKrylovSolver('gmres', 'amg')

        prm_v = self.solver_v.parameters
        prm_p = self.solver_p.parameters

        TransportProblemBase.set_default_solver_parameters(prm_p)

    def get_residual(self):
        """"""
        residual_momentum = assemble(self.residual_momentum_form)
        residual_mass = assemble(self.residual_mass_form)

        for bc in self.velocity_bc:
            bc.apply(residual_momentum, self.u0.vector())

        return residual_momentum.norm('l2') + residual_mass.norm('l2')

    def solve_flow(self, target_residual: float, max_steps: int):
        """"""
        steps = 0

        while(self.get_residual() > target_residual or steps < max_steps):
            assemble(self.L_v, tensor=self.b_v)
            [bc.apply(self.A_v, self.b_v) for bc in self.velocity_bc]
            self.solver_v.solve(self.A_v, self.u1.vector(), self.b_v)

            assemble(self.L_p, tensor=self.b_p)
            self.solver_p.solve(self.A_p, self.p1.vector(), self.b_p)

            self.u0.assign(self.u1)
            self.p0.assign(self.p1)

            steps+=1

        return self.u0, self.p0
