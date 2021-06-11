from . import *

class DarcyFlowUzawa(TransportProblemBase, FluidProperty):
    """This class utilizes the Augmented Lagrangian Uzawa method to solve
    the pressure and velocity of Darcy flow.
    """

    def __init__(self, mesh, boundary_markers, domain_markers):
        self.set_mesh(mesh)
        self.set_boundary_markers(boundary_markers)
        self.set_domain_markers(domain_markers)

        self.velocity_bc = []

    def mark_flow_boundary(self, **kwargs):
        """This method gives boundary markers physical meaning.

        Keywords
        --------
        velocity: Sets the boundary flow rate.
        pressure: Sets the DirichletBC of pressure.
        """

        self.__boundary_dict = kwargs

    def set_pressure_ic(self, init_cond_pressure: Expression):
        """Sets up the initial condition of pressure."""
        self.init_cond_pressure = init_cond_pressure

    def set_form_and_pressure_bc(self, pressure_bc_val: list):
        """Sets up the FeNiCs form of Darcy flow."""

        V = self.velocity_func_space
        Q = self.pressure_func_space

        self.__u, self.__p = TrialFunction(V), TrialFunction(Q)
        self.__v, self.__q = TestFunction(V), TestFunction(Q)

        u, p = self.__u, self.__p
        v, q = self.__v, self.__q

        self.v0 = Function(V) # Test function used for residual calculations
        self.v0.vector()[:] = 1.0
        v0 = self.v0

        self.__u0, self.__u1 = Function(V), Function(V)
        self.__p0 = interpolate(self.init_cond_pressure, Q)
        self.__p1 = Function(Q)

        u0, u1, p0, p1 = self.__u0, self.__u1, self.__p0, self.__p1

        u0.rename('velocity', 'fluid velocity')
        p0.rename('pressure', 'fluid pressure')

        mu, k, rho, g = self._mu, self._k, self._rho, self._g

        self.r = Constant(0.0)
        self.omega = Constant(1.0)
        r, omega = self.r, self.omega

        n = self.n
        dx, ds, dS = self.dx, self.ds, self.dS

        self.form_update_velocity = mu/k*inner(v, u)*dx \
                                    + r*inner(div(v), div(u))*dx \
                                    - inner(p0, div(v))*dx \
                                    - inner(v, rho*g)*dx

        self.form_update_pressure = q*p*dx - q*p0*dx + omega*q*(div(u1))*dx

        self.residual_momentum_form = mu/k*inner(v0, u0)*dx \
                                      - inner(div(v0), p0)*dx \
                                      - inner(v0, rho*g)*dx
        self.residual_mass_form = q*div(u0)*dx

        for i, marker in enumerate(self.__boundary_dict['pressure']):
            self.form_update_velocity += pressure_bc_val[i]*inner(n, v) \
                                         *ds(marker)

            self.residual_momentum_form += pressure_bc_val[i]*inner(n, v0) \
                                           *ds(marker)

    def add_momentum_source(self, sources: list):
        v = self.__v
        v0 = self.v0

        for source in sources:
            self.form_update_velocity -= inner(v, source)*self.dx
            self.residual_momentum_form -= inner(v0, source)*self.dx

    def set_uzawa_parameters(self, r_val: float, omega_val: float):
        """When r = 0, it converges for omega < 2. Try 1.5 first.
        For 0 < omega < 2r, the augmented system converges.
        One can choose r >> 1.
        """

        self.r.assign(r_val)
        self.omega.assign(omega_val)

    def set_velocity_bc(self, velocity_bc_val: list):
        """
        Arguments
        ---------
        velocity_bc_val : list of Constants,
                          e.g., [Constant((1.0, -1.0)), Constant((0.0, -2.0))]
        """

        self.velocity_bc = []

        for i, marker in enumerate(self.__boundary_dict['velocity']):
            self.velocity_bc.append(DirichletBC(self.velocity_func_space,
                                                velocity_bc_val[i],
                                                self.boundary_markers, marker))

    def get_residual(self):
        """"""

        #TODO: Fix this method.
        residual_momentum = assemble(self.residual_momentum_form)
        residual_mass = assemble(self.residual_mass_form)

        for bc in self.velocity_bc:
            bc.apply(self.__u0.vector())

        residual = residual_momentum#.norm('l2')
        residual += residual_mass.norm('l2')

        return residual

    def get_relative_error(self):
        """"""

        relative_error = assemble((self.__u1 - self.__u0)**2*self.dx) \
                         /(assemble(self.__u0**2*self.dx) + DOLFIN_EPS)
        relative_error += assemble((self.__p1 - self.__p0)**2*self.dx) \
                          /(assemble(self.__p0**2*self.dx) + DOLFIN_EPS)

        return relative_error


    def assemble_matrix(self):
        F_velocity = self.form_update_velocity
        F_pressure = self.form_update_pressure

        a_v, self.L_v = lhs(F_velocity), rhs(F_velocity)
        a_p, self.L_p = lhs(F_pressure), rhs(F_pressure)

        self.A_v = assemble(a_v)
        self.A_p = assemble(a_p)
        self.b_v, self.b_p = PETScVector(), PETScVector()

    def set_solver(self):
        # Users can override this method.
        # Or, TODO: make this method more user friendly.

        self.solver_v = PETScKrylovSolver('gmres', 'ilu')
        self.solver_p = PETScKrylovSolver('gmres', 'amg')

        prm_v = self.solver_v.parameters
        prm_p = self.solver_p.parameters

        TransportProblemBase.set_default_solver_parameters(prm_v)
        TransportProblemBase.set_default_solver_parameters(prm_p)

    def solve_flow(self, target_error: float, max_steps: int):
        steps = 0
        relative_error = target_error + 1

        while relative_error > target_error and steps < max_steps:
            assemble(self.L_v, tensor=self.b_v)
            for bc in self.velocity_bc:
                bc.apply(self.A_v, self.b_v)

            self.solver_v.solve(self.A_v, self.__u1.vector(), self.b_v)

            assemble(self.L_p, tensor=self.b_p)
            self.solver_p.solve(self.A_p, self.__p1.vector(), self.b_p)

            relative_error = self.get_relative_error()
            steps+=1

            self.__u0.assign(self.__u1)
            self.__p0.assign(self.__p1)

        self.fluid_velocity.assign(self.__u0)
        self.fluid_pressure.assign(self.__p0)
