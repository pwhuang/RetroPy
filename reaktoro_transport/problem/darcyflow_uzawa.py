from . import *

class DarcyFlowUzawa(TransportProblemBase, DarcyFlowBase):
    """This class utilizes the Augmented Lagrangian Uzawa method to solve
    the pressure and velocity of Darcy flow.
    """

    def __init__(self, mesh, boundary_markers, domain_markers):
        DarcyFlowBase.__init__(self, mesh, boundary_markers, domain_markers)
        self.init_cond_pressure = Constant(0.0)

    def generate_form(self):
        """Sets up the FeNiCs form of Darcy flow."""

        V = self.velocity_func_space
        Q = self.pressure_func_space

        self.__u, self.__p = TrialFunction(V), TrialFunction(Q)
        self.__v, self.__q = TestFunction(V), TestFunction(Q)

        u, p = self.__u, self.__p
        v, q = self.__v, self.__q

        self.__u0 = self.fluid_velocity
        self.fluid_pressure.assign(interpolate(self.init_cond_pressure, Q))
        self.__p0 = self.fluid_pressure
        self.__u1, self.__p1 = Function(V), Function(Q)

        u0, u1, p0, p1 = self.__u0, self.__u1, self.__p0, self.__p1

        mu, k, rho, g, phi = self._mu, self._k, self._rho, self._g, self._phi

        self.__r = Constant(1.0)
        self.omega = Constant(1.0)
        r, omega = self.__r, self.omega

        n = self.n
        dx, ds, dS = self.dx, self.ds, self.dS

        self.form_update_velocity = mu/k*inner(v, u)*dx \
                                    + r*inner(div(v), div(rho*phi*u))*dx \
                                    - inner(p0, div(v))*dx \
                                    - inner(v, rho*g)*dx

        self.form_update_pressure = q*(p-p0)*dx + omega*q*(div(rho*phi*u1))*dx

        for i, marker in enumerate(self.darcyflow_boundary_dict['pressure']):
            self.form_update_velocity += self.pressure_bc[i]*inner(n, v) \
                                         *ds(marker)

        self.functions_to_save = [self.fluid_pressure, self.fluid_velocity]

    def add_mass_source(self, sources: list):
        q, v, r, omega = self.__q, self.__v, self.__r, self.omega
        dx = self.dx

        for source in sources:
            self.form_update_velocity -= r*inner(div(v), source)*dx
            self.form_update_pressure -= q*omega*source*dx

    def add_momentum_source(self, sources: list):
        v = self.__v

        for source in sources:
            self.form_update_velocity -= inner(v, source)*self.dx

    def set_additional_parameters(self, r_val: float, omega_by_r: float):
        """For 0 < omega/r < 2, the augmented system converges."""

        self.__r.assign(r_val)
        self.omega.assign(r_val*omega_by_r)

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

    def set_solver(self, **kwargs):
        # Users can override this method.
        # Or, TODO: make this method more user friendly.

        self.solver_v = PETScKrylovSolver('bicgstab', 'sor')
        self.solver_p = PETScKrylovSolver('gmres', 'none')

        prm_v = self.solver_v.parameters
        prm_p = self.solver_p.parameters

        TransportProblemBase.set_default_solver_parameters(prm_v)
        TransportProblemBase.set_default_solver_parameters(prm_p)

    def solve_flow(self, target_residual: float, max_steps: int):
        steps = 0

        while (residual := self.get_residual()) > target_residual\
               and steps < max_steps:

            info('Darcy flow residual = ' + str(residual))
            
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

        print('Steps used: ', steps)
